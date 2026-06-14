import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer
from .model_builder import TransformerLM, get_modules_info
from .trainer import Trainer
from .experiment_store import ExperimentStore
from .gpu_monitor import get_gpu_snapshot

logger = logging.getLogger(__name__)


class _SuppressGpuAccessLog(logging.Filter):
    marker = "architecture_lab_suppress_gpu_access_log"

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args if isinstance(record.args, tuple) else ()
        if len(args) >= 3:
            method = args[1]
            path = args[2]
            if method == "GET" and isinstance(path, str) and path.split("?", 1)[0] == "/api/gpu":
                return False

        message = record.getMessage()
        return '"GET /api/gpu ' not in message and '"GET /api/gpu?' not in message


def _install_access_log_filter():
    access_logger = logging.getLogger("uvicorn.access")
    already_installed = any(
        getattr(item, "marker", None) == _SuppressGpuAccessLog.marker
        for item in access_logger.filters
    )
    if not already_installed:
        access_logger.addFilter(_SuppressGpuAccessLog())


_install_access_log_filter()


DATA_PATH = Path(__file__).parent.parent / "data" / "train.bin"  # 默认训练数据路径
EXPERIMENTS_PATH = Path(__file__).parent.parent / "data" / "experiments.json"  # 实验记录存储路径
TOKENIZER_PATH = Path(__file__).parent / "tokenizer"
DEFAULT_VOCAB_SIZE = len(AutoTokenizer.from_pretrained(str(TOKENIZER_PATH)))
trainer = Trainer()
experiment_store = ExperimentStore(EXPERIMENTS_PATH)
ws_connections: list[WebSocket] = []  # 当前在线的 WebSocket 客户端连接列表


# 应用生命周期管理，确保在应用关闭时停止训练任务
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if trainer.is_training():
        trainer.stop()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])  # 允许跨域请求


# ------------- 字段定义 -------------
class LayerConfig(BaseModel):
    attention_type: str
    attention_params: dict = {}
    ffn_type: str
    ffn_params: dict = {}


class ModelConfig(BaseModel):
    hidden_size: int = 256
    vocab_size: int = DEFAULT_VOCAB_SIZE
    max_seq_len: int = 64
    rms_norm_eps: float = 1e-6
    share_embedding_head: bool = False
    layers: list[LayerConfig]


class TrainConfig(BaseModel):
    batch_size: int = 16
    learning_rate: float = 3e-4
    max_steps: Optional[int] = None
    warmup_steps: int = 50
    data_order_seed: int = 0
    model_init_seed: int = 0


class TrainRequest(BaseModel):
    model_cfg: ModelConfig
    train_config: TrainConfig = TrainConfig()
    run_name: Optional[str] = None
    param_count: Optional[int] = None


class ExperimentRecord(BaseModel):
    data: dict


class ClearFinishedTrainRequest(BaseModel):
    run_id: str


# ------------- API 端点定义 -------------
@app.get("/api/modules")
def api_modules():
    return get_modules_info()


@app.get("/api/defaults")
def api_defaults():
    return {
        "model": {
            "vocab_size": DEFAULT_VOCAB_SIZE,
        }
    }


@app.get("/api/gpu")
def api_gpu():
    return get_gpu_snapshot()


@app.post("/api/estimate_params")
def api_estimate_params(config: ModelConfig):
    try:
        model = TransformerLM(config.model_dump())
        return model.parameter_breakdown()
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/api/train")
async def api_train(req: TrainRequest):
    if trainer.is_training():
        return JSONResponse(status_code=409, content={"error": "Training already in progress"})

    model_config = req.model_cfg.model_dump()
    train_config = req.train_config.model_dump()
    try:
        param_count = TransformerLM(model_config).parameter_breakdown()["param_count"]
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    loop = asyncio.get_event_loop()

    def remove_ws(ws: WebSocket):
        if ws in ws_connections:
            ws_connections.remove(ws)

    def broadcast(data: dict):
        for ws in ws_connections[:]:
            try:
                future = asyncio.run_coroutine_threadsafe(ws.send_json(data), loop)
            except Exception:
                remove_ws(ws)
                continue

            def discard_failed_send(done_future, target_ws=ws):
                try:
                    done_future.result()
                except Exception:
                    loop.call_soon_threadsafe(remove_ws, target_ws)

            future.add_done_callback(discard_failed_send)

    try:
        run_id = trainer.train(
            model_config,
            train_config,
            DATA_PATH,
            broadcast,
            broadcast,
            run_name=req.run_name,
            param_count=param_count,
        )
    except RuntimeError as e:
        return JSONResponse(status_code=409, content={"error": str(e)})
    return {"status": "started", "run_id": run_id}


@app.get("/api/train/status")
def api_train_status():
    return trainer.snapshot()


@app.post("/api/train/clear_finished")
def api_clear_finished_train(req: ClearFinishedTrainRequest):
    if trainer.clear_finished(req.run_id):
        return {"status": "cleared"}
    return {"status": "not_cleared"}


@app.post("/api/stop")
def api_stop():
    if trainer.is_training():
        trainer.stop()
        return {"status": "stopping"}
    return {"status": "not_training"}


@app.get("/api/experiments")
def api_list_experiments():
    return experiment_store.list()


@app.post("/api/experiments")
def api_save_experiment(record: ExperimentRecord):
    try:
        return experiment_store.save(record.data)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.patch("/api/experiments/{experiment_id}")
def api_update_experiment(experiment_id: str, record: ExperimentRecord):
    result = experiment_store.update(experiment_id, record.data)
    if result is None:
        return JSONResponse(status_code=404, content={"error": "Experiment not found"})
    return result


@app.delete("/api/experiments/{experiment_id}")
def api_delete_experiment(experiment_id: str):
    if experiment_store.delete(experiment_id):
        return {"status": "deleted"}
    return JSONResponse(status_code=404, content={"error": "Experiment not found"})


@app.websocket("/ws/train")
async def ws_train(websocket: WebSocket):
    await websocket.accept()
    ws_connections.append(websocket)
    try:
        await websocket.send_json(trainer.snapshot())
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in ws_connections:
            ws_connections.remove(websocket)
    except Exception:
        logger.exception("Training WebSocket connection failed")
        if websocket in ws_connections:
            ws_connections.remove(websocket)


# 使用前端构建产物作为静态文件服务，让后端服务器直接提供前端页面服务
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="static")
