from __future__ import annotations
import copy
import logging
import math
import random
import time
import threading
import uuid
from typing import Callable, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model_builder import TransformerLM
from .dataset.dataset import PretrainDataset

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._reset_state_locked()

    def is_training(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _reset_state_locked(self):
        self._run_id: str | None = None
        self._status = "idle"
        self._loss_history: list[dict] = []
        self._progress: dict | None = None
        self._done: dict | None = None
        self._error: dict | None = None
        self._model_config: dict | None = None
        self._train_config: dict | None = None
        self._run_name: str | None = None
        self._param_count: int | None = None
        self._started_at: float | None = None
        self._updated_at: float | None = None

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "type": "snapshot",
                "run_id": self._run_id,
                "status": self._status,
                "training": self.is_training(),
                "loss_history": copy.deepcopy(self._loss_history),
                "progress": copy.deepcopy(self._progress),
                "done": copy.deepcopy(self._done),
                "error": copy.deepcopy(self._error),
                "model_config": copy.deepcopy(self._model_config),
                "train_config": copy.deepcopy(self._train_config),
                "run_name": self._run_name,
                "param_count": self._param_count,
                "started_at": self._started_at,
                "updated_at": self._updated_at,
            }

    def clear_finished(self, run_id: str) -> bool:
        with self._lock:
            if self._run_id != run_id or self._status not in {"done", "error"} or self.is_training():
                return False
            self._reset_state_locked()
            return True

    def train(
        self,
        model_config: dict,
        train_config: dict,
        data_path: str,
        on_step: Callable[[dict], None],
        on_done: Callable[[dict], None],
        run_name: str | None = None,
        param_count: int | None = None,
    ):
        with self._lock:
            if self.is_training():
                raise RuntimeError("Training already in progress")
            self._stop_event.clear()
            self._reset_state_locked()
            self._run_id = uuid.uuid4().hex
            self._status = "running"
            self._model_config = copy.deepcopy(model_config)
            self._train_config = copy.deepcopy(train_config)
            self._run_name = run_name
            self._param_count = param_count
            self._started_at = time.time()
            self._updated_at = self._started_at
            self._thread = threading.Thread(
                target=self._train_loop,
                args=(model_config, train_config, data_path, on_step, on_done),
                daemon=True,
            )
        self._thread.start()
        return self._run_id

    def stop(self):
        self._stop_event.set()
        with self._lock:
            if self._status == "running":
                self._status = "stopping"
                self._updated_at = time.time()

    def _notify(self, callback: Callable[[dict], None], data: dict):
        try:
            callback(data)
        except Exception:
            logger.exception("Training callback failed")

    def _record_step(self, data: dict, on_step: Callable[[dict], None]):
        payload = {**data, "run_id": self._run_id}
        with self._lock:
            self._loss_history.append({
                "step": payload["step"],
                "loss": payload["loss"],
                "lr": payload["lr"],
            })
            self._progress = copy.deepcopy(payload)
            self._status = "stopping" if self._stop_event.is_set() else "running"
            self._updated_at = time.time()
        self._notify(on_step, payload)

    def _record_done(self, data: dict, on_done: Callable[[dict], None]):
        payload = {**data, "run_id": self._run_id}
        with self._lock:
            if payload.get("type") == "error":
                self._error = copy.deepcopy(payload)
                self._status = "error"
            else:
                self._done = copy.deepcopy(payload)
                self._status = "done"
            self._updated_at = time.time()
        self._notify(on_done, payload)

    def _train_loop(
        self,
        model_config: dict,
        train_config: dict,
        data_path: str,
        on_step: Callable[[dict], None],
        on_done: Callable[[dict], None],
    ):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            max_steps = train_config["max_steps"]
            
            # 固定随机数种子，确保不同架构可对比
            model_init_seed = train_config.get("model_init_seed", 0)
            data_order_seed = train_config.get("data_order_seed", 0)
            random.seed(model_init_seed)
            np.random.seed(model_init_seed)
            torch.manual_seed(model_init_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(model_init_seed)
            
            model = TransformerLM(model_config).to(device)
            seq_len = model_config["max_seq_len"]
            dataset = PretrainDataset(data_path, seq_len)
            dataset_total_steps = len(dataset) // train_config["batch_size"]

            # max_steps 为 None 时，训完一个 epoch
            if max_steps is None:
                max_steps = dataset_total_steps

            if max_steps == 0:
                self._record_done({
                    "type": "done",
                    "final_loss": None,
                    "target_total_steps": 0,
                    "dataset_total_steps": dataset_total_steps,
                    "elapsed_seconds": 0.0,
                    "total_steps": 0,
                    "stopped_early": False,
                }, on_done)
                return
            if dataset_total_steps == 0:
                raise ValueError("Dataset does not contain a full batch for the current seq_len and batch_size")
            
            data_generator = torch.Generator()
            data_generator.manual_seed(data_order_seed)
            dataloader = DataLoader(
                dataset,
                batch_size=train_config["batch_size"],
                shuffle=True,
                drop_last=True,
                generator=data_generator,
            )

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=train_config["learning_rate"],
                weight_decay=0.01,
            )

            warmup_steps = train_config.get("warmup_steps", max_steps // 20)

            step = 0
            start_time = time.time()
            data_iter = iter(dataloader)
            loss_value = None

            while step < max_steps and not self._stop_event.is_set():
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    try:
                        batch = next(data_iter)
                    except StopIteration as exc:
                        raise ValueError("Dataset does not contain a full batch for the current seq_len and batch_size") from exc

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                if step < warmup_steps:
                    lr = train_config["learning_rate"] * (step + 1) / warmup_steps
                else:
                    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
                    lr = train_config["learning_rate"] * 0.5 * (1 + math.cos(math.pi * progress))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"]
                main_loss = outputs.get("main_loss", loss)
                loss_value = main_loss.item()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                step += 1
                elapsed = time.time() - start_time
                tokens_per_sec = step * train_config["batch_size"] * seq_len / elapsed if elapsed > 0 else 0
                eta_seconds = max(0.0, elapsed / step * max_steps - elapsed) if step > 0 else None
                progress_pct = step / max_steps * 100 if max_steps > 0 else 100.0

                self._record_step({
                    "type": "step",
                    "step": step,
                    "target_total_steps": max_steps,
                    "dataset_total_steps": dataset_total_steps,
                    "progress_pct": round(progress_pct, 2),
                    "elapsed_seconds": round(elapsed, 1),
                    "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
                    "loss": round(main_loss.item(), 4),
                    "lr": round(lr, 6),
                    "tokens_per_sec": round(tokens_per_sec, 0),
                }, on_step)

            elapsed = time.time() - start_time
            done_payload = {
                "type": "done",
                "final_loss": round(loss_value, 4) if loss_value is not None else None,
                "target_total_steps": max_steps,
                "dataset_total_steps": dataset_total_steps,
                "elapsed_seconds": round(elapsed, 1),
                "total_steps": step,
                "stopped_early": self._stop_event.is_set(),
            }

            # 释放显存
            del model, optimizer, dataloader, dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._record_done(done_payload, on_done)
        except Exception as e:
            logger.exception("Training failed")
            # 异常时也尽量释放显存
            if "model" in dir():
                del model
            if "optimizer" in dir():
                del optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._record_done({"type": "error", "message": str(e)}, on_done)
