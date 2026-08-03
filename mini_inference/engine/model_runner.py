import pickle

import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from mini_inference.config import Config
from mini_inference.engine.sequence import Sequence
from mini_inference.models.mini_llama3 import MiniLlama3ForCausalLM
from mini_inference.models.qwen3 import Qwen3ForCausalLM
from mini_inference.layers.sampler import Sampler
from mini_inference.utils.context import set_context, get_context, reset_context
from mini_inference.utils.loader import load_model


MODEL_REGISTRY = {
    "mini_llama3": MiniLlama3ForCausalLM,
    "qwen3": Qwen3ForCausalLM,
}


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager  # 走 eager 模式，不使用 CUDA graph 优化
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        model_class = MODEL_REGISTRY.get(hf_config.model_type)
        if model_class is None:
            raise ValueError(f"Unsupported model type: {hf_config.model_type}")
        if hf_config.model_type == "mini_llama3" and self.world_size != 1:
            raise ValueError("MiniLlama3 inference only supports tensor_parallel_size=1")

        self.use_distributed = hf_config.model_type != "mini_llama3"
        if self.use_distributed:
            dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()  # 当前默认的浮点类型，通常是 torch.float32
        torch.set_default_dtype(config.dtype)      # Flash Attention 要求模型使用 torch.float16 或 torch.bfloat16
        torch.set_default_device("cuda")           # 设置默认的设备为 CUDA，这样后续创建的张量会默认分配到 GPU 上
        self.model = model_class(
            hf_config,
            attention_backend=config.attention_backend,
        )
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()  # 预热模型
        self.allocate_kv_cache()  # 分配 kv cache
        if not self.enforce_eager:
            self.capture_cudagraph()  # 捕获 CUDA graph，优化推理性能
        torch.set_default_device("cpu")         # 初始化完成后，将默认设备设置回 CPU
        torch.set_default_dtype(default_dtype)  # 初始化完成后，将默认浮点类型设置回原来的默认值

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="mini_inference", create=True, size=2**20)  # 创建共享内存对象，大小为 1MB
                dist.barrier()  # 所有进程同步后，再进行下一步执行
            else:
                dist.barrier()
                self.shm = SharedMemory(name="mini_inference")  # 连接到 rank 0 创建的共享内存对象
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()  # 删除共享内存对象，释放资源
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()  # 确保所有 CUDA 操作完成，避免在进程退出时出现未完成的 CUDA 操作
        if self.use_distributed:
            dist.destroy_process_group()  # 销毁分布式进程组，释放资源

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()  # 阻塞等待
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()  # 清除事件标志，准备下一次等待
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])  # 将方法名和参数序列化为字节流
        n = len(data)  # 数据的字节长度
        self.shm.buf[0:4] = n.to_bytes(4, "little")  # 先把数据长度写入共享内存的前 4 个字节
        self.shm.buf[4:n+4] = data  # 再把数据写入共享内存的后续字节
        for event in self.event:
            event.set()  # 唤醒所有等待的子进程，通知它们共享内存中有新的数据可读

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)  # 如果是多进程，则由 rank 0 的进程将方法名和参数写入共享内存，通知其他进程
        method = getattr(self, method_name, None)  # 获取指定名称的方法对象
        return method(*args)

    def warmup_model(self):
        # 首先清空 GPU 缓存和峰值计数
        torch.cuda.empty_cache()  # 清空 GPU 缓存，释放未使用的显存
        torch.cuda.reset_peak_memory_stats()  # 重置 GPU 的峰值显存使用统计信息
        
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        # 选取一个合法且尽量大的单序列长度
        # seq_len 必须同时满足 seq_len <= max_num_batched_tokens 和 seq_len <= max_model_len，因此取两者的最小值
        seq_len = min(max_num_batched_tokens, max_model_len)
        # 计算序列数量，确保总的 token 数量不超过 max_num_batched_tokens，同时不超过配置中的最大序列数
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]  # 创建 num_seqs 个长度为 seq_len 的 warmup 序列
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)  # 执行 prefill warmup
        
        torch.cuda.empty_cache()  # 再次清空 GPU 缓存，释放未使用的显存，此时有新统计的峰值显存使用信息
        
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()  # 获取当前 GPU 的可用显存和总显存
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]  # 获取自上次 reset_peak_memory_stats() 以来的峰值显存使用量
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]  # 获取当前由 PyTorch 张量实际分配中的字节数，主要是本进程里 PyTorch 管的 allocated 内存，不等于整卡已用
        
        num_kv_heads = hf_config.num_key_value_heads // self.world_size  # 每个 GPU 上的 kv 头数
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)  # 每个头的维度
        # 计算单个 block 需要多少字节显存，其中 dtype.itemsize 表示每个元素的字节数，例如 fp16=2，bf16=2，fp32=4
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * self.config.dtype.itemsize
        # 计算当前 GPU 上可以分配的 kv cache block 数量，num_kvcache_blocks = budget_bytes // block_bytes
        # 其中， budget_bytes = (total * gpu_memory_utilization - used) - (peak - current)
        # (total * gpu_memory_utilization - used) 表示当前配额下可用的显存
        # (peak - current) 表示 warmup 期间中间激活、通信等瞬时涨幅的显存占用
        # 因此 budget_bytes 对应的是常驻 kv cache 的显存预算，(peak - current) 是预留的瞬时峰值余量，防止 OOM
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        
        # kv cache 的形状为 (2, num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim)
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            dtype=config.dtype,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                # 将每层的 k_cache 和 v_cache 指向 kv_cache 中对应的切片
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)  # 计算本轮调度的序列中，block_table 的最大长度
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]  # 将每条序列的 block_table 补齐到最大长度，未使用的部分填充为 -1
        # 将 block_tables 转换为张量，pin_memory=True + cuda(non_blocking=True) 是常见的尽量把数据搬运与计算重叠的标准搭配
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []  # 记录本次调度的所有 token id，形状为 (num_batched_tokens,) 例如: [a1, a2, a3, b1, b2, b3, b4, c1, c2]
        positions = []  # 记录本次调度的所有 token 的位置索引，形状为 (num_batched_tokens,) 例如: [0, 1, 2, 0, 1, 2, 3, 0, 1]
        cu_seqlens_q = [0]  # 记录每条序列的累计长度，形状为 (num_seqs + 1,) 例如 [0, 3, 7, 9] 表示有 3 条序列，长度分别为 3、4、2
        cu_seqlens_k = [0]  # 记录每条序列 kv 的累计长度(=num_cached_tokens+num_scheduled_tokens)，形状为 (num_seqs+1,)
                            # 无 prefix cache 时通常与 cu_seqlens_q 相同，例如 [0,3,7,9];
                            # 有 prefix cache 时会更大，例如若三条序列 cached 分别为 [5,10,20]、本轮 q 长度为 [3,4,2]，则 cu_seqlens_k=[0,8,22,44]
        max_seqlen_q = 0  # 记录本轮调度的最大序列长度，例如 4
        max_seqlen_k = 0  # 记录本轮调度的最大 kv 序列长度，例如 22 (对应 cached[5,10,20] + q[3,4,2])
        slot_mapping = []  # 记录本轮调度的每个 token 在 kv cache 中的物理槽位索引，形状为 (num_batched_tokens,) positions 是逻辑位置，而 slot_mapping 是物理位置
        block_tables = None  # 记录本轮调度的每条序列的 block table，形状为 (num_seqs, max_num_blocks)
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens  # 本轮需要新计算的 token 数量
            end = start + seqlen_q
            seqlen_k = end  # 本轮的 kv length = cached + q
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:  # warmup 不写入 kv cache 池
                continue
            start_block = start // self.block_size  # 逻辑 block 起始索引，向下取整
            end_block = (end + self.block_size - 1) // self.block_size  # 逻辑 block 结束索引，向上取整
            for i in range(start_block, end_block):  # 至少会循环一次
                # block_table[i] 是逻辑 block i 对应的物理 block id，乘以 block_size 得到该 block 在 kv cache 中的起始槽位索引
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:  # 如果是第一个 block，要跳过此 block 中已经缓存的 token
                    slot_start += start % self.block_size
                if i != end_block - 1:  # 如果不是最后一个 block，则 slot_end 为该 block 的末尾
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:  # 如果是最后一个 block，则 slot_end 为本轮真正结束的位置，不一定写满整个 block
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # 此时说明有 prefix cache
            # 如果有 prefix cache，内核需要知道每条序列的历史块编号映射
            # 没有 prefix cache 时，block_tables 传 None，kv 就是新算出来的这部分
            block_tables = self.prepare_block_tables(seqs)
        # 将 input_ids、positions、cu_seqlens_q、cu_seqlens_k、slot_mapping 转换为张量，并移动到 GPU 上
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))  # 每条序列的当前上下文长度
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        has_greedy = any(seq.temperature == 0 for seq in seqs)
        all_greedy = all(seq.temperature == 0 for seq in seqs)
        use_temperature = any(seq.temperature > 0 and seq.temperature != 1.0 for seq in seqs)
        use_filter = not all_greedy and any(seq.top_k > 0 or 0.0 < seq.top_p < 1.0 for seq in seqs)
        use_penalty = any(seq.repetition_penalty != 1.0 or seq.frequency_penalty != 0.0 for seq in seqs)

        temperatures = None
        if has_greedy or use_temperature:
            temperatures = torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

        top_ks = top_ps = None
        if use_filter:
            top_ks = torch.tensor([seq.top_k for seq in seqs], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            top_ps = torch.tensor([seq.top_p for seq in seqs], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

        repetition_penalties = frequency_penalties = None
        penalty_token_ids = penalty_token_counts = None
        if use_penalty:
            repetition_penalties = torch.tensor([seq.repetition_penalty for seq in seqs], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
            frequency_penalties = torch.tensor([seq.frequency_penalty for seq in seqs], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

            token_counts = [seq.completion_token_counts for seq in seqs]
            max_unique_tokens = max(1, max(len(counts) for counts in token_counts))  # len(counts) 是该序列里已生成过的不同 token 数，这里取批内最大值
            penalty_token_ids = []
            penalty_token_counts = []
            for counts in token_counts:
                pad_size = max_unique_tokens - len(counts)
                penalty_token_ids.append([*counts.keys(), *([-1] * pad_size)])
                penalty_token_counts.append([*counts.values(), *([0] * pad_size)])
            penalty_token_ids = torch.tensor(penalty_token_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            penalty_token_counts = torch.tensor(penalty_token_counts, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

        return (
            temperatures,
            top_ks,
            top_ps,
            repetition_penalties,
            frequency_penalties,
            penalty_token_ids,
            penalty_token_counts,
            has_greedy,
            all_greedy,
            use_temperature,
            use_filter,
            use_penalty,
        )

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # 如果是 prefill 阶段、强制使用 eager 模式，或者 batch size 超过 512，则直接使用 eager 模式计算 logits
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # 否则使用 CUDA Graph 优化 decode 阶段的前向计算
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]  # 选择能够容纳当前 batch size 的最小 CUDA Graph
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()  # 调用 CUDA Graph 的 replay 方法，执行捕获的 GPU 操作
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # 执行模型前向计算，返回每个序列的下一个 token 的 id
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        sampling_metadata = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        if self.rank == 0:
            token_ids = self.sampler(logits, *sampling_metadata).tolist()
        else:
            token_ids = None  # 其他 rank 只负责计算 logits
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        预捕获不同 batch size 下的 CUDA Graph, 用于加速 LLM 的 decode 阶段

        - CUDA Graph 会将一次模型前向过程中提交到 GPU 的整套操作, 包括各个 CUDA kernel、执行顺序、依赖关系以及输入输出显存地址, 
        记录为一张可重复执行的图; 后续推理时只需要调用 graph.replay(), 即可整体提交这批操作, 避免 PyTorch 和 CPU 每一步重新
        调度、逐个发射大量 CUDA kernel

        - CUDA Graph 主要减少的是 CPU 端的 kernel launch 开销, LLM decode 每个序列每轮通常只处理一个 token, 单个 kernel 计算
        量较小, 因此 launch 开销占比较高, 特别适合使用 CUDA Graph

        - CUDA Graph 捕获后要求计算结构、Tensor shape 和显存地址保持不变, 因此一张图不能直接处理任意 batch size, 通常预先为一系
        列离散 batch size 分别捕获: [1, 2, 4, 8, 16, 32, 48, ..., max_bs]; 推理时根据实际请求数量, 选择能够容纳它的最小 graph
        batch size; 例如实际 batch size 为 13, 可以使用 batch size 16 的图, 并将多出的 3 个位置作为 padding 或无效槽位处理

        - graph_vars 中的 Tensor 必须在 CUDA Graph 整个生命周期内保持存在, 因为图中记录的是这些 Tensor 对应的固定显存地址, 不能在
        每次推理时重新创建输入 Tensor

        - 这里按 batch size 从大到小捕获, 并让所有图共享 graph_pool, 可以让不同 CUDA Graph 尽量复用同一个私有显存池, 降低为多张图
        分别保留临时显存所带来的总显存开销
        
        捕获流程如下:

        1. 创建固定显存地址的 input_ids、positions、KV Cache 元数据和输出缓冲区;
        2. 对每个 batch size 先执行一次 warmup, 使 CUDA context、kernel、workspace 和临时显存分配等延迟初始化操作在捕获前完成;
        3. 在 torch.cuda.graph 上下文中再次执行模型前向并记录 GPU 操作;
        4. 保存每个 batch size 对应的 CUDA Graph;
        5. 推理时将新请求的数据复制到这些固定缓冲区, 再调用 graph.replay()
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)  # 限制最大 batch size 为 512
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size  # 单条序列最多需要多少个 block 索引位
        
        # graph_vars
        input_ids = torch.zeros(max_bs, dtype=torch.int64)  # 由于是 decode 阶段的 CUDA Graph, 每个序列每轮只处理一个 token, 因此 shape 为 (max_bs,)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)  # 当前这个新 token 要写入 kv cache 的物理槽位索引
        context_lens = torch.zeros(max_bs, dtype=torch.int32)  # 当前上下文长度，即当前序列已经写入 kv cache 的 token 数量
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)  # (max_bs, max_num_blocks)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)  # 输出 token 的 hidden states, shape 为 (max_bs, hidden_size)
        
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))  # 预定义一组 batch size，用于捕获不同 batch size 下的 CUDA graph
        self.graphs = {}
        self.graph_pool = None  # 用于存放 CUDA Graph 的私有显存池

        # 这里进行反向遍历， 先捕获大 batch size 的 CUDA Graph, 再捕获小 batch size 的 CUDA Graph
        # 这样可以让不同 batch size 的图尽量复用同一个 graph_pool, 降低显存开销
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()  # 创建一个新的 CUDA Graph 对象
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()  # 第一次捕获的 graph_pool 用于后续所有 CUDA Graph 的显存分配
            self.graphs[bs] = graph  # 保存当前 batch size 对应的 CUDA Graph
            torch.cuda.synchronize()  # 确保当前 CUDA Graph 捕获完成，避免后续捕获被干扰
            reset_context()  # 重置上下文

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
