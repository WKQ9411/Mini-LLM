import atexit
from collections.abc import Iterator
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from mini_inference.config import Config
from mini_inference.sampling_params import SamplingParams
from mini_inference.engine.sequence import Sequence
from mini_inference.engine.scheduler import Scheduler
from mini_inference.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, tokenizer_path=None, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)  # 只保留 Config 中定义的参数，其他参数忽略
        Sequence.block_size = config.kvcache_block_size
        self.ps = []  # 存储子进程对象的列表
        self.events = []  # 存储主进程和子进程之间的事件对象的列表，用于进程间通信和同步
        ctx = mp.get_context("spawn")  # 指定多进程启动方式为 spawn，启动子进程时，不继承父进程当前运行状态，而是新开一个干净解释器，再导入模块并执行目标函数
        for i in range(1, config.tensor_parallel_size):  # 如果 tp 为 1，则此循环被跳过
            event = ctx.Event()  # 每个子进程都有一个独立的事件对象，用于主进程和子进程之间的通信和同步
            process = ctx.Process(target=ModelRunner, args=(config, i, event))  # 子进程对象为 ModelRunner
            process.start()
            self.ps.append(process)  # 将子进程对象添加到子进程列表中
            self.events.append(event)  # 将事件对象添加到事件列表中
        self.model_runner = ModelRunner(config, 0, self.events)  # rank 为 0 的进程为主进程，传入的是事件对象列表
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)  # 注册退出函数，在程序退出时调用 self.exit()，确保子进程被正确终止

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()  # 等待子进程结束，确保所有子进程都被正确终止

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> Sequence:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)  # 将字符串 prompt 转换为 token id 列表
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()  # 获取本次调度的序列列表和是否为 prefill 的标志
        # num_tokens 用于吞吐统计，prefill 时直接计算本次总调度的 token 数量
        # decode 时由于只调度 1 个 token，因此总的就是 len(seqs)，加负号是用于后续区分 num_tokens 是属于 prefill 还是 decode
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        # 这里产生的 token_ids 有以下几层含义：
        # 1. 对于 decode 阶段，很好理解，就是模型生成的新的 token
        # 2. 对于完整的 prefill，这里的 token_ids 是基于完整 prompt 生成的第一个输出 token
        # 3. 对于 chunked prefill，这里的 token_ids 是基于部分 prompt 生成的第一个输出 token
        #    但后面其实还有 prompt 没处理完，所以这个 token 会被丢弃，参考 scheduler postprocess() 中的逻辑
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]  # 获取已经生成完毕的 seq
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def stream_generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
    ) -> Iterator[str]:
        if not self.is_finished():
            raise RuntimeError("stream_generate only supports one request on an idle engine")

        seq = self.add_request(prompt, sampling_params)
        num_streamed_tokens = 0
        pending_token_ids = []

        while not seq.is_finished:
            self.step()
            new_token_ids = seq.completion_token_ids[num_streamed_tokens:]
            if not new_token_ids:
                # Chunked Prefill 的临时预测不会 append 到 Sequence，因此这里没有可输出 token
                continue
            num_streamed_tokens += len(new_token_ids)
            pending_token_ids.extend(new_token_ids)
            text = self.tokenizer.decode(pending_token_ids, skip_special_tokens=True)
            if text.endswith("\uFFFD"):
                continue
            pending_token_ids.clear()
            if text:
                yield text

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)  # 逐个添加请求
        outputs = {}
        prefill_throughput = decode_throughput = 0.  # 用于统计 prefill 和 decode 的吞吐
        while not self.is_finished():
            t = perf_counter()  # perf_counter() 返回一个高精度的时间戳，用于计算本轮调度的耗时
            output, num_tokens = self.step()
            if num_tokens > 0:  # prefill
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:  # decode
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)} tok/s",
                "Decode": f"{int(decode_throughput)} tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]  # 按 seq_id 排序，保证输出顺序与输入顺序一致
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
