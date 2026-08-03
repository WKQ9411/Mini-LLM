from collections import deque

from mini_inference.config import Config
from mini_inference.engine.sequence import Sequence, SequenceStatus
from mini_inference.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs  # 单次调度里最多同时处理多少条序列
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 单次调度里最多同时处理多少个 token
        self.eos = config.eos
        self.block_size = config.kvcache_block_size  # 分页大小
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running  # 如果等待队列和运行队列都为空，则表示所有序列都已完成

    def add(self, seq: Sequence):
        self.waiting.append(seq)  # 加入等待队列，等待调度

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]  # 获取等待队列的第一条序列
            remaining = self.max_num_batched_tokens - num_batched_tokens  # 计算剩余可调度的 token 数量
            
            if remaining == 0:  # 如果剩余的 token 数量为 0，则跳出循环
                break
            if not seq.block_table:  # 如果序列的 block_table 为空，则表示该序列还没有分配缓存块
                num_cached_blocks = self.block_manager.can_allocate(seq)  # 返回可复用的缓存块数量
                if num_cached_blocks == -1:  # 如果无法分配缓存块，则跳出循环
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size  # 计算该序列还需要调度的 token 数量
            else:  # 如果序列的 block_table 不为空，则表示该序列已经分配了缓存块
                num_tokens = seq.num_tokens - seq.num_cached_tokens  # 计算该序列还需要调度的 token 数量
            if remaining < num_tokens and scheduled_seqs:
                # 如果剩余的 token 数量小于该序列还需要调度的 token 数量，此时应当进行 chunked prefill
                # 但是为了避免复杂的调度逻辑，只允许对第一条序列进行 chunked prefill
                # 因此如果当前序列不是队首序列，则不进行 chunked prefill，而是跳出循环
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)  # 如果该序列还没有分配缓存块，则进行缓存块的分配
            
            seq.num_scheduled_tokens = min(num_tokens, remaining)  # 该序列本次调度的 token 数量
            num_batched_tokens += seq.num_scheduled_tokens  # 累加本次调度的 token 数量
            
            # 如果该序列本次调度的 token 数量与缓存的 token 数量之和等于该序列的总 token 数量
            # 则说明该序列已经完成 prefill，可以从等待队列中移除，并加入运行队列中进行下一步 decode
            # 如果该序列还没有完成 prefill，例如位于队首的执行了 chunked prefill 的序列，则继续等待下一次调度
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)  # 将该序列加入本次调度的序列列表中

        if scheduled_seqs:
            return scheduled_seqs, True  # (本轮需要执行的序列列表, 是否是 prefill 阶段)

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()  # 直接从取出队首序列
            while not self.block_manager.can_append(seq):  # 如果当前序列无法继续 append，则需要 preempt 该序列，释放其占用的缓存块
                if self.running:
                    # decode 循环里用 popleft() 从左取，prefill 时 append 到右
                    # 所以最左 = 最早开始 decode（可能快生成完了），最右 = 最新加入
                    # 弹出最新加入的 seq，优先释放最新加入的 seq 的缓存块
                    self.preempt(self.running.pop())
                else:
                    # 如果 running 队列为空，则说明当前序列无法继续 append，且没有其他序列可以 preempt
                    # 此时只能抢占自己，
                    self.preempt(seq)
                    break  # 不会走下面的 else 分支
            # while-else 语法：只有 while 循环正常结束（条件变 False）而不是被 break 跳出时，才执行 else
            # 如果当前序列可以继续 append，或抢占之后可以 append，则将其加入本轮调度的序列列表中
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)  # 如果需要，就给 block_table 尾部追加一个新 block
                scheduled_seqs.append(seq)  # 将该序列加入本轮调度的序列列表中
        
        # 如果 decode 第一个取出的 seq 就无法 append，且 running 没有别的可抢
        # 那 preempt(seq); break 后 scheduled_seqs 为空、running 也空，外层 while 退出，会撞上 assert scheduled_seqs
        # 意味着显存紧张到连一条序列都续不了一个 block 时直接报错，而不是更复杂地再从别处腾内存
        assert scheduled_seqs
        # 把本轮处理过的序列按原顺序塞回 running 队列左端
        # 关键在于 reversed 是用来抵消 extendleft 自带的反转行为
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        抢占策略：将序列的状态退回到 waiting, 标记为 prefill, 并释放其占用的缓存块
        这是基于重计算(recomputation)的抢占, 区别于把 kv cache 换出到 CPU 内存的 swapping 方案
        preempt 和 deallocate 完全不碰 seq.token_ids, 因此已生成的 token 不会丢, 丢的是 kv cache
        """
        seq.status = SequenceStatus.WAITING  # 状态退回 waiting
        seq.is_prefill = True  # 标记为 prefill，恢复时需要重计算
        self.block_manager.deallocate(seq)  # 释放该序列占用的缓存块
        self.waiting.appendleft(seq)  # 放回等待队列的队首

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """
        seqs 就是本轮 schedule() 返回的 scheduled_seqs, token_ids 是模型给每条 seq 生成的 1 个 token, 在此进行一一配对
        把模型生成的 token 正式落到每条序列上, 并维护每条序列的状态
        """
        for seq, token_id in zip(seqs, token_ids):
            # 本轮调度的 token 可能让某些 block 刚好写满，这些 block 内容现在稳定了，计算它们的链式哈希并登记到 hash_to_block_id
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens  # 更新序列的 num_cached_tokens 为已缓存的 token 数量
            seq.num_scheduled_tokens = 0  # 清空本轮调度的 token 数量
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue  # 如果是 prefill 阶段，且该序列还没有完成 prefill (chunked prefill)，则不进行后续处理
            seq.append_token(token_id)  # 把模型生成的 token 正式落到序列上，NOTE: 此时有 len(token_ids) - num_cached_tokens = 1，本轮的 token_id 会在下一轮 decode forward 时产生 cache
            # 如果生成的 token 是 eos，或者该序列的 completion token 数量已经达到 max_tokens，则标记该序列为 finished，并释放其占用的缓存块
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
