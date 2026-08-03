from collections import deque
import xxhash
import numpy as np

from mini_inference.engine.sequence import Sequence


# 内存块类，只存储元数据，而非 kv cache
class Block:

    def __init__(self, block_id):
        self.block_id = block_id  # 物理 block 编号
        self.ref_count = 0        # 引用计数，有多少个序列在共享这个 block
        self.hash = -1            # 该 block 内容的哈希，用于 prefix caching，-1 表示未计算
        self.token_ids = []       # 存储 block 中的 token id

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


# 内存块管理器
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()  # 哈希到 block id 的映射，用于前缀缓存
        self.free_block_ids: deque[int] = deque(range(num_blocks))  # 空闲 block 队列
        self.used_block_ids: set[int] = set()  # 已使用 block 集合

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        # 计算当前 block 的哈希值，考虑前一个 block 的哈希值，实现链式哈希
        # 一个 block 的哈希 = hash(前一个 block 的哈希 + 本 block 的 token)
        # 这样哈希值天然编码了从序列开头到本 block 的完整前缀
        # 两条序列只要前缀相同，对应 block 的哈希就相同 → 可以共享同一个物理 block，这就是 prefix caching 的原理
        # 前缀不同，即使当前块 token 一样，最终哈希也会不同
        h = xxhash.xxh64()  # 使用 xxhash 计算哈希值，极快的非加密哈希
        # 只对满 block 算哈希；partial block（最后没填满的）哈希为 -1，因为它的内容还会变，无法稳定共享
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))  # 把前一个块的哈希值转成 8 字节二进制，然后喂给当前哈希器
        h.update(np.array(token_ids).tobytes())  # 把 token_ids 转成 int32 的二进制，然后追加哈希
        return h.intdigest()  # 取出当前 xxh64 的最终整数哈希（64 位无符号整数）并返回

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()  # 从空闲队列中取出一个 block id
        block = self.blocks[block_id]
        assert block.ref_count == 0  # 确保这个 block 没有被任何序列引用
        # 如果这个 block 之前有哈希值并且存在映射，则删除映射
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()  # 重置 block 的状态，引用计数设为 1，哈希设为 -1，token_ids 清空
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)  # 从已使用 block 集合中移除
        self.free_block_ids.append(block_id)  # 加入空闲 block 队列，注意，此时没有清空 block 的哈希和 token_ids，作为可命中的空闲缓存块

    def can_allocate(self, seq: Sequence) -> int:
        """
        分为两步:
        1. 先沿着序列前缀, 看有多少个整块已经在缓存里(hash_to_block_id)并且内容完全一致
        2. 再算还需要新占用多少块, 看空闲块够不够; 不够就返回 -1, 够就返回可复用的块数
        """
        h = -1
        num_cached_blocks = 0  # 表示有多少个 block 可以直接复用，首先假设没有任何 block 可以复用
        num_new_blocks = seq.num_blocks  # 表示为了处理这个序列，还需要新增占用多少 block，首先假设全部都要新分配
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:  # 未命中缓存，或者命中但是内容不一致（哈希碰撞）
                break
            num_cached_blocks += 1  # 命中则缓存 block 数量加 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1  # 如果这个 block 已经在使用中，则不需要新分配，所需新分配的 block 数量减 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """
        allcate 只负责首次建表，不负责在已有表上增量修改
        Args:
            seq: 需要分配 block 的序列
            num_cached_blocks: 已经在缓存里可以复用的 block 数量, 前 num_cached_blocks 个 block 已经在缓存里, 直接复用, 之后的 block 则需要新分配
        """
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):  # 前 num_cached_blocks 个 block 已经在缓存里，直接复用
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:  # 如果这个 block 已经在使用中，则增加引用计数
                block.ref_count += 1
            else:  # 如果这个 block 之前没有在使用中，则设置引用计数为 1，并从 free_block_ids 中移除，加入 used_block_ids
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):  # 后续的 block 需要新分配
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size  # 更新序列的 num_cached_tokens 为已缓存的 token 数量

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1  # 减少引用计数
            if block.ref_count == 0:
                self._deallocate_block(block_id)  # 如果引用计数为 0，则释放该 block
        seq.num_cached_tokens = 0
        seq.block_table.clear()  # 清空 block_table，表示该序列不再引用任何 block

    def can_append(self, seq: Sequence) -> bool:
        # decode 阶段按需给序列尾部补一个新的 block
        # 生成阶段每次只会 append 1 个 token，一个 block 没满之前，新 token 继续写进当前最后一个 block
        # 只有当上一个 block 刚好写满、下一个 token 需要落到新页时，才需要再分配一个 block
        # 如果当前长度对 block_size 取模不等于 1，则 len(seq) % self.block_size == 1 为 False，此时最后一个 block 还没满
        # 比较的是 free >= 0，一定成立
        # 如果当前长度对 block_size 取模等于 1，则 len(seq) % self.block_size == 1 为 True，此时最后一个 block 刚好满了，需要再分配一个 block
        # 此时需要判断 free >= 1，才能继续 append
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        # 如果需要，就给 block_table 尾部追加一个新 block
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        # 把本轮写满的块计算链式前缀哈希并登记到 hash_to_block_id，从而让后续请求可以命中并复用这些 prefix cache 块
        # 对于 prefill，如果未命中 prefix，则 num_cached_tokens 为 0，num_scheduled_tokens 通常是 prompt 长度
        # 对于 decode，num_cached_tokens 为上轮已经缓存的 token 数量，num_scheduled_tokens 是本轮调度的 token 数量
        # 对于 decode，seq.block(i) 取出的 token，已经包含了本轮的输入 token
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
