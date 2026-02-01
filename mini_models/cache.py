import torch
from transformers.cache_utils import Cache
from .mini_qwen3_next.configuration_mini_qwen3_next import MiniQwen3NextConfig


class MiniQwen3NextDynamicCache:
    """
    MiniQwen3Next 的动态缓存, 可以同时缓存标准注意力层的 kv cache 和线性注意力层的 conv_state 和 recurrent state

    该缓存包含两组张量列表：
    - key_cache 和 value_cache 用于标准注意力
    - conv_states 和 recurrent_states 用于线性注意力
    
    每组列表包含 num_layers 个张量，各张量的预期形状如下：
    - 对于标准注意力层: 
        - key_cache 和 value_cache 的形状为 (batch_size, num_heads, seq_len, head_dim)
        - conv_states 和 recurrent_states 的形状为 (batch_size, 0) (空张量)
    - 对于线性注意力层: 
        - key_cache 和 value_cache 的形状为 (batch_size, 0) (空张量)
        - conv_states 表示卷积状态，形状为 (batch_size, conv_dim, conv_kernel_size)
        - recurrent_states 表示循环状态，形状为 (batch_size, num_heads, head_k_dim, head_v_dim)
    """

    is_compileable = False  # 显式声明无需编译

    def __init__(self, config: MiniQwen3NextConfig):
        super().__init__()
        self.layer_types = config.layer_types
        self.transformer_layers = [i for i in range(config.num_hidden_layers) if self.layer_types[i] == "full_attention"]  # 标准注意力层索引
        self.last_linear_layer = len(self.layer_types) - 1 - self.layer_types[::-1].index("linear_attention")  # 找到最后一个线性注意力层的索引

        # 全部初始化为 None
        self.conv_states = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_states = [None for _ in range(config.num_hidden_layers)]
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]

    def __len__(self):
        return len(self.layer_types)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """更新指定层的 key 和 value cache, 仅用于标准注意力层"""
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            # 在 seq_len 维度上拼接
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """
        给定 beam_idx, 重新排列 cache 以适应 beam search
        束搜索需要重排是因为每步保留的 top-k 候选可能来自不同的历史 beam, 因此必须让 kv cache 与新的 beam 对应
        此方法在 transformers 的 generate() 中可能会用到, 本项目的自定义 generate 暂时未实现 beam search 功能
        
        示例:
        假设 beam_width=2, 初始输入 "我":
        Step 1: 
            输入: "我" (batch_size=1) 模型输出 logits 取 top-2 token: ["是", "爱"]
            现在有 2 个候选序列:
                beam 0: "我是"
                beam 1: "我爱"
            kv cache 需要复制成 2 份 (batch_size 变为 2)
        Step 2:
            输入: ["是", "爱"] (batch_size=2, 分别是两个 beam 的最后一个 token, "我" 已经被缓存在 cache 中)
            模型输出 2 组 logits
                beam 0 ("我是") 的 top-2: ["人"(0.4), "谁"(0.3)]
                beam 1 ("我爱") 的 top-2: ["你"(0.6), "他"(0.2)]
            所有候选的联合概率:
                "我是人": 0.5 x 0.4 = 0.20  (来自 beam 0)  ✓
                "我是谁": 0.5 x 0.3 = 0.15  (来自 beam 0)
                "我爱你": 0.3 x 0.6 = 0.18  (来自 beam 1)  ✓
                "我爱他": 0.3 x 0.2 = 0.06  (来自 beam 1)
            保留 top-2:
                新 beam 0: "我是人" (来自旧 beam 0)
                新 beam 1: "我爱你" (来自旧 beam 1)
            此时 beam_idx = [0, 1] 新 beam 0 来自旧 beam 0, 新 beam 1 来自旧 beam 1
            kv cache 不需要重排
        Step 3:
            输入: ["人", "你"] (batch_size=2, 此时 "我是" 和 "我爱" 分别是两个 beam 的 cache)
            模型输出 2 组 logits
                beam 0 ("我是人") 的 top-2: ["类"(0.3), "啊"(0.2)]
                beam 1 ("我爱你") 的 top-2: ["们"(0.5), "呀"(0.1)]
            所有候选的联合概率:
                "我是人类": 0.20 x 0.3 = 0.060  (来自 beam 0)  ✓
                "我是人啊": 0.20 x 0.2 = 0.040  (来自 beam 0)
                "我爱你们": 0.18 x 0.5 = 0.090  (来自 beam 1)  ✓
                "我爱你呀": 0.18 x 0.1 = 0.018  (来自 beam 1)
            保留 top-2:
                新 beam 0: "我爱你们" (来自旧 beam 1)
                新 beam 1: "我是人类" (来自旧 beam 0)
            此时 beam_idx = [1, 0] 新 beam 0 来自旧 beam 1, 新 beam 1 来自旧 beam 0
            kv cache 必须重排, 因为:
                新 beam 0 需要 "我爱你" 的历史 kv cache (旧 beam 1 的 cache)
                新 beam 1 需要 "我是人" 的历史 kv cache (旧 beam 0 的 cache)
        """
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                device = self.key_cache[layer_idx].device
                beam_idx = beam_idx.to(device)
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx)  # 在 dim=0 也就是 batch_size 上进行重排
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx)

            if self.conv_states[layer_idx] is not None:
                device = self.conv_states[layer_idx].device
                beam_idx = beam_idx.to(device)
                self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx)
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(0, beam_idx)

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        """返回指定层的已缓存序列长度, 仅用于标准注意力层"""
        # 确保 layer_idx 在有效范围内，如果不在，就返回第一个标准注意力层的缓存序列长度
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        此函数用于计算注意力掩码的长度 kv_length 和偏移量 kv_offset
        - kv_length 表示当前 kv cache 的总长度, 即当前输出长度 + 历史缓存长度
        - kv_offset 表示 kv cache 的起始位置偏移量, 一般在类似滑动窗口注意力等场景下会用到, 这里固定为 0
        
        Args:
            cache_position (torch.Tensor): cache_position 是当前输入序列的位置索引，索引范围为 [past_seen_tokens, past_seen_tokens + query_length], 形状为 (query_length,)
            layer_idx (int): 指定层的索引
        
        Returns:
            tuple[int, int]: kv_length 和 kv_offset 组成的元组
        """
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    @property
    def has_previous_state(self):
        """如果最后一个线性注意力层的 conv_state 不为 None, 则表示存在 previous state"""
        return self.conv_states[self.last_linear_layer] is not None