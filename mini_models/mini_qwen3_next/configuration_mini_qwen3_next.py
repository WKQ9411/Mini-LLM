from transformers import PretrainedConfig


class MiniQwen3NextConfig(PretrainedConfig):
    """
    mini_qwen3_next 模型配置参数
    
    Attributes:
        vocab_size (int): 词典大小
        hidden_size (int): 隐藏层大小
        intermediate_size (int): MLP 中间维度
        num_hidden_layers (int): 层数
        num_attention_heads (int): 注意力头数
        num_key_value_heads (int): 标准注意力 kv 头数
        max_position_embeddings (int): 最大位置编码数
        initializer_range (float): 初始化参数范围
        rms_norm_eps (float): RMSNorm 的 eps
        use_cache (bool): 是否使用缓存
        rope_theta (float): RoPE 的底, 默认为 10000.0
        attention_bias (bool): 是否使用注意力偏置
        head_dim (int): 每个头的维度
        linear_conv_kernel_dim (int): 卷积核的维度
        linear_key_head_dim (int): 线性注意力 Key 头的维度
        linear_value_head_dim (int): 线性注意力 Value 头的维度
        linear_num_key_heads (int): 线性注意力 Key 头的数量
        linear_num_value_heads (int): 线性注意力 Value 头的数量
        decoder_sparse_step (int): 每多少步一个 MoE, 1 表示每层都是 MoE
        moe_intermediate_size (int): MoE 中间维度
        shared_expert_intermediate_size (int): 共享专家中间维度
        num_experts_per_tok (int): 每个 token 的专家数量
        num_experts (int): 专家数量
        norm_topk_prob (bool): 是否对选中的专家得分进行归一化
        output_router_logits (bool): 是否输出路由 logits, 同时决定是否计算负载均衡辅助损失
        router_aux_loss_coef (float): 路由辅助损失系数
        mlp_only_layers (list[int]): 用于控制哪些层使用 MLP 而不是 MoE
        layer_types (list[str]): 手动设置层类型
    """
    model_type = "mini_qwen3_next"
    
    def __init__(
        self,
        vocab_size: int = -1,  # 加载时覆盖
        hidden_size: int | None = 768,
        intermediate_size: int | None = 2112,
        num_hidden_layers: int | None = 12,
        num_attention_heads: int | None = 12,
        num_key_value_heads: int | None = 2,
        max_position_embeddings: int | None = 512,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-6,
        use_cache: bool | None = True,
        rope_theta: float = 10000.0,
        attention_bias: bool | None = False,
        head_dim: int | None = 64,
        linear_conv_kernel_dim: int | None = 4,
        linear_key_head_dim: int | None = 64,
        linear_value_head_dim: int | None = 64,
        linear_num_key_heads: int | None = 6,
        linear_num_value_heads: int | None = 12,
        decoder_sparse_step: int | None = 1,
        moe_intermediate_size: int | None = 512,
        shared_expert_intermediate_size: int | None = 512,
        num_experts_per_tok: int | None = 2,
        num_experts: int | None = 8,
        norm_topk_prob: bool | None = True,
        output_router_logits: bool | None = True,
        router_aux_loss_coef: float | None = 0.01,
        mlp_only_layers: list[int] | None = [0,1,2],
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.layer_types = layer_types
        if self.layer_types is None:
            interval_pattern = kwargs.get("full_attention_interval", 4)
            self.layer_types = [
                "linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"  # 默认每 3 层 linear_attention 加入 1 层 full_attention
                for i in range(self.num_hidden_layers)
            ]

        # linear attention
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        # MoE
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = mlp_only_layers
        
        # 父类初始化
        super().__init__(**kwargs)