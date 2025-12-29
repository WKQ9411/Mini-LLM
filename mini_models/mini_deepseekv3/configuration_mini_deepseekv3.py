from transformers import PretrainedConfig


class MiniDeepSeekV3Config(PretrainedConfig):
    """
    mini_deepseekv3 模型配置参数

    Attributes:
        max_seq_len (int): 最大序列长度
        vocab_size (int): 词典大小
        dim (int): 嵌入维度
        inter_dim (int): MLP层的中间维度
        moe_inter_dim (int): MoE层专家的中间维度
        n_layers (int): Transformer层的数量
        n_dense_layers (int): 模型中密集层的数量, 前几层负载均衡收敛慢, 因此设置为密集层
        n_heads (int): 注意力头数
        n_routed_experts (int): MoE层中路由的专家数量
        n_shared_experts (int): MoE层中共享的专家数量
        n_activated_experts (int): MoE层中激活的专家数量
        n_expert_groups (int): 专家的分组数量
        n_limited_groups (int): 路由限制, 每次最多从限制的专家组里选择专家
        route_scale (float): 路由权重的缩放因子
        use_noaux_load_balance (bool): 是否使用无辅助损失的负载均衡策略
        bias_update_speed (float): 偏置更新速度
        use_seq_aux (bool): 是否使用序列级别的辅助损失
        seq_aux_alpha (float): 序列级别的辅助损失的权重
        q_lora_rank (int): query 的下投影维度【对应论文中的 d_c'】
        kv_lora_rank (int): key/value 的下投影维度【对应论文中的 d_c】
        qk_nope_head_dim (int): 没有位置编码的 q_t^C 和 k_t^C 的每个头的维度【对应论文中的 d_h】
        qk_rope_head_dim (int): 解耦的带 RoPE 的 q_t^R 和 k_t^R 的每个头的维度【对应论文中的 d_h^R】
        v_head_dim (int): value 的每个头的维度, 可以与 qk_nope_head_dim 不同【但在论文中也同样设定为 d_h】
        rope_theta (float): 旋转位置编码的基底【即 θ_d=b^(-2d/D) 中的 b】
        use_mtp (bool): 是否使用 MTP 策略
        mtp_loss_lambda (float): MTP 损失的权重
    """
    model_type = "mini_deepseekv3"

    def __init__(
        self,
        # ---- 通用 ----
        vocab_size: int = -1,  # 加载时覆盖
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        moe_intermediate_size: int = 512,
        num_hidden_layers: int = 12,
        num_dense_layers: int = 3,
        num_attention_heads: int = 12,
        attn_impl: str = "absorb",
        attention_bias: bool = False,
        max_position_embeddings: int = 512,
        rms_norm_eps: float = 1e-6,
        # ---- MoE ----
        n_routed_experts: int = 8,
        n_shared_experts: int = 1,
        n_activated_experts: int = 2,
        n_expert_groups: int = 4,
        n_limited_groups: int = 2,
        route_scale: float = 1.0,
        use_noaux_load_balance: bool = True,
        bias_update_speed: float = 0.001,
        use_seq_aux: bool = True,
        seq_aux_alpha: float = 0.0001,
        # ---- MLA ----
        q_lora_rank: int = 384,  # 源码中若 q_lora_rank=0, 则不使用下投影，这里我们使用下投影，并略去若 q_lora_rank=0 的逻辑
        kv_lora_rank: int = 256,  # 论文中 d_c = 4 * d_h
        qk_nope_head_dim: int = 64,  # d_h
        qk_rope_head_dim: int = 32,  # 论文中 d_h^R = d_h / 2
        v_head_dim: int = 64,
        # ---- RoPE ----
        rope_theta: float = 10000.0,
        # ---- MTP ----
        use_mtp: bool = True,
        mtp_loss_lambda: float = 0.0001,
        # ---- 父类通用字段 ----
        **kwargs
        ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_dense_layers = num_dense_layers
        self.num_attention_heads = num_attention_heads
        self.attn_impl = attn_impl
        self.attention_bias = attention_bias
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps

        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_activated_experts = n_activated_experts
        self.n_expert_groups = n_expert_groups
        self.n_limited_groups = n_limited_groups
        self.route_scale = route_scale
        self.use_noaux_load_balance = use_noaux_load_balance
        self.bias_update_speed = bias_update_speed
        self.use_seq_aux = use_seq_aux
        self.seq_aux_alpha = seq_aux_alpha

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.rope_theta = rope_theta

        self.use_mtp = use_mtp
        self.mtp_loss_lambda = mtp_loss_lambda

        # 父类初始化
        super().__init__(**kwargs)