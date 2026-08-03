from huggingface_hub.dataclasses import strict
from transformers import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters


@strict
class MiniDeepSeekV3Config(PreTrainedConfig):
    """
    mini_deepseekv3 模型配置参数

    Attributes:
        vocab_size (int): 词典大小
        hidden_size (int): 隐藏层维度
        intermediate_size (int): Dense MLP 中间维度
        moe_intermediate_size (int): MoE 专家的中间维度
        num_hidden_layers (int): Transformer 层数
        num_dense_layers (int): 模型前部使用 Dense MLP 的层数
        num_attention_heads (int): 注意力头数
        attn_impl (str): MLA 推理实现，支持 naive 或 absorb
        attention_bias (bool): 注意力投影是否使用偏置
        max_position_embeddings (int): 最大位置编码长度
        rms_norm_eps (float): RMSNorm 的 epsilon
        n_routed_experts (int): MoE 路由专家数量
        n_shared_experts (int): MoE 共享专家数量
        n_activated_experts (int): 每个 token 激活的路由专家数量
        n_expert_groups (int): 路由专家分组数量
        n_limited_groups (int): 每个 token 最多选择的专家组数量
        route_scale (float): 路由权重缩放因子
        use_noaux_load_balance (bool): 是否使用无辅助损失负载均衡策略
        bias_update_speed (float): 无辅助损失负载均衡偏置的更新速度
        use_seq_aux (bool): 是否使用序列级负载均衡辅助损失
        seq_aux_alpha (float): 序列级负载均衡辅助损失系数
        q_lora_rank (int): Query 低秩压缩维度
        kv_lora_rank (int): Key/Value 低秩压缩维度
        qk_nope_head_dim (int): Query/Key 中不应用 RoPE 部分的每头维度
        qk_rope_head_dim (int): Query/Key 中应用 RoPE 部分的每头维度
        v_head_dim (int): Value 的每头维度
        rope_parameters (dict): RoPE 参数，包含 rope_theta 和可选缩放参数
        use_mtp (bool): 是否启用 Multi-Token Prediction 模块
        mtp_loss_lambda (float): MTP 损失系数
        pad_token_id (int): Padding token ID
        bos_token_id (int): 序列开始 token ID
        eos_token_id (int | list[int]): 序列结束 token ID
        tie_word_embeddings (bool): 是否共享输入词嵌入与输出投影权重
    """

    model_type = "mini_deepseekv3"

    vocab_size: int = -1
    hidden_size: int = 768
    intermediate_size: int = 3072
    moe_intermediate_size: int = 512
    num_hidden_layers: int = 12
    num_dense_layers: int = 3
    num_attention_heads: int = 12
    attn_impl: str = "absorb"
    attention_bias: bool = False
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-6

    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    n_expert_groups: int = 4
    n_limited_groups: int = 2
    route_scale: float = 1.0
    use_noaux_load_balance: bool = True
    bias_update_speed: float = 0.001
    use_seq_aux: bool = True
    seq_aux_alpha: float = 0.0001

    q_lora_rank: int = 384
    kv_lora_rank: int = 256
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64

    rope_parameters: RopeParameters | dict | None = None

    use_mtp: bool = True
    mtp_loss_lambda: float = 0.0001
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
