from huggingface_hub.dataclasses import strict
from transformers import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters


@strict
class MiniQwen3NextConfig(PreTrainedConfig):
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
        rope_parameters (dict): RoPE 参数，包含 rope_theta 和可选缩放参数
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
        pad_token_id (int): Padding token ID
        bos_token_id (int): 序列开始 token ID
        eos_token_id (int | list[int]): 序列结束 token ID
        tie_word_embeddings (bool): 是否共享输入词嵌入与输出投影权重
    """

    model_type = "mini_qwen3_next"

    vocab_size: int = -1
    hidden_size: int = 768
    intermediate_size: int = 2112
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    max_position_embeddings: int = 512
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    head_dim: int = 64

    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 64
    linear_value_head_dim: int = 64
    linear_num_key_heads: int = 6
    linear_num_value_heads: int = 12

    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    num_experts_per_tok: int = 2
    num_experts: int = 8
    norm_topk_prob: bool = True
    output_router_logits: bool = True
    router_aux_loss_coef: float = 0.01
    mlp_only_layers: list[int] | None = None
    layer_types: list[str] | None = None

    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.mlp_only_layers = [0, 1, 2] if self.mlp_only_layers is None else self.mlp_only_layers
        if self.layer_types is None:
            full_attention_interval = kwargs.pop("full_attention_interval", 4)
            self.layer_types = [
                "linear_attention" if (layer_idx + 1) % full_attention_interval else "full_attention"
                for layer_idx in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)
