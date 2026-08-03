from huggingface_hub.dataclasses import strict
from transformers import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters


@strict
class MiniLlama3Config(PreTrainedConfig):
    """
    mini_llama3 模型配置参数

    Attributes:
        vocab_size (int): 词典大小
        hidden_size (int): 隐藏层维度
        intermediate_size (int): MLP 中间维度
        num_hidden_layers (int): 模型层数
        num_attention_heads (int): 注意力头数
        num_key_value_heads (int): Key-Value 头数
        head_dim (int): 每个注意力头的维度，为 None 时根据 hidden_size 自动计算
        rms_norm_eps (float): RMSNorm 的 epsilon
        attention_bias (bool): 注意力投影是否使用偏置
        rope_parameters (dict): RoPE 参数，包含 rope_theta 和可选缩放参数
        use_cache (bool): 是否使用 KV Cache
        max_position_embeddings (int): 最大位置编码长度
        pad_token_id (int): Padding token ID
        bos_token_id (int): 序列开始 token ID
        eos_token_id (int | list[int]): 序列结束 token ID
        tie_word_embeddings (bool): 是否共享输入词嵌入与输出投影权重
    """

    model_type = "mini_llama3"

    vocab_size: int = -1
    hidden_size: int = 768
    intermediate_size: int = 2064
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    head_dim: int | None = None
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    rope_parameters: RopeParameters | dict | None = None
    use_cache: bool = True
    max_position_embeddings: int = 512
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if self.head_dim is None:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError(
                    "hidden_size must be divisible by num_attention_heads when head_dim is not specified"
                )
            self.head_dim = self.hidden_size // self.num_attention_heads

        super().__post_init__(**kwargs)
