from transformers import PretrainedConfig


class MiniLlama3Config(PretrainedConfig):
    """
    mini_llama3 模型配置参数

    Attributes:
        vocab_size (int): 词典大小
        hidden_size (int): 隐藏层维度
        intermediate_size (int): MLP中间维度
        num_hidden_layers (int): 模型层数
        num_attention_heads (int): 注意力头数
        num_key_value_heads (int): key-value头数
        head_dim (int): 每个头的维度
        rms_norm_eps (float): RMSNorm正则化系数
        attention_bias (bool): 是否使用注意力偏置
        rope_theta (int): ROPE的底数
        use_cache (bool): 是否使用KV Cache
        max_position_embeddings (int): 最大位置编码长度
    """
    model_type = "mini_llama3"

    def __init__(
        self,
        vocab_size: int = -1,  # 加载时覆盖
        hidden_size: int = 768,
        intermediate_size: int = 2064,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,
        head_dim: int = None,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        rope_theta: int = 10000.0,
        use_cache: bool = True,
        max_position_embeddings: int = 512,
        **kwargs,
        ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        # 严格来说 hidden_size 可以不与 num_attention_heads * head_dim 相同，这里采用更兼容的 head_dim 设计
        if head_dim is None:
            assert hidden_size % num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads when head_dim is not specified"
            self.head_dim = hidden_size // num_attention_heads
        else:
            self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings
        # 父类初始化
        super().__init__(**kwargs)