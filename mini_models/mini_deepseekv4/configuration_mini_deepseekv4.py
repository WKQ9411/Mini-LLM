from huggingface_hub.dataclasses import strict
from transformers import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters


_COMPRESS_RATIO_TO_LAYER_TYPE = {
    0: "sliding_attention",
    4: "compressed_sparse_attention",
    128: "heavily_compressed_attention",
}


@strict
class MiniDeepSeekV4Config(PreTrainedConfig):
    """
    mini_deepseekv4 模型配置参数

    Attributes:
        vocab_size (int): 词典大小
        hidden_size (int): 隐藏层维度
        num_hidden_layers (int): Transformer 层数
        num_attention_heads (int): 注意力头数
        max_position_embeddings (int): 最大位置编码长度
        rms_norm_eps (float): RMSNorm 的 epsilon
        attention_bias (bool): 注意力投影是否使用偏置
        initializer_range (float): 参数初始化标准差
        q_lora_rank (int): Query 低秩压缩维度
        head_dim (int): 每个注意力头的维度
        rope_head_dim (int): 每个注意力头中应用 RoPE 的维度
        o_groups (int): 分组输出投影的组数
        o_lora_rank (int): 输出投影的低秩维度
        window_size (int): Sliding Attention 的滑动窗口大小
        sliding_window (int): Transformers 掩码接口使用的滑动窗口大小
        index_num_attention_heads (int): CSA Indexer 的注意力头数
        index_head_dim (int): CSA Indexer 的每头维度
        index_topk (int): CSA Indexer 为每个 Query 选择的压缩块数量
        index_score_bias_alpha (float): CSA Indexer 训练时分数偏置系数
        csa_ratio (int): Compressed Sparse Attention 的压缩比例
        hca_ratio (int): Heavily Compressed Attention 的压缩比例
        moe_intermediate_size (int): MoE 专家的中间维度
        compress_ratios (dict[str, int]): 各注意力层类型对应的压缩比例
        ratio_list (list[int]): 各 Transformer 层对应的压缩比例
        layer_types (list[str]): 各 Transformer 层对应的注意力类型
        n_hash_layers (int): 模型前部使用 Hash-based Routing 的层数
        n_routed_experts (int): MoE 路由专家数量
        n_shared_experts (int): MoE 共享专家数量
        n_activated_experts (int): 每个 token 激活的路由专家数量
        route_scale (float): 路由权重缩放因子
        use_noaux_load_balance (bool): 是否使用无辅助损失负载均衡策略
        bias_update_speed (float): 无辅助损失负载均衡偏置的更新速度
        use_seq_aux (bool): 是否使用序列级负载均衡辅助损失
        seq_aux_alpha (float): 序列级负载均衡辅助损失系数
        score_func (str): MoE 路由分数函数
        swiglu_limit (float): SwiGLU 中间激活值的裁剪上限
        rope_parameters (dict): RoPE 参数
        compress_rope_theta (float): CSA 和 HCA 压缩位置编码使用的 RoPE 底数
        hc_mult (int): mHC 超连接残差流的扩展倍数
        hc_sinkhorn_iters (int): mHC Sinkhorn-Knopp 归一化迭代次数
        hc_eps (float): mHC 数值稳定项
        use_mtp (bool): 是否启用 Multi-Token Prediction 模块
        mtp_loss_lambda (float): MTP 损失系数
        pad_token_id (int): Padding token ID
        bos_token_id (int): 序列开始 token ID
        eos_token_id (int | list[int]): 序列结束 token ID
        tie_word_embeddings (bool): 是否共享输入词嵌入与输出投影权重
    """

    model_type = "mini_deepseekv4"

    vocab_size: int = -1
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    initializer_range: float = 0.02

    q_lora_rank: int = 192
    head_dim: int = 64
    rope_head_dim: int = 16
    o_groups: int = 4
    o_lora_rank: int = 96
    window_size: int = 128
    sliding_window: int | None = None
    index_num_attention_heads: int = 8
    index_head_dim: int = 16
    index_topk: int = 16
    index_score_bias_alpha: float = 0.1
    csa_ratio: int = 4
    hca_ratio: int = 128

    moe_intermediate_size: int = 512
    compress_ratios: dict[str, int] | None = None
    ratio_list: list[int] | None = None
    layer_types: list[str] | None = None
    n_hash_layers: int = 3
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    route_scale: float = 1.0
    use_noaux_load_balance: bool = True
    bias_update_speed: float = 0.001
    use_seq_aux: bool = True
    seq_aux_alpha: float = 0.0001
    score_func: str = "sqrtsoftplus"
    swiglu_limit: float = 10.0

    rope_parameters: RopeParameters | dict | None = None
    compress_rope_theta: float = 40000.0

    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    use_mtp: bool = True
    mtp_loss_lambda: float = 0.0001
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.sliding_window = self.window_size if self.sliding_window is None else self.sliding_window
        if self.compress_ratios is None:
            self.compress_ratios = {
                "sliding_attention": 0,
                "compressed_sparse_attention": self.csa_ratio,
                "heavily_compressed_attention": self.hca_ratio,
            }

        if self.layer_types is None:
            if self.ratio_list is None:
                csa_hca_list = [self.csa_ratio, self.hca_ratio] * ((self.num_hidden_layers - 4) // 2)
                self.ratio_list = [0, 0, 0, *csa_hca_list, 0]
            self.layer_types = [_COMPRESS_RATIO_TO_LAYER_TYPE[ratio] for ratio in self.ratio_list]

        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("The length of layer_types must be equal to num_hidden_layers")

        super().__post_init__(**kwargs)
