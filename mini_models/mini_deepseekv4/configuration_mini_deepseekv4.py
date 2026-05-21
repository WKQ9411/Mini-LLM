from transformers import PretrainedConfig


_COMPRESS_RATIO_TO_LAYER_TYPE = {
    0: "sliding_attention",
    4: "compressed_sparse_attention",
    128: "heavily_compressed_attention",
}


class MiniDeepSeekV4Config(PretrainedConfig):
    model_type = "mini_deepseekv4"

    def __init__(
        self,
        # ---- 通用 ----
        vocab_size: int = -1,  # 加载时覆盖
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        max_position_embeddings: int = 512,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        flash_attention: bool = False,
        initializer_range: float = 0.02,
        # ---- Attn ----
        q_lora_rank: int = 192,
        head_dim: int = 64,
        rope_head_dim: int = 16,
        o_groups: int = 4,
        o_lora_rank: int = 96,
        window_size: int = 128,
        index_num_attention_heads: int = 8,
        index_head_dim: int = 16,
        index_topk: int = 16,
        index_score_bias_alpha: float = 0.1,
        csa_ratio: int = 4,
        hca_ratio: int = 128,
        # ---- MoE ----
        moe_intermediate_size: int = 512,
        compress_ratios: dict = None,
        ratio_list: list = None,
        n_hash_layers: int = 3,
        n_routed_experts: int = 8,
        n_shared_experts: int = 1,
        n_activated_experts: int = 2,
        route_scale: float = 1.0,
        use_noaux_load_balance: bool = True,
        bias_update_speed: float = 0.001,
        use_seq_aux: bool = True,
        seq_aux_alpha: float = 0.0001,
        score_func: str = "sqrtsoftplus",
        swiglu_limit: float = 10.0,
        # ---- RoPE ----
        rope_theta: float = 10000.0,
        compress_rope_theta: float = 40000.0,
        rope_scaling: dict = None,
        # ---- mHC ----
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        # ---- MTP ----
        use_mtp: bool = True,
        mtp_loss_lambda: float = 0.0001,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.flash_attention = flash_attention
        self.initializer_range = initializer_range
        
        self.q_lora_rank = q_lora_rank
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.window_size = window_size
        self.sliding_window = window_size  # 兼容配置
        self.index_num_attention_heads = index_num_attention_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.index_score_bias_alpha = index_score_bias_alpha
        self.csa_ratio = csa_ratio
        self.hca_ratio = hca_ratio
        
        self.moe_intermediate_size = moe_intermediate_size
        self.compress_ratios = compress_ratios
        self.n_hash_layers = n_hash_layers
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_activated_experts = n_activated_experts
        self.route_scale = route_scale
        self.use_noaux_load_balance = use_noaux_load_balance
        self.bias_update_speed = bias_update_speed
        self.use_seq_aux = use_seq_aux
        self.seq_aux_alpha = seq_aux_alpha
        self.score_func = score_func
        self.swiglu_limit = swiglu_limit
        
        self.rope_theta = rope_theta
        self.compress_rope_theta = compress_rope_theta
        self.rope_scaling = rope_scaling
        
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        
        self.use_mtp = use_mtp
        self.mtp_loss_lambda = mtp_loss_lambda
        
        # mini_deepseekv4 暂不支持自实现的 triton flash attention
        if self.flash_attention:
            raise ValueError("flash_attention is not supported in mini_deepseekv4.")
        
        if self.compress_ratios is None:
            self.compress_ratios = {
                "sliding_attention": 0,
                "compressed_sparse_attention": csa_ratio,
                "heavily_compressed_attention": hca_ratio,
            }
        
        # 为了使总模型层数与本项目其他模型保持一致，这里前三层均使用 sliding attention，原文是前两层
        # 然后 CSA 与 HCA 交替使用，最后一层使用 sliding attention
        if ratio_list is None:
            csa_hca_list = [csa_ratio, hca_ratio] * ((num_hidden_layers - 4) // 2)
            ratio_list = [0, 0, 0] + csa_hca_list + [0]
        self.layer_types = [_COMPRESS_RATIO_TO_LAYER_TYPE[ratio] for ratio in ratio_list]
        
        assert len(self.layer_types) == self.num_hidden_layers, "The length of layer_types must be equal to num_hidden_layers."
        

        super().__init__(**kwargs)
