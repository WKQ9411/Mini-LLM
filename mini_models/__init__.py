from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers import PreTrainedModel
from typing import Tuple
from .generator import Generator
from .mini_llama3.modeling_mini_llama3 import MiniLlama3ForCausalLM, MiniLlama3Model
from .mini_llama3.configuration_mini_llama3 import MiniLlama3Config
from .mini_deepseekv3.modeling_mini_deepseekv3 import MiniDeepSeekV3ForCausalLM, MiniDeepSeekV3Model
from .mini_deepseekv3.configuration_mini_deepseekv3 import MiniDeepSeekV3Config


# 注册 AutoConfig
AutoConfig.register("mini_llama3", MiniLlama3Config)
AutoConfig.register("mini_deepseekv3", MiniDeepSeekV3Config)

# 注册 AutoModel
AutoModel.register(MiniLlama3Config, MiniLlama3Model)
AutoModel.register(MiniDeepSeekV3Config, MiniDeepSeekV3Model)

# 注册 AutoModelForCausalLM
AutoModelForCausalLM.register(MiniLlama3Config, MiniLlama3ForCausalLM)
AutoModelForCausalLM.register(MiniDeepSeekV3Config, MiniDeepSeekV3ForCausalLM)


def list_models():
    return ["mini_llama3", "mini_deepseekv3"]


def get_model_and_config(model_name: str, **kwargs):
    """
    根据模型名称获取模型类和配置类

    Args:
        model_name (str): 模型名称，如 "mini_llama3"

    Returns:
        (PreTrainedModel, PretrainedConfig): 返回对应的模型类和配置类
    """
    if model_name == "mini_llama3":
        model = MiniLlama3ForCausalLM
        config = MiniLlama3Config
    elif model_name == "mini_deepseekv3":
        model = MiniDeepSeekV3ForCausalLM
        config = MiniDeepSeekV3Config
    else:
        raise ValueError(f"Unsupported model: {model_name}. Available models: {list_models()}")
    return model, config


def get_model_info(model: PreTrainedModel) -> Tuple[int, dict]:
        """
        计算模型可训练参数量, 参数量说明及模型架构类型

        Args:
            model (PreTrainedModel): 模型

        Returns:
            参数量 (Tuple[int, dict]): (精确可训练参数量: int, 大致参数量及说明: dict)
        """
        def get_approximate_params(params: int) -> str:
            return f"{params / 1_000_000 if params < 1_000_000_000 else params / 1_000_000_000:.1f}{' M' if params < 1_000_000_000 else ' B'}"
        
        config = model.config
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if hasattr(model, "architecture_type"):
            architecture_type = model.architecture_type
        else:
            architecture_type = "unknown"
        
        if config.model_type == "mini_deepseekv3":
            # MTP 模块的参数量，这部分参数只是预训练参数量，推理时不需要
            # MTP 模块中的 embed_tokens 和 lm_head 是共享的，这里不计入 MTP 参数量
            if hasattr(model, "mtp") and model.mtp is not None:
                mtp_params = 0
                for name, param in model.mtp.named_parameters():
                    if param.requires_grad:
                        # 检查是否是共享参数
                        is_shared = False
                        if name == "embed_tokens.weight":
                            # embed_tokens 与主模型共享
                            if model.mtp.embed_tokens is model.model.embed_tokens:
                                is_shared = True
                        elif name == "lm_head.weight":
                            # lm_head 与主模型共享
                            if model.mtp.lm_head is model.lm_head:
                                is_shared = True
                        
                        if not is_shared:
                            mtp_params += param.numel()
            else:
                mtp_params = 0

            # 计算 MoE 层的数量和激活比例
            activation_ratio = config.n_activated_experts / config.n_routed_experts

            # 计算所有路由专家的参数
            routed_experts_total_params = 0
            for i in range(config.num_dense_layers, config.num_hidden_layers):
                layer = model.model.layers[i]
                if hasattr(layer, 'mlp'):
                    from .mini_deepseekv3.modeling_mini_deepseekv3 import MiniDeepSeekV3MoE
                    if isinstance(layer.mlp, MiniDeepSeekV3MoE):
                        # 计算所有路由专家的参数
                        for expert in layer.mlp.experts:
                            routed_experts_total_params += sum(p.numel() for p in expert.parameters() if p.requires_grad)

            # 激活参数量 = 总参数 - MTP参数 - 总路由专家参数 + 激活的路由专家参数
            activated_params = trainable_params - mtp_params - routed_experts_total_params + routed_experts_total_params * activation_ratio
            approx_trainable_params = get_approximate_params(trainable_params)
            approx_mtp_params = get_approximate_params(mtp_params)
            approx_activated_params = get_approximate_params(activated_params)
            approx_actual_params = get_approximate_params(trainable_params - mtp_params)
            approx_params_info = {
                "architecture": architecture_type,
                "trainable_params": approx_trainable_params + f" (including MTP: {approx_mtp_params})",
                "total_params": approx_actual_params + f" (activated: {approx_activated_params})",
            }
        else:
            # Dense 架构模型默认 info
            approx_params_info = {
                "architecture": architecture_type,
                "trainable_params": get_approximate_params(trainable_params),
            }
        return {"specific_params": trainable_params}, approx_params_info


__all__ = ['list_models', 'get_model_and_config', 'get_model_info', 'Generator']