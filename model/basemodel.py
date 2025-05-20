import torch
import torch.nn as nn
from dataclasses import dataclass
import inspect
from typing import Tuple, Optional, List
import torch.nn.functional as F
from collections import Counter


# ---------------- 基础模型和配置 ----------------
@dataclass
class BaseModelArgs:
    # 默认训练参数
    epochs: int = 2
    max_lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_iters: Optional[int] = None  # 如果不为 None，则优先使用
    warmup_ratio: float = 0.1  # 前 10% 步使用 warm up
    lr_decay_iters: Optional[int] = None  # 如果不为 None，则优先使用
    lr_decay_ratio: float = 0.98  # 余弦衰减到总训练步数的 98% 步，之后使用最小学习率
    weight_decay: float = 1e-1
    betas: Tuple[float, float] = (0.9, 0.95)

    # 默认推理参数
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

    def _apply_temperature(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """根据温度缩放 logits, temperature 为 0 或负数时不应用"""
        if temperature > 0:
            return logits / temperature
        return logits
    
    def _apply_top_k(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """将概率分布限制在 top_k 个 token"""
        if top_k > 0:
            # 找出 top_k 个 logits 的值和索引
            top_k = min(top_k, logits.size(-1)) # 确保 top_k 不超过词汇表大小
            values, _ = torch.topk(logits, top_k)
            # 获取第 k 个元素的值作为阈值
            kth_value = values[:, -1].unsqueeze(-1)
            # 将所有低于阈值的 logits 设置为负无穷大，这样它们在 softmax 后概率为0
            logits[logits < kth_value] = -float('Inf')
        return logits
    
    def _apply_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """应用 Top-P (Nucleus Sampling) 过滤 logits。"""
        # 如果 top_p 无效或不进行过滤，则直接返回
        if top_p <= 0.0 or top_p >= 1.0:
            return logits

        # 1. 按 logit 值降序排序，同时获取排序后的索引
        # sorted_logits: 排序后的 logit 值
        # sorted_indices: 排序后的 logit 值对应的原始词汇表索引
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        # 2. 对排序后的 logits 计算 softmax 概率
        sorted_probs = F.softmax(sorted_logits, dim=-1)

        # 3. 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 4. 创建一个 boolean mask，标记那些累积概率超过 top_p 的 token
        # 我们需要移除那些 *超出* P 核的 token。
        # 首先标记累积概率 > top_p 的位置
        sorted_indices_to_remove = cumulative_probs > top_p

        # --- 重要调整：确保 P 核至少包含概率最高的那个 token ---
        # 将标记右移一位，这样第一个超过阈值的 token 会被保留下来
        # 我们总是希望保留概率最高的 token (即排序后的第一个)，所以将第一个位置的移除标记设为 False
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # 5. 使用 scatter_ 将移除标记应用回原始（未排序）的索引位置
        # 创建一个与 logits 形状相同、初始全为 False 的 mask
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        # 将 sorted_indices_to_remove 中的 True 值放回它们在原始 logits 中的位置
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

        # 6. 将被标记为移除的 token 的 logit 值设置为负无穷大
        # torch.where(condition, value_if_true, value_if_false)
        filtered_logits = torch.where(
            indices_to_remove,
            torch.full_like(logits, -float('inf')), # 设置为负无穷
            logits  # 保留原始 logit
        )

        return filtered_logits

    def _apply_repetition_penalty(self, logits: torch.Tensor, generated_token_ids: List[int], penalty: float):
        """
        应用存在惩罚, 只要是已生成过的 token, 就无差别应用惩罚

        Args:
            logits (torch.Tensor): 当前步的 logits (batch_size, vocab_size), 这里 batch_size=1
            generated_token_ids (List[int]): 到目前为止已生成的 token ID 列表
            penalty (float): 重复惩罚因子 (>= 1.0)

        Returns:
            torch.Tensor: 应用惩罚后的 logits
        """
        if not generated_token_ids or penalty == 1.0:
            return logits

        # 计算 logits 张量中所有对应于已经生成的 token 的原始分数
        score = torch.gather(logits, 1, torch.tensor([generated_token_ids], device=logits.device)) # (1, num_generated)

        # 原始 logits 可正可负，如果是正，就减小，如果是负，就更负
        # 如果 score < 0, 则 score = score * penalty
        # 如果 score > 0, 则 score = score / penalty
        score = torch.where(score < 0, score * penalty, score / penalty)

        logits.scatter_(1, torch.tensor([generated_token_ids], device=logits.device), score)
        return logits
    
    def _apply_frequency_penalty(self, logits: torch.Tensor, generated_token_ids: List[int], penalty: float):
        """
        应用频率惩罚 (Frequency Penalty), 惩罚力度与 token 在已生成序列中的出现次数成正比

        Args:
            logits (torch.Tensor): 当前步的 logits (batch_size, vocab_size), 这里 batch_size=1
            generated_token_ids (List[int]): 到目前为止已生成的 token ID 列表
            penalty (float): 频率惩罚因子 (通常 >= 0.0) 0.0 表示无惩罚, 正值会降低重复 token 的 logit。

        Returns:
            torch.Tensor: 应用惩罚后的 logits
        """
        if not generated_token_ids or penalty == 0.0:
            return logits

        # 1. 计算已生成 token 的频率
        token_counts = Counter(generated_token_ids)  # {token_id: count, ...}
        # 提取出现过的 unique token IDs 和它们的频率
        unique_ids = list(token_counts.keys())
        frequencies = [token_counts[token_id] for token_id in unique_ids]

        # 转换为 tensor
        unique_ids_tensor = torch.tensor([unique_ids], device=logits.device) # Shape: (1, num_unique_ids)
        # 确保 frequencies tensor 类型与 logits 一致，以便进行数学运算
        frequencies_tensor = torch.tensor([frequencies], dtype=logits.dtype, device=logits.device) # Shape: (1, num_unique_ids)

        # 2. 计算要从 logit 中减去的惩罚量
        # 惩罚量 = 频率 * 惩罚因子
        penalty_amounts = frequencies_tensor * penalty

        # 3. 从对应的 logit 中减去惩罚量
        # 使用 scatter_add_ 并传入负的惩罚量来实现减法
        # 注意：这里直接修改 logits tensor (in-place)
        logits.scatter_add_(1, unique_ids_tensor, -penalty_amounts)

        return logits

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        context_length: int,  # 最大序列长度
        tokenizer,
        max_new_tokens: int | None = None,  # 最大生成的新 token 数量
        stream: bool = True,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        task: str = "chat",
        ):
        """
        生成文本

        Args:
            prompt (str): 输入的提示文本
            context_length (int, optional): 模型的最大上下文长度, 即预训练时的最大序列长度
            tokenizer: 分词器
            max_new_tokens (int | None, optional): 最大生成的新 token 数量, 如果为 None 或超过最大可生成长度, 则使用最大可生成长度, 默认为 None
            stream (bool, optional): 是否流式输出, 默认为 True
            temperature (float, optional): 控制采样随机性的温度值 ( > 0), 值越小越确定，值越大越随机, 默认为 0.8
            top_k (int, optional): Top-K 采样, 仅考虑概率最高的 K 个 token, 0 表示禁用, 默认为 50
            top_p (float, optional): Top-P 采样, 仅考虑累积概率超过 P 的最小 token 集, 0 或 1.0 表示禁用。 Defaults to 0.9
            repetition_penalty (float, optional): 重复惩罚因子 (>= 1.0), 默认为 1.0, 表示无惩罚, 推荐 1.0-1.5
            frequency_penalty (float, optional): 频率惩罚因子 (>= 0.0), 默认为 0.0, 表示无惩罚, 推荐 0.0-0.5
            task (str, optional): 任务类型, 若为 chat, 则应用聊天模板, 若为 generate, 则进行补全, 默认为 chat (generate 模式主要用于观察预训练模型的效果)
            
        Yields:
            str: 如果 stream=True, 逐个 yield 生成的文本块, 如果 stream=False, 最后 yield 整个生成的文本字符串
        """
        # --- 参数验证 ---
        if temperature < 0:
            raise ValueError("temperature 必须大于等于 0")
        if top_k < 0:
            raise ValueError("top_k 必须大于等于 0")
        if top_p < 0.0 or top_p > 1.0:
            raise ValueError("top_p 必须在 [0.0, 1.0] 范围内")
        if max_new_tokens is not None and max_new_tokens <= 0:
            raise ValueError("max_new_tokens 必须大于 0")
        if repetition_penalty < 1.0:
            raise ValueError("repetition_penalty (存在惩罚因子)必须大于等于 1.0")
        if frequency_penalty < 0.0:
            raise ValueError("frequency_penalty (频率惩罚因子) 必须大于等于 0.0")
        if repetition_penalty != 1.0 and frequency_penalty != 0.0:
            raise ValueError("不建议同时使用 repetition_penalty (存在惩罚因子) 和 frequency_penalty (频率惩罚因子)")
        
        # --- Tokenize 输入 ---
        if task == "chat":  # 聊天任务，应用聊天模板
            messages = [
                {"role": "user", "content": f'{prompt}'}
            ]
            input_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", add_generation_prompt=True).to(next(self.parameters()).device)
        elif task == "generate":  # 生成任务，直接使用输入预测下一个 token
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(next(self.parameters()).device)  # (1, prompt_len)
        else:
            raise ValueError("task 必须为 chat 或 generate")
        
        prompt_len = len(input_ids[0])
        max_allowed = context_length - prompt_len  # 允许的最大生成长度

        # --- 计算最大生成长度 ---
        if max_allowed <= 0:
            raise ValueError(f"prompt 长度 ({prompt_len}) 超过最大上下文长度 ({context_length})")
        if max_new_tokens is None or max_new_tokens > max_allowed:
            max_tokens_to_generate = context_length - prompt_len
        else:
            max_tokens_to_generate = max_new_tokens
        
        # --- 生成循环 ---
        # 1) prefill 阶段，将 input_ids 一次性输入，前向传播的过程中会将前 prompt_len-1 个 token 的 KV 加入到模型的 buffer 中，因果 mask 在模型内部实现
        _, _, _ = self(input_ids=input_ids[:, :-1])  # (1, prompt_len-1)

        # 2) decode 阶段，从 input_ids 的最后一个 token 开始，每次只输入一个 token，将当前 token 的 KV 补到 buffer 中，并预测下一个 token
        generated_tokens = [] # 用于非流式输出时收集
        prompt_last_pos = prompt_len - 1
        prev_id = input_ids[:, [-1]]  # (1, 1)

        # --- 流式输出相关变量 ---
        # 对于多字节编码（如 UTF-8 中的中文、emoji 等），单个 token ID 可能只代表字符的一部分。
        # 直接解码单个 token ID 可能会产生无效字符或 Unicode 替换字符（例如 �），直到组成完整字符的所有 token 都被解码。
        yielded_text = ""  # 记录已经 yield 出去的文本
        token_buffer = []  # 临时存储 token ID 用于解码

        for start_pos in range(prompt_last_pos, prompt_last_pos + max_tokens_to_generate):  # start_pos 从 prompt_last_pos 开始
            logits, _, _ = self(input_ids=prev_id, start_pos=start_pos)  # (1, 1, vocab_size)
            next_token_logits = logits[:, -1, :]  # (1, vocab_size)

            # 在 temperature 和 top-k/p 之前应用惩罚是常见做法
            if repetition_penalty > 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits,
                    generated_tokens, # 只惩罚已生成的 token
                    repetition_penalty
                 )
            if frequency_penalty > 0.0:
                next_token_logits = self._apply_frequency_penalty(
                    next_token_logits,
                    generated_tokens, # 只惩罚已生成的 token
                    frequency_penalty
                 )

            # 应用采样策略
            next_token_logits = self._apply_temperature(next_token_logits, temperature)
            next_token_logits = self._apply_top_k(next_token_logits, top_k)
            next_token_logits = self._apply_top_p(next_token_logits, top_p)
            probs = F.softmax(next_token_logits, dim=-1)  # (1, vocab_size)
            next_token_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token_id.item())

            if stream:
                token_buffer.append(next_token_id.item()) # 加入 buffer
                # 尝试解码当前 buffer
                current_decoded_text = tokenizer.decode(token_buffer, skip_special_tokens=True)

                # 检查是否有新的、完整的文本可以 yield
                # 判断解码出的文本末尾是否是因字符不完整而产生的替换符，如果不完整，解码的将是�，此时不 yield
                if len(current_decoded_text) > len(yielded_text) and not current_decoded_text.endswith('\uFFFD'):
                    new_text_chunk = current_decoded_text[len(yielded_text):]
                    yield new_text_chunk
                    yielded_text = current_decoded_text # 更新已 yield 文本

            prev_id = next_token_id
        
        if not stream:
            full_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            yield full_response

    def configure_optimizer(self, weight_decay: float, learning_rate: float, betas: Tuple[float, float], device_type: str='cuda'):
        """
        配置 AdamW 优化器, 并对参数分组, 以应用不同的优化策略, 通常权重矩阵(2D及以上)应用权重衰减, 而偏置(bias)和层归一化(LayerNorm)的参数不应用权重衰减

        Args:
            weight_decay (float): 权重衰减系数
            learning_rate (float): 学习率
            betas (Tuple[float, float]): AdamW 优化器的 beta1 和 beta2 参数
            device_type (str): 设备类型, 用于指定优化器的设备

        Returns:
            torch.optim.AdamW: 优化器
        """
        # 获取模型参数并过滤不需要梯度的参数
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # 维度大于等于 2 的参数（如权重矩阵、嵌入层参数），这些参数会应用权重衰减
        # 这些参数通常是模型的主要可学习参数，直接影响模型的表达能力
        # 维度小于 2 的参数（如偏置、LayerNorm 参数），这些参数不会应用权重衰减
        # 这些参数通常用于调整模型的输出分布，而不是直接参与特征变换
        decay_params = []
        no_decay_params = []

        for name, param in param_dict.items():
            if param.dim() < 2 or "bias" in name or isinstance(param, torch.nn.LayerNorm):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # 创建优化器参数组
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        # 检查是否支持融合 AdamW
        # 融合 AdamW（Fused AdamW） 是 PyTorch 提供的一种优化 AdamW 实现的高性能版本，通过将多个操作融合为一个内核（kernel）来加速计算
        # 它特别适用于 GPU 上的大规模深度学习训练任务
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    def count_parameters(self):
        """
        计算模型可训练参数量

        Returns:
            参数量 (Tuple[int, str]): (精确参数量, 大致参数量)
        """
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        approx_params = f"{trainable_params / 1_000_000 if trainable_params < 1_000_000_000 else trainable_params / 1_000_000_000:.1f}{' M' if trainable_params < 1_000_000_000 else ' B'}"
        return trainable_params, approx_params
    
