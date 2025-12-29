from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
import torch.nn.functional as F
from collections import Counter
from transformers.modeling_outputs import CausalLMOutputWithPast


class Generator:

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def _apply_temperature(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """根据温度缩放 logits, temperature 为 0 或负数时不应用"""
        if temperature > 0:
            return logits / temperature
        return logits
    
    def _apply_top_k(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """将概率分布限制在 top_k 个 token"""
        if top_k > 0:
            # 找出 top_k 个 logits 的值和索引
            top_k = min(top_k, logits.size(-1))  # 确保 top_k 不超过词汇表大小
            values, _ = torch.topk(logits, top_k, dim=-1)
            # 获取第 k 个元素的值作为阈值
            kth_value = values[:, -1].unsqueeze(-1)
            # 将所有低于阈值的 logits 设置为负无穷大，这样它们在 softmax 后概率为0
            logits = torch.where(logits < kth_value, torch.full_like(logits, -float('inf')), logits)
        return logits
    
    def _apply_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """应用 Top-P (Nucleus Sampling) 过滤 logits"""
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
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # 5. 使用 scatter_ 将移除标记应用回原始未排序的索引位置
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

    def _apply_repetition_penalty(self, logits: torch.Tensor, generated_token_ids: list[int], penalty: float):
        """
        应用存在惩罚, 只要是已生成过的 token, 就无差别应用惩罚

        Args:
            logits (torch.Tensor): 当前步的 logits (batch_size, vocab_size), 这里 batch_size=1
            generated_token_ids (list[int]): 到目前为止已生成的 token id 列表
            penalty (float): 重复惩罚因子 (>= 1.0)

        Returns:
            torch.Tensor: 应用惩罚后的 logits
        """
        if not generated_token_ids or penalty == 1.0:
            return logits

        # 去重，确保每个 token 只应用一次惩罚
        unique_token_ids = list(dict.fromkeys(generated_token_ids))  # 保持顺序的去重
        
        # 计算 logits 张量中所有对应于已经生成的 token 的原始分数
        score = torch.gather(logits, 1, torch.tensor([unique_token_ids], device=logits.device))  # (1, num_unique)

        # 原始 logits 可正可负，如果是正，就减小，如果是负，就更负
        # 如果 score < 0, 则 score = score * penalty
        # 如果 score > 0, 则 score = score / penalty
        score = torch.where(score < 0, score * penalty, score / penalty)

        logits.scatter_(1, torch.tensor([unique_token_ids], device=logits.device), score)
        return logits
    
    def _apply_frequency_penalty(self, logits: torch.Tensor, generated_token_ids: list[int], penalty: float):
        """
        应用频率惩罚 (Frequency Penalty), 惩罚力度与 token 在已生成序列中的出现次数成正比

        Args:
            logits (torch.Tensor): 当前步的 logits (batch_size, vocab_size), 这里 batch_size=1
            generated_token_ids (list[int]): 到目前为止已生成的 token id 列表
            penalty (float): 频率惩罚因子 (通常 >= 0.0) 0.0 表示无惩罚, 正值会降低重复 token 的 logit。

        Returns:
            torch.Tensor: 应用惩罚后的 logits
        """
        if not generated_token_ids or penalty == 0.0:
            return logits

        # 1. 计算已生成 token 的频率
        token_counts = Counter(generated_token_ids)  # {token_id: count, ...}
        # 提取出现过的 unique token id 和它们的频率
        unique_ids = list(token_counts.keys())
        frequencies = [token_counts[token_id] for token_id in unique_ids]

        # 转换为 tensor
        unique_ids_tensor = torch.tensor([unique_ids], device=logits.device)  # Shape: (1, num_unique_ids)
        # 确保 frequencies tensor 类型与 logits 一致，以便进行数学运算
        frequencies_tensor = torch.tensor([frequencies], dtype=logits.dtype, device=logits.device)  # Shape: (1, num_unique_ids)

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
        input_ids: torch.LongTensor,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
    ):
        """
        生成文本

        Args:
            input_ids (torch.LongTensor): 输入的 token ids, 当前仅支持单样本输入, 形状为 (1, seq_len)
            max_new_tokens (int, optional): 最大生成的新 token 数量, 默认为 512
            temperature (float, optional): 控制采样随机性的温度值 ( > 0), 值越小越确定，值越大越随机, 默认为 1.0, 为 0 时使用贪婪解码
            top_k (int, optional): Top-K 采样, 仅考虑概率最高的 K 个 token, 0 表示禁用, 默认为 20
            top_p (float, optional): Top-P 采样, 仅考虑累积概率超过 P 的最小 token 集, 0 或 1.0 表示禁用, 默认为 0.9
            repetition_penalty (float, optional): 重复惩罚因子 (>= 1.0), 默认为 1.0, 表示无惩罚, 推荐 1.0-1.5
            frequency_penalty (float, optional): 频率惩罚因子 (>= 0.0), 默认为 0.0, 表示无惩罚, 推荐 0.0-0.5
            
        Yields:
            str: 逐个 yield 生成的文本块
        """
        # --- 参数验证 ---
        if temperature < 0:
            raise ValueError("`temperature` must >= 0")
        if top_k < 0:
            raise ValueError("`top_k` must >= 0")
        if top_p < 0.0 or top_p > 1.0:
            raise ValueError("`top_p` must be in the range [0.0, 1.0]")
        if max_new_tokens is not None and max_new_tokens <= 0:
            raise ValueError("`max_new_tokens` must > 0")
        if repetition_penalty < 1.0:
            raise ValueError("`repetition_penalty` (repetition penalty factor) must >= 1.0")
        if frequency_penalty < 0.0:
            raise ValueError("`frequency_penalty` (frequency penalty factor) must >= 0.0")
        if repetition_penalty != 1.0 and frequency_penalty != 0.0:
            raise ValueError("It is not recommended to use both `repetition_penalty` (repetition penalty factor) and `frequency_penalty` (frequency penalty factor) at the same time")
        if input_ids.shape[0] != 1 or len(input_ids.shape) != 2:
            raise ValueError("`input_ids` must be a 2D tensor with shape (1, seq_len)")
        input_ids = input_ids.to(self.model.device)

        # --- 生成循环 ---
        # 1) prefill 阶段，将 input_ids 一次性输入，前向传播的过程中会将前 seq_len-1 个 token 的 KV 加入到模型的 cache 中，mask 在模型内部实现
        outputs: CausalLMOutputWithPast = self.model(
            input_ids=input_ids[:, :-1],
            use_cache=True,
            )
        past_key_values = outputs.past_key_values

        # 2) decode 阶段，从 input_ids 的最后一个 token 开始，每次只输入一个 token，将当前 token 的 KV 补到 cache 中，并预测下一个 token
        generated_tokens = []  # 收集已生成的 token ids
        decoded_tokens = []  # 记录已经成功解码的 token ids，用于增量解码
        input_ids = input_ids[:, [-1]]  # (1, 1)
        yielded_text = ""  # 记录已经 yield 出去的文本

        for _ in range(max_new_tokens):
            outputs: CausalLMOutputWithPast = self.model(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )
            next_token_logits = outputs.logits[:, -1, :]  # (1, vocab_size)

            # 在 temperature 和 top-k/p 之前应用重复惩罚或频率惩罚
            if repetition_penalty > 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits,
                    generated_tokens,  # 只惩罚已生成的 token
                    repetition_penalty
                 )
            if frequency_penalty > 0.0:
                next_token_logits = self._apply_frequency_penalty(
                    next_token_logits,
                    generated_tokens,  # 只惩罚已生成的 token
                    frequency_penalty
                 )

            # 应用采样策略
            next_token_logits = self._apply_temperature(next_token_logits, temperature)
            next_token_logits = self._apply_top_k(next_token_logits, top_k)
            next_token_logits = self._apply_top_p(next_token_logits, top_p)
            
            # temperature=0 时使用贪婪解码，否则使用采样
            if temperature == 0.0:
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (1, 1)
            else:
                probs = F.softmax(next_token_logits, dim=-1)  # (1, vocab_size)
                next_token_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
            generated_tokens.append(next_token_id.item())

            # 对于多字节编码（如 UTF-8 中的中文、emoji 等），单个 token id 可能只代表字符的一部分。
            # 直接解码单个 token id 可能会产生无效字符或 unicode 替换符（例如 �），因此需要累计直到能够完整解码一个 token 才 yield
            # 增量解码优化，只解码新增的 token
            if len(generated_tokens) > len(decoded_tokens):
                new_tokens = generated_tokens[len(decoded_tokens):]
                new_decoded_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                # 检查解码是否完整，即末尾没有 unicode 替换符
                if not new_decoded_text.endswith('\uFFFD'):
                    # 解码完整，将新增的 token 添加到 decoded_tokens
                    decoded_tokens.extend(new_tokens)
                    current_decoded_text = yielded_text + new_decoded_text
                else:
                    current_decoded_text = None
            else:
                current_decoded_text = None

            # 检查是否有新的、完整的文本可以 yield
            if current_decoded_text is not None and len(current_decoded_text) > len(yielded_text):
                new_text_chunk = current_decoded_text[len(yielded_text):]
                yield new_text_chunk
                yielded_text = current_decoded_text  # 更新已 yield 文本

            input_ids = next_token_id