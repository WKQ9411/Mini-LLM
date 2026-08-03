from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0  # 默认禁用
    top_p: float = 1.0  # 默认禁用
    repetition_penalty: float = 1.0  # 默认禁用
    frequency_penalty: float = 0.0  # 默认禁用
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("`temperature` must >= 0")
        if self.top_k < 0:
            raise ValueError("`top_k` must >= 0")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("`top_p` must be in the range [0.0, 1.0]")
        if self.repetition_penalty < 1.0:
            raise ValueError("`repetition_penalty` must >= 1.0")
        if self.frequency_penalty < 0.0:
            raise ValueError("`frequency_penalty` must >= 0.0")
        if self.repetition_penalty != 1.0 and self.frequency_penalty != 0.0:
            raise ValueError("repetition_penalty and frequency_penalty cannot be enabled together")
        if self.max_tokens <= 0:
            raise ValueError("`max_tokens` must > 0")
