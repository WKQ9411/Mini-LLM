import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from .basemodel import BaseModelArgs, BaseModel


# ----------------------------------------------【参数配置】---------------------------------------------- #
@dataclass
class Llama3ModelArgs(BaseModelArgs):
    """
    配置模型参数

    Attributes:
        max_batch_size (int): 最大批量大小
        max_seq_len (int): 最大序列长度
        vocab_size (int): 词典大小
        dim (int): 嵌入维度
        n_layers (int): Block 层数
        n_heads (int): query头数
        n_kv_heads (int): key-value头数
        norm_eps (float): 正则化系数
        multiple_of (int): 用于计算中间维度
        ffn_dim_multipler (int): MLP中间维度的倍数
        rope_theta (int): ROPE的底数
    """
    max_batch_size: int = 32
    max_seq_len: int = 512
    vocab_size: int = -1  # 加载模型时传入
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    norm_eps: float = 1e-6
    multiple_of: int = 256
    ffn_dim_multipler = None
    rope_theta: int = 10000.0


# ----------------------------------------------------------------------------
# RMSNorm
# ----------------------------------------------------------------------------
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 输入维度为(batch_size, seq_len, dim),对最后一个维度进行归一化
        # x.pow(2)用于对每个元素进行平方运算
        # torch.rsqrt()是计算倒数平方根
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# ----------------------------------------------------------------------------
# ROPE
# ----------------------------------------------------------------------------
# 预先计算旋转矩阵的各个角度
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """计算频率矩阵, 并将其表示为复数的极坐标表示, 函数名中的cis指cos(θ)+i·sin(θ), 表示一个复数位于单位圆上的位置

    Args:
        dim (int): Embedding的维度
        end (int): 序列长度
        theta (float, optional): 计算θ的底数值【θ=10000^(-2i/d)】. Defaults to 10000.0.

    Returns:
        代表各个位置m旋转角度的复数矩阵, 形状为(end, dim//2), 每两个维度对应一个旋转角度
    """
    # 计算旋转矩阵中的θ值, 原文中θ=10000^(-2i/d)【这里源代码[: (dim // 2)]的操作似乎是冗余的？】
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # 计算位置信息m的序列
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)

    # torch.outer用于计算外积, 就得到不同位置m和不同θ值的所有组合m*θ
    # 得到的freqs矩阵形状为(end, dim//2), 索引含义为freqs[mi][θi]=mi*θi
    freqs = torch.outer(t, freqs)

    # 生成一个模长为1, 幅角为freqs的复数矩阵
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


# 调整freqs_cis以方便其与x进行广播计算
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """调整freqs_cis以方便其与x进行广播计算

    Args:
        freqs_cis (torch.Tensor): 旋转矩阵, 初始形状为(end, head_dim//2)
        x (torch.Tensor): query, 初始形状为(batch_size, seq_len, n_heads, head_dim//2)

    Returns:
        调整形状后的旋转矩阵, 形状为(1, seq_len, 1, head_dim//2)
    """
    ndim = x.ndim  # 获取x的维度数
    assert 0 <= 1 < ndim  # 确保x至少为2维【这里0<=1似乎也是冗余】

    # x形状一般为(batch_size, seq_len, n_heads, head_dim//2)
    # 这里确保freqs_cis与x的seq_len, head_dim//2维度一致, RoPE是对每个头分别进行的
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])

    # 将第二维度和最后一维度分别变为seq_len和head_dim//2, 其余维度均为1，即(1, seq_len, 1, head_dim//2)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


# 应用RoPE
def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
    """应用RoPE, llama3是通过转换成复数形式来旋转角度的

    Args:
        xq (torch.Tensor): query
        xk (torch.Tensor): key
        freqs_cis (torch.Tensor): 旋转矩阵

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: query和key的旋转结果
    """
    # 将xq和xk由(batch_size, seq_len, n_(kv)_heads, head_dim)转换为(batch_size, seq_len, n_(kv)_heads, head_dim//2, 2)
    # 即每个头的维度两两一组, 以此作为复数的实部和虚部, 转换为复数
    # xq_和xk_的形状为(batch_size, seq_len, n_(kv)_heads, head_dim//2), 里面保存的是复数, 这样转换后最后一维就与freqs_cis的最后一维一致了
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # (batch_size, seq_len, n_heads, head_dim//2)
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  # (batch_size, seq_len, n_kv_heads, head_dim//2)

    # 按照xq_将freqs_cis的维度变为(1, seq_len, 1, head_dim//2)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # 通过复数乘法实现角度旋转
    # 复数张量转换为实数张量后, 通常为(..., 2)的形状, 即最后一维代表实部与虚部
    # 因此使用flatten将索引为3的维度展平, 形状由(batch_size, seq_len, n_(kv)_heads, head_dim//2, 2)变为(batch_size, seq_len, n_(kv)_heads, head_dim)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)  # (batch_size, seq_len, n_heads, head_dim)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)  # (batch_size, seq_len, n_kv_heads, head_dim)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# ----------------------------------------------------------------------------
# Attention (GQA/KV Cache)
# ----------------------------------------------------------------------------
# 复制kv heads
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """当key和value的头数量n_kv_heads小于查询头(query heads)数量时, 需要将key和value进行重复, 以匹配查询头的数量

    Args:
        x (torch.Tensor): key/value: (batch_size, seq_len, n_kv_heads, head_dim)
        n_rep (int): 重复的次数

    Returns:
        key/value: (batch_size, seq_len, n_kv_heads*n_rep, head_dim)
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # x[:, :, :, None, :]用于插入一个维度, 使得形状变为: (batch_size, seq_len, n_kv_heads, 1, head_dim)
    # expand()用于扩展张量的维度, 使得形状变为: (batch_size, seq_len, n_kv_heads, n_rep, head_dim)
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# 源码仅用于推理, 且使用了分布式训练方法, 这里进行了部分修改
class Attention(nn.Module):
    def __init__(self, args: Llama3ModelArgs):
        super().__init__()
    
        self.args = args
        self.n_heads = args.n_heads  # query的头数
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads  # key/value的头数, 未设置kv头数时, 默认与n_heads一致, 即MHA
        self.head_dim = args.dim // args.n_heads
        self.n_rep = args.n_heads // self.n_kv_heads  # query heads必须是kv heads的整数倍

        # 初始化权重矩阵
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)  # GQA也产生n_heads个头的attention

        # 实现KV Cache, 用于存储KV矩阵, 包括prompt部分和生成部分的KV, 因此形状为(max_batch_size, max_seq_len*2, n_kv_heads, head_dim)
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len*2, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len*2, self.n_kv_heads, self.head_dim))

    # 源代码仅有推理模式, 这里区分训练与推理
    def forward(self, x: torch.Tensor, start_pos, inference, freqs_cis):
        # 输入维度为(batch_size, seq_len, dim)
        bsz, seq_len, _ = x.shape
        # mask只在训练时使用, 由于使用了KV Cache, 因此在推理模式下不需要使用mask
        mask = None
        
        # 由于只对线性层只对dim做变换，因此实际上跟seq_len无关，可以接受任意长度的seq_len
        xq = self.wq(x)  # (batch_size, seq_len, dim) -> (batch_size, seq_len, n_heads * head_dim)
        xk = self.wk(x)  # (batch_size, seq_len, dim) -> (batch_size, seq_len, n_kv_heads * head_dim)
        xv = self.wv(x)  # (batch_size, seq_len, dim) -> (batch_size, seq_len, n_kv_heads * head_dim)

        # 转换形状
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)      # (batch_size, seq_len, n_heads, head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)   # (batch_size, seq_len, n_kv_heads, head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)   # (batch_size, seq_len, n_kv_heads, head_dim)

        # 推理模式, KV Cache仅在推理模式下使用
        if inference:
            # 【推理模式中使用max_seq_len*2是为了同时容纳prompt和生成内容, 因此需要乘以2】
            # 【推理时只考虑当前位置token在序列长度范围内的旋转矩阵】
            freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
            
            # xq:(batch_size, seq_len, n_heads, head_dim), xk:(batch_size, seq_len, n_kv_heads, head_dim)
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            # 将当前位置新产生的key和value存入KV Cache
            self.cache_k[:bsz, start_pos:start_pos + seq_len] = xk
            self.cache_v[:bsz, start_pos:start_pos + seq_len] = xv

            # 取出所有的历史key和value
            keys = self.cache_k[:bsz, :start_pos + seq_len]
            values = self.cache_v[:bsz, :start_pos + seq_len]

            # 使用repeat_kv函数将key/value的维度变为与query一致
            keys = repeat_kv(keys, self.n_rep)  # (batch_size, seq_len, n_heads, head_dim)
            values = repeat_kv(values, self.n_rep)  # (batch_size, seq_len, n_heads, head_dim)

            if seq_len > 1:  # 此时必定是prefill阶段
                mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device)  # (seq_len, seq_len)的全为负无穷的张量
                mask = torch.triu(mask, diagonal=1).to(x.device)

        # 训练模式, 无需使用KV Cache
        else:
            # xq:(batch_size, seq_len, n_heads, head_dim), xk:(batch_size, seq_len, n_kv_heads, head_dim)
            # 预训练时，这里使训练的输入序列和freq_cis都按照max_seq_len进行计算，因此预训练的输入长度必须为max_seq_len
            # 而推理时，进行了freqs_cis = freqs_cis[start_pos : start_pos + seq_len]截取，因此可以接受任意长度的输入序列
            # 类比到transformer的绝对位置编码，实际也是可以计算更大的freqs_cis，然后根据序列长度来截取的
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

            # 使用repeat_kv函数将key/value的维度变为与query一致
            keys = repeat_kv(xk, self.n_rep)  # (batch_size, seq_len, n_heads, head_dim)
            values = repeat_kv(xv, self.n_rep)  # (batch_size, seq_len, n_heads, head_dim)

            # 生成因果掩码(causal mask / sequence mask)
            mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device)  # (seq_len, seq_len)的全为负无穷的张量
            mask = torch.triu(mask, diagonal=1).to(x.device)  # 生成上三角矩阵, 对角线上方不变, 对角线及下方全为0

        # 调整形状进行注意力计算
        xq = xq.transpose(1,2)  # (batch_size, n_heads, seq_len, head_dim)
        keys = keys.transpose(1,2)  # (batch_size, n_heads, seq_len, head_dim)
        values = values.transpose(1,2)  # (batch_size, n_heads, seq_len, head_dim)

        # 计算注意力分数
        scores = torch.matmul(xq, keys.transpose(2,3)).to(x.device)/math.sqrt(self.head_dim)  # (batch_size, n_heads, seq_len, seq_len)
        if mask is not None:
            scores = scores + mask

        # 应用softmax
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # 乘value
        output = torch.matmul(scores, values).to(x.device)  # (batch_size, n_heads, seq_len, head_dim)

        # (batch_size, n_heads, seq_len, head_dim) -> (batch_size, seq_len, n_heads * head_dim)
        output = output.transpose(1,2).contiguous().view(bsz, seq_len, -1)

        return self.wo(output)  # (batch_size, seq_len, n_heads * head_dim) -> (batch_size, seq_len, dim)

# ----------------------------------------------------------------------------
# FFN
# ----------------------------------------------------------------------------
# 前馈网络
class FeedForward(nn.Module):
    def __init__(self, dim:int, hidden_dim:int, multiple_of:int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        self.dim = dim

        # 以下hidden dim计算方式源于源码, 用于保证hidden dim是256的倍数
        # 其中传入的初始hidden dim为4 * dim, multiple_of为256
        hidden_dim = int(2 * hidden_dim/3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # 定义线性层
        self.w1 = nn.Linear(self.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, self.dim, bias=False)
        self.w3 = nn.Linear(self.dim, hidden_dim, bias=False)

    def forward(self, x):
        # (batch_size, seq_len, dim)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))  # silu是beta=1的Swish

# ----------------------------------------------------------------------------
# Transformer Block
# ----------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: Llama3ModelArgs):
        super().__init__()
    
        # 定义参数
        self.args = args
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.layer_id = layer_id
    
        # 定义attention部分
        self.attention = Attention(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)

        # 定义feedforward部分
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multipler
            )
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, start_pos, inference, freqs_cis):
        # (batch_size, seq_len, dim)
        h = x + self.attention(self.attention_norm(x), start_pos, inference, freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

# ----------------------------------------------------------------------------
# Llama
# ----------------------------------------------------------------------------
class Llama3Model(BaseModel):
    model_name = "mini_llama3"
    def __init__(self, params: Llama3ModelArgs):
        super().__init__()
        
        # 定义参数
        self.params = params

        # 定义embedding层
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        # 定义transformer模块
        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id=layer_id, args=params))

        # 定义输出模块的RMSNorm及线性层
        self.norm = RMSNorm(params.dim, eps = params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # 在模型初始化时，预先计算好旋转矩阵，区分训练时使用的旋转矩阵和推理时使用的旋转矩阵
        self.head_dim = params.dim // params.n_heads
        freqs_cis_for_train = precompute_freqs_cis(
            dim=self.head_dim, 
            end=self.params.max_seq_len, 
            theta=self.params.rope_theta
            )  # (max_seq_len, head_dim//2)
        freqs_cis_for_inference = precompute_freqs_cis(
            dim=self.head_dim, 
            end=self.params.max_seq_len*2, 
            theta=self.params.rope_theta
            )  # (max_seq_len*2, head_dim//2)
        self.register_buffer('freqs_cis_for_train', freqs_cis_for_train)
        self.register_buffer('freqs_cis_for_inference', freqs_cis_for_inference)
        self.freqs_cis = None

    def forward(self, input_ids, targets=None, start_pos=0):

        # start_pos: 推理模式下, 当前token的位置索引
        # x:(batch_size, seq_len) -> h:(batch_size, seq_len, dim)
        h = self.tok_embeddings(input_ids)

        # 根据是否传入targets，确定是否是推理模式
        if targets is None:
            inference = True
            self.freqs_cis = self.freqs_cis_for_inference
        else:
            inference = False
            self.freqs_cis = self.freqs_cis_for_train

        # 依次传入各个transformer block
        for layer in self.layers:
            h = layer(h, start_pos, inference, self.freqs_cis)

        # 传入输出模块
        h = self.norm(h)
        # h:(batch_size, seq_len, dim) -> logits:(batch_size, seq_len, vocab_size)
        logits = self.output(h).float()
        loss = None

        # 如果是训练模式, 就计算loss
        if targets is None:
            loss = None
        else:
            # logits:(batch_size, seq_len, vocab_size)
            # targets:(batch_size, seq_len)
            loss = F.cross_entropy(logits.view(-1, self.params.vocab_size), targets.view(-1))

        return logits, loss, None  # 第三部分用于给 tensorboard 提供需要记录的模型内部变量