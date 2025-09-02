# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

"""
文件用途: 定义 MiniMind 模型的配置类 (MiniMindConfig)
主要功能:
1. 继承自 HuggingFace 的 PretrainedConfig，用于保存和传递模型的超参数
2. 包含基础 Transformer 参数 (如 hidden_size, num_layers, vocab_size 等)
3. 包含可选的 Mixture-of-Experts (MoE) 相关配置

"""

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    """
    MiniMind 模型配置类
    作用:
        - 保存模型的超参数 (hidden_size, num_layers 等)
        - 指定模型结构 (attention heads, vocab size 等)
        - 支持 MoE (Mixture of Experts) 配置
    使用场景:
        在初始化模型时，传入该配置，决定模型的结构与功能
    """
    model_type = "minimind"  # 标识模型类型 (给 HuggingFace 框架使用)

    def __init__(
            self,
            dropout: float = 0.0,                  # dropout 概率
            bos_token_id: int = 1,                 # 句子起始 token
            eos_token_id: int = 2,                 # 句子结束 token
            hidden_act: str = 'silu',              # 激活函数
            hidden_size: int = 512,                # 隐藏层维度
            intermediate_size: int = None,         # FFN 中间层维度 (默认 None, 自动计算)
            max_position_embeddings: int = 32768,  # 最大序列长度
            num_attention_heads: int = 8,          # 注意力头数
            num_hidden_layers: int = 8,            # Transformer 层数
            num_key_value_heads: int = 2,          # KV cache 的头数
            vocab_size: int = 6400,                # 词表大小
            rms_norm_eps: float = 1e-05,           # RMSNorm 的 epsilon
            rope_theta: int = 1000000.0,           # RoPE 旋转位置编码的 theta
            flash_attn: bool = True,               # 是否启用 FlashAttention (高效注意力)
            ####################################################
            # 下方为 MoE (Mixture of Experts) 的专用参数
            # 当 use_moe = False 时，这些参数不生效
            ####################################################
            use_moe: bool = False,                 # 是否启用 MoE
            num_experts_per_tok: int = 2,          # 每个 token 选择多少个专家
            n_routed_experts: int = 4,             # 总专家数量
            n_shared_experts: int = 1,             # 共享专家数量 (所有 token 都可用)
            scoring_func: str = 'softmax',         # MoE 评分函数
            aux_loss_alpha: float = 0.1,           # 辅助损失权重系数
            seq_aux: bool = True,                  # 是否基于整个序列计算辅助损失
            norm_topk_prob: bool = True,           # 是否标准化 top-k 专家选择的概率
            **kwargs
    ):
        # 调用父类构造函数
        super().__init__(**kwargs)

        # ========== 基础 Transformer 配置 ==========
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn

        # ========== MoE (Mixture of Experts) 专用配置 ==========
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok    # 每个 token 选择的专家数量
        self.n_routed_experts = n_routed_experts          # 总专家数量
        self.n_shared_experts = n_shared_experts          # 共享专家数量
        self.scoring_func = scoring_func                  # 专家选择的评分函数 (默认 softmax)
        self.aux_loss_alpha = aux_loss_alpha              # 辅助损失系数
        self.seq_aux = seq_aux                            # 是否使用序列级别辅助损失
        self.norm_topk_prob = norm_topk_prob              # 是否对 top-k 概率归一化



# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

"""
文件用途：实现 MiniMind 模型主体（Transformer + 可选 MoE）与因果语言建模头
主要模块：
- RMSNorm：RMS 归一化
- RoPE：旋转位置编码的预计算与应用
- Attention：多头自注意力（支持 FlashAttention 与 KV Cache）
- FeedForward：前馈网络（SwiGLU 风格）
- MoEGate/MOEFeedForward：门控专家路由与推理/训练
- MiniMindBlock：单层 Transformer（Pre-Norm + 残差）
- MiniMindModel：堆叠多层 + 位置编码缓存 + 汇总辅助损失
- MiniMindForCausalLM：接入 HuggingFace 框架，输出 CausalLMOutputWithPast

受众：有一定 PyTorch/Transformers 基础的小白
"""

import math
import torch
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    """
    RMSNorm：仅归一化均方根，不减均值；参数量更小，数值更稳
    公式：y = w * x / sqrt(mean(x^2) + eps)
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数 γ

    def _norm(self, x):
        # 在最后一维做 RMS 归一化
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 用权重缩放，保持 dtype 与输入一致
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """
    预计算 RoPE 所需的 cos/sin（复用以减少重复计算）
    参数：
        dim: 每个注意力头的维度（必须偶数）
        end: 预计算的最大序列长度
        theta: RoPE 频率基数（越大可支持更长上下文）
    返回：
        freqs_cos, freqs_sin: [end, dim] 的表格，索引即为 position_id
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)          # 位置索引 0..end-1
    freqs = torch.outer(t, freqs).float()               # [end, dim/2]
    # 拼成 [end, dim]，前半与后半一致，便于 rotate_half 操作
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    对 (q, k) 应用 RoPE。这里使用“半维旋转”的实现：
    rotate_half([a, b]) = [-b, a]
    """
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # cos/sin 在 head 与 batch 维度上广播
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将 KV 头复制以匹配 Q 头数量（GQA 情形）
    输入 x: [bs, slen, kv_heads, head_dim]
    返回  : [bs, slen, kv_heads*n_rep, head_dim]
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    多头自注意力，支持：
      - GQA（num_key_value_heads <= num_attention_heads）
      - FlashAttention（PyTorch 2.0+ 的 scaled_dot_product_attention）
      - KV Cache（增量解码）
      - attention_mask（左上三角 + padding）
    """
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        # KV 头数（若未显式设置，等于注意力头数）
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # KV 复制倍数（GQA）
        self.head_dim = args.hidden_size // args.num_attention_heads

        # Q/K/V/O 投影
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # 若环境支持且开启开关，则使用高效注意力
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # 若需要，可在此打印警告：旧版 PyTorch 将退回慢速实现

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # (cos, sin)
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        """
        输入：
            x: [bs, seq_len, hidden]
            position_embeddings: 预切片好的 (cos[seq_len, dim], sin[seq_len, dim])
            past_key_value: 过往 KV 缓存 (k, v)，形状与当前批兼容
            use_cache: 是否返回 present KV 以便后续增量解码
            attention_mask: [bs, seq_len]，1=保留，0=mask（将被广播并转为 -inf）
        返回：
            output: [bs, seq_len, hidden]
            past_kv: (k, v) 或 None
        """
        bsz, seq_len, _ = x.shape

        # 线性投影并重排为多头
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 应用 RoPE
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # 追加 KV Cache（用于增量生成）
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 将 KV 扩展为与 Q 相同头数；并将 [bs, seq, heads, dim] 转为 [bs, heads, seq, dim]
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        # 路径1：FlashAttention（高效/内置掩码）
        if self.flash and seq_len != 1:
            dropout_p = self.dropout if self.training else 0.0
            attn_mask = None
            if attention_mask is not None:
                # 扩展为 [bs, heads, q_len, k_len]，True=保留，False=mask
                attn_mask = attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1)
                attn_mask = attn_mask.bool()
            output = F.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True
            )
        else:
            # 路径2：常规注意力实现（手动算 softmax + 下三角掩码 + padding 掩码）
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [bs, h, q, k]
            # 因果掩码：仅允许看见历史（含自身）
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)

            # padding 掩码（0 位置加 -1e9）
            if attention_mask is not None:
                extended = attention_mask.unsqueeze(1).unsqueeze(2)     # [bs,1,1,k]
                extended = (1.0 - extended) * -1e9
                scores = scores + extended

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv  # [bs, h, q, dim]

        # 汇合多头并做输出投影
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    """
    前馈网络：gate_proj(x) 激活后与 up_proj(x) 相乘（SwiGLU 风格），再 down_proj 回 hidden_size
    若未显式给出 intermediate_size，则按 8/3 * hidden_size 并对齐到 64 倍数
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)  # 64 对齐
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # 激活(gate) * 线性(up) 的门控，再投回 hidden_size，并做 dropout
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    """
    MoE 门控：
      - 对每个 token 打分并选出 top-k 专家（索引与权重）
      - 可选概率归一化（norm_topk_prob）
      - 训练时计算负载均衡辅助损失（aux_loss），缓解专家倾斜
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """
        输入：hidden_states [bs, seq, hidden]
        输出：
            topk_idx    [bs*seq, top_k]：被选中的专家索引
            topk_weight [bs*seq, top_k]：对应权重
            aux_loss：负载均衡损失（训练时>0，否则为0）
        """
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)                 # [bs*seq, h]
        logits = F.linear(hidden_states, self.weight, None)       # [bs*seq, n_experts]

        # 打分 -> 概率
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # 选择 top-k 专家
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 多专家时可对权重归一化（数值更稳）
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # 负载均衡辅助损失（两种：序列级/整体级）
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)  # [bs, seq*topk]

            if self.seq_aux:
                # 序列级：统计每个 batch 内各专家分配的相对频率，与平均分布对齐
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)  # [bs, seq, n_experts]
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(
                    1, topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # 整体级：全局 one-hot 频次与平均分布的差异
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)                   # 每个专家的平均被选中概率
                Pi = scores_for_aux.mean(0)                    # 门控平均概率
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """
    MoE 版本的 FFN：
      - 训练：按 top-k 将 token 分配给专家并加权合并（repeat_interleave + scatter）
      - 推理：按专家分组批量前向（moe_infer，无梯度）
      - 可选 shared_experts：对残差进行额外的共享专家修正
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 路由专家池
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)

        # 共享专家（所有 token 都会走）
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape

        # 1) 门控选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)

        # 2) 训练/推理两条路径
        x = x.view(-1, x.shape[-1])          # [bs*seq, hidden]
        flat_topk_idx = topk_idx.view(-1)    # [bs*seq*topk]

        if self.training:
            # 训练：复制 token，按专家掩码分别前向，再按权重聚合
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)  # 缓存各专家输出（节省显存）
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)
            # 按 topk 权重加权融合
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理：无梯度 + 分组批量前向，减少小批次 kernel 启动开销
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        # 3) 共享专家（残差修正）
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)

        # 暴露 aux_loss，供上层汇总
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        推理路径：按专家聚合连续索引批量计算，并将结果 scatter 回原位
        输入：
            x: [bs*seq, hidden]
            flat_expert_indices: [bs*seq*topk] 每个复制 token 对应的专家 id
            flat_expert_weights: [bs*seq*topk, 1] 对应权重
        返回：
            expert_cache: [bs*seq, hidden]
        """
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()                   # 将相同专家的样本聚在一起
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok   # 还原到原 token 下标（去除复制带来的展开）

        # 逐专家处理连续区间
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])  # 加权
            # 累加到对应 token 位置（可能多个专家贡献）
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    """
    Transformer 基本层（Pre-Norm）
    结构：x -> LN -> Attention -> +res -> LN -> FFN/MoE -> +res
    """
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.self_attn = Attention(config)
        self.layer_id = layer_id

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 按配置选择普通 FFN 或 MoE FFN
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        # Pre-Norm + Self-Attention
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual

        # Pre-Norm + FFN/MoE
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """
    MiniMind 主体：
      - 词嵌入 + 多层 MiniMindBlock + Final RMSNorm
      - 预计算并注册 RoPE cos/sin（缓存在 buffer）
      - 汇总所有 MoE 层的辅助损失 aux_loss
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算 RoPE 并注册为 buffer（不随状态字典保存）
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, theta=config.rope_theta)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        """
        MiniMindModel.forward
        输入：
            input_ids: [bs, seq]  输入 token 序列
            attention_mask: [bs, seq]  1=保留，0=mask（用于 padding/可变长）
            past_key_values: List[(k, v)]  每层的 KV Cache，用于增量生成
            use_cache: bool  是否在本次前向后返回新的 KV Cache
        返回：
            hidden_states: [bs, seq, hidden]  最后一层归一化后的隐藏状态
            presents: 长度为 num_layers 的列表；每层的 (k, v)（当 use_cache=False 时为 None）
            aux_loss: float  所有 MoE 层的辅助损失之和（未启用 MoE 时为 0）
        说明：
            - 支持“从 past 继续”的增量解码：start_pos 决定 RoPE 的偏移
            - RoPE 的 cos/sin 事先预计算并缓存在 buffer 中，按需切片
        """
        batch_size, seq_length = input_ids.shape
        # 若未提供 past_kv，则按层数填 None（常规全量前向）
        past_key_values = past_key_values or [None] * len(self.layers)

        # 起始位置（增量解码场景下，从已有 K/V 的长度继续位置编码）
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 词嵌入 + dropout：[bs, seq, hidden]
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # RoPE：根据 start_pos 与 seq_length 切片出本次需要的 cos/sin（形状：[seq, head_dim]）
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        # 逐层前向：Self-Attn(+KV) -> FFN/MoE（在 block 内部）
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)  # 可能是 (k, v) 或 None

        # 最终层归一化（RMSNorm）
        hidden_states = self.norm(hidden_states)

        # 汇总 MoE 辅助损失（未启用 MoE 时列表为空，求和为 0）
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        return hidden_states, presents, aux_loss
    
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    因果语言模型封装：
      - 集成 HuggingFace 训练/推理范式（支持 .generate）
      - 权重 tying：embedding 与 lm_head 共享权重
      - 统一返回 CausalLMOutputWithPast
    """
    config_class = MiniMindConfig  

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)

        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # 权重共享：输出层与词嵌入层共享参数，减少参数量并常带来泛化好处
        self.model.embed_tokens.weight = self.lm_head.weight

        # 预分配输出对象，避免频繁创建字典
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        """
        MiniMindForCausalLM.forward
        输入：
            input_ids, attention_mask, past_key_values, use_cache 同上
            logits_to_keep:
                - int：仅返回最后 K 个 time steps 的 logits（K=0 表示返回全部）
                - Tensor：可传入切片索引（如 torch.arange 等）
        返回：
            CausalLMOutputWithPast，其中包含：
              - last_hidden_state: [bs, seq, hidden]
              - logits: [bs, K(or seq), vocab]  依据 logits_to_keep 截取
              - aux_loss: MoE 辅助损失
              - past_key_values: 用于增量解码的 KV
        """
        # 主体模型前向：得到最后隐藏态、KV cache 以及 MoE 辅助损失
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )

        # 仅保留最后 K 个位置的 logits（或使用自定义切片），以降低显存/带宽占用
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, slice_indices, :])

        # 填充标准输出对象
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT
