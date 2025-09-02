"""
文件用途：
  基于 PyTorch/Transformers 的 **MiniMind 知识蒸馏 (Knowledge Distillation)** 训练脚本。
  核心逻辑：让小规模"学生模型"学习大规模"教师模型"的知识，通过结合：
    - 标准交叉熵损失（CE Loss，基于真实标签）
    - 蒸馏损失（KL散度，基于教师模型的软标签分布）
  支持：
    - 混合精度训练（AMP）
    - 分布式训练（DDP）
    - 余弦退火学习率调度
    - 梯度累积与梯度裁剪
  适用场景：通过蒸馏让小模型逼近大模型性能，平衡精度与效率。
"""
import os
import sys

# 设置包路径，保证可以从父目录导入模块
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings

import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import SFTDataset

warnings.filterwarnings('ignore')


def Logger(content):
    """
    日志输出函数
    - 只在非分布式模式或 rank=0（主进程）时打印日志
    """
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    余弦学习率调度函数
    - 公式：lr(t) = lr/10 + 0.5 * lr * (1 + cos(π * t / T))
      其中 t 为当前步数，T 为总步数
    - 特性：初始值为 lr/10，随训练逐步上升至 lr（约前1/4周期），随后余弦衰减至 lr/10
    - 作用：避免训练初期大学习率导致的不稳定，后期精细调整参数
    """
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def distillation_loss_fn(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    """
    知识蒸馏损失函数（KL 散度）
    - 原理：衡量学生模型与教师模型输出分布的差异，引导学生模仿教师的"软标签"（概率分布）
    - 温度参数：temperature 越大，教师输出分布越平滑（软化），学生更易学习全局模式
    - 温度平方：蒸馏损失需乘以 T² 以抵消温度对梯度的缩放（保持损失梯度量级与CE损失匹配）
    - 参数：
        student_logits: 学生模型输出的 [B, T, V] 特征
        teacher_logits: 教师模型输出的 [B, T, V] 特征（需 detach 避免梯度传播）
    """
    with torch.no_grad():
        # 教师输出做 softmax 概率分布（温度缩放）
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    # 学生模型输出 log-softmax（温度缩放）
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # 计算 KL 散度
    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    # 蒸馏损失需要乘以温度平方
    return (temperature ** 2) * kl


def train_epoch(epoch, wandb, alpha=0.0, temperature=1.0):
    """
    单个 epoch 的训练逻辑
    - 损失组合：loss = alpha * CE_loss + (1-alpha) * distill_loss
      alpha 控制真实标签与教师软标签的权重（alpha=1 退化为纯SFT，alpha=0 纯蒸馏）
    - 教师模型处理：
      1. 设为 eval 模式：固定 BatchNorm/ Dropout 等参数，保证输出稳定
      2. 禁用梯度计算：避免教师模型参数被更新（仅作为"知识提供者"）
    - 蒸馏温度：temperature 通常设为 2-10（平滑教师分布，增强泛化性）
    """
    start_time = time.time()

    # 教师模型设置为 eval 模式，且不计算梯度
    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 将 batch 数据移动到 GPU/CPU
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 更新学习率
        lr = get_lr(epoch * iter_per_epoch + step,
                    args.epochs * iter_per_epoch,
                    args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ================== 学生模型前向传播 ==================
        with ctx:
            res = model(X)
            student_logits = res.logits

        # ================== 教师模型前向传播 ==================
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(X).logits
                # 保证教师和学生 vocab 对齐（截断到学生的 vocab 大小）
                vocab_size_student = student_logits.size(-1)
                teacher_logits = teacher_logits[..., :vocab_size_student]

        # ================== 损失函数计算 ==================
        # 1) 交叉熵损失（基于 Ground-Truth）
        loss_mask_flat = loss_mask.view(-1)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            Y.view(-1),
            ignore_index=0,  # 忽略 padding token
            reduction='none'
        )
        ce_loss = torch.sum(ce_loss * loss_mask_flat) / loss_mask_flat.sum()
        if lm_config_student.use_moe:
            # 如果是 MoE 架构，还会加上辅助损失
            ce_loss += res.aux_loss

        # 2) 蒸馏损失（基于 KL 散度）
        if teacher_model is not None:
            # 只在有效 token 位置计算蒸馏损失
            distill_loss = distillation_loss_fn(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            distill_loss = torch.tensor(0.0, device=args.device)

        # 3) 总损失 = alpha * CE + (1-alpha) * Distill
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps

        # 反向传播（梯度累积）
        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            # 反梯度裁剪，防止梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 优化器更新
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # ========== 日志打印 ==========
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.4f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch,
                    args.epochs - 1,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                )
            )

            # wandb 记录指标
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "loss": loss.item(),
                    "ce_loss": ce_loss.item(),
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "lr": optimizer.param_groups[-1]['lr'],
                    "last-time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                })

        # ========== 定期保存模型 ==========
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/full_dist_{lm_config_student.hidden_size}{moe_path}.pth'

            # 分布式模式下要取出 module.state_dict
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 半精度保存，节省显存与存储
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            model.train()


def init_student_model(lm_config):
    """
    初始化学生模型
    - 从本地保存的全量 SFT checkpoint 加载参数
    """
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    model = MiniMindForCausalLM(lm_config)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    Logger(f'学生模型(LLM)总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    return model, tokenizer


def init_teacher_model(lm_config):
    """
    初始化教师模型
    - 一般是大模型，用于提供软标签（logits 分布）
    """
    model = MiniMindForCausalLM(lm_config)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    Logger(f'教师模型(LLM)总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    return model


def init_distributed_mode():
    """
    初始化分布式训练（DDP）
    - backend: nccl（GPU 通信）
    - 设置 rank / local_rank / world_size
    """
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    # ========== 参数解析 ==========
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--data_path", type=str, default="../dataset/sft_xxx.jsonl")

    args = parser.parse_args()

    # 定义学生模型与教师模型的配置
    lm_config_student = MiniMindConfig(hidden_size=512, num_hidden_layers=8)
    lm_config_teacher = MiniMindConfig(hidden_size=768, num_hidden_layers=16)

    # 输出目录
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # wandb 运行名称
    args.wandb_run_name = f"MiniMind-Dist-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 混合精度上下文
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    ddp = int(os.environ.get("RANK", -1)) != -1  # 是否启用分布式训练
    ddp_local_rank, DEVICE = 0, "cuda:0"

    # 固定随机种子
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        torch.cuda.manual_seed(base_seed + rank)

    # wandb 初始化
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # ========== 模型初始化 ==========
    model, tokenizer = init_student_model(lm_config_student)
    teacher_model = init_teacher_model(lm_config_teacher)

    # ========== 数据加载 ==========
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # 混合精度梯度缩放器
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 分布式封装
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # ========== 训练循环 ==========
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
