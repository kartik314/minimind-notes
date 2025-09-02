"""
文件用途：
  使用 Direct Preference Optimization (DPO) 对 MiniMind 进行偏好对齐训练。
  核心原理：
    - 无需强化学习（RL）中的奖励模型（RM）和PPO复杂流程，直接通过成对偏好数据（chosen/rejected）优化模型。
    - 核心思想：让模型对“人类偏好的回答（chosen）”输出概率高于“不偏好的回答（rejected）”，同时以参考模型（ref_model）为基准。
  特点：
    - 参考模型 ref_model 固定不训练；被训练模型与其对比 log 概率比
    - 支持 AMP 混合精度、DDP 分布式、余弦退火学习率
    - 数据集为成对样本（chosen / rejected），由 DPODataset 提供
    - 仅对 assistant 回复区域计入损失（由 mask 控制）

"""
import os
import sys
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
from dataset.lm_dataset import DPODataset


warnings.filterwarnings('ignore')


def Logger(content):
    """仅在非分布式或 rank=0 进程打印日志。"""
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """余弦退火调度：lr/10 为下限，训练中间抬升后再缓慢下降。"""
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def logits_to_probs(logits, labels):
    """
    取出 labels 位置对应的 token 对数概率（log-prob）
      - logits: 模型输出的 [B, T, V] 特征（未归一化）
      - labels: 目标序列 [B, T]（通常是 chosen/rejected 的回答文本）
    返回：
      - probs: [B, T] 每个 token 位置的对数概率（用于后续计算序列级平均概率）
    作用：将模型输出转换为目标序列的概率分布，用于衡量模型对该序列的“置信度”
    """
    log_probs = F.log_softmax(logits, dim=2)                     # -> [B, T, V]
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)  # 取出标签对应列
    return probs


def dpo_loss(ref_probs, probs, mask, beta):
    """
    DPO 目标函数（batch 内前半为 chosen，后半为 rejected）：
      - ref_probs: 参考模型在标签上的 log-prob，形状 [B, T]
      - probs:     被训练模型在标签上的 log-prob，形状 [B, T]
      - mask:      仅对回答区域（assistant）计损，形状 [B, T]，取值{0,1}
      - beta:      温度超参（越大越强调“偏好差异”）

    步骤：
      1) 对每条样本，按 mask 求均值 log-prob
      2) 切分成 chosen/rejected 两半
      3) 计算 log 概率比差值：(π_c - π_r) - (π_ref_c - π_ref_r)
      4) 用 -logsigmoid(β * ·) 求损失并取均值
    """
    # 对有效 token（mask=1）求“平均” log-prob（避免长序列偏置）
    seq_lengths = mask.sum(dim=1, keepdim=True)                 # [B, 1]
    ref_probs = (ref_probs * mask).sum(dim=1) / seq_lengths.squeeze()  # [B]
    probs = (probs * mask).sum(dim=1) / seq_lengths.squeeze()          # [B]

    # 切分 batch：前半 chosen，后半 rejected
    batch_size = ref_probs.shape[0]
    chosen_ref_probs = ref_probs[:batch_size // 2]
    reject_ref_probs = ref_probs[batch_size // 2:]
    chosen_probs = probs[:batch_size // 2]
    reject_probs = probs[batch_size // 2:]

    # DPO 的核心：被训模型与参考模型的“相对偏好比”差异
    pi_logratios = chosen_probs - reject_probs
    ref_logratios = chosen_ref_probs - reject_ref_probs
    logits = pi_logratios - ref_logratios

    # 损失：-log σ(β * logits)
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def train_epoch(epoch, wandb):
    """
    单个 epoch 训练：
      - 拼接 chosen/rejected 批次
      - 前向得到 ref_model 与 model 在标签处的 log-prob
      - 计算 DPO 损失并反传
      - 日志与定期保存
    """
    start_time = time.time()
    for step, batch in enumerate(train_loader):
        # 取出成对样本并拼接（约定前半 chosen，后半 rejected）
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)

        x = torch.cat([x_chosen, x_rejected], dim=0)   # [B, T]
        y = torch.cat([y_chosen, y_rejected], dim=0)   # [B, T]
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)  # [B, T]

        # 学习率调度
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            # 参考模型仅推理，不参加训练
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_probs = logits_to_probs(ref_logits, y) * mask

            # 当前模型前向
            outputs = model(x)
            logits = outputs.logits
            probs = logits_to_probs(logits, y) * mask

            # DPO 损失（mask 内的平均 log-prob，chosen vs rejected）
            loss = dpo_loss(ref_probs, probs, mask, beta=0.1)
            loss = loss / args.accumulation_steps

        # AMP 反传 + 累积
        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 日志
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 定期半精度保存
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/rlhf_{lm_config.hidden_size}{moe_path}.pth'

            state_dict = model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict()
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    """
    初始化待训练模型与参考模型：
      - 待训练模型：从SFT（有监督微调）权重初始化，将通过DPO优化偏好。
      - 参考模型（ref_model）：与待训练模型初始参数完全一致（同SFT权重），但固定不训练，用于：
        1) 提供基准概率分布，避免模型在偏好优化中遗忘原有能力。
        2) 计算“相对偏好差异”，使DPO损失更稳定。
      - 两者均移动到目标设备，并设置参考模型为 eval 模式（禁用 dropout 等随机操作）。
    返回：model, ref_model, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    model = MiniMindForCausalLM(lm_config)

    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)

    # 参考模型
    ref_model = MiniMindForCausalLM(lm_config)
    ref_model.load_state_dict(state_dict, strict=False)
    ref_model.eval()
    ref_model.requires_grad_(False)

    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    ref_model = ref_model.to(args.device)

    return model, ref_model, tokenizer


def init_distributed_mode():
    """
    初始化 DDP：设置 backend / rank / local_rank / world_size，并绑定设备。
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
    # ================== 参数解析 ==================
    parser = argparse.ArgumentParser(description="MiniMind RLHF (DPO)")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    # 提示：SFT 长度 512 学习率可 5e-6→5e-7；DPO（偏好对齐）建议更小 lr（≤1e-8）避免灾难遗忘
    parser.add_argument("--learning_rate", type=float, default=1e-8)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-RLHF-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=1024, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl")
    args = parser.parse_args()

    # ================== 训练前准备 ==================
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)

    # 目录
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Full-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # AMP 上下文（CPU 不支持）
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # 是否分布式
    ddp = int(os.environ.get("RANK", -1)) != -1
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

    # wandb
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # ================== 模型 & 数据 ==================
    model, ref_model, tokenizer = init_model(lm_config)

    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,                 # 分布式下通常由 sampler 控制打乱
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # AMP 梯度缩放 & 优化器
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # DDP 包装
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # ================== 训练循环 ==================
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
