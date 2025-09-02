"""
MiniMind 项目 - 预训练脚本 (1-pretrain.py)

功能：
1. 从零开始构建 MiniMind 模型并进行预训练；
2. 支持分布式训练 (DDP)、梯度累积、混合精度 (AMP)；
3. 训练过程中定期保存 checkpoint，采用半精度权重以减少存储占用；
4. 可选接入 wandb 记录 loss / 学习率 / 训练时间等指标。
核心目标：
  - 从零开始训练模型掌握基础语言能力，通过大规模无标注文本学习：
    1) 词汇、语法等语言结构规律；
    2) 世界知识（事实、常识等）；
    3) 基本的序列生成能力。
  - 为后续的监督微调（SFT）和偏好对齐（如DPO）提供基础模型权重。
适合读者：熟悉 PyTorch 的基础用户，想理解从数据加载到分布式预训练的完整流程。
"""

import os
import sys
__package__ = "trainer"
# 保证项目内模块可以正确导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset

warnings.filterwarnings('ignore')


# -------------------------
# 工具函数
# -------------------------
def Logger(content):
    """
    日志输出函数：
    - 单卡模式：直接打印；
    - 多卡 DDP 模式：仅 rank=0 的进程打印，避免重复日志。
    """
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    学习率调度 (Cosine Decay + warmup)。
    参数：
        current_step: 当前步数
        total_steps: 总训练步数
        lr: 初始学习率
    返回：
        当前步对应的学习率
    """
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


# -------------------------
# 训练单个 Epoch
# -------------------------
def train_epoch(epoch, wandb):
    """
    单个训练 Epoch：
    - 前向传播计算 loss；
    - 反向传播累积梯度；
    - 达到 accumulation_steps 时更新一次参数；
    - 定期打印日志、保存 checkpoint。
    """
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # -------- 数据准备 --------
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # -------- 动态学习率 --------
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # -------- 前向传播 --------
        with ctx:  # AMP 自动混合精度
            res = model(X)
            loss = loss_fct(res.logits.view(-1, res.logits.size(-1)), Y.view(-1)).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss  # 模型额外正则项
            loss = loss / args.accumulation_steps  # 梯度累积缩放

        # -------- 反向传播 --------
        scaler.scale(loss).backward()

        # 累积到一定步数后更新一次参数
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 反缩放梯度，便于裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 防止梯度爆炸
            scaler.step(optimizer)  # 参数更新
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # -------- 日志输出 --------
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1, args.epochs, step, iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # -------- 保存 checkpoint --------
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'

            # DDP 模式下需要取 module.state_dict()
            state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()

            # 半精度保存，节省空间
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            model.train()


# -------------------------
# 模型初始化
# -------------------------
def init_model(lm_config):
    """
    初始化模型与分词器：
    - 构建 MiniMindForCausalLM；
    - 打印可训练参数规模。
    """
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    # 从零初始化模型：
    # 预训练阶段模型权重随机生成，前几轮loss会较高（如10+），随着训练逐步下降，
    # 这与微调阶段（从预训练权重开始）的低初始loss形成明显区别。
    model = MiniMindForCausalLM(lm_config).to(args.device)
    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


# -------------------------
# 分布式初始化
# -------------------------
def init_distributed_mode():
    """
    初始化分布式训练 (DDP)：
    - NCCL 后端；
    - 获取 RANK/LOCAL_RANK/WORLD_SIZE；
    - 绑定当前进程到对应 GPU。
    """
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# -------------------------
# 主函数入口
# -------------------------
if __name__ == "__main__":
    # -------- 参数解析 --------
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=1, help="若仅做zero实验可设1；常规建议2~6")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")
    args = parser.parse_args()

    # -------- 配置初始化 --------
    lm_config = MiniMindConfig(hidden_size=args.hidden_size,
                               num_hidden_layers=args.num_hidden_layers,
                               use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # AMP 混合精度上下文
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # 是否 DDP 运行
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"

    # 随机种子
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        torch.cuda.manual_seed(base_seed + rank)

    # -------- wandb --------
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # -------- 模型 & 数据 --------
    model, tokenizer = init_model(lm_config)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              pin_memory=True,
                              drop_last=False,
                              shuffle=False,
                              num_workers=args.num_workers,
                              sampler=train_sampler)

    # -------- 优化器 & AMP --------
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 包装 DDP
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)

    # -------- 训练循环 --------
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
