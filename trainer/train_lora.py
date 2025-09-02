"""
MiniMind 项目 - LoRA SFT 训练脚本 (main_lora.py)

功能：
1. 在已完成的 Full SFT 基础上，加载全模型权重；
2. 使用 LoRA（低秩适配器）方法对模型进行增量微调（只训练 LoRA 层，冻结其他参数）；
3. 支持分布式训练 (DDP) 与梯度累积、混合精度训练；
4. 训练过程中定期保存 LoRA 权重，支持 wandb 记录日志。
核心优势：
  - 参数效率高：仅训练模型中添加的低秩适配器（LoRA层），冻结原有99%以上参数，大幅减少计算量和显存占用。
  - 避免灾难性遗忘：不修改基础模型权重，仅通过适配器微调，保留全量SFT模型的通用能力。
  - 灵活部署：可针对不同任务训练多个LoRA权重，部署时动态切换，无需保存完整模型副本。
适合读者：对 PyTorch 有一定基础的小白，可理解如何在大模型上应用 LoRA 微调。
"""

import os
import sys

__package__ = "trainer"
# 添加上级目录到 Python 搜索路径，保证可以导入项目中的其他模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import torch
from torch import optim, nn
import torch.distributed as dist
from contextlib import nullcontext
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import SFTDataset
from model.model_lora import load_lora, save_lora, apply_lora

warnings.filterwarnings('ignore')


# -------------------------
# 工具函数
# -------------------------
def Logger(content):
    """
    日志输出函数：
    - 在单卡模式下直接打印；
    - 在多卡 DDP 模式下，仅 rank=0 的进程打印，避免重复日志。
    """
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    动态学习率调度函数 (Cosine Decay + warmup)。
    参数:
        current_step: 当前迭代步数
        total_steps: 总训练步数
        lr: 初始学习率
    返回:
        当前步数对应的学习率
    """
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


# -------------------------
# 训练单个 Epoch
# -------------------------
def train_epoch(epoch, wandb):
    """
    单个训练 Epoch 的过程：
    - 前向传播计算 loss
    - 反向传播计算梯度
    - 只更新 LoRA 参数
    - 日志输出与 wandb 可视化
    - 定期保存 LoRA 权重
    """
    loss_fct = nn.CrossEntropyLoss(reduction='none')  # 逐 token 交叉熵损失
    start_time = time.time()

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 将输入数据移动到 GPU/CPU
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 根据当前 step 计算学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ------------------ 前向传播 ------------------
        with ctx:  # 混合精度加速 (autocast)
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),  # [batch*seq, vocab]
                Y.view(-1)                                # [batch*seq]
            ).view(Y.size())                               # reshape 回原始 [batch, seq_len]

            # 按 mask 位置统计平均 loss
            loss = (loss * loss_mask).sum() / loss_mask.sum()

            # 模型可能自带额外正则项 loss
            loss += res.aux_loss

            # 梯度累积
            loss = loss / args.accumulation_steps

        # ------------------ 反向传播 ------------------
        scaler.scale(loss).backward()

        # 梯度累积到一定步数才更新一次
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 反缩放梯度，便于裁剪
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)  # 防止梯度爆炸

            scaler.step(optimizer)  # 更新参数（仅 LoRA）
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # ------------------ 日志输出 ------------------
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

        # ------------------ 定期保存 LoRA 权重 ------------------
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            lora_save_path = f'{args.save_dir}/lora/{args.lora_name}_{lm_config.hidden_size}.pth'
            os.makedirs(os.path.dirname(lora_save_path), exist_ok=True)
            save_lora(model, lora_save_path)  # 【区别于 full_sft】只保存 LoRA 权重
            model.train()


# -------------------------
# 模型初始化
# -------------------------
def init_model(lm_config):
    """
    初始化模型与分词器：
    - 加载全量 SFT 权重作为基底
    - 返回模型和 tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    model = MiniMindForCausalLM(lm_config)

    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)

    # strict=False -> 允许部分参数缺失 (LoRA 参数后面会补充)
    model.load_state_dict(state_dict, strict=False)
    return model.to(args.device), tokenizer


# -------------------------
# 分布式初始化
# -------------------------
def init_distributed_mode():
    """
    初始化分布式训练 (DDP)
    - 使用 NCCL 后端
    - 设置 rank, local_rank, world_size
    - 绑定当前进程到对应 GPU
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
    parser = argparse.ArgumentParser(description="MiniMind SFT with LoRA")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="../dataset/lora_medical.jsonl")
    parser.add_argument("--lora_name", type=str, default="lora_medical", help="LoRA 权重保存名称 (区分任务，如医学/心理等)")
    args = parser.parse_args()

    # -------- 配置初始化 --------
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    ddp = int(os.environ.get("RANK", -1)) != -1  # 是否分布式训练
    ddp_local_rank, DEVICE = 0, "cuda:0"

    # 设置随机种子，保证可复现
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        torch.cuda.manual_seed(base_seed + rank)

    # -------- wandb 可视化 --------
    args.wandb_run_name = f"MiniMind-Lora-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # -------- 模型 & LoRA 初始化 --------
    model, tokenizer = init_model(lm_config)
    apply_lora(model)  # 给模型添加 LoRA 层

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    if not ddp or dist.get_rank() == 0:
        print(f"LLM 总参数量: {total_params}")
        print(f"LoRA 参数量: {lora_params_count}")
        print(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")

    # 冻结非 LoRA 参数，只训练 LoRA
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
    lora_params = [p for name, p in model.named_parameters() if 'lora' in name]

    # -------- 数据集 & 优化器 --------
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
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

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    iter_per_epoch = len(train_loader)

    # -------- 训练循环 --------
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
