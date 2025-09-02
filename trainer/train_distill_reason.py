"""
文件用途：
  基于 PyTorch/Transformers 的 **MiniMind 推理蒸馏 (Distill Reasoning)** 训练脚本。
  核心目标：让模型学习“推理过程”（如分步思考），但在输出时不直接暴露推理内容，而是生成简洁回答。
  支持：
    - AMP 混合精度训练
    - 单机/分布式 DDP 训练
    - 余弦退火学习率
    - SFT 风格数据集 (jsonl conversations)，动态 loss mask
    - 对 <think>/<answer> 等特殊标签位置附加额外损失权重（抑制把“思考过程”原样输出）

读者对象：
  有基础的深度学习/分布式训练小白，便于快速上手与二次开发。
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
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import SFTDataset

warnings.filterwarnings('ignore')


def Logger(content):
    """
    仅在单卡或 DDP 的 rank=0 进程打印日志，避免重复打印。
    """
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    余弦退火 + 最低 lr/10：
    lr(t) = lr/10 + 0.5 * lr * (1 + cos(pi * t / T))
    """
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    """
    单个 epoch 的训练逻辑：
      - 从 dataloader 取 (X, Y, loss_mask)
      - 前向计算 + 交叉熵逐 token 损失
      - 对 <think>/<answer> 相关标签位置附加更大权重
      - 累加 MoE 的 aux_loss
      - AMP 反传 + 梯度裁剪 + 累积步
      - 定期日志/保存 ckpt（半精度）
    """
    # “思考/回答”标签，用于在这些 token 位置施加更大损失（鼓励模型**不要**把思考过程原样输出）
    # 注意：实际训练中，start_of_think/end_of_think等应为不同符号（如“<think>”“</think>”），此处用相同符号仅为示例
    start_of_think_ids = tokenizer('<think>').input_ids
    end_of_think_ids = tokenizer('</think>').input_ids
    start_of_answer_ids = tokenizer('<answer>').input_ids
    end_of_answer_ids = tokenizer('</answer>').input_ids

    loss_fct = nn.CrossEntropyLoss(reduction='none')  # 逐 token 损失，方便与 mask 相乘加权
    start_time = time.time()

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 移动到设备
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 动态学习率（按全局 step）
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 混合精度上下文
        with ctx:
            res = model(X)  # 前向，返回含 logits/aux_loss 等的标准输出

            # 交叉熵逐位置 loss：[B, T, V] vs [B, T] -> [B, T]
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            # 找到 Y 中属于特殊标签的位置（例如 <think> / </think> / <answer> / </answer>）
            sp_ids = torch.isin(
                Y.view(-1),
                torch.tensor(
                    start_of_think_ids + end_of_think_ids + start_of_answer_ids + end_of_answer_ids
                ).to(args.device)
            )

            # 原始 mask 展平成一维便于操作
            loss_mask = loss_mask.view(-1)
            loss_mask_sum = loss_mask.sum()  # 归一化分母（保持尺度稳定）

            # 在特殊标签处加大权重（这里直接把 mask 值改大）
            loss_mask[sp_ids] = 10
            loss_mask = loss_mask.view(Y.size())

            # 应用掩码并做归一化
            loss = (loss * loss_mask).sum() / loss_mask_sum

            # 累加 MoE 的负载均衡损失
            loss += res.aux_loss

            # 梯度累积
            loss = loss / args.accumulation_steps

        # AMP 反向
        scaler.scale(loss).backward()

        # 每 accumulation_steps 更新一次参数
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 先反缩放再裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 日志：loss / lr / 估算 epoch 用时
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,   # 还原未除以累积步前的损失
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                )
            )

            # 仅在主进程写 wandb
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "loss": loss * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr'],
                    "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                })

        # 定期保存 checkpoint（半精度，节省空间/加速加载）
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/reason_{lm_config.hidden_size}{moe_path}.pth'

            # DDP 下取 module 的权重
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 半精度保存
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    """
    加载 tokenizer 与模型权重：
      - 从 ../out/rlhf_*.pth 加载（默认路径可按需修改）
      - 支持 MoE/非 MoE
    返回：
      model (to device), tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained('../model')
    model = MiniMindForCausalLM(lm_config)

    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/rlhf_{lm_config.hidden_size}{moe_path}.pth'

    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)

    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

    model = model.to(args.device)
    return model, tokenizer


def init_distributed_mode():
    """
    初始化分布式 DDP 环境（NCCL 后端）：
      - 从环境变量 RANK/LOCAL_RANK/WORLD_SIZE 读取信息
      - 绑定本地 GPU
    """
    if not ddp:
        return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Distill Reasoning")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)  # 预留：若需 warmup，可自行在 get_lr 中扩展
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=1024, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="../dataset/r1_mix_1024.jsonl")
    args = parser.parse_args()

    # 构建模型配置（此处仅用到部分字段）
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe
    )

    # 目录准备
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # 统计每 iter 训练 token 数（便于估算吞吐）
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # wandb 运行名
    args.wandb_run_name = (
        f"MiniMind-Distill-Reasoning-"
        f"Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
    )

    # AMP 上下文：CPU 无法用 AMP
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # 是否 DDP：依据环境变量 RANK 是否存在
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"

    # 全局随机种子（DDP 会在各 rank 上做不同偏移）
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        # 初始化分布式，并为各 rank 设置不同随机种子
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        torch.cuda.manual_seed(base_seed + rank)

    # wandb 初始化（仅主进程）
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 加载模型 & tokenizer
    model, tokenizer = init_model(lm_config)

    # 训练数据（SFT 格式，内部会构建 ChatML 模板与 loss_mask）
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,              # 分布式场景建议 False + DistributedSampler
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # AMP 梯度缩放器（bf16/fp16 开启）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))

    # AdamW 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # DDP 封装（忽略不需要同步的 buffer）
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 训练循环
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
