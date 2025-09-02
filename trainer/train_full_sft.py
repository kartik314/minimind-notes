"""
文件用途：
  - 使用监督微调（Supervised Fine-Tuning, SFT）对 MiniMind 模型进行全量训练/继续训练
    核心目标：让预训练模型学习遵循人类指令的能力，通过标注的对话数据（如"问题-回答"对）
    调整模型参数，使其生成符合预期的、高质量的回答。
  - 支持 AMP 混合精度、DDP 分布式训练、余弦退火学习率、按步保存检查点

整体流程：
  1) 解析命令行参数，构造模型配置 MiniMindConfig
  2) 初始化分布式（可选）、随机种子、AMP 上下文
  3) 加载 tokenizer、模型（从预训练权重加载）、SFT 数据集与 DataLoader
  4) 训练循环：前向 -> 损失 -> 反向（支持梯度累积）-> 日志 -> 定期保存
  5) 可选 WandB 记录指标

"""
import os
import sys

# 为了能从上级目录导入项目内模块（如 model/, dataset/）
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
    仅在单卡或 DDP 的主进程（rank=0）打印日志，避免多卡重复输出。
    """
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    余弦退火学习率：
      lr(t) = lr/10 + 0.5 * lr * (1 + cos(pi * t / T))
    说明：
      - 前期从较高值平滑过渡，后期逐步减小到 lr/10，训练更稳定。
    """
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    """
    单个 epoch 的训练逻辑：
      - 读入一个 batch 的 (X, Y, loss_mask)
      - 前向计算 logits，按 token 交叉熵 + 掩码求 Loss
      - 累加模型可能返回的 aux_loss（如 MoE 负载均衡损失）
      - AMP 反传 + 梯度裁剪 + 梯度累积 + 优化步
      - 定期打印日志、保存权重

    参数：
      epoch: 当前轮次（从 0 开始）
      wandb: 可选的 Weights & Biases 记录器
    """
    loss_fct = nn.CrossEntropyLoss(reduction='none')  # 不在这里做平均，为了后续与 mask 相乘
    start_time = time.time()

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # -------- 数据搬运到目标设备 --------
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # -------- 按步更新学习率（余弦退火）--------
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # -------- 前向 & 损失计算（AMP）--------
        with ctx:  # GPU 上启用 autocast，可显著加速并节省显存
            res = model(X)  # 前向，返回含 logits/aux_loss 的标准输出对象

            # 逐 token 交叉熵：形状对齐为 (B*T, V) vs (B*T)
            ce = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())  # 还原回 (B, T)，方便和 mask 相乘

            # 仅对有效位置计损：mask=1 的位置（如 assistant 回复部分）
            loss = (ce * loss_mask).sum() / loss_mask.sum()

            # 如模型包含 MoE 等额外正则/辅助损失，在此相加
            loss += res.aux_loss

            # 梯度累积：把有效 loss 均分到 accumulation_steps 次反传中
            loss = loss / args.accumulation_steps

        # -------- 反向传播（AMP + 梯度缩放器）--------
        scaler.scale(loss).backward()

        # 每累计到指定步数才做一次真正的优化器更新
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 先反缩放，再做梯度裁剪更稳定
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)  # 执行优化器更新（按 AMP 缩放）
            scaler.update()         # 更新缩放系数
            optimizer.zero_grad(set_to_none=True)  # 清空梯度，set_to_none 节省显存

        # -------- 训练日志输出 --------
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,                        # 人类友好的 1-based 显示
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,  # 还原到未累积前的 loss
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                )
            )

            # 可选：上报到 WandB（仅主进程）
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "loss": loss * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr'],
                    "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                })

        # -------- 定期保存检查点（半精度）--------
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth'

            # DDP 场景下需要从 model.module 取 state_dict
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict()

            # 半精度保存（fp16/bf16），减少文件体积并加快后续加载
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    """
    加载 tokenizer 与模型，并从预训练权重初始化。

    返回：
      model: 已移动到目标设备的 MiniMindForCausalLM
      tokenizer: 用于编码/解码的分词器
    """
    tokenizer = AutoTokenizer.from_pretrained('../model')  # 读取自建 tokenizer 目录
    model = MiniMindForCausalLM(lm_config)

    # 从预训练 checkpoint 加载（支持 MoE 与非 MoE 两种命名）
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)  # strict=False：允许略微形状不匹配（比如改了 head）

    # 打印“可训练参数量”做基本 sanity check
    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

    model = model.to(args.device)  # 移动到目标设备
    return model, tokenizer


def init_distributed_mode():
    """
    初始化分布式训练（DDP）：
      - 使用 NCCL 后端（GPU 最优）
      - 读取环境变量 RANK/LOCAL_RANK/WORLD_SIZE
      - 绑定本地 device 到相应 GPU
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
    # ========================== 1) 解析参数 ==========================
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")                 # 是否启用分布式
    parser.add_argument("--accumulation_steps", type=int, default=1)  # 梯度累积步数
    parser.add_argument("--grad_clip", type=float, default=1.0)       # 梯度裁剪阈值
    parser.add_argument("--warmup_iters", type=int, default=0)        # 预留：若需 warmup，可扩展 get_lr
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)         # 兼容 torchrun 传参
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl")

    args = parser.parse_args()

    # ========================== 2) 构建配置 & 目录 ==========================
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe
    )

    # 输出目录（保存 ckpt）
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    tokens_per_iter = args.batch_size * args.max_seq_len  # 便于评估训练吞吐
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # WandB 运行名
    args.wandb_run_name = (
        f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
    )

    # AMP 上下文：CPU 无法用 autocast，给一个空上下文以统一写法
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # 判断是否处于分布式环境（torchrun 会注入 RANK 等环境变量）
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"

    # 固定随机性（不同 rank 用不同种子，增强多卡训练稳定性）
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        torch.cuda.manual_seed(base_seed + rank)

    # ========================== 3) 初始化 WandB（可选） ==========================
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # ========================== 4) 加载模型 & 分词器 ==========================
    model, tokenizer = init_model(lm_config)

    # ========================== 5) 构建数据集与 DataLoader ==========================
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None  # 分布式按 rank 切分数据
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,           # 提高 host->device 拷贝效率
        drop_last=False,           # 可根据需要改为 True，保证每步 batch 大小一致
        shuffle=False,             # DDP 通常交给 DistributedSampler 随机打散
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # ========================== 6) 优化器 & AMP 梯度缩放器 ==========================
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========================== 7) DDP 封装（可选） ==========================
    if ddp:
        # 如果模型里有不需同步的 buffer，可以在这里忽略（示例名 pos_cis 仅作占位）
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # ========================== 8) 训练循环 ==========================
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)