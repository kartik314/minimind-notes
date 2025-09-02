"""
文件用途: 定义 LoRA (Low-Rank Adaptation) 模块，并提供相关工具函数
主要功能:
1. LoRA 模块定义 (LoRA class)
   - 用低秩矩阵 A、B 替代全参数更新
   - 显著减少可训练参数数量
2. apply_lora: 将 LoRA 模块注入到模型的 Linear 层中
3. load_lora: 从文件加载 LoRA 权重
4. save_lora: 将当前模型的 LoRA 权重保存到文件
"""

import torch
from torch import optim, nn


# 定义 LoRA 网络结构
class LoRA(nn.Module):
    """
    LoRA 模块 (低秩适配器)
    思想:
        将一个大的线性层 weight 矩阵 (out_features × in_features)，
        分解为两个小矩阵 A (in_features × rank) 和 B (rank × out_features)。
        在微调时，只训练 A 和 B，而原始大矩阵保持冻结（需外部配合冻结原始层参数）。
    参数:
        - in_features: 输入维度
        - out_features: 输出维度
        - rank: 低秩矩阵大小（秩），控制模型适配能力与参数量的平衡
    """
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA 的秩 (rank)，控制低秩近似的能力

        # 低秩分解矩阵 A, B
        # 注意：PyTorch 线性层权重形状为 [out_dim, in_dim]，与数学定义相反
        self.A = nn.Linear(in_features, rank, bias=False)   # A: 输入 -> 降维到 rank（权重 shape: [rank, in_features]）
        self.B = nn.Linear(rank, out_features, bias=False)  # B: rank -> 升维回输出（权重 shape: [out_features, rank]）

        # 初始化方式：A 用高斯分布，B 用全零
        # 全零初始化 B 可确保初始时 LoRA 模块不影响原始模型输出
        self.A.weight.data.normal_(mean=0.0, std=0.02)  # 常用初始化方法
        self.B.weight.data.zero_()                      # 保证一开始不影响原始模型

    def forward(self, x):
        """
        前向传播:
        - 输入先通过 A (降维)
        - 再通过 B (升维)
        - 输出作为 LoRA 的额外增量，需与原始线性层输出相加
        """
        return self.B(self.A(x))


def apply_lora(model, rank=8):
    """
    将 LoRA 模块注入到模型的所有 Linear 层中
    条件: 只对方阵 Linear 层 (输入维度 == 输出维度) 应用 LoRA
          （通常针对 Transformer 中的注意力投影层等对称结构）
    实现细节:
        1. 为每个符合条件的 Linear 层创建 LoRA 子模块，并通过 `setattr` 绑定为层的属性
        2. 替换原始 forward 方法，新输出 = 原始线性层输出 + LoRA 模块输出
    参数:
        - model: 原始模型（需确保 Linear 层未被冻结，否则需先解冻再注入）
        - rank: LoRA 秩，值越小参数量越少，通常取 4-32
    """
    for name, module in model.named_modules():
        # 仅处理输入输出维度相同的线性层（方阵）
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 创建 LoRA 模块并移动到模型相同设备
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            # 将 LoRA 模块绑定为当前 Linear 层的属性（命名为 'lora'）
            setattr(module, "lora", lora)

            # 保存原始 forward 方法，用于后续拼接
            original_forward = module.forward

            # 定义新的 forward 方法：原始输出 + LoRA 增量
            # 使用默认参数绑定 original_forward 和 lora，避免闭包变量捕获问题
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            # 替换 Linear 层的 forward 方法
            module.forward = forward_with_lora


def load_lora(model, path):
    """
    从文件加载 LoRA 权重，并应用到模型中
    前提: 模型已通过 apply_lora 注入 LoRA 模块
    实现逻辑:
        1. 加载保存的 state_dict，按模块名称匹配对应的 LoRA 子模块
        2. 提取子模块权重并通过 load_state_dict 加载
    参数:
        - model: 已注入 LoRA 的模型
        - path: 权重文件路径 (.pth)
    """
    state_dict = torch.load(path, map_location=model.device)
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 提取属于该模块的 LoRA 权重（通过键名中的模块路径匹配）
            lora_state = {
                k.replace(f'{name}.lora.', ''): v
                for k, v in state_dict.items()
                if f'{name}.lora.' in k
            }
            # 加载到对应的 LoRA 子模块
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """
    将模型中的 LoRA 权重保存到文件
    实现逻辑:
        1. 遍历所有含 LoRA 子模块的层，收集其 state_dict
        2. 按 "模块路径.lora.参数名" 的格式整理键名，确保加载时可正确匹配
    参数:
        - model: 已注入 LoRA 的模型
        - path: 保存路径 (.pth)
    """
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 收集当前模块的 LoRA 子模块权重，添加模块路径前缀
            lora_state = {
                f'{name}.lora.{k}': v 
                for k, v in module.lora.state_dict().items()
            }
            state_dict.update(lora_state)
    torch.save(state_dict, path)
