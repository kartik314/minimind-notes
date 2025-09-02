"""
文件用途:  
  本脚本实现 **MiniMind 模型权重** 与 **HuggingFace Transformers 格式** 的相互转换。  
  主要功能：
    1. 将 PyTorch 保存的 MiniMind 模型权重，转为 HuggingFace Transformers-MiniMind 格式
    2. 将 PyTorch 保存的模型，转为 HuggingFace LLaMA 结构（方便生态兼容）
    3. 将 Transformers 模型权重，反向保存为 PyTorch .pth 格式

适用场景：
  - 在训练完成后，需要将模型导出，供 HuggingFace `transformers` 或本地推理框架使用
  - 在不同生态之间切换（MiniMind ↔ LLaMA ↔ PyTorch）
"""

import os
import sys

# 兼容脚本调用路径，确保可以找到上一级目录中的 model 包
__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

# 屏蔽无关的警告（比如权重加载中的一些提示）
warnings.filterwarnings('ignore', category=UserWarning)


# ------------------------------
# 功能1: 将 PyTorch MiniMind 模型转为 Transformers 格式
# ------------------------------
def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.bfloat16):
    """
    参数:
        torch_path:  输入的 PyTorch 模型路径（.pth 文件）
        transformers_path: 输出的 Transformers 保存目录
        dtype: 模型精度（默认 bfloat16，可改为 float16 / float32）

    步骤:
        1. 注册 MiniMind 的 AutoClass，使其能被 transformers 框架识别
        2. 加载 PyTorch state_dict 到 MiniMind 模型
        3. 转换精度并保存为 Transformers 权重
        4. 保存分词器，保证模型能正确加载
    """
    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    lm_model = MiniMindForCausalLM(lm_config)  # 初始化模型结构
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载 PyTorch 权重
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)

    # 转换精度（bfloat16 更省显存，兼容性好）
    lm_model = lm_model.to(dtype)

    # 打印参数量
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6:.2f} 百万 = {model_params / 1e9:.2f} B (Billion)')

    # 保存为 HuggingFace Transformers 格式
    lm_model.save_pretrained(transformers_path, safe_serialization=False)

    # 同时保存 tokenizer
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)
    print(f"✅ 模型已保存为 Transformers-MiniMind 格式: {transformers_path}")


# ------------------------------
# 功能2: 将 PyTorch 模型转为 LLaMA 结构 (兼容生态)
# ------------------------------
def convert_torch2transformers_llama(torch_path, transformers_path, dtype=torch.bfloat16):
    """
    参数:
        torch_path:  输入的 PyTorch 模型路径
        transformers_path: 输出的 Transformers-LLaMA 格式目录
        dtype: 转换后的精度

    说明:
        - 这里会构造一个 LLaMAConfig，并用 MiniMind 配置中的超参来初始化
        - 输出的模型权重结构兼容 HuggingFace LLaMA 系列，方便使用第三方生态
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载 PyTorch 权重（strict=False 允许权重与模型结构存在部分不匹配，可能导致部分参数未加载）
    state_dict = torch.load(torch_path, map_location=device)

    # 构造 LLaMA 配置（部分参数来源于 MiniMindConfig）
    llama_config = LlamaConfig(
        vocab_size=lm_config.vocab_size,
        hidden_size=lm_config.hidden_size,
        intermediate_size=64 * ((int(lm_config.hidden_size * 8 / 3) + 64 - 1) // 64),  # 近似对齐FFN
        num_hidden_layers=lm_config.num_hidden_layers,
        num_attention_heads=lm_config.num_attention_heads,
        num_key_value_heads=lm_config.num_key_value_heads,
        max_position_embeddings=lm_config.max_seq_len,
        rms_norm_eps=lm_config.rms_norm_eps,
        rope_theta=lm_config.rope_theta,
    )

    # 初始化 LLaMA 模型并加载权重
    llama_model = LlamaForCausalLM(llama_config)
    llama_model.load_state_dict(state_dict, strict=False)

    # 转换精度
    llama_model = llama_model.to(dtype)

    # 保存为 Transformers-LLaMA 格式
    llama_model.save_pretrained(transformers_path)

    # 打印参数量
    model_params = sum(p.numel() for p in llama_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6:.2f} 百万 = {model_params / 1e9:.2f} B (Billion)')

    # 保存分词器
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)
    print(f"✅ 模型已保存为 Transformers-LLaMA 格式: {transformers_path}")


# ------------------------------
# 功能3: 将 Transformers 模型转回 PyTorch 格式
# ------------------------------
def convert_transformers2torch(transformers_path, torch_path):
    """
    参数:
        transformers_path: 输入的 Transformers 模型目录
        torch_path: 输出的 PyTorch 文件路径 (.pth)

    步骤:
        1. 使用 AutoModel 加载 Transformers 格式模型
        2. 保存其 state_dict() 为 .pth
    """
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save(model.state_dict(), torch_path)
    print(f"✅ 模型已保存为 PyTorch 格式: {torch_path}")


# ------------------------------
# 主函数入口
# ------------------------------
if __name__ == '__main__':
    # 定义默认的 MiniMind 配置
    # hidden_size: 隐藏层维度，决定模型容量
    # num_hidden_layers: 隐藏层数量，影响模型深度
    # max_seq_len: 最大序列长度，决定模型可处理的文本长度
    # use_moe: 是否使用混合专家模型（MoE）结构

    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=16, max_seq_len=8192, use_moe=False)

    # 输入的 PyTorch 权重路径
    torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"

    # 输出的 Transformers 目录
    transformers_path = '../MiniMind2'

    # 执行转换 (PyTorch → Transformers-MiniMind)
    convert_torch2transformers_minimind(torch_path, transformers_path)

    # 如需执行反向转换，可取消注释：
    # convert_transformers2torch(transformers_path, torch_path)
