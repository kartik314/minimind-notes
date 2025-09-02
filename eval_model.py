"""
文件用途: MiniMind 模型推理/对话脚本
主要功能:
1. 初始化并加载 MiniMind 模型（支持 LoRA、MoE 等配置）
2. 提供预设测试问题或用户手动输入
3. 使用 HuggingFace 的 TextStreamer 实时打印生成结果
4. 支持携带历史上下文对话（history_cnt）
核心作用：
  - 作为模型训练后的"使用接口"，验证预训练/SFT/LoRA微调等不同阶段模型的效果。
  - 支持不同配置（基础模型、LoRA适配器、MoE结构）的模型加载与测试。
  - 模拟真实对话场景，通过上下文管理实现多轮交互。
"""

import argparse
import random
import warnings
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *  # 引入 LoRA 相关函数 (apply_lora, load_lora)

warnings.filterwarnings('ignore')


def init_model(args):
    """
    初始化并加载 MiniMind 模型
    参数:
        args: argparse 解析后的命令行参数
    核心逻辑：
      - 支持两种加载模式：
        1) 原生torch权重（args.load=0）：加载预训练/SFT/RLHF等阶段保存的pth文件，
           可叠加LoRA适配器（通过apply_lora和load_lora实现）。
        2) transformers接口（args.load=1）：加载转换为HuggingFace格式的完整模型，
           兼容transformers生态的生成函数。
      - 自动适配MoE结构（根据args.use_moe）和不同模型规模（hidden_size/num_hidden_layers）。
    返回:
        - model: 已加载权重并切换到eval()的模型（禁用dropout等训练特有操作）
        - tokenizer: 对应的分词器（负责文本编码/解码）
    """
    tokenizer = AutoTokenizer.from_pretrained('./model/')
    if args.load == 0:
        # 模型模式与权重文件路径映射
        moe_path = '_moe' if args.use_moe else ''
        modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason', 4: 'grpo'}
        ckp = f'./{args.out_dir}/{modes[args.model_mode]}_{args.hidden_size}{moe_path}.pth'

        # 初始化 MiniMind 模型配置与结构
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=args.use_moe
        ))

        # 加载保存好的权重
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)

        # 如果使用 LoRA，加载 LoRA 权重
        if args.lora_name != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.out_dir}/lora/{args.lora_name}_{args.hidden_size}.pth')
    else:
        # 通过 transformers 接口加载完整模型（默认路径 ./MiniMind2）
        transformers_model_path = './MiniMind2'
        tokenizer = AutoTokenizer.from_pretrained(transformers_model_path)
        model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)

    # 打印模型参数量 (以百万为单位)
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
    return model.eval().to(args.device), tokenizer


def get_prompt_datas(args):
    """
    根据模型模式与 LoRA 配置，获取测试用的 prompt 数据
    参数:
        args: argparse 参数
    返回:
        prompt_datas: 一个字符串列表
    """
    if args.model_mode == 0:
        # 预训练模型：只能做续写（无对话能力）
        prompt_datas = [
            '马克思主义基本原理',
            '人类大脑的主要功能',
            '万有引力原理是',
            '世界上最高的山峰是',
            '二氧化碳在空气中',
            '地球上最大的动物有',
            '杭州市的美食有'
        ]
    else:
        if args.lora_name == 'None':
            # 通用对话问题
            prompt_datas = [
                '请介绍一下自己。',
                '你更擅长哪一个学科？',
                '鲁迅的《狂人日记》是如何批判封建礼教的？',
                '我咳嗽已经持续了两周，需要去医院检查吗？',
                '详细的介绍光速的物理概念。',
                '推荐一些杭州的特色美食吧。',
                '请为我讲解“大语言模型”这个概念。',
                '如何理解ChatGPT？',
                'Introduce the history of the United States, please.'
            ]
        else:
            # 特定领域 LoRA 微调模型
            lora_prompt_datas = {
                'lora_identity': [
                    "你是ChatGPT吧。",
                    "你叫什么名字？",
                    "你和openai是什么关系？"
                ],
                'lora_medical': [
                    '我最近经常感到头晕，可能是什么原因？',
                    '我咳嗽已经持续了两周，需要去医院检查吗？',
                    '服用抗生素时需要注意哪些事项？',
                    '体检报告中显示胆固醇偏高，我该怎么办？',
                    '孕妇在饮食上需要注意什么？',
                    '老年人如何预防骨质疏松？',
                    '我最近总是感到焦虑，应该怎么缓解？',
                    '如果有人突然晕倒，应该如何急救？'
                ],
            }
            prompt_datas = lora_prompt_datas[args.lora_name]

    return prompt_datas


def setup_seed(seed):
    """
    设置随机种子，保证结果可复现
    参数:
        seed: 整数随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # 固定卷积计算
    torch.backends.cudnn.benchmark = False      # 禁用动态优化


def main():
    """
    主函数入口：
    流程链条：
      1) 参数解析 → 2) 模型/分词器加载 → 3) 测试prompt生成 → 
      4) 交互模式选择（自动测试/手动输入） → 5) 对话上下文管理 → 
      6) 输入编码 → 7) 流式生成 → 8) 输出解码与上下文更新
    核心特性：
      - 流式输出（TextStreamer）：模拟人类打字效果，提升交互体验。
      - 上下文管理（history_cnt）：支持多轮对话，通过截断控制上下文长度。
      - 生成参数可调（temperature/top_p）：控制输出的随机性和多样性。
    """
    parser = argparse.ArgumentParser(description="Chat with MiniMind")
    parser.add_argument('--lora_name', default='None', type=str)
    parser.add_argument('--out_dir', default='out', type=str)
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)

    # 模型结构相关参数
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)

    # 对话历史参数
    # history_cnt 必须为偶数 (用户+助手为一组)
    parser.add_argument('--history_cnt', default=0, type=int)

    # 权重加载模式
    parser.add_argument('--load', default=0, type=int, help="0: 原生torch权重，1: transformers加载")

    # 模型模式 (决定加载权重和 prompt 生成方式)
    parser.add_argument('--model_mode', default=1, type=int,
                        help="0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型，4: RLAIF-Chat模型")

    args = parser.parse_args()

    # 初始化模型与分词器
    model, tokenizer = init_model(args)

    # 获取 prompt 数据
    prompts = get_prompt_datas(args)

    # 模式选择：0 = 自动测试（预设问题），1 = 手动输入
    test_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))

    # HuggingFace 自带的实时输出器
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    messages = []
    for idx, prompt in enumerate(prompts if test_mode == 0 else iter(lambda: input('👶: '), '')):
        # 设置随机种子，保证每轮输出的随机性（或固定性）
        setup_seed(random.randint(0, 2048))
        # setup_seed(2025)  # 如果需要固定结果，可写死种子

        if test_mode == 0:
            print(f'👶: {prompt}')

        # 维护历史上下文（如果 history_cnt > 0）
        messages = messages[-args.history_cnt:] if args.history_cnt else []
        messages.append({"role": "user", "content": prompt})

        # 构建对话模板
        if args.model_mode != 0:
            # 构建对话模板：
            # tokenizer.apply_chat_template 将历史消息转换为模型预期的格式（如添加角色标记<user>/<assistant>），
            # 不同模型可能需要不同模板，确保模型能正确区分对话角色和轮次。
            new_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True # 自动添加生成提示（如"assistant:"）
            )
        else:
            # 预训练模型只做简单的续写
            new_prompt = tokenizer.bos_token + prompt

        # 编码输入
        inputs = tokenizer(
            new_prompt,
            return_tensors="pt",
            truncation=True
        ).to(args.device)

        # 生成回答
        print('🤖️: ', end='')
        # 生成回答后，可从以下维度评估：
        # 1) 相关性：回答是否紧扣问题，无冗余信息。
        # 2) 流畅性：语句是否通顺，无语法错误。
        # 3) 准确性：事实性内容是否正确（如医疗建议、知识问答）。
        # 4) 安全性：是否包含不当内容（针对对话模型）。
        generated_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=args.max_seq_len,
            num_return_sequences=1,
            do_sample=True,
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            top_p=args.top_p,
            temperature=args.temperature
        )

        # 解码生成内容（去掉输入部分，只保留新生成的）
        response = tokenizer.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        messages.append({"role": "assistant", "content": response})
        print('\n\n')


if __name__ == "__main__":
    main()
