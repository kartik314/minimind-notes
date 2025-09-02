"""
脚本功能：
1. 使用 BPE (Byte-Pair Encoding) 训练一个分词器（Tokenizer）。
2. 设置特殊符号（system/user/assistant 开始与结束标记）。
3. 保存 tokenizer 的配置文件，确保能与 HuggingFace Transformers 生态兼容。
4. 提供一个简单的 eval_tokenizer 函数验证 tokenizer 的效果。

适合对象：对 NLP 和分词器训练有一定基础的小白用户。
"""

import random
import json
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
import os

# 固定随机种子，保证每次训练结果一致
random.seed(42)


def train_tokenizer():
    """
    功能：
        训练一个 BPE 分词器，并保存到 ../model/ 文件夹下。

    步骤：
        1. 读取 JSONL 格式的语料（每行一个 JSON，其中包含 "text" 字段）。
        2. 初始化 BPE 模型，并设置字节级别预切分（ByteLevel）。
        3. 定义并添加特殊符号：<|endoftext|> <|im_start|> <|im_end|>
        4. 训练 tokenizer，并保存模型和配置文件。
    """

    # 读取JSONL文件并提取文本数据
    def read_texts_from_jsonl(file_path):
        """
        输入：
            file_path: JSONL 文件路径
        输出：
            每行的 'text' 内容（生成器方式逐行返回）
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data['text']

    # 训练语料路径
    data_path = '../dataset/pretrain_hq.jsonl'

    # 初始化 BPE 模型
    tokenizer = Tokenizer(models.BPE())
    # 设置预切分器：按字节级别切分（保证对所有字符鲁棒）
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 定义特殊符号，用于对话场景
    special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

    # 定义 BPE 训练器
    trainer = trainers.BpeTrainer(
        vocab_size=6400,               # 词汇表大小
        special_tokens=special_tokens, # 加入特殊符号
        show_progress=True,            # 显示训练进度
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 读取训练数据
    texts = read_texts_from_jsonl(data_path)

    # 开始训练
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 设置解码器：从 token 序列恢复原始文本
    tokenizer.decoder = decoders.ByteLevel()

    # 检查特殊符号的 ID 是否正确分配
    assert tokenizer.token_to_id("<|endoftext|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2

    # 保存 tokenizer 到本地
    tokenizer_dir = "../model/"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save("../model/")

    # 手动创建配置文件（兼容 HuggingFace Transformers）
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {"content": "<|endoftext|>", "special": True},
            "1": {"content": "<|im_start|>", "special": True},
            "2": {"content": "<|im_end|>", "special": True},
        },
        "bos_token": "<|im_start|>",         # 开始符号
        "eos_token": "<|im_end|>",           # 结束符号
        "pad_token": "<|endoftext|>",        # 填充符号
        "unk_token": "<|endoftext|>",        # 未知符号
        "model_max_length": 32768,           # 最大序列长度
        "tokenizer_class": "PreTrainedTokenizerFast",
        # 定义 chat 模板，用于拼接 system/user/assistant 消息
        "chat_template": (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set system_message = messages[0]['content'] %}"
            "{{ '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}"
            "{% else %}"
            "{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}"
            "{% endif %}"
            "{% for message in messages %}"
            "{% set content = message['content'] %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ content + '<|im_end|>' + '\\n' }}"
            "{% endif %}"
            "{% endfor %}"
        )
    }

    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("✅ Tokenizer training completed and saved.")


def eval_tokenizer():
    """
    功能：
        测试训练好的 tokenizer 是否能正确工作。

    步骤：
        1. 加载已保存的 tokenizer。
        2. 构造 system/user/assistant 对话。
        3. 应用 chat_template 拼接为一个字符串。
        4. 编码 → 解码，检查是否一致。
    """
    from transformers import AutoTokenizer

    # 加载预训练的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("../model/")

    # 定义一个对话示例
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]

    # 将对话转为统一的 prompt
    new_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print("拼接后的对话：\n", new_prompt)

    # 查看词表长度
    actual_vocab_size = len(tokenizer)
    print('tokenizer 实际词表长度：', actual_vocab_size)

    # 编码测试
    model_inputs = tokenizer(new_prompt)
    print('encoder 输入长度：', len(model_inputs['input_ids']))

    # 解码测试：验证分词器编码→解码过程是否无信息丢失（保证文本完整性）
    input_ids = model_inputs['input_ids']
    response = tokenizer.decode(input_ids, skip_special_tokens=False)
    print('decoder 和原始文本是否一致：', response == new_prompt)


def main():
    """入口函数：先训练 tokenizer，再进行简单评估"""
    train_tokenizer()
    eval_tokenizer()


if __name__ == '__main__':
    main()
