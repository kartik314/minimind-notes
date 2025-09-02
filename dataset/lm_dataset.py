"""
文件用途：提供 MiniMind 训练阶段用到的多种 Dataset：
- PretrainDataset：无监督预训练（纯文本按语言模型方式构造 X、Y）
- SFTDataset：监督微调（ChatML 对话格式，动态生成损失掩码）
- DPODataset：DPO/RLHF 类偏好训练（chosen vs rejected）
- RLAIFDataset：用于基于 AI 反馈的强化学习阶段的 prompt/answer 取样

核心概念：
- input_ids：分词后的 token 序列
- X / Y：语言模型训练常见的“前一位预测后一位”的输入与标签
- loss_mask：掩码张量（1 表示该位置参与损失，0 不参与）
"""
import json
import random
import re
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import ast

# 关闭 tokenizer 的多进程并行（某些环境下会有警告/死锁风险）
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    """
    用途：无监督预训练数据集（纯文本）
    数据格式：jsonl，每行至少包含字段 'text'
    训练目标：构造自回归语言模型的 (X, Y)，即用前一个 token 预测后一个 token
    参数：
        - data_path: jsonl 文件路径
        - tokenizer: 分词器（需提供 pad_token_id）
        - max_length: 每条样本的最大长度（超出会被截断；不足会被 pad）
    返回：
        - X: input_ids[:-1]
        - Y: input_ids[1:]
        - loss_mask: 与 Y 对齐的参与损失的掩码（pad 位置为 0）
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        """从 jsonl 逐行读取数据到内存（每行一个 dict）。"""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        将文本转换为定长的 token 序列，并构造 (X, Y, loss_mask)：
        - input_ids: [t0, t1, ..., t_{L-1}]
        - X = [t0, ..., t_{L-2}]
        - Y = [t1, ..., t_{L-1}]
        - loss_mask 对应 Y（pad 位置为 0）
        """
        sample = self.samples[index]

        # 将原始文本编码为定长 token 序列；padding='max_length' 会补到 max_length
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()          # [max_length]
        loss_mask = (input_ids != self.tokenizer.pad_token_id)  # pad 位置为 False

        # 构造自回归训练的输入/标签，右移一位
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 与 Y 对齐
        return X, Y, loss_mask


class SFTDataset(Dataset):
    """
    用途：监督微调数据集（SFT）
    数据格式：jsonl，每行形如：
        {
            "conversations": [
                {"content": "..."},
                {"content": "..."},
                ...
            ]
        }
      假定对话轮次按 user/assistant 交替排列（第 0 条为 user）
    机制：
      - 使用 tokenizer.apply_chat_template 生成 ChatML 序列
      - 仅对 assistant 的回答部分计算 loss（通过动态 loss_mask 控制）
    参数：
        - jsonl_path: 数据路径
        - tokenizer: 分词器（需支持 apply_chat_template）
        - max_length: 最大序列长度
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)

        # 这些特殊 token 序列用于在 input_ids 中定位 assistant 片段
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        """读取 jsonl，逐行解析为 dict。"""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """
        将原始轮次转换为 ChatML 文本：
        - 偶数轮设为 user，奇数轮设为 assistant
        - 使用 tokenizer.apply_chat_template 生成模板化字符串
        """
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False  # 训练时不额外加“请继续生成”的提示
        )

    def _generate_loss_mask(self, input_ids):
        """
        动态生成 loss_mask：只在 assistant 的回答范围内置 1，其他为 0。
        思路：
        - 在 token 序列中查找每段 '<|im_start|>assistant' ... '<|im_end|>'
        - 对该区间内（通常从答案第一个 token 开始）标记为 1
        注意：此函数按“已分词后的 id 序列”工作。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 命中一段 assistant 开头
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)  # 回答正文起点（在特殊标记之后）
                end = start
                # 寻找对应的 '<|im_end|>' 结束位置
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 置 1 的范围：答案正文到结束标记（含缓冲，受 max_length 限制）
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                # 跳到该段结束之后，继续找下一段
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        """
        产出定长训练样本：
        - 先模板化对话 -> 分词 -> 截断/补齐到 max_length
        - 生成只对 assistant 片段计损失的 loss_mask
        - 构造 (X, Y, loss_mask) 并与 Y 对齐
        """
        sample = self.samples[index]
        # 1) 构建 ChatML 文本
        prompt = self._create_chat_prompt(sample['conversations'])

        # 2) 编码为 input_ids 并补齐到定长
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 3) 生成动态损失掩码（只在答案区域为 1）
        loss_mask = self._generate_loss_mask(input_ids)

        # 4) 右移构造 (X, Y) 并对齐 loss_mask
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 与 Y 对齐

        return X, Y, loss_mask


class DPODataset(Dataset):
    """
    用途：偏好优化（DPO/RLHF）数据集
    数据格式：jsonl，每行包含：
        {
          "chosen":   [{"role": "...", "content": "..."}, ...],
          "rejected": [{"role": "...", "content": "..."}, ...]
        }
      - chosen：更优答案对应的对话
      - rejected：较差答案对应的对话
    目标：同时返回两条样本（chosen/rejected）各自的 (X, Y, loss_mask)，
          供损失函数比较两者的对数似然等指标。
    """
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # 与 SFT 相同，用特殊 token 确定 assistant 文本范围
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

        # 读取 jsonl 到内存
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        返回一个 dict，包含 chosen 与 rejected 两套 (x, y, mask)：
        - x_* = input_ids[:-1]
        - y_* = input_ids[1:]
        - mask_* = 对应 y_* 的 loss_mask（只在 assistant 答案处为 1）
        """
        item = self.data[index]
        chosen = item['chosen']      # list[{"role","content"}, ...]
        rejected = item['rejected']  # 同上

        # 将 role/content 对话转为 ChatML 文本
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )

        # 编码并补到定长
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)

        # 构造 X/Y 与 mask（右移对齐）
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)

        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def _generate_loss_mask(self, input_ids):
        """
        与 SFTDataset 中一致：只在 assistant 回答区间置 1。
        复用相同的查边界逻辑，确保训练目标一致。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    """
    用途：基于 AI 反馈的强化学习（RLAIF/RLHF）的数据准备
    数据格式：jsonl，每行形如：
        {
            "conversations": [
                {"content": "用户问题"},
                {"content": "助手回答"},
                ...
            ]
        }
    行为：
      - 只构造“到助手需要生成的位置”为止的 prompt（add_generation_prompt=True）
      - 单独返回最后一轮的真实 answer（可做参考/评估）
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)

        # 用于与前两类数据集保持一致的标记（本类不直接用到）
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        """读取 jsonl，逐行解析为 dict。"""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """
        生成 “到需要生成的 assistant 位置为止” 的 prompt，并返回参考答案：
        - messages[:-1]：去掉最后一轮（通常是已有答案），让模型来生成
        - add_generation_prompt=True：告诉模板“接下来轮到 assistant 说话”
        返回：(prompt_str, answer_str)
        """
        messages = []
        answer = ''
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
            answer = turn['content']  # 保留最后一条作为参考答案
        return self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True
        ), answer

    def __getitem__(self, index):
        """
        输出：
            {
              'prompt': 模型要生成的位置之前的对话上下文（字符串）,
              'answer': 最后一轮的参考答案（字符串）
            }
        说明：该数据集返回原始字符串，通常在采样/在线阶段再做分词处理。
        """
        sample = self.samples[index]
        prompt, answer = self._create_chat_prompt(sample['conversations'])
        return {
            'prompt': prompt,
            'answer': answer
        }


if __name__ == "__main__":
    # 作为库被导入时不会执行；可在此做简单的快速检查或单元测试入口
    pass
