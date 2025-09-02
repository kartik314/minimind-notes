"""
文件用途:
  本脚本实现了一个基于 **FastAPI** 的 Web 服务器，用于启动 MiniMind 模型的对话接口。
  提供 OpenAI 风格的 `/v1/chat/completions` API，可支持：
    1. 普通对话模式（一次性返回完整答案）
    2. 流式输出模式（逐字返回，类似 ChatGPT 的输出方式）

核心功能:
  - 模型初始化（支持从原始 torch 权重 或 Transformers 格式加载）
  - LoRA 参数加载（低秩适配，用于模型微调）
  - FastAPI 端点定义，接收并处理用户请求
  - 支持 SSE（Server-Sent Events）流式返回
"""

import argparse
import json
import os
import sys

# 允许脚本直接调用上一级目录中的 model 包
__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
import warnings
import uvicorn

from threading import Thread
from queue import Queue
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora

warnings.filterwarnings('ignore')

# 初始化 FastAPI 应用
app = FastAPI()


# ------------------------------
# 模型初始化函数
# ------------------------------
def init_model(args):
    """
    功能:
        根据传入参数初始化 MiniMind 模型和 tokenizer
    参数:
        args: argparse 解析得到的配置参数
    返回:
        model: 已加载参数的模型 (eval 模式)
        tokenizer: 对应的分词器
    """
    if args.load == 0:
        # 从原生 PyTorch 权重加载（.pth 文件），需手动构建模型结构并匹配配置参数
        tokenizer = AutoTokenizer.from_pretrained('../model/')
        moe_path = '_moe' if args.use_moe else ''
        modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason'}

        # 拼接权重路径，例如 "../out/full_sft_768.pth"
        ckp = f'../{args.out_dir}/{modes[args.model_mode]}_{args.hidden_size}{moe_path}.pth'

        # 构建 MiniMind 模型
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            max_seq_len=args.max_seq_len,
            use_moe=args.use_moe
        ))

        # 加载权重
        model.load_state_dict(torch.load(ckp, map_location=device), strict=True)

        # 如果指定了 LoRA 微调权重，则应用
        if args.lora_name != 'None':
            apply_lora(model)
            load_lora(model, f'../{args.out_dir}/{args.lora_name}_{args.hidden_size}.pth')
    else:
       # 从 Transformers 格式加载（通过 AutoModel 自动识别模型结构，无需手动配置参数）
        model_path = '../MiniMind2'
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 打印模型参数量
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M')

    return model.eval().to(device), tokenizer


# ------------------------------
# 请求体定义 (遵循 OpenAI API 格式)
# ------------------------------
class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    top_p: float = 0.92
    max_tokens: int = 8192
    stream: bool = False  # 是否启用流式输出
    tools: list = []      # 预留字段，可扩展工具调用


# ------------------------------
# 自定义流式输出器
# ------------------------------
class CustomStreamer(TextStreamer):
    """
    用于将模型输出逐步放入队列，以实现流式响应。
    """
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)  # 将生成的文本放入队列
        if stream_end:
            self.queue.put(None)  # 用 None 表示结束


# ------------------------------
# 生成流式响应
# ------------------------------
def generate_stream_response(messages, temperature, top_p, max_tokens):
    """
    根据输入消息生成流式响应，逐步 yield 输出。
    """
    try:
        # 构造 prompt
        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)[-max_tokens:]
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        queue = Queue()
        streamer = CustomStreamer(tokenizer, queue)

        # 后台线程负责生成文本
        def _generate():
            model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                attention_mask=inputs.attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer
            )
        # 启动后台线程生成文本：避免生成过程阻塞主线程，保证服务器能同时处理其他请求
        Thread(target=_generate).start()

        # 主循环从队列取数据：后台线程生成的文本逐段放入队列，主线程实时取出并返回给客户端，实现"流式输出"
        while True:
            text = queue.get()
            if text is None:  # 生成结束
                yield json.dumps({
                    "choices": [{
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }, ensure_ascii=False)
                break

            yield json.dumps({
                "choices": [{"delta": {"content": text}}]
            }, ensure_ascii=False)

    except Exception as e:
        yield json.dumps({"error": str(e)})


# ------------------------------
# API 路由：/v1/chat/completions
# ------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    模仿 OpenAI ChatCompletion API 的接口。
    支持普通输出和流式输出两种模式。
    """
    try:
        if request.stream:
            # 流式返回
            return StreamingResponse(
                (f"data: {chunk}\n\n" for chunk in generate_stream_response(
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens
                )),
                media_type="text/event-stream"
            )
        else:
            # 一次性返回完整回答
            new_prompt = tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True
            )[-request.max_tokens:]
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    inputs["input_ids"],
                    max_length=inputs["input_ids"].shape[1] + request.max_tokens,
                    do_sample=True,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=request.top_p,
                    temperature=request.temperature
                )
                # 截取生成的新增部分
                answer = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "minimind",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop"
                    }
                ]
            }
    except Exception as e:
             # 捕获所有异常（如模型生成失败、输入参数无效等），返回 500 错误及详细信息
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------
# 程序入口
# ------------------------------
if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser(description="Server for MiniMind")
    parser.add_argument('--out_dir', default='out', type=str)
    parser.add_argument('--lora_name', default='None', type=str)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=16, type=int)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument('--load', default=0, type=int, help="0: 从原生torch权重，1: 利用transformers加载")
    parser.add_argument('--model_mode', default=1, type=int,
                        help="0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型")

   # 决定模型运行设备（优先使用 GPU 以提升性能，无 GPU 时自动切换到 CPU）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化模型和分词器
    model, tokenizer = init_model(parser.parse_args())

    # 启动 Uvicorn 服务
    uvicorn.run(app, host="0.0.0.0", port=8998)
