"""
文件用途：
  使用 Streamlit 构建一个简洁的 MiniMind 聊天前端，支持两种模型来源：
    1) 本地模型（通过 Transformers 直接加载）
    2) 远端 API（OpenAI 兼容接口）
  功能点：
    - 自定义样式（圆形操作按钮、标题栏等）
    - 对话历史记忆（可配置携带轮数）
    - 流式输出（本地/远端均支持）
    - “推理内容”(<think>...</think>) 折叠展示

"""

import random
import re
from threading import Thread

import torch
import numpy as np
import streamlit as st

# 页面基础设置
st.set_page_config(page_title="MiniMind", initial_sidebar_state="collapsed")

# 页面样式：主要用于圆形按钮、边距微调
st.markdown("""
    <style>
        /* 侧栏按钮样式（圆形、小号、浅色） */
        .stButton button {
            border-radius: 50% !important;
            width: 32px !important;
            height: 32px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #666 !important;
            margin: 5px 10px 5px 0 !important;
        }
        .stButton button:hover {
            border-color: #999 !important;
            color: #333 !important;
            background-color: #f5f5f5 !important;
        }
        .stMainBlockContainer > div:first-child { margin-top: -50px !important; }
        .stApp > div:last-child { margin-bottom: -35px !important; }

        /* 重置按钮基础样式（更小的圆形 × 按钮） */
        .stButton > button {
            all: unset !important;
            box-sizing: border-box !important;
            border-radius: 50% !important;
            width: 18px !important;
            height: 18px !important;
            min-width: 18px !important;
            min-height: 18px !important;
            max-width: 18px !important;
            max-height: 18px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #888 !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            margin: 0 2px !important;
        }
    </style>
""", unsafe_allow_html=True)

# 系统级消息（可注入 system 指令）
system_prompt = []
# 推断设备
device = "cuda" if torch.cuda.is_available() else "cpu"


def process_assistant_content(content):
    """
    用途：
        处理助手输出的文本，将 <think> ... </think> 包裹的推理过程转换为可折叠 HTML。
    逻辑：
        - 仅当模型名包含 'R1' 时展示 think 区域（本地/远端都做检测）
        - 同时兼容：只有起始或只有结束标签的边界情况
    """
    if model_source == "API" and 'R1' not in api_model_name:
        return content
    if model_source != "API" and 'R1' not in MODEL_PATHS[selected_model][1]:
        return content

    # 完整 <think>...</think>
    if '<think>' in content and '</think>' in content:
        content = re.sub(
            r'(<think>)(.*?)(</think>)',
            r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理内容（展开）</summary>\2</details>',
            content,
            flags=re.DOTALL
        )

    # 只有 <think> 起始
    if '<think>' in content and '</think>' not in content:
        content = re.sub(
            r'<think>(.*?)$',
            r'<details open style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理中...</summary>\1</details>',
            content,
            flags=re.DOTALL
        )

    # 只有 </think> 结束
    if '<think>' not in content and '</think>' in content:
        content = re.sub(
            r'(.*?)</think>',
            r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理内容（展开）</summary>\1</details>',
            content,
            flags=re.DOTALL
        )

    return content


@st.cache_resource
def load_model_tokenizer(model_path):
    """
    作用：
        加载本地 Transformers 模型与分词器，并在首次调用后进行缓存（避免重复加载）。
    参数：
        model_path: 模型目录（包含 config、权重、tokenizer）
    返回：
        (model.eval().to(device), tokenizer)
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = model.eval().to(device)
    return model, tokenizer


def clear_chat_messages():
    """ 清空会话缓存（messages & chat_messages） """
    del st.session_state.messages
    del st.session_state.chat_messages


def init_chat_messages():
    """
    初始化并回放历史消息：
      - assistant 消息使用 chat_message UI 渲染并可单条删除
      - user 消息右对齐灰底气泡
    """
    if "messages" in st.session_state:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant":
                with st.chat_message("assistant", avatar=image_url):
                    st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
                    # 单条删除（会删除一对 Q/A）
                    if st.button("🗑", key=f"delete_{i}"):
                        st.session_state.messages.pop(i)
                        st.session_state.messages.pop(i - 1)
                        st.session_state.chat_messages.pop(i)
                        st.session_state.chat_messages.pop(i - 1)
                        st.rerun()
            else:
                st.markdown(
                    f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px;  background-color: #ddd; border-radius: 10px; color: black;">{message["content"]}</div></div>',
                    unsafe_allow_html=True)
    else:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    return st.session_state.messages


def regenerate_answer(index):
    """ 重新生成：删除最后一条 A（和对应的渲染），触发重渲染 """
    st.session_state.messages.pop()
    st.session_state.chat_messages.pop()
    st.rerun()


def delete_conversation(index):
    """ 删除某一轮对话（Q/A 各一条） """
    st.session_state.messages.pop(index)
    st.session_state.messages.pop(index - 1)
    st.session_state.chat_messages.pop(index)
    st.session_state.chat_messages.pop(index - 1)
    st.rerun()


# ========== 侧边栏：模型与推理参数 ==========
st.sidebar.title("模型设定调整")
# 历史对话轮数（必须为偶数，Q+A 为一组；0 表示不携带历史）
st.session_state.history_chat_num = st.sidebar.slider("Number of Historical Dialogues", 0, 6, 0, step=2)
# 最大新生成 token（用于限制上下文与生成长度）
st.session_state.max_new_tokens = st.sidebar.slider("Max Sequence Length", 256, 8192, 8192, step=1)
# 采样温度
st.session_state.temperature = st.sidebar.slider("Temperature", 0.6, 1.2, 0.85, step=0.01)

# 模型来源选择：本地 / API
model_source = st.sidebar.radio("选择模型来源", ["本地模型", "API"], index=0)

if model_source == "API":
    # 远端 API 参数（OpenAI 兼容）
    api_url = st.sidebar.text_input("API URL", value="http://127.0.0.1:8000/v1")
    api_model_id = st.sidebar.text_input("Model ID", value="minimind")
    api_model_name = st.sidebar.text_input("Model Name", value="MiniMind2")
    api_key = st.sidebar.text_input("API Key", value="none", type="password")
    slogan = f"Hi, I'm {api_model_name}"
else:
    # 本地模型集合（名称 -> [路径, 展示名]）
    MODEL_PATHS = {
        "MiniMind2-R1 (0.1B)": ["../MiniMind2-R1", "MiniMind2-R1"],
        "MiniMind2-Small-R1 (0.02B)": ["../MiniMind2-Small-R1", "MiniMind2-Small-R1"],
        "MiniMind2 (0.1B)": ["../MiniMind2", "MiniMind2"],
        "MiniMind2-MoE (0.15B)": ["../MiniMind2-MoE", "MiniMind2-MoE"],
        "MiniMind2-Small (0.02B)": ["../MiniMind2-Small", "MiniMind2-Small"]
    }
    # 默认 MiniMind2
    selected_model = st.sidebar.selectbox('Models', list(MODEL_PATHS.keys()), index=2)
    model_path = MODEL_PATHS[selected_model][0]
    slogan = f"Hi, I'm {MODEL_PATHS[selected_model][1]}"

# 顶部标题与 logo
image_url = "https://www.modelscope.cn/api/v1/studio/gongjy/MiniMind/repo?Revision=master&FilePath=images%2Flogo2.png&View=true"
st.markdown(
    f'<div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0; padding: 0;">'
    '<div style="font-style: italic; font-weight: 900; margin: 0; padding-top: 4px; display: flex; align-items: center; justify-content: center; flex-wrap: wrap; width: 100%;">'
    f'<img src="{image_url}" style="width: 45px; height: 45px; "> '
    f'<span style="font-size: 26px; margin-left: 10px;">{slogan}</span>'
    '</div>'
    '<span style="color: #bbb; font-style: italic; margin-top: 6px; margin-bottom: 10px;">内容完全由AI生成，请务必仔细甄别<br>Content AI-generated, please discern with care</span>'
    '</div>',
    unsafe_allow_html=True
)


def setup_seed(seed):
    """
    固定随机性，便于调试与复现。
    注意：开启 deterministic=True 会禁用某些 cuDNN 加速。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """
    页面主逻辑：
      - 加载本地模型（或准备 API 调用参数）
      - 回放历史消息
      - 处理新输入（本地/远端两条推理路径，均支持流式渲染）
      - 记录并渲染本轮问答
    """
    # 仅当选择“本地模型”时加载；API 模式下由服务端推理
    if model_source == "本地模型":
        model, tokenizer = load_model_tokenizer(model_path)
    else:
        model, tokenizer = None, None

    # 初始化状态存储
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    messages = st.session_state.messages

    # 回放历史消息（包含删除按钮）
    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar=image_url):
                st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
                # 删除当前这轮（Q/A 两条）
                if st.button("×", key=f"delete_{i}"):
                    st.session_state.messages = st.session_state.messages[:i - 1]
                    st.session_state.chat_messages = st.session_state.chat_messages[:i - 1]
                    st.rerun()
        else:
            # 右对齐气泡渲染用户消息
            st.markdown(
                f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px;  background-color: gray; border-radius: 10px; color:white; ">{message["content"]}</div></div>',
                unsafe_allow_html=True)

    # 输入框
    prompt = st.chat_input(key="input", placeholder="给 MiniMind 发送消息")

    # “重新生成”场景：用最后一次用户消息作为当前 prompt
    if hasattr(st.session_state, 'regenerate') and st.session_state.regenerate:
        prompt = st.session_state.last_user_message
        regenerate_index = st.session_state.regenerate_index
        delattr(st.session_state, 'regenerate')
        delattr(st.session_state, 'last_user_message')
        delattr(st.session_state, 'regenerate_index')

    if prompt:
        # 立即显示用户消息（右对齐）
        st.markdown(
            f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px;  background-color: gray; border-radius: 10px; color:white; ">{prompt}</div></div>',
            unsafe_allow_html=True)

        # 记录消息（裁剪长度，避免超长）
        messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})
        st.session_state.chat_messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})

        # 助手消息容器（用于流式更新）
        with st.chat_message("assistant", avatar=image_url):
            placeholder = st.empty()

            if model_source == "API":
                # ===== 远端 API 推理（OpenAI 兼容）=====
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key, base_url=api_url)

                    # 历史对话条数（+1 包含当前这条）
                    history_num = st.session_state.history_chat_num + 1
                    conversation_history = system_prompt + st.session_state.chat_messages[-history_num:]

                    answer = ""
                    response = client.chat.completions.create(
                        model=api_model_id,
                        messages=conversation_history,
                        stream=True,
                        temperature=st.session_state.temperature
                    )
                    # 流式渲染
                    for chunk in response:
                        content = chunk.choices[0].delta.content or ""
                        answer += content
                        placeholder.markdown(process_assistant_content(answer), unsafe_allow_html=True)

                except Exception as e:
                    answer = f"API调用出错: {str(e)}"
                    placeholder.markdown(answer, unsafe_allow_html=True)

            else:
                # ===== 本地模型推理（Transformers generate + TextIteratorStreamer）=====
                random_seed = random.randint(0, 2 ** 32 - 1)
                setup_seed(random_seed)

                # 带入系统提示与有限历史
                st.session_state.chat_messages = system_prompt + st.session_state.chat_messages[-(st.session_state.history_chat_num + 1):]

                # chat_template 生成 prompt
                new_prompt = tokenizer.apply_chat_template(
                    st.session_state.chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # 编码为张量
                inputs = tokenizer(
                    new_prompt,
                    return_tensors="pt",
                    truncation=True
                ).to(device)

                # 流式输出器（逐增量 token 推送）
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                generation_kwargs = {
                    "input_ids": inputs.input_ids,
                    "max_length": inputs.input_ids.shape[1] + st.session_state.max_new_tokens,
                    "num_return_sequences": 1,
                    "do_sample": True,
                    "attention_mask": inputs.attention_mask,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "temperature": st.session_state.temperature,
                    "top_p": 0.85,
                    "streamer": streamer,
                }

                # 后台线程生成
                Thread(target=model.generate, kwargs=generation_kwargs).start()

                # 前台逐段渲染
                answer = ""
                for new_text in streamer:
                    answer += new_text
                    placeholder.markdown(process_assistant_content(answer), unsafe_allow_html=True)

            # 记录助手回答
            messages.append({"role": "assistant", "content": answer})
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})

            # 当前条目就地提供删除按钮（删除最近一轮）
            with st.empty():
                if st.button("×", key=f"delete_{len(messages) - 1}"):
                    st.session_state.messages = st.session_state.messages[:-2]
                    st.session_state.chat_messages = st.session_state.chat_messages[:-2]
                    st.rerun()


if __name__ == "__main__":
    # 推迟导入以加快首次页面加载
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    main()
