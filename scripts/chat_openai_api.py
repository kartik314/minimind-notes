"""
文件用途: 使用 OpenAI 兼容接口调用本地部署的 MiniMind 模型，实现多轮对话
主要功能:
1. 通过 OpenAI Python SDK 指定 base_url，连接本地 http://127.0.0.1:8998/v1
2. 支持历史对话记忆（可设置携带多少轮上下文）
3. 支持两种模式：
   - 非流式: 一次性返回完整回答
   - 流式: 边生成边输出

"""

from openai import OpenAI

# 初始化OpenAI客户端，配置连接本地部署的模型服务
# api_key设置为"ollama"是为了兼容接口格式，本地服务通常不验证该值
# base_url指定本地模型服务的API地址
client = OpenAI(
    api_key="ollama",  # 形式上的API密钥，本地服务可忽略验证
    base_url="http://127.0.0.1:8998/v1"  # 本地Ollama/MiniMind API的OpenAI兼容端点
)

# 输出模式控制：True为流式输出（逐段显示回答），False为非流式输出（完整显示回答）
stream = True
# 存储完整的对话历史记录（包含所有轮次的提问与回答）
conversation_history_origin = []
# 用于实际交互的对话历史副本
conversation_history = conversation_history_origin.copy()

# 控制每次请求携带的历史对话轮数（必须为偶数，因为一问一答为一组）
# 例如：设置为2表示携带最近1轮对话（1个问题+1个回答）
# 设置为0表示不携带任何历史，每次都是全新的对话
history_messages_num = 2

# 主交互循环：持续接收用户输入并获取模型响应
while True:
    # 获取用户输入的问题
    query = input('[Q]: ')
    # 将用户的问题添加到对话历史中
    conversation_history.append({"role": "user", "content": query})

    # 调用本地模型API获取回答
    # model参数指定要使用的模型名称，需与本地部署的模型名称一致
    # messages参数传入需要携带的历史对话（最近的history_messages_num条）
    # stream参数控制是否启用流式输出
    response = client.chat.completions.create(
        model="minimind",  # 模型名称，对应本地部署的模型
        messages=conversation_history[-history_messages_num:],  # 截取需要携带的历史对话
        stream=stream  # 是否启用流式输出
    )

    if not stream:
        # 非流式处理：一次性获取完整回答
        # 从响应中提取助手的回答内容
        assistant_res = response.choices[0].message.content
        # 打印完整回答
        print('[A]: ', assistant_res)
    else:
        # 流式处理：逐块接收并显示回答内容
        print('[A]: ', end='')  # 不换行，准备后续逐段输出
        assistant_res = ''  # 用于拼接完整的回答内容
        # 遍历流式响应的每个数据块
        for chunk in response:
            # 提取当前块的内容（可能为空，需处理）
            chunk_content = chunk.choices[0].delta.content or ""
            # 输出当前块内容，不换行
            print(chunk_content, end="")
            # 将当前块内容添加到完整回答中
            assistant_res += chunk_content

    # 将助手的完整回答添加到对话历史中，用于维护上下文
    conversation_history.append({"role": "assistant", "content": assistant_res})
    # 打印空行，分隔不同轮次的对话
    print('\n\n')