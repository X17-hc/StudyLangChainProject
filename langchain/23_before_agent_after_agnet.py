from dotenv import load_dotenv
load_dotenv()


# before_agent 和 after_agent 是 Agent 级别的钩子，分别在 Agent 执行之前和完成之后各执行一次。适合做初始化、预处理、后处理和统计分析。

# before_agent —— Agent 开始前的准备工作
# before_agent 在 Agent 正式开始执行前运行，只执行一次。你可以在这里做输入预处理、用户信息验证、资源初始化等。

# 1.输入预处理——自动修正用户输入
from langchain.agents import create_agent
from langchain.agents.middleware import before_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool

@before_agent
def preprocess_input(state, runtime):
    """在 Agent 开始前处理用户输入"""
    messages = state.get("messages", [])
    if not messages:
        return None

    # 获取用户最后一条消息
    last_msg = messages[-1]
    content = str(last_msg.content) if hasattr(last_msg, 'content') else ""

    # 自动添加礼貌用语（如果用户直接问问题）
    greetings = ["你好", "您好", "hi", "hello", "嗨"]
    if content and not any(content.lower().startswith(g) for g in greetings):
        # 不修改，直接返回
        pass

    return None


@tool
def search_course(keyword: str) -> str:
    """在菜鸟教程 RUNOOB 搜索课程"""
    courses = {
        "python": "Python3 基础教程（免费，30章）",
        "html": "HTML 基础教程（免费，25章）",
    }
    return courses.get(keyword.lower(), f"未找到 {keyword} 相关课程")


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[search_course],
    middleware=[preprocess_input],
    system_prompt="你是菜鸟教程 RUNOOB 的课程顾问。",
)

# result = agent.invoke({
#     "messages": [HumanMessage(content="Python 课程")]
# })
# print(f"回复: {result['messages'][-1].content}")

step = 0
for chunk in agent.stream(
    {"messages": [HumanMessage(content="Python 课程")]},
    stream_mode="updates",
):
    step += 1
    print(f"--- 步骤 {step} ---")
    for node_name, update in chunk.items():
        print(f"节点: {node_name}")
        if update is None:
            continue          # 跳过 None 更新
        if "messages" in update:
            for msg in update["messages"]:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  → 请求调用工具: {tc['name']}({tc['args']})")
                elif msg.type == "tool":
                    print(f"  → 工具结果 [{msg.name}]: {msg.content}")
                elif msg.type == "ai" and msg.content:
                    print(f"  → AI 回复: {msg.content[:100]}")



# 2.访问控制 —— 权限检查
from langchain.agents.middleware import before_agent

@before_agent
def access_control(state, runtime):
    """检查用户是否有权限使用 Agent"""
    # 从 runtime.context 获取用户信息
    context = runtime.context
    if context is None:
        return None

    user_role = context.get("user_role", "guest")

    # 访客用户只能使用有限功能
    if user_role == "guest":
        messages = state.get("messages", [])
        if messages:
            last_content = str(messages[-1].content)
            # 检查是否涉及限制功能
            restricted_keywords = ["删除", "管理", "配置", "admin"]
            if any(kw in last_content for kw in restricted_keywords):
                return {
                    "jump_to": "end",
                    "messages": [HumanMessage(
                        content="您当前的权限不足，无法执行此操作。请登录后重试。"
                    )]
                }

    return None



# after_agent —— Agent 完成后的处理。after_agent
# 在 Agent 完成所有处理后执行（只执行一次）。你可以在这里格式化最终输出、记录统计信息、清理资源等。

# 3.统计分析 —— 记录对话数据
from langchain.agents.middleware import after_agent

@after_agent
def conversation_stats(state, runtime):
    """统计对话信息并追加到结果中"""
    messages = state.get("messages", [])

    # 统计数据
    model_calls = 0
    tool_calls = 0
    total_chars = 0

    for msg in messages:
        if msg.type == "ai":
            model_calls += 1
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_calls += len(msg.tool_calls)
        if hasattr(msg, 'content') and msg.content:
            total_chars += len(str(msg.content))

    # 通过 custom stream 发送统计信息
    runtime.stream_writer({
        "type": "stats",
        "model_calls": model_calls,
        "tool_calls": tool_calls,
        "total_messages": len(messages),
        "total_chars": total_chars,
    })

    return None


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[search_course],
    middleware=[conversation_stats],
    system_prompt="你是菜鸟教程 RUNOOB 的课程顾问。",
)

# 使用 stream_mode=["updates", "custom"] 接收自定义事件
for mode, chunk in agent.stream(
    {"messages": [HumanMessage(content="查一下 Python 课程")]},
    stream_mode=["updates", "custom"],
):
    if mode == "custom" and chunk.get("type") == "stats":
        print(f"统计信息: {chunk}")




# 4.格式化输出 —— 统一回复风格
from langchain.agents.middleware import after_agent
from langchain.messages import AIMessage


@after_agent
def format_output(state, runtime):
    """在结果中追加格式化的总结信息"""
    messages = state.get("messages", [])
    if not messages:
        return None

    # 找到最后一条 AI 消息（最终回复）
    last_ai = None
    for msg in reversed(messages):
        if msg.type == "ai" and msg.content:
            last_ai = msg
            break

    if last_ai:
        # 统计信息
        tool_msgs = [m for m in messages if m.type == "tool"]
        tool_count = len(tool_msgs)

        footer = (
            f"\n\n---\n"
            f"> 本次对话共进行 {len(messages)} 条消息，"
            f"调用了 {tool_count} 次工具。\n"
            f"> 由菜鸟教程 RUNOOB AI 助手提供支持。"
        )

        # 追加到最终回复
        return {
            "messages": [
                AIMessage(content=last_ai.content + footer)
            ]
        }

    return None


model = init_chat_model("deepseek:deepseek-v4-flash",
                        temperature=0,
                        model_kwargs={
                                "extra_body": {
                                    "thinking": {"type": "disabled"}   # 关闭思考模式（deepseek的思考模式与结构化输出不兼容）
                                }
                            }
                        )
agent = create_agent(
    model=model,
    tools=[search_course],
    middleware=[format_output],
    system_prompt="你是菜鸟教程 RUNOOB 的课程顾问。",
)

result = agent.invoke({
    "messages": [HumanMessage(content="Python 有什么课程？")]
})
print(result["messages"][-1].content)