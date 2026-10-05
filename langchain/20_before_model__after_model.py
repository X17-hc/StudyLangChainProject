from dotenv import load_dotenv
load_dotenv()


# LangChain 中间件钩子 -- @before_model 与 @after_model
# before_model 和 after_model 是最常用的两个 中间件（Middleware） 钩子。它们在每次模型调用前后执行，适合做内容过滤、消息预处理、响应审核等。


# 1.@before_model —— 模型调用前拦截
# @before_model 在每次调用模型之前执行。你可以在这里修改消息、注入上下文条件、或直接跳过模型调用。

# 1.1. 消息预处理 —— 限制对话长度
from langchain.agents import create_agent
from langchain.agents.middleware import before_model
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage


@before_model
def limit_context(state, runtime):
    """限制消息历史长度，防止上下文过长"""
    messages = state.get("messages", [])

    # 保留 system message + 最近 6 条消息
    MAX_MESSAGES = 6
    if len(messages) > MAX_MESSAGES:
        # 裁剪到最近的消息
        trimmed = messages[-MAX_MESSAGES:]
        # 确保第一条是用户消息
        if trimmed and trimmed[0].type != "human":
            trimmed = trimmed[1:]
        return {"messages": trimmed}

    return None


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    middleware=[limit_context],
    system_prompt="你是菜鸟教程 RUNOOB 的助手。",
)

# 模拟多轮对话
result = agent.invoke({
    "messages": [
        HumanMessage(content="第一轮"),
        HumanMessage(content="第二轮"),
        HumanMessage(content="第三轮"),
        HumanMessage(content="第四轮"),
        HumanMessage(content="第五轮"),
        HumanMessage(content="第六轮"),
        HumanMessage(content="第七轮"),
    ]
})
print(f"消息数: {len(result['messages'])}")
print(f"回复: {result['messages'][-1].content}")



# 1.2 内容过滤——屏蔽敏感词
from langchain.agents.middleware import before_model

SENSITIVE_WORDS = ["密码", "银行卡号", "身份证号"]

@before_model
def content_filter(state, runtime):
    """检查用户消息是否包含敏感词，如果包含则拦截"""
    messages = state.get("messages", [])
    if not messages:
        return None

    last_msg = messages[-1]
    content = str(last_msg.content) if hasattr(last_msg, 'content') else ""

    for word in SENSITIVE_WORDS:
        if word in content:
            print(f"[拦截] 检测到敏感词: {word}")
            # jump_to="end" 直接结束，不让模型回复
            return {
                "jump_to": "end",
                "messages": [
                    HumanMessage(content=f"抱歉，为了安全，不能处理包含「{word}」的请求。")
                ]
            }

    return None


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    middleware=[content_filter],
    system_prompt="你是菜鸟教程 RUNOOB 的助手。",
)

# 正常问题
result = agent.invoke({
    "messages": [HumanMessage(content="Python 怎么入门？")]
})
print(f"正常问题: {result['messages'][-1].content[:80]}")

# 敏感问题
result = agent.invoke({
    "messages": [HumanMessage(content="我的银行卡号是多少？你能帮我查吗？")]
})
print(f"\n敏感问题: {result['messages'][-1].content}")




# 2.@after_model——模型调用后处理
# @after_model 在模型回复后执行，适合审核模型输出、提取关键信息、追加后续指令等。

# 2.1 响应内容审核
from langchain.agents.middleware import after_model

FORBIDDEN_TOPICS = ["政治", "暴力", "色情"]

@after_model
def response_audit(state, runtime):
    """审核模型回复，如果涉及禁止话题则替换"""
    messages = state.get("messages", [])
    if not messages:
        return None

    last_msg = messages[-1]
    content = str(last_msg.content) if hasattr(last_msg, 'content') else ""

    for topic in FORBIDDEN_TOPICS:
        if topic in content:
            runtime.stream_writer({
                "type": "warning",
                "message": f"检测到回复包含「{topic}」相关内容，已被替换"
            })
            # 返回一个覆盖的消息（通过 add_messages reducer）
            from langchain.messages import AIMessage
            return {
                "messages": [
                    AIMessage(content="抱歉，我无法回答这个问题。"
                                      "请询问编程学习相关的内容。")
                ]
            }

    return None


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    middleware=[response_audit],
    system_prompt="你是菜鸟教程 RUNOOB 的助手。",
)

result = agent.invoke({
    "messages": [HumanMessage(content="Python 有哪些学习资源？")]
})
print(f"正常回复: {result['messages'][-1].content}")




# 2.2 自动追加提示信息
from langchain.agents.middleware import after_model
from langchain.messages import AIMessage


@after_model
def append_disclaimer(state, runtime):
    """在每次模型回复后自动追加免责声明"""
    messages = state.get("messages", [])
    if not messages:
        return None

    last_msg = messages[-1]

    # 只在 AI 的最终回复（没有 tool_calls）时追加
    if (last_msg.type == "ai"
        and last_msg.content
        and not (hasattr(last_msg, 'tool_calls') and last_msg.tool_calls)):
        # 替换最后一条 AI 消息，加上免责声明
        return {
            "messages": [
                AIMessage(
                    content=(
                        last_msg.content
                        + "\n\n---\n*以上内容由菜鸟教程 RUNOOB AI 助手生成，仅供参考。*"
                    )
                )
            ]
        }

    return None


agent = create_agent(
    model=model,
    middleware=[append_disclaimer],
    system_prompt="你是菜鸟教程的助手。",
)

result = agent.invoke({
    "messages": [HumanMessage(content="Python 容易学吗？")]
})
print(result["messages"][-1].content)




# 3.can_jump_to——流程跳转控制.
# 在 before_model 和 after_model 中，你可以通过 can_jump_to 参数和 jump_to 状态来控制 Agent 的流程
from langchain.agents.middleware import before_model


@before_model(can_jump_to=["end"])  # 声明可以跳转到的目标
def conditional_exit(state, runtime):
    """在特定条件下直接结束 Agent"""
    messages = state.get("messages", [])
    if not messages:
        return None

    # 如果用户说"再见"，直接结束
    last_content = str(messages[-1].content)
    if last_content.strip() in ["再见", "拜拜", "bye"]:
        return {
            "jump_to": "end",  # 直接结束 Agent
            "messages": [{"role": "assistant", "content": "再见！期待下次为您服务。"}]
        }

    return None

"""
    can_jump_to         值	                含义	适用场景
    ["end"]	            可跳转到结束	        条件退出、安全拦截
    ["model"]	        可跳转回模型	        需要让模型重新处理
    ["tools"]	        可跳转到工具节点	    跳过模型直接执行工具
    ["model", "end"]	可跳转到模型或结束	    多种条件分支
"""
