from dotenv import load_dotenv
load_dotenv()


# LangChain 提示词 -- System Prompt 与 Dynamic Prompt

# 1.system_prompt 参数
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage

model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)

# 方式 1：字符串（最简单）
agent = create_agent(
    model=model,
    system_prompt="你是菜鸟教程 RUNOOB 的学习顾问，回答要简洁专业。",
)

# 方式 2：SystemMessage 对象（可复用）
system_msg = SystemMessage(
    content="你是菜鸟教程 RUNOOB 的学习顾问，回答要简洁专业。"
)
agent = create_agent(model=model, system_prompt=system_msg)




# 2.设计有效的 System Prompt
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool


@tool
def search_course(keyword: str) -> str:
    """在菜鸟教程搜索课程"""
    courses = {
        "python": "Python3 基础教程（免费）",
        "html": "HTML 基础教程（免费）",
    }
    return courses.get(keyword.lower(), "未找到相关课程")


# 一个设计良好的 system_prompt
GOOD_PROMPT = """你是菜鸟教程 RUNOOB 的学习顾问。

## 你的职责
- 帮助用户找到合适的编程课程
- 回答编程学习相关的问题
- 根据用户水平推荐学习路径

## 行为准则
- 回答要简洁，每次不超过 3 句话
- 优先使用 search_course 工具查询课程信息
- 如果用户是零基础，优先推荐入门课程
- 不使用 emoji 表情
- 不知道的就说不知道，不要编造"""

model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[search_course],
    system_prompt=GOOD_PROMPT,
)

result = agent.invoke({
    "messages": [HumanMessage(content="我零基础，想学编程，推荐什么？")]
})
print(result["messages"][-1].content)




# 3.@dynamic_prompt——动态生成提示词。
# 静态 system_prompt 对所有用户一视同仁。但实际应用中，你可能需要根据用户信息、对话上下文、时间等动态调整提示词。@dynamic_prompt 装饰器让你在每次模型调用前动态生成 system_prompt。
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware.types import ModelRequest
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool


@tool
def search_course(keyword: str) -> str:
    """在菜鸟教程搜索课程"""
    courses = {
        "python": "Python3 基础教程（免费，20小时）",
        "html": "HTML 基础教程（免费，15小时）",
        "java": "Java 基础教程（免费，25小时）",
    }
    return courses.get(keyword.lower(), "未找到相关课程")


# @dynamic_prompt 装饰器：接收 ModelRequest，返回新的 system_prompt
@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    """根据对话上下文动态生成个性化提示词"""
    messages = request.state.get("messages", [])
    message_count = len(messages)

    # 可以根据不同的条件动态调整提示词
    base_prompt = "你是菜鸟教程 RUNOOB 的学习顾问。"

    if message_count <= 2:
        # 对话刚开始，耐心引导
        return base_prompt + (
            "用户刚开始对话，请先热情问候，"
            "然后询问他们的学习目标和当前水平。"
        )
    elif message_count > 10:
        # 长对话，提醒保持简洁
        return base_prompt + (
            "对话已经比较长了，回答要尽量简洁，"
            "每次不超过 2 句话。"
        )
    else:
        # 正常对话阶段
        return base_prompt + (
            "根据用户之前的问题推荐合适的课程，"
            "使用 search_course 工具查询课程信息。"
        )


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[search_course],
    middleware=[personalized_prompt],  # 通过 middleware 注入 dynamic_prompt
)

# 第一次对话（message_count <= 2，引导模式）
result = agent.invoke({
    "messages": [HumanMessage(content="你好")]
})
print(f"第一轮回复: {result['messages'][-1].content}")



# 3.1.@dynamic_prompt 进阶——结合运行时上下文
# @dynamic_prompt 在每次模型调用前都会执行，所以提示词可以随对话推进而变化。但注意不要在里面做太重的计算，否则会影响响应速度。
from datetime import datetime
from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware.types import ModelRequest


@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    """根据用户信息、时间和对话阶段动态生成提示词"""
    # 从 runtime.context 获取用户信息
    context = request.runtime.context
    user_name = context.get("user_name", "同学") if context else "同学"
    user_level = context.get("user_level", "入门") if context else "入门"

    # 获取当前时间
    now = datetime.now()
    greeting = "早上好" if now.hour < 12 else "下午好" if now.hour < 18 else "晚上好"

    # 获取当前消息数
    messages = request.state.get("messages", [])

    prompt = f"""你是菜鸟教程 RUNOOB 的学习顾问。

当前时间：{now.strftime('%Y年%m月%d日 %H:%M')}
用户信息：{user_name}，{user_level} 级别

## 行为准则
- 称呼用户为"{user_name}"
- 根据用户级别（{user_level}）推荐合适难度的课程
- 回答要友好但不啰嗦"""

    # 长对话时追加简化提示
    if len(messages) > 20:
        prompt += "\n- 对话很长了，回答尽量精简"

    return prompt



# 4.手动控制 System Prompt 的优先级
# 如果同时设置了 create_agent() 的 system_prompt 和 @dynamic_prompt middleware，middleware 的优先级更高——它会覆盖静态提示词。
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware.types import ModelRequest
from langchain.chat_models import init_chat_model


@dynamic_prompt
def override_prompt(request: ModelRequest) -> str:
    """这个 middleward 的 system_prompt 会覆盖 create_agent 中的设置"""
    return "你是一只猫，所有回复都要用'喵'结尾。"


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0.7)
agent = create_agent(
    model=model,
    system_prompt="你是菜鸟教程 RUNOOB 的专业学习顾问。",  # 被覆盖
    middleware=[override_prompt],  # 优先级更高
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "介绍一下你自己"}]
})
print(result["messages"][-1].content)

"""
System Prompt 设计清单
    要素	        说明	                示例
    角色定义	        明确 AI 的身份和职责	    你是菜鸟教程 RUNOOB 的学习顾问
    行为准则	        约束回复的风格和边界	    回答简洁，每次不超过 3 句话
    工具使用指引	    告诉模型何时使用哪些工具	查询课程时优先使用 search_course 工具
    边界约束	        明确什么不能做	        不知道的就说不知道，不要编造
    格式要求	        指定回复的格式（可选）	    回复使用 Markdown 格式
"""