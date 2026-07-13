from dotenv import load_dotenv
load_dotenv()

# 1.LangChain 工具访问 -- InjectedState 与 InjectedStore
# InjectedState——在工具中访问 Agent 状态
from typing import Annotated, Any
from langchain.tools import tool, InjectedState
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

@tool
def remember_preference(
    preference: str,
    state: Annotated[dict[str, Any], InjectedState],
) -> str:
    """记住用户的偏好设置。

    Args:
        preference: 用户的偏好内容
        state: 系统自动注入的当前 Agent 状态
    """
    # 从状态中获取之前的消息历史
    messages = state.get("messages", [])
    message_count = len(messages)

    # 可以读取状态中的任何字段
    previous_prefs = state.get("user_preferences", "无")

    return (
        f"已记住偏好: {preference}。"
        f"(当前对话共 {message_count} 条消息，"
        f"之前偏好: {previous_prefs})"
    )


# 将工具放入 Agent 后，Agent 运行时会自动注入 state
agent = create_agent(
    model=init_chat_model("deepseek:deepseek-v4-flash"),
    tools=[remember_preference],
    system_prompt="你是偏好管理助手"
)

# Agent 调用过程中，工具会收到包含 messages 的状态
result = agent.invoke({"messages": [{"role": "user", "content": "记住我喜欢暗色主题"}]})
print(result.get("messages"))




# 1.1 在 Agent 中使用 InjectedState
from typing import Annotated, Any
from langchain.tools import tool, InjectedState
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage


@tool
def conversation_stats(
    state: Annotated[dict[str, Any], InjectedState],
) -> str:
    """获取当前对话的统计信息，如消息数量、对话长度等。

    不需要任何参数，统计信息从当前状态中自动读取。
    """
    messages = state.get("messages", [])
    human_msgs = [m for m in messages if m.type == "human"]
    ai_msgs = [m for m in messages if m.type == "ai"]
    tool_msgs = [m for m in messages if m.type == "tool"]

    return (
        f"对话统计：共 {len(messages)} 条消息 | "
        f"用户消息 {len(human_msgs)} 条 | "
        f"AI 回复 {len(ai_msgs)} 条 | "
        f"工具调用 {len(tool_msgs)} 次"
    )




# 2.InjectedStore——在工具中访问持久化存储
# Agent 状态（state）是对话级别的，对话结束就没了。而 Store 是跨会话的持久化存储，可以用来保存用户偏好、学习进度等长期信息。
# InjectedStore 让工具可以直接读写 Store。
from typing import Annotated
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langchain.tools import tool, InjectedStore
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage


# 2.1. 创建持久化存储并预置数据
store = InMemoryStore()
store.put(("users", "user_001"), "profile", {
    "data": {
        "name": "小明",
        "level": "入门",
        "completed_courses": ["HTML 基础教程"]
    }
})

# 2.2. 定义工具（Store 由框架注入）
@tool
def get_user_profile(
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """获取当前用户的学习档案信息"""
    item = store.get(("users", "user_001"), "profile")
    if item is None:
        return "未找到用户档案"
    profile = item.value["data"]
    return (
        f"用户档案：姓名={profile['name']}，"
        f"水平={profile['level']}，"
        f"已完成课程={', '.join(profile['completed_courses'])}"
    )

@tool
def save_course_progress(
    course_name: str,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """保存用户的学习进度到持久化存储"""
    item = store.get(("users", "user_001"), "profile")
    profile = item.value["data"] if item else {
        "name": "小明", "level": "入门", "completed_courses": []
    }
    if course_name not in profile["completed_courses"]:
        profile["completed_courses"].append(course_name)
    store.put(("users", "user_001"), "profile", {"data": profile})
    return (
        f"学习进度已更新！已完成 {len(profile['completed_courses'])} 门课程："
        f"{', '.join(profile['completed_courses'])}"
    )

# 2.3. 创建 Agent（使用 LangGraph 的 create_react_agent）
#    关键：将 store 通过 checkpointer 或 store 参数注入
model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)

# LangGraph 的 create_react_agent 可以接受一个 store 参数
# 注意：create_react_agent 返回的是一个编译好的 LangGraph 应用
agent = create_react_agent(
    model=model,
    tools=[get_user_profile, save_course_progress],
    store=store,   # ← 这里把 store 传给 Agent，工具调用时会自动注入
)

# 2.4. 测试：通过 Agent 调用工具
print("=== 查询用户档案 ===")
result = agent.invoke({
    "messages": [HumanMessage(content="帮我查一下我的学习档案")]
})
print(result["messages"][-1].content)

print("\n=== 保存课程进度 ===")
result = agent.invoke({
    "messages": [HumanMessage(content="我刚学完了 Python3 基础教程，帮我记录一下")]
})
print(result["messages"][-1].content)

print("\n=== 再次查询档案，验证持久化 ===")
result = agent.invoke({
    "messages": [HumanMessage(content="再查一下我的档案")]
})
print(result["messages"][-1].content)





# 3.在 Agent 中结合 Store
from typing import Annotated
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langchain.tools import tool, InjectedStore
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

# 创建 Store 并预置数据
store = InMemoryStore()
store.put(("runoob", "courses"), "catalog", {
    "data": {
        "Python3 基础教程": {"price": "免费", "duration": "20小时"},
        "Python 数据分析": {"price": "会员", "duration": "30小时"},
        "HTML 基础教程": {"price": "免费", "duration": "15小时"},
    }
})


@tool
def query_course_price(
    course_name: str,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """查询菜鸟教程 RUNOOB 中指定课程的价格信息。

    Args:
        course_name: 课程名称
    """
    item = store.get(("runoob", "courses"), "catalog")
    catalog = item.value["data"] if item else {}

    if course_name in catalog:
        info = catalog[course_name]
        return f"《{course_name}》- 价格：{info['price']}，学习时长：{info['duration']}"
    return f"未找到课程《{course_name}》"


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[query_course_price],
    store=store,  # 将 Store 传入 Agent
    system_prompt="你是菜鸟教程 RUNOOB 的课程顾问。",
)

result = agent.invoke({
    "messages": [HumanMessage(content="Python3 基础教程和 Python 数据分析分别多少钱？")]
})
print(result["messages"][-1].content)


"""
    维度	    InjectedState	                InjectedStore
    作用域	    当前对话（单次 Agent 运行）	    跨会话（多次 Agent 运行共享）
    生命周期	    对话结束即消失	                持久化存储
    典型用途	    读取消息历史、当前对话的中间结果	用户偏好、学习进度、配置信息
    传入方式	    InjectedState（自动注入）	    InjectedStore()（需要括号）
    数据组织	    扁平字典                        	命名空间 + 键的层级结构
"""




# 4.InjectedToolArg——标记通用注入参数
from typing import Annotated
from langchain.tools import tool, InjectedToolArg


@tool
def my_tool(
    normal_param: str,
    injected_param: Annotated[str, InjectedToolArg],
) -> str:
    """一个包含注入参数的示例工具。

    Args:
        normal_param: 这个参数由模型提供
        injected_param: 这个参数由框架注入（Agent 不需要提供）
    """
    return f"normal={normal_param}, injected={injected_param}"
