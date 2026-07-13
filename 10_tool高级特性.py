from dotenv import load_dotenv
load_dotenv()

# 1.return_direct —— 直接返回最终结果。设置 return_direct=True 后，工具执行完就立即结束 Agent 循环，工具返回内容直接作为最终输出
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage


# 普通工具：结果返回给模型，模型再做总结
@tool
def search_normal(keyword: str) -> str:
    """搜索菜鸟教程 RUNOOB 的课程（普通模式）"""
    return f"搜索结果：Python3 基础教程、Python 数据分析、Python 爬虫入门"


# return_direct 工具：结果直接作为最终输出
@tool(return_direct=True)
def search_direct(keyword: str) -> str:
    """搜索菜鸟教程 RUNOOB 的课程（直接返回模式）。

    当用户只需要搜索结果，不需要额外分析时使用此工具。
    """
    return f"搜索结果：Python3 基础教程、Python 数据分析、Python 爬虫入门"


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)

# 对比：普通模式 vs 直接返回模式
agent_normal = create_agent(
    model=model,
    tools=[search_normal],
    system_prompt="你是菜鸟教程的学习顾问。",
)

agent_direct = create_agent(
    model=model,
    tools=[search_direct],
    system_prompt="你是菜鸟教程的学习顾问。",
)

# 普通模式：模型会基于搜索结果再生成一段总结
result = agent_normal.invoke({
    "messages": [HumanMessage(content="搜索 Python 课程")]
})
print("=== 普通模式（模型会再加工）===")
print(result["messages"][-1].content[:150])

# 直接返回模式：工具结果就是最终答案
result = agent_direct.invoke({
    "messages": [HumanMessage(content="搜索 Python 课程")]
})
print("\n=== 直接返回模式（工具结果即最终答案）===")
print(result["messages"][-1].content[:150])

"""
    模式	                    工具执行后	                                适用场景
    return_direct=False（默认）	模型收到工具结果 → 模型继续思考 → 生成最终回复	需要分析/总结/进一步决策
    return_direct=True	        工具执行后立即结束 → 工具结果就是最终输出	    查询类、数据获取类、已格式化好的结果
"""




# 2.InjectedToolCallId——获取工具调用 ID。InjectedToolCallId——获取工具调用 ID
from typing import Annotated
from langchain.tools import tool, InjectedToolCallId


@tool
def log_user_action(
    action: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """记录用户操作到日志系统。

    Args:
        action: 用户操作描述
        tool_call_id: 系统自动注入的工具调用 ID
    """
    # 实际项目中这里会写入数据库或发送到日志服务
    return f"操作已记录 (调用ID: {tool_call_id}): {action}"


# InjectedToolCallId 参数会被自动注入，不需要手动传入
tool_input = {
    "args": {"action": "用户查询了 Python 课程"},
    "name": "log_user_action",
    "type": "tool_call",
    "id": "test-call-123"
}
result = log_user_action.invoke(tool_input)
print(result)




# 3.ToolException - 工具异常处理
from langchain.tools import tool, ToolException

@tool
def get_user_info(user_id: int) -> str:
    """根据用户 ID 查询用户信息。

    Args:
        user_id: 用户 ID，必须是正整数
    """
    # 数据校验
    if user_id <= 0:
        # 抛出 ToolException，而不是普通 Exception
        # ToolException 会被 Agent 捕获并告知模型
        raise ToolException(f"用户 ID 必须为正整数，收到了: {user_id}")

    # 模拟数据库查询
    users = {
        1: "张三（VIP 会员，注册于 2024-01-15）",
        2: "李四（普通用户，注册于 2024-03-20）",
    }

    if user_id not in users:
        raise ToolException(f"未找到 ID 为 {user_id} 的用户")

    return users[user_id]


# 正常调用
print(get_user_info.invoke({"user_id": 1}))

# 异常调用 1：无效 ID
try:
    get_user_info.invoke({"user_id": -1})
except ToolException as e:
    print(f"工具异常: {e}")

# 异常调用 2：用户不存在
try:
    get_user_info.invoke({"user_id": 999})
except ToolException as e:
    print(f"工具异常: {e}")





# 4.handle_tool_errors——让 Agent 自动处理工具错误
from langchain.tools import tool, ToolException
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。

    Args:
        city: 城市名称，必须是中文全称，如 "杭州"、"北京"
    """
    weather_data = {
        "杭州": "晴，25°C",
        "北京": "多云，18°C",
        "上海": "小雨，22°C",
    }
    if city not in weather_data:
        # 城市不在数据中时抛出 ToolException
        raise ToolException(
            f"未收录城市 '{city}'。"
            f"可使用城市：{', '.join(weather_data.keys())}。"
            f"请使用中文城市全称。"
        )
    return f"{city}天气：{weather_data[city]}"


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)

# 在不使用 Agent 时，也可以直接用 handle_tool_errors 控制行为
# 方式 1：返回错误信息字符串（模型能看到并修正）
tool_with_feedback = get_weather.with_config(
    handle_tool_errors=True  # 捕获异常并返回错误信息给模型
)

# 方式 2：直接抛出异常（Agent 终止）
tool_without_feedback = get_weather.with_config(
    handle_tool_errors=False  # 异常直接上抛
)

# 测试：用错误城市名调用
# handle_tool_errors=True 时，错误信息会返回给模型
result = tool_with_feedback.invoke({"city": "北境"})
print(f"handle_tool_errors=True: {result}")

"""
    handle_tool_errors	    行为	                                        适用场景
    True	                捕获所有异常，将错误信息作为 ToolMessage 返回给模型	希望 Agent 自行修正错误
    False	                异常直接上抛，Agent 执行中断	                    不可恢复的错误
    str	                    捕获异常后用指定字符串替换错误信息	                自定义错误提示
    tuple[type]	            只捕获指定类型的异常	                            只处理特定异常
"""




# 5.完整示例
from langchain.tools import tool, ToolException
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage


@tool
def book_course(user_name: str, course_name: str) -> str:
    """为用户预订菜鸟教程 RUNOOB 的课程。

    Args:
        user_name: 用户姓名
        course_name: 课程名称
    """
    # 校验用户是否存在
    valid_users = {"张三", "李四", "王五"}
    if user_name not in valid_users:
        raise ToolException(
            f"用户 '{user_name}' 不存在。"
            f"有效用户：{', '.join(sorted(valid_users))}"
        )

    # 校验课程是否存在
    valid_courses = {"Python3 基础教程", "HTML 基础教程", "Java 面向对象"}
    if course_name not in valid_courses:
        raise ToolException(
            f"课程 '{course_name}' 不存在。"
            f"有效课程：{', '.join(sorted(valid_courses))}"
        )

    return f"已为 {user_name} 成功预订《{course_name}》"



model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)

agent = create_agent(
    model=model,
    tools=[book_course],
    system_prompt="你是菜鸟教程 RUNOOB 的课程顾问。",
)

# 正常调用
result = agent.invoke({
    "messages": [HumanMessage(content="帮张三预订 Python3 基础教程")]
})
print(f"成功: {result['messages'][-1].content[:100]}")

# 错误调用：用户不存在
result = agent.invoke({
    "messages": [HumanMessage(content="帮赵六预订 Python3 基础教程")]
})
print(f"\n错误-用户不存在:")
for msg in result["messages"]:
    if msg.type == "tool":
        print(f"  [{msg.type}] {msg.content[:80]}")
    elif msg.type == "ai" and msg.content:
        print(f"  [{msg.type}] {msg.content[:100]}")