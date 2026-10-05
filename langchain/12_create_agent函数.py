from dotenv import load_dotenv
load_dotenv()

"""
from langchain.agents import create_agent

agent = create_agent(
    model,                     # str | BaseChatModel：语言模型
    tools=None,                # Sequence：工具列表
    *,
    system_prompt=None,        # str | SystemMessage：系统提示
    middleware=(),             # Sequence[AgentMiddleware]：中间件列表
    response_format=None,      # ResponseFormat | type：结构化输出配置
    state_schema=None,         # type[AgentState]：自定义状态结构
    context_schema=None,       # type：运行时上下文结构
    checkpointer=None,         # Checkpointer：对话持久化
    store=None,                # BaseStore：跨会话存储
    interrupt_before=None,     # list[str]：在哪些节点前暂停
    interrupt_after=None,      # list[str]：在哪些节点后暂停
    debug=False,               # bool：是否输出详细日志
    name=None,                 # str：Agent 名称
    cache=None,                # BaseCache：缓存配置
)
"""

# 1.model参数 - 模型配置
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

# 方式 1：传字符串（最常用）
# 推荐使用方式 1（传字符串）。create_agent() 内部会自动处理模型初始化、工具绑定、结构化输出等逻辑。方式 2 适合你需要在 Agent 之外也使用同一个模型实例的场景。
# create_agent 内部会调用 init_chat_model() 处理
agent = create_agent(
    model="deepseek:deepseek-v4-flash",
    system_prompt="你是菜鸟教程 RUNOOB 的助手",
)

# 方式 2：传已构建好的模型实例
# 适合需要精细控制模型参数的场景
model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0.3, max_tokens=500)
agent = create_agent(
    model=model,
    system_prompt="你是菜鸟教程 RUNOOB 的助手",
)

# 方式 3：传已绑定工具的模型实例
# less common，通常让 create_agent 自己管理工具绑定
model_with_tools = init_chat_model("deepseek:deepseek-v4-flash").bind_tools([...])



# 2.tools参数 - 工具列表
from langchain.tools import tool
from langchain.agents import create_agent

# 格式 1：@tool 装饰的函数（最常用）
@tool
def search_course(keyword: str) -> str:
    """搜索菜鸟教程课程"""
    return f"搜索结果：{keyword} 相关课程"


# 格式 2：Pydantic BaseModel 类
from pydantic import BaseModel, Field

class WeatherQuery(BaseModel):
    """查询天气"""
    city: str = Field(description="城市名称")


# 格式 3：字典（描述远程工具或内置工具）
mcp_tool = {
    "type": "mcp",
    "server_label": "weather_server",
    "server_url": "https://weather.example.com/sse",
    "allowed_tools": ["get_forecast"],
}

# 混合使用
agent = create_agent(
    model="deepseek:deepseek-v4-flash",
    tools=[search_course, WeatherQuery, mcp_tool],
)

# 传 None 或空列表表示 Agent 无工具可用，此时它就是一个纯粹的对话模型：
from langchain.agents import create_agent
from langchain.messages import HumanMessage

# 无工具 Agent——等价于直接调用模型
agent = create_agent(
    model="deepseek:deepseek-v4-flash",
    tools=None,
    system_prompt="你是菜鸟教程 RUNOOB 的助手",
)

result = agent.invoke({
    "messages": [HumanMessage(content="Python 适合零基础学习吗？")]
})
print(result["messages"][-1].content)


# 3.system_prompt 参数——系统提示。定义 Agent 的行为角色和约束规则。支持字符串和 SystemMessage 对象。
from langchain.agents import create_agent
from langchain.messages import SystemMessage

# 方式 1：字符串（简单直接）
agent = create_agent(
    model="deepseek:deepseek-v4-flash",
    system_prompt="你是菜鸟教程 RUNOOB 的学习顾问。回答要简洁，不超过 100 字。",
)

# 方式 2：SystemMessage 对象（可在多个 Agent 间复用）
system_msg = SystemMessage(
    content="你是菜鸟教程 RUNOOB 的学习顾问。回答要简洁，不超过 100 字。"
)
agent = create_agent(
    model="deepseek:deepseek-v4-flash",
    system_prompt=system_msg,
)




# 4.state_schema 参数——自定义状态
from typing import Annotated
from langchain.agents import create_agent, AgentState
from langchain.messages import HumanMessage
from langchain.tools import tool, InjectedState
from typing_extensions import TypedDict


# 扩展 AgentState，添加自定义字段
class LearningAgentState(AgentState):
    """自定义状态，增加学习进度相关字段"""
    user_level: str                       # 用户等级
    completed_topics: list[str]           # 已完成的主题列表


@tool
def track_progress(
    topic: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """记录用户的学习进度。

    Args:
        topic: 刚学完的主题名称
    """
    completed = state.get("completed_topics", [])
    completed.append(topic)
    return (
        f"已记录学习进度。当前已完成 {len(completed)} 个主题："
        f"{', '.join(completed)}"
    )


agent = create_agent(
    model="deepseek:deepseek-v4-flash",
    tools=[track_progress],
    state_schema=LearningAgentState,  # 使用自定义状态
    system_prompt="你是菜鸟教程 RUNOOB 的学习助手。",
)

# 运行时需要提供自定义状态的初始值
result = agent.invoke({
    "messages": [HumanMessage(content="我学完了 Python 基础，帮我记录一下")],
    "user_level": "入门",
    "completed_topics": ["HTML 基础"],
})

print(f"用户等级: {result.get('user_level')}")
print(f"已完成主题: {result.get('completed_topics')}")
print(f"回复: {result['messages'][-1].content[:100]}")

"""
    方法	                            说明	                适用场景
    invoke(input, config)	            同步运行，等待完整结果	    脚本、简单接口
    ainvoke(input, config)	            异步运行，等待完整结果	    Web 服务
    stream(input, config, stream_mode)	同步流式运行	            实时展示中间步骤
    astream(input, config, stream_mode)	异步流式运行	            WebSocket、SSE
    get_state(config)	                获取当前状态	            查看/恢复对话状态
    update_state(config, values)	    更新状态	                手动修改对话状态
"""