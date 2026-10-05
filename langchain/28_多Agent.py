from dotenv import load_dotenv
load_dotenv()


"""
    单个 Agent 的问题：
        1.system_prompt 太长会导致模型注意力分散
        2.工具太多会增加模型选择工具的出错概率
        3.不同类型的任务需要不同的专业知识和行为风格
    多 Agent 的方案：每个 Agent 专注于一个领域，通过协作完成复杂任务。
"""


# 方式1：子Agent作为工具
# 将 Agent 编译成 CompiledStateGraph，然后作为一个工具注册给父 Agent
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)

# 子 Agent 1：天气专家
@tool
def get_weather(city: str) -> str:
    """查询天气"""
    data = {"杭州": "晴，25°C", "北京": "多云，18°C"}
    return data.get(city, f"{city}: 数据暂缺")

weather_agent = create_agent(
    model=model,
    tools=[get_weather],
    name="weather_expert",  # 名字用于标识和日志
    system_prompt="你是天气专家，专门回答天气相关问题。回答要简洁。",
)

# 子 Agent 2：计算专家
@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    result = eval(expression, {"__builtins__": {}}, {})
    return f"{expression} = {result}"

math_agent = create_agent(
    model=model,
    tools=[calculate],
    name="math_expert",
    system_prompt="你是数学专家，专门进行数学计算。回答要简洁。",
)

# 父 Agent：协调者
# 将子 Agent 作为工具注册
@tool
def ask_weather_expert(question: str) -> str:
    """向天气专家咨询天气相关问题。

    Args:
        question: 关于天气的问题
    """
    result = weather_agent.invoke(
        {"messages": [HumanMessage(content=question)]}
    )
    return result["messages"][-1].content


@tool
def ask_math_expert(question: str) -> str:
    """向数学专家咨询数学计算问题。

    Args:
        question: 数学计算问题
    """
    result = math_agent.invoke(
        {"messages": [HumanMessage(content=question)]}
    )
    return result["messages"][-1].content


coordinator = create_agent(
    model=model,
    tools=[ask_weather_expert, ask_math_expert],
    system_prompt="""你是协调助手。根据用户问题选择合适的专家：
- 天气相关问题 → 使用 ask_weather_expert
- 数学计算问题 → 使用 ask_math_expert
- 如果同时涉及多个领域，依次咨询各个专家""",
)

# 测试复合问题
result = coordinator.invoke({
    "messages": [HumanMessage(
        content="杭州今天天气怎么样？如果温度是 25 度，换算成华氏度是多少？"
        "（公式：华氏度 = 摄氏度 × 9/5 + 32）"
    )]
})
print(result["messages"][-1].content)


# 方式2：用name参数区分Agent。当你将子 Agent 作为工具嵌入时，设置 name 参数有助于追踪消息来源：
# name 参数的作用：
# 1. 编译后的图中使用该名称
# 2. 作为子图节点嵌入父图时使用该名称
# 3. 所有 AI 消息被标记为该名称
agent = create_agent(
    model=model,
    tools=[...],
    name="customer_service",  # 给 Agent 命名
)


# 方式3：Middleware 实现 Agent 路由。更复杂的多 Agent 场景可以通过 Middleware 实现动态路由：
from langchain.agents.middleware import before_model

# 定义不同专家使用的工具集
general_tools = [tool_a, tool_b]
admin_tools = [tool_c, tool_d]

@before_model
def route_by_user_role(state, runtime):
    """根据用户角色动态切换可用工具"""
    context = runtime.context
    if context is None:
        return None

    user_role = context.get("user_role", "user")

    # 不同角色看到不同的工具
    if user_role == "admin":
        available_tools = general_tools + admin_tools
    else:
        available_tools = general_tools

    # 注意：before_model 不能直接修改 tools，
    # 需要配合 wrap_model_call 或 request.override 来实现
    return None

# 多Agent 结构模式
"""
    模式	        结构	                            适用场景
    协调者模式	一个父 Agent → 多个子 Agent 工具	任务类型明确可分类
    接力模式	    Agent A 的输出 → Agent B 的输入	流水线式处理（生成→审核→润色）
    辩论模式	    多个 Agent 并行输出 → 汇总决策	    需要多角度分析的问题
"""