from dotenv import load_dotenv
load_dotenv()

"""
LangChain 提供了三种结构化输出策略。
三种策略概述
    策略	                原理	                                                    模型支持	                        响应速度
    ToolStrategy	    将 Schema 伪装成工具，模型"调用"这个工具来输出结构化数据	    所有支持 function calling 的模型	较慢（多一次工具调用）
    ProviderStrategy	使用模型原生的结构化输出能力（如 OpenAI 的 response_format）	部分模型（GPT-4o+、Claude 3+等）	较快（直接输出）
    AutoStrategy	    自动检测模型能力，选择最佳策略	                            自动适配	                        自动选择最优
"""

# 1.ToolStrategy——工具调用模式
# ToolStrategy 是兼容性最好的方式。它将你的 Schema 转换为一个"假工具"，模型通过调用这个工具来输出结构化数据。
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage


class WeatherReport(BaseModel):
    """天气报告"""
    city: str = Field(description="城市名称")
    temperature: float = Field(description="温度（摄氏度）")
    condition: str = Field(description="天气状况")
    humidity: int = Field(description="湿度百分比")


model = init_chat_model("deepseek:deepseek-v4-flash",
                        temperature=0,
                        model_kwargs={
                            "extra_body":{
                                "thinking":{"type": "disabled"}
                        }
                    }
                )

# 显式指定使用 ToolStrategy
agent = create_agent(
    model=model,
    response_format=ToolStrategy(schema=WeatherReport),
    system_prompt="你是天气助手，根据用户描述生成结构化天气报告。",
)

result = agent.invoke({
    "messages": [HumanMessage(content="杭州今天晴天，温度25度，湿度60%")]
})

report = result["structured_response"]
print(f"城市: {report.city}")
print(f"温度: {report.temperature}°C")
print(f"状况: {report.condition}")
print(f"湿度: {report.humidity}%")

# 查看执行过程——可以发现多了一条工具调用的消息
print(f"\n消息总数: {len(result['messages'])}")
for msg in result["messages"]:
    print(f"  [{msg.type}]", end="")
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        print(f" 调用: {[tc['name'] for tc in msg.tool_calls]}")
    elif msg.type == "tool":
        print(f" {msg.content[:60]}")
    else:
        print(f" {str(msg.content)[:60]}")

# 1.1 handle_errors —— 错误重试
from langchain.agents.structured_output import ToolStrategy
# handle_errors=True：输出格式错误时，将错误信息反馈给模型重试
strategy_with_retry = ToolStrategy(
    schema=WeatherReport,
    handle_errors=True,  # 默认 False
)



# 2.ProviderStrategy——原生结构化输出
# ProviderStrategy 使用模型提供商的原生能力（如 OpenAI 的 response_format 参数）。不是所有模型都支持。
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage


class CourseInfo(BaseModel):
    """课程信息"""
    name: str = Field(description="课程名称")
    level: str = Field(description="难度：入门/进阶/高级")
    price: str = Field(description="价格信息")


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)

# 显式指定 ProviderStrategy
agent = create_agent(
    model=model,
    response_format=ProviderStrategy(schema=CourseInfo),
    system_prompt="你是菜鸟教程 RUNOOB 的课程助手。",
)

result = agent.invoke({
    "messages": [HumanMessage(content="Python3 基础教程是入门级的免费课程")]
})

course = result["structured_response"]
print(f"课程: {course.name}")
print(f"难度: {course.level}")
print(f"价格: {course.price}")
print(f"\n消息数: {len(result['messages'])}")  # 比 ToolStrategy 少




# 3.AutoStrategy——自动选择
# 这是最推荐的方式。传入 Pydantic 模型（而不是策略对象），LangChain 会自动选择最佳策略：
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.agents.structured_output import ToolStrategy

class Analysis(BaseModel):
    """分析结果"""
    summary: str = Field(description="一句话总结")
    score: int = Field(description="评分 1~10")
    pros: list[str] = Field(description="优点列表")
    cons: list[str] = Field(description="缺点列表")


# 直接传入 Pydantic 模型——LangChain 自动选择策略
model = init_chat_model("deepseek:deepseek-v4-flash",
                        temperature=0,
                        model_kwargs={
                            "extra_body":{
                                "thinking":{"type": "disabled"}
        }
    }
)

agent = create_agent(
    model=model,
    response_format=Analysis,  # 直接传模型，自动选择策略,DeepSeek不支持
    system_prompt="你是课程评估专家，评估用户描述的课程。",
)

result = agent.invoke({
    "messages": [HumanMessage(
        content="菜鸟教程 RUNOOB 的 Python 课程：内容系统全面，"
                "实例丰富，而且完全免费。但视频教程较少，"
                "高级内容覆盖不够。"
    )]
})

analysis = result["structured_response"]
print(f"总结: {analysis.summary}")
print(f"评分: {analysis.score}/10")
print(f"优点: {', '.join(analysis.pros)}")
print(f"缺点: {', '.join(analysis.cons)}")




"""
三种策略选择指南
    场景	                    推荐策略	                            原因
    不确定模型是否支持原生输出	AutoStrategy（直接传 Pydantic）	    自动选择最优策略
    需要兼容各种模型	        ToolStrategy	                    所有支持 function calling 的模型都可用
    追求极致性能	            ProviderStrategy	                跳过工具调用环节，速度更快
    需要错误重试	            ToolStrategy(handle_errors=True)	只有 ToolStrategy 支持 handle_errors

大多数情况下，直接传入 Pydantic 模型（即使用 AutoStrategy）就够了。只有在需要错误重试或明确控制策略行为时，
才需要显式指定 ToolStrategy 或 ProviderStrategy。
"""