from dotenv import load_dotenv
load_dotenv()


# 1.最简单的用法 —— 传入 Pydantic 模型
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage


# 定义期望的输出结构
class CourseInfo(BaseModel):
    """菜鸟教程 RUNOOB 课程提取结果"""
    course_name: str = Field(description="课程名称")
    difficulty: str = Field(description="难度：入门/进阶/高级")
    estimated_hours: int = Field(description="预计学习时长（小时）")
    is_free: bool = Field(description="是否免费")


model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature=0,
    model_kwargs={
        "extra_body": {
            "thinking": {"type": "disabled"}   # 关闭思考模式（deepseek的思考模式与结构化输出不兼容）
        }
    }
)
agent = create_agent(
    model=model,
    response_format=ToolStrategy(schema=CourseInfo),  # 显式用 ToolStrategy，绕过 ProviderStrategy（DeepSeek 不支持 json_schema）
    system_prompt="你是菜鸟教程 RUNOOB 的课程助手，从用户描述中提取课程信息。",
)

# 用户输入一段非结构化的描述
result = agent.invoke({
    "messages": [HumanMessage(
        content="我最近在学习 Python3 基础教程，是入门级别的，"
                "大概要学 20 个小时，而且是完全免费的"
    )]
})

# 从 structured_response 获取结构化结果
if "structured_response" in result:
    course = result["structured_response"]
    print(f"课程名: {course.course_name}")
    print(f"难度: {course.difficulty}")
    print(f"预计时长: {course.estimated_hours} 小时")
    print(f"免费: {'是' if course.is_free else '否'}")
    print(f"对象类型: {type(course)}")

"""
    返回的 structured_response 是 Pydantic 模型实例，而不是普通字典。这意味着你可以使用 .course_name 等属性访问，
    IDE 也能提供自动补全。
"""



# 2.与工具共存的Structured Output
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain.agents.structured_output import ToolStrategy


@tool
def search_course(keyword: str) -> str:
    """在菜鸟教程 RUNOOB 搜索课程信息"""
    courses = {
        "python": "Python3 基础教程 | 入门 | 免费 | 30章 | 约20小时",
        "java": "Java 基础教程 | 入门 | 免费 | 35章 | 约25小时",
        "数据分析": "Python 数据分析 | 进阶 | 会员 | 25章 | 约30小时",
    }
    return courses.get(keyword.lower(), f"未找到 '{keyword}' 相关课程")


class CourseRecommendation(BaseModel):
    """课程推荐结果"""
    course_name: str = Field(description="推荐课程名称")
    reason: str = Field(description="推荐理由")
    difficulty: str = Field(description="难度：入门/进阶/高级")


model = init_chat_model(
    "deepseek:deepseek-v4-flash",
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
    response_format=ToolStrategy(CourseRecommendation),
    system_prompt="你是菜鸟教程 RUNOOB 的课程顾问。先查询课程再给出推荐。",
)

result = agent.invoke({
    "messages": [HumanMessage(content="我想学 Python，有什么推荐？")]
})

rec = result["structured_response"]
print(f"推荐课程: {rec.course_name}")
print(f"推荐理由: {rec.reason}")
print(f"难度: {rec.difficulty}")

# 查看完整过程
print("\n=== 执行过程 ===")
for msg in result["messages"]:
    if msg.type == "tool":
        print(f"  调用 {msg.name}: {msg.content}")




# 4.复杂嵌套结构
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.agents.structured_output import ToolStrategy


class Topic(BaseModel):
    """知识点"""
    name: str = Field(description="知识点名称")
    order: int = Field(description="学习顺序，从 1 开始")
    minutes: int = Field(description="建议学习分钟数")


class LearningPlan(BaseModel):
    """学习计划"""
    goal: str = Field(description="学习目标概述")
    level: Literal["入门", "进阶", "高级"] = Field(description="难度级别")
    total_hours: float = Field(description="总时长（小时）")
    topics: list[Topic] = Field(description="知识点列表")


model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature=0,
    model_kwargs={
        "extra_body": {
            "thinking": {"type": "disabled"}  # 关闭思考模式（deepseek的思考模式与结构化输出不兼容）
        }
    }
)
agent = create_agent(
    model=model,
    response_format=ToolStrategy(LearningPlan),
    system_prompt="你是菜鸟教程 RUNOOB 的学习规划师。",
)

result = agent.invoke({
    "messages": [HumanMessage(
        content="帮我制定一个 Python 入门学习计划，总时长控制在 10 小时以内"
    )]
})

plan = result["structured_response"]
print(f"目标: {plan.goal}")
print(f"难度: {plan.level}")
print(f"总时长: {plan.total_hours} 小时")
print(f"\n知识点列表 ({len(plan.topics)} 个):")
for topic in plan.topics:
    print(f"  {topic.order}. {topic.name} ({topic.minutes}分钟)")