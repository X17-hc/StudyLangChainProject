# 加载api key
from dotenv import load_dotenv
load_dotenv()

# 1.@tool基本语法
from langchain.tools import tool

# 最简单的工具：一个普通函数 + @tool 装饰器
@tool
def hello_tool(name: str) -> str:
    """向指定的人打招呼。

    Args:
        name: 要打招呼的人的名字
    """
    return f"你好，{name}！欢迎来到菜鸟教程 RUNOOB。"

# 工具也是普通的 Python 函数，可以直接调用
result = hello_tool.invoke({"name": "小明"})
print(result)

# 工具包含自动生成的描述信息
print(f"\n工具名称: {hello_tool.name}")
print(f"工具描述: {hello_tool.description}")




# 2.使用不同参数类型。包括int、float、bool和枚举值
from langchain.tools import tool
from typing import Literal


@tool
def search_courses(
    keyword: str,
    level: Literal["入门", "进阶", "高级"],
    max_results: int = 5,
    free_only: bool = True,
) -> str:
    """在菜鸟教程 RUNOOB 中搜索课程。

    Args:
        keyword: 搜索关键词
        level: 课程难度级别
        max_results: 最多返回多少条结果
        free_only: 是否只显示免费课程
    """
    # 模拟搜索结果
    courses = {
        ("python", "入门"): "Python3 基础教程（免费）",
        ("python", "进阶"): "Python 数据分析实战（免费）",
        ("python", "高级"): "Python 机器学习进阶（会员）",
        ("html", "入门"): "HTML 基础教程（免费）",
    }
    key = (keyword.lower(), level)
    course = courses.get(key, f"未找到 {level} 级别的 {keyword} 课程")
    if free_only and "会员" in course:
        return f"搜索结果：{course} —— 该课程需要会员，已为你过滤"
    return f"搜索结果：{course}"


# 测试不同参数的调用
print(search_courses.invoke({
    "keyword": "python",
    "level": "入门",
}))
# 输出：搜索结果：Python3 基础教程（免费）

print(search_courses.invoke({
    "keyword": "python",
    "level": "高级",
    "free_only": False,
}))
# 输出：搜索结果：Python 机器学习进阶（会员）




# 3.注册工具到Agent。将定义好的工具传给create_agent() 的 tools 参数，Agent 就能使用了。
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage


@tool
def search_courses(keyword: str) -> str:
    """在菜鸟教程 RUNOOB 中搜索课程。传入关键词返回相关课程列表。

    Args:
        keyword: 搜索关键词，如 python、html、java
    """
    courses = {
        "python": "Python3 基础教程、Python 数据分析、Python 爬虫入门",
        "html": "HTML 基础教程、HTML5 新特性、HTML 表单实战",
        "java": "Java 基础教程、Java 面向对象、Java Spring 框架",
    }
    return courses.get(keyword.lower(), f"未找到与 {keyword} 相关的课程")


@tool
def get_course_detail(course_name: str) -> str:
    """获取指定课程的详细信息，包括章节数和学习时长。

    Args:
        course_name: 课程名称，如 "Python3 基础教程"
    """
    details = {
        "python3 基础教程": "共 30 章，预计学习时长 20 小时，适合零基础入门",
        "html 基础教程": "共 25 章，预计学习时长 15 小时，适合零基础入门",
    }
    return details.get(
        course_name.lower(),
        f"{course_name} 详情：适合初学者，内容丰富，附带实战案例"
    )
# 创建Agent
model = init_chat_model("deepseek:deepseek-v4-flash",temperature = 0)
agent = create_agent(
    model = model,
    tools=[
        search_courses, get_course_detail,
    ]
)

# 运行Agent
def ask(question: str) -> str:
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    print(f"用户: {question}")
    print(f"顾问: {result['messages'][-1].content}")
    print("-" * 60)

ask("我想学 Python，有什么课程推荐？")
ask("Python3 基础教程学完需要多久？")



#########
# 多个工具协作。一个Agent可以注册多个工具，模型自动判断何时使用哪个工具
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage


@tool
def get_stock_price(symbol: str) -> str:
    """查询股票当前价格。

    Args:
        symbol: 股票代码，如 AAPL、GOOGL、TSLA
    """
    # 模拟股票数据
    prices = {"AAPL": 185.50, "GOOGL": 142.30, "TSLA": 245.80}
    price = prices.get(symbol.upper())
    if price is None:
        return f"未找到股票代码 {symbol}"
    return f"{symbol.upper()} 当前价格：${price}"


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """货币换算。

    Args:
        amount: 金额
        from_currency: 原货币代码，如 USD、CNY
        to_currency: 目标货币代码，如 CNY、USD
    """
    # 模拟汇率
    rates = {
        ("USD", "CNY"): 7.25,
        ("CNY", "USD"): 0.138,
    }
    rate = rates.get((from_currency.upper(), to_currency.upper()))
    if rate is None:
        return f"不支持 {from_currency} → {to_currency} 的换算"
    result = round(amount * rate, 2)
    return f"{amount} {from_currency} = {result} {to_currency}"


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[get_stock_price, convert_currency],
    system_prompt="你是一个金融助手，帮助用户查询股价和换算货币。",
)

# Agent 会自动决定调用哪个工具
result = agent.invoke({
    "messages": [HumanMessage(content="苹果股价是多少？换算成人民币是多少？")]
})

for msg in result["messages"]:
    if msg.type == "tool":
        print(f"[调用工具 {msg.name}] {msg.content}")
    elif msg.type == "ai" and msg.content:
        print(f"\n最终回答: {msg.content}")




# 3.工具的可选参数与默认值
from langchain.tools import tool


@tool
def recommend_tutorial(
    language: str,
    level: str = "入门",
    count: int = 3,
) -> str:
    """根据编程语言和难度推荐菜鸟教程 RUNOOB 的课程。

    Args:
        language: 编程语言，如 Python、Java、HTML
        level: 难度级别，可选 "入门"、"进阶"、"高级"。默认 "入门"
        count: 推荐的课程数量，默认 3
    """
    tutorials = {
        "python": ["Python3 基础", "Python 面向对象", "Python 爬虫", "Python 数据分析"],
        "java": ["Java 基础", "Java 面向对象", "Java 集合框架", "Java 多线程"],
    }
    all_tutorials = tutorials.get(language.lower(), [f"{language} 基础教程"])

    selected = all_tutorials[:count]
    return f"推荐 {language} {level} 级别课程：{'、'.join(selected)}"


# 调用时只需要传必填参数
print(recommend_tutorial.invoke({"language": "Python"}))
# 输出：推荐 Python 入门 级别课程：Python3 基础、Python 面向对象、Python 爬虫

# 也可以覆盖默认参数
print(recommend_tutorial.invoke({
    "language": "Java",
    "level": "进阶",
    "count": 2,
}))
# 输出：推荐 Java 进阶 级别课程：Java 基础、Java 面向对象




# 4.工具的 args_schema——自定义参数校验
from pydantic import BaseModel, Field
from langchain.tools import tool


# 定义参数模型（提供更精细的参数控制）
class CourseSearchInput(BaseModel):
    """搜索课程参数"""
    keyword: str = Field(
        description="搜索关键词，支持模糊匹配",
        min_length=1,            # 最少 1 个字符
        max_length=50,           # 最多 50 个字符
    )
    category: str = Field(
        default="all",
        description="课程类别：all（全部）、frontend（前端）、backend（后端）、data（数据科学）",
        pattern=r"^(all|frontend|backend|data)$",  # 限定可选值
    )
    page: int = Field(
        default=1,
        description="页码，从 1 开始",
        ge=1,                    # 大于等于 1
        le=100,                  # 小于等于 100
    )


@tool(args_schema=CourseSearchInput)
def search_course(keyword: str, category: str = "all", page: int = 1) -> str:
    """在菜鸟教程 RUNOOB 中搜索课程"""
    return f"搜索 '{keyword}' (分类: {category}, 第 {page} 页)：共找到 15 条结果"


# 有效调用
print(search_course.invoke({"keyword": "Python", "category": "backend", "page": 1}))

# 无效调用（category 不在允许的值内）
try:
    search_course.invoke({"keyword": "Python", "category": "invalid"})
except Exception as e:
    print(f"参数校验失败: {e}")



"""
    方式	                代码量	        适用场景	                示例
    @tool 装饰器	            最少	        简单到中等复杂度的工具	    大多数场景
    @tool + args_schema	    中等	        需要精细参数校验的工具	    API 封装、数据库操作
    Pydantic 类作为工具	    较多	        复杂业务逻辑的工具	    内部包含状态的工具
    字典格式                	最少（但不推荐）	描述远程/内置工具	        MCP 工具、服务端工具
"""