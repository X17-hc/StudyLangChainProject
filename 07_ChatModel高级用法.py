# 加载api key
from dotenv import load_dotenv
load_dotenv()

#1.bind_tools———让模型自动可以使用哪些工具。调用 bind_tools() 后，模型能在回复中返回 tool_call，告诉程序"我需要调用这个工具"。
from langchain.chat_models import init_chat_model

model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)

# 用字典描述工具（OpenAI function calling 格式）
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 杭州、北京"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# bind_tools() 将工具绑定到模型
# 模型现在"知道"有 get_weather 这个工具可用
model_with_tools = model.bind_tools(tools)

# 问一个需要工具的问题
response = model_with_tools.invoke("杭州今天天气怎么样？")

# 检查模型是否请求调用工具
if response.tool_calls:
    print("模型请求调用以下工具：")
    for tc in response.tool_calls:
        print(f"  工具名: {tc['name']}")
        print(f"  参数: {tc['args']}")
        print(f"  调用ID: {tc['id']}")
else:
    print(f"模型直接回复: {response.content}")




# 2.用 Pydantic 模型描述工具。对于复杂工具，使用Pydantic模型定义参数结构化比手写字典更清晰
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

# 用 Pydantic 定义工具的参数结构
class WeatherInput(BaseModel):
    """查询指定城市的天气情况"""
    city: str = Field(description="城市名称，如 杭州、北京")
    unit: str = Field(
        default="celsius",
        description="温度单位，celsius（摄氏度）或 fahrenheit（华氏度）"
    )

class CalculatorInput(BaseModel):
    """执行数学计算"""
    expression: str = Field(
        description="要计算的数学表达式，如 '(3 + 5) * 2'"
    )

model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)

# 传入 Pydantic 模型，LangChain 自动转换为工具描述
model_with_tools = model.bind_tools([WeatherInput, CalculatorInput])

# 测试复杂场景
response = model_with_tools.invoke(
    "北京今天多少度？顺便帮我算一下 123 * 456"
)

print(f"模型请求了 {len(response.tool_calls)} 个工具调用：")
for tc in response.tool_calls:
    print(f"  {tc['name']}({tc['args']})")





# 3.with_structured_output()——让模型返回结构化数据。with_structured_output() 是比 tool_calling 更直接的方式。它让模型按你指定的格式（Schema）返回数据，而不是返回 tool_call。
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

class PersonInfo(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    occupation: str = Field(description="职业")
    skills: list[str] = Field(description="技能列表")

# 关键：通过 extra_body 禁用 thinking
model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature=0,
    model_kwargs={
        "extra_body": {
            "thinking": {"type": "disabled"}   # 关闭思考模式（deepseek的思考模式与结构化输出不兼容）
        }
    }
)

structured_model = model.with_structured_output(PersonInfo)

text = "张三今年28岁，是一名全栈工程师，精通 Python、React 和 Docker"
result = structured_model.invoke(text)

print(f"姓名: {result.name}")
print(f"年龄: {result.age}")
print(f"职业: {result.occupation}")
print(f"技能: {', '.join(result.skills)}")
print(f"类型: {type(result)}")




# 4.with_structured_output() vs bind_tools()
"""
    用途	    从文本中提取结构化数据	            让模型知道可用的工具列表
    返回格式	    直接返回 Pydantic 对象	        返回 AIMessage，其中包含 tool_calls
    适用场景	    信息提取、数据解析	            Agent 工具调用、需要外部执行的场景
    模型支持	    需模型支持原生structured output	所有支持 function calling 的模型
"""

# 5.嵌套结构化输出
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model


class Ingredient(BaseModel):
    """食材信息"""
    name: str = Field(description="食材名称")
    amount: str = Field(description="用量，如 '200g'、'2个'")


class CookingStep(BaseModel):
    """烹饪步骤"""
    step_number: int = Field(description="步骤编号")
    description: str = Field(description="步骤描述")
    duration_minutes: int = Field(description="此步骤需要的时间（分钟）")


class Recipe(BaseModel):
    """菜谱"""
    dish_name: str = Field(description="菜名")
    difficulty: str = Field(description="难度：简单、中等、困难")
    ingredients: list[Ingredient] = Field(description="食材列表")
    steps: list[CookingStep] = Field(description="烹饪步骤")


model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature=0,
    model_kwargs={
        "extra_body": {
            "thinking": {"type": "disabled"}   # 关闭思考模式
        }
    }
)

structured_model = model.with_structured_output(Recipe)

# 输入一个菜谱描述
recipe_text = """
今天来教大家做一道经典的番茄炒蛋，这道菜难度适中。
需要准备：番茄 2 个、鸡蛋 3 个、葱花少许、盐适量、糖少许。
步骤：
1. 先把番茄切块，鸡蛋打散，大概需要 5 分钟
2. 热锅放油，先把鸡蛋炒熟盛出，大概 3 分钟
3. 锅中再放油，炒番茄至出汁，加盐和糖，大概 5 分钟
4. 倒入炒好的鸡蛋，翻炒均匀，撒上葱花，大概 2 分钟
"""

result = structured_model.invoke(recipe_text)

print(f"菜名: {result.dish_name}")
print(f"难度: {result.difficulty}")
print(f"食材 ({len(result.ingredients)} 种):")
for ing in result.ingredients:
    print(f"  - {ing.name}: {ing.amount}")
print(f"步骤 ({len(result.steps)} 步):")
for step in result.steps:
    print(f"  {step.step_number}. {step.description} ({step.duration_minutes}分钟)")




# 6.JSON Schema模式
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature=0,
    model_kwargs={
        "extra_body": {
            "thinking": {"type": "disabled"}   # 关闭思考模式
        }
    }
)

# 直接传入 JSON Schema
json_schema = {
    "title": "SentimentAnalysis",
    "description": "情感分析结果",
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
            "description": "情感倾向"
        },
        "confidence": {
            "type": "number",
            "description": "置信度，0~1"
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "关键情感词"
        }
    },
    "required": ["sentiment", "confidence"]
}

structured_model = model.with_structured_output(json_schema)
result = structured_model.invoke("菜鸟教程 RUNOOB 真的太棒了，强烈推荐给所有编程新手！")

print(f"情感: {result['sentiment']}")
print(f"置信度: {result['confidence']}")
print(f"关键词: {result['keywords']}")


# 7.ConfigurableModel 上的 bind_tools 和 with_structured_output。可配置模型也支持这两个方法，用法完全相同。
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

# 创建可配置模型
configurable_model = init_chat_model(
    "deepseek-v4-flash",
    configurable_fields=("model", "model_provider"),
    temperature=0,
)

# 链式调用：先绑定工具，再调用
configurable_with_tools = configurable_model.bind_tools([...])

# 运行时可以用不同模型来执行
result = configurable_with_tools.invoke(
    "查询天气",
    config={"configurable": {"model": "claude-sonnet-4-5"}}
)