from typing import List
from unittest import result

from dotenv import load_dotenv
from langchain.agents.structured_output import ToolStrategy

load_dotenv()


"""
系统设计
    三个工具：天气查询、日程管理、邮件发送
    结构化输出：日程汇总格式化为 Markdown
    流式输出：实时显示 AI 思考和处理过程
    对话记忆：记住用户偏好和上下文
"""

from datetime import datetime
from pydantic import BaseModel, Field

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage



# ========== 1. 模拟数据 ==========

calendar_events = [
    {"id": 1, "title": "Python 学习", "date": "2024-03-25",
     "time": "14:00", "duration": "2小时"},
    {"id": 2, "title": "团队周会", "date": "2024-03-25",
     "time": "10:00", "duration": "1小时"},
    {"id": 3, "title": "代码审查", "date": "2024-03-26",
     "time": "15:00", "duration": "1.5小时"},
]

weather_db = {
    "杭州": {"condition": "晴", "temp": 25, "humidity": 60},
    "北京": {"condition": "多云", "temp": 18, "humidity": 45},
    "上海": {"condition": "小雨", "temp": 22, "humidity": 80},
}


# ========== 2. 定义工具 ==========

@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气。

    Args:
        city: 城市名称，如 杭州、北京、上海
    """
    data = weather_db.get(city)
    if not data:
        return f"暂不支持查询 {city} 的天气。支持的城市：{', '.join(weather_db.keys())}"

    return f"{city}天气：{data['condition']}，"f"温度 {data['temp']}°C，湿度 {data['humidity']}%"

@tool
def query_schedule(date: str = None) -> str:
    """查询指定日期的日程安排。不指定日期则查询今天的日程。

    Args:
        date: 日期，格式 YYYY-MM-DD，如 2024-03-25。不传则查询今天
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    events = [e for e in calendar_events if e["date"] == date]
    if not events:
        return f"{date} 没有日程安排"


    events.sort(key=lambda e: e["time"])
    lines = [f"&#x1f4c5; {date} 日程安排："]
    for e in events:
        lines.append(f"  - {e['time']} {e['title']}（{e['duration']}）")
    return "\n".join(lines)

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件（模拟）。

    Args:
        to: 收件人邮箱
        subject: 邮件主题
        body: 邮件正文
    """
    # 模拟发送
    email_id = f"MSG-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return f"邮件已发送！收件人：{to}，主题：{subject}，邮件ID：{email_id}"



# ========== 3. 结构化输出模型 ==========

class DailySummary(BaseModel):
    """ 每日摘要 """
    date: str = Field(description="日期")
    weather_summary: str = Field(description="天气概述")
    event_count: int = Field(description="日程数量")
    key_events: list[str] = Field(description="重要日程列表")
    suggestion: str = Field(description="今日建议")



# ========== 4. 定义 Middleware ==========

@before_model
def inject_date_context(state, runtime):
    """ 自动注入当前日期信息 """
    now = datetime.now()
    date_hint = HumanMessage(
        content=f"[系统提示] 当前日期是 {now.strftime('%Y年%m月%d日')}，"
                f"星期{['一', '二', '三', '四', '五', '六', '日'][now.weekday()]}。"
                f"如果用户没有指定日期，默认查询今天。"
    )
    # 将日期提示注入到消息中
    msgs = list(state.get("messages", []))
    msgs.insert(-1, date_hint)  # 在最后一条用户消息前插入
    return {"messages": msgs}



# ========== 5. 创建 Agent ==========

model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature = 0.3,
    model_kwargs={
        "extra_body": {
            "thinking": {"type": "disabled"}   # 关闭思考模式（deepseek的思考模式与结构化输出不兼容）
        }
    }
)

agent = create_agent(
    model=model,
    tools=[get_weather, query_schedule, send_email],
    middleware=[inject_date_context],
    response_format=ToolStrategy(DailySummary),
    system_prompt="""
        你是个人助手"小助"。你可以查天气、管理日程、发送邮件。

        ## 工作方式
        1. 当用户问"今天怎么样"或类似问题时：
           - 先查询今天的天气（get_weather）
           - 再查询今天的日程（query_schedule）
           - 然后生成每日摘要

        2. 当用户要求发邮件时，使用 send_email 工具

        3. 当用户只问天气或只问日程时，只调用对应的工具

        ## 风格
        - 语气亲切自然
        - 优先使用工具获取实时数据，不要编造
    """
)




# ========== 6. 交互函数 ==========

def chat(message: str):
    """ 与助手对话 """
    result = agent.invoke({
        "messages": [HumanMessage(content=message)],
    })

    # 获取结构化输出
    if "structured_response" in result:
        summary = result["structured_response"]
        print("\n=== 结构化摘要 ===")
        print(f"日期: {summary.date}")
        print(f"天气: {summary.weather_summary}")
        print(f"日程数: {summary.event_count}")
        print(f"重要事项: {', '.join(summary.key_events)}")
        print(f"建议: {summary.suggestion}")

    print(f"\n助手：{result['messages'][-1].content}")
    return result




# ========== 7. 测试 ==========

if __name__ == "__main__":
    chat("杭州今天天气怎么样？看看我的日程，然后给我一个今日总结")
    chat("帮我发一封邮件给 team@runoob.com，主题是'今日总结'，内容是今天日程已确认")



"""
输出结果

=== 结构化摘要 ===
日期: 2026-08-01
天气: 杭州今天晴，温度25°C，湿度60%
日程数: 0
重要事项: 
建议: 今天没有日程安排，天气晴朗舒适，适合出门走走或安排一些休闲活动。

助手：Returning structured response: date='2026-08-01' weather_summary='杭州今天晴，温度25°C，湿度60%' event_count=0 key_events=[] suggestion='今天没有日程安排，天气晴朗舒适，适合出门走走或安排一些休闲活动。'

=== 结构化摘要 ===
日期: 2026-08-01
天气: 杭州今天天气晴朗，温度25°C，湿度60%
日程数: 0
重要事项: 
建议: 今天没有日程安排，天气晴好，适合安排一些户外活动或休息放松。

助手：Returning structured response: date='2026-08-01' weather_summary='杭州今天天气晴朗，温度25°C，湿度60%' event_count=0 key_events=[] suggestion='今天没有日程安排，天气晴好，适合安排一些户外活动或休息放松。'

"""


"""
    特性	            实现方式
    多工具协作	    天气+日程+邮件，Agent 自动选择调用顺序
    结构化输出	    DailySummary Pydantic 模型，程序可直接使用
    日期注入	        Middleware 自动注入当前日期上下文
    自然语言交互	    用户用自然语言描述需求，Agent 自主规划
"""