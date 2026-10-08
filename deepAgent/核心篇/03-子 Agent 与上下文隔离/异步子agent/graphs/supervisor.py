from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
import os

from deepagents import AsyncSubAgent, create_deep_agent

graph = create_deep_agent(
    model=ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL_NAME"),
        api_key=os.getenv("DEEPSEEK_API"),
        base_url="https://api.deepseek.com",
    ),
    system_prompt=(
        "You are a supervisor agent for an async-subagent demo. "
        "When the user asks for a long-running research task, you must delegate "
        "to the async subagent named researcher immediately. "
        "After calling start_async_task, return the task_id to the user and stop. "
        "Do not call check_async_task unless the user explicitly asks for progress. "
        "If the user asks to revise the background task, call update_async_task."
    ),
    subagents=[
        AsyncSubAgent(
            name="researcher",
            description=(
                "Use for any long-running background research or async demo task. "
                "This agent intentionally sleeps before returning so the async "
                "behavior is easy to observe."
            ),
            graph_id="researcher",
        )
    ]

)


"""
最佳实践
    1.本地开发要把worker pool调大
        例如： langgraph dev --n-jobs-per-worker 10
    2.描述要具体，行为导向
        # ✅ 好
        AsyncSubAgent(
            name="researcher",
            description="深度网络调研，需要多次搜索 + 信息综合时使用",
            graph_id="researcher",
        )
    
        # ❌ 差
        AsyncSubAgent(
            name="helper",
            description="帮你处理事情",
            graph_id="helper",
        )
    3. 用 Thread ID 串联追踪
    
"""