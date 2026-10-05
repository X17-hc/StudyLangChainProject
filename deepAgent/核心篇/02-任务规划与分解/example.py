import os
from typing import Literal

from langchain.agents.middleware import TodoListMiddleware
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from deepagents import create_deep_agent
from pathlib import Path
from dotenv import load_dotenv

# 固定读项目根目录 .env，避免从子目录运行时读不到
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

# 1. 配置模型（.env 里是 DeepSeek 官方密钥，不能走硅基流动）
DEEPSEEK_API = os.getenv("DEEPSEEK_API")
DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME")

if not DEEPSEEK_API or not DEEPSEEK_MODEL_NAME:
    raise RuntimeError("请在项目根目录 .env 中配置 DEEPSEEK_API 和 DEEPSEEK_MODEL_NAME")

model = ChatOpenAI(
    model=DEEPSEEK_MODEL_NAME,
    api_key=DEEPSEEK_API,
    base_url="https://api.deepseek.com",
)

# 搜索工具
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

def internet_search(query:str, max_results:int=5) -> dict:
    """ 搜索互联网获取最新信息。 """
    return tavily_client.search(query, max_results=max_results)


# 创建Agent,并显式启用 write_todos
agent = create_deep_agent(
    model=model,
    tools=[internet_search],
    middleware=[TodoListMiddleware()],
    system_prompt="""你是一位专业的技术研究员。
面对复杂研究任务时，你会：
1. 先用 write_todos 制定研究计划
2. 逐步执行每个步骤，及时更新进度
3. 将搜索结果写入文件系统整理
4. 最终输出完整的研究报告
""",
)

# 发起一个需要规划的复杂任务
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "请调研 Agent 开发领域的三大 Harness 框架（Deep Agents、Claude Agent SDK、Codex SDK），对比它们的核心能力差异，写一份简要分析报告。"
    }]
})

print(result["messages"][-1].content)