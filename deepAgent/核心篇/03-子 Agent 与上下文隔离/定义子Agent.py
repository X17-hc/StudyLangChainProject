import os

from deepagents import create_deep_agent
from tavily import TavilyClient

# 搜索工具
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

def internet_search(query:str, max_results:int=5) -> dict:
    """ 搜索互联网获取最新信息。 """
    return tavily_client.search(query, max_results=max_results)

# 定义一个研究型子agent
research_subagent = {
    "name": "research_subagent",  # 必填：唯一标识符
    "description": "深入研究特定主题，搜索多个信息源并整理成摘要",  # 必填：主 Agent 靠它决定何时委派
    "system_prompt": """你是一位专业的研究员。你的任务是：
1. 把研究问题拆解为多个搜索查询
2. 用 internet_search 搜索相关信息
3. 整理发现，写成简洁摘要
4. 列出关键发现和信息来源

注意：返回结果控制在 500 字以内，只返回核心发现。""",  # 必填：子 Agent 自己的指令
    "tools": [internet_search], # 可选，默认继承；显式指定后完全替换（不合并）
    "skills": ["/skills/research/"],     # 可选，不继承主 Agent；指定后独立运行
}

agent = create_deep_agent(
    model= os.getenv("DEEP_AGENT_MODEL"),
    api_key= os.getenv("DEEPSEEK_API"),
    base_url="https://api.deepseek.com",
)