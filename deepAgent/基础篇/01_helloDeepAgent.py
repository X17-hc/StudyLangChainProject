import os

from deepagents import create_deep_agent
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI


api_key = os.getenv("DEEPSEEK_API")
model_name = os.getenv("DEEPSEEK_MODEL_NAME")

if not api_key or not model_name:
    raise RuntimeError("请在项目根目录 .env 中配置 DEEPSEEK_API 和 DEEPSEEK_MODEL_NAME")

# DeepSeek 是 OpenAI 兼容接口，必须显式传 api_key 和 base_url
model = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

def get_weather(city: str) -> str:
    """ Get weather data from city """
    return f"It's always sunny in {city}!"

agent = create_deep_agent(
    model=model,
    tools=[get_weather],
    system_prompt="you are a helpful assistant",
)

result =  agent.invoke(
    {"messages": [{"role":"user", "content":"佛山今天天气怎么样？"}]}
)

print("\n")
# 输出也是一个字典，result["messages"] 包含了完整的对话历史，最后一条消息就是 Agent 的最终回复
print(result["messages"][-1].content)