import os
from unittest import result

from create_agent.templates.default.app.utils.prompts import system_prompt_template
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

def calculate(expression: str) -> float:
    """Evaluate a math expression and return the result.

        Args:
            expression: A math expression, e.g. "1 + 2 * 3".
    """
    # 仅做演示，实际项目应使用安全的解析库而非 eval
    return eval(expression)

def convert_currency(amount: float, from_currency: str, to_currency:str = "CNY") -> dict:
    """Convert an amount from one currency to another.

    Args:
        amount: The amount to convert.
        from_currency: The source currency code, e.g. "USD".
        to_currency: The target currency code, defaults to "CNY".
    """
    # 使用固定汇率做演示，真是场景下接入API
    rates = {"USD": 7.2, "CNY": 1.0, "EUR": 7.8}
    cny = amount * rates[from_currency] / rates[to_currency]

    return {"amount": round(cny, 2), "currency": to_currency }

agent = create_deep_agent(
    model=model,
    tools=[
        calculate,
        convert_currency,
    ],
    system_prompt="你是一个计算助手，能帮用户做数学运算和货币换算。",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我把 100 美元换算成人民币，再用它乘以 1.08 的通胀系数。"}]}
)

print(result["messages"][-1].content)
