# 加载api key
from dotenv import load_dotenv
load_dotenv()
#
# 1.temperature——控制创造性与确定性
from langchain.chat_models import init_chat_model

# 同一问题，不同 temperature 的对比
question = "用一句话介绍菜鸟教程 RUNOOB"

# temperature=0：输出非常确定，几乎每次结果一样
model_low = init_chat_model("deepseek:deepseek-v4-flash", temperature=0,model_kwargs={"seed": 42})
resp1 = model_low.invoke(question)
resp2 = model_low.invoke(question)
print(f"temperature=0 第1次: {resp1.content}")
print(f"temperature=0 第2次: {resp2.content}")
print(f"两次结果相同: {resp1.content == resp2.content}")
print()

# temperature=1.5：输出多样化，每次可能不同
model_high = init_chat_model("deepseek:deepseek-v4-flash", temperature=1.5,model_kwargs={"seed": 42})
resp1 = model_high.invoke(question)
resp2 = model_high.invoke(question)
print(f"temperature=1.5 第1次: {resp1.content}")
print(f"temperature=1.5 第2次: {resp2.content}")

"""
temperature数值：
    0 ~ 0.3	    输出稳定、确定，每次结果几乎一致	数据提取、分类、代码生成、翻译
    0.5 ~ 0.7	适度的创造性，输出自然但不偏离主题	日常对话、内容总结
    0.8 ~ 1.2	输出多样化，有较多发挥空间	        创意写作、头脑风暴
    1.3 ~ 2.0	输出非常随机，可能出现意外内容	    探索性生成（不太推荐用于生产）
"""



# 2.max_tokens——控制输出长度与成本
from langchain.chat_models import init_chat_model

model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)

# 限制 30 tokens —— 使用 bind() 动态设置
response_short = model.bind(max_tokens=300).invoke(
    "详细介绍一下菜鸟教程 RUNOOB 平台"
)
print(f"限制 300 tokens ({len(response_short.content)} 字符):")
print(response_short.content)
print()

# 限制 200 tokens —— 使用 bind()
response_long = model.bind(max_tokens=1000).invoke(
    "详细介绍一下菜鸟教程 RUNOOB 平台"
)
print(f"限制 1000 tokens ({len(response_long.content)} 字符):")
print(response_long.content)



# 3.timeout 与 max_retries——网络可靠性
from langchain.chat_models import init_chat_model

# 生产环境推荐配置
model = init_chat_model(
    "deepseek:deepseek-v4-flash",

    # 单次请求最多等待 30 秒
    timeout=30,

    # 失败后最多重试 3 次（总共 4 次请求机会）
    max_retries=3,
)

# 模拟正常调用
try:
    response = model.invoke("菜鸟教程 RUNOOB 是什么？")
    print(f"调用成功: {response.content[:50]}...")
except Exception as e:
    print(f"调用失败: {e}")

"""
    timeout	        单次请求的最大等待时间（秒）。None 表示不限制	30~60（太短容易超时，太长用户体验差）
    max_retries	    失败后的重试次数。0 表示不重试	                2~3（足够处理偶发网络问题）
"""



# 4.base_url ———— 自定义API地址
from langchain.chat_models import init_chat_model

# 场景 1：通过代理访问 OpenAI
model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    base_url="https://your-proxy-domain.com/v1",  # 代理地址
)

# 场景 2：使用兼容 OpenAI 接口的第三方服务
# 很多国产模型提供了 OpenAI 兼容接口
model = init_chat_model(
    "deepseek:deepseek-v4-flash",           # provider 写 openai
    base_url="https://api.third-party.com/v1",  # 但实际指向第三方
    api_key="your-third-party-key",  # 第三方 API Key
)

# 场景 3：连接本地模型（如 vLLM、Ollama）
model = init_chat_model(
    "openai:qwen2.5",               # 本地模型名
    base_url="http://localhost:8000/v1",  # 本地服务地址
    api_key="not-needed",           # 本地通常不需要 Key
)



# 5.其他常用参数
# 5.1 top_p —— 核采样
from langchain.chat_models import init_chat_model

# top_p 是另一种控制随机性的方式
# 模型只会从累积概率达到 top_p 的词中采样
# top_p=0.1 表示只从最高概率的 10% 的词中选择
model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    top_p=0.9,       # 只考虑累积概率前 90% 的词
)

response = model.invoke("介绍菜鸟教程 RUNOOB")
print(response.content[:100])

# 5.2 stop——停止序列
from langchain.chat_models import init_chat_model

model = init_chat_model("deepseek:deepseek-v4-flash")

# stop 参数指定停止序列，模型遇到这些词时会立即停止生成
response = model.invoke(
    "列出五个编程学习网站，每个一行",
    stop=["\n"]  # 遇到换行就停止，只返回第一个
)
print(f"限制 stop=['\\n']: {response.content}")


# 5.3 seed——可重复性（部分模型支持）
from langchain.chat_models import init_chat_model

# 某些模型支持 seed 参数，用于获得确定性的输出
# 相同的 seed + 相同的输入 = 相同的输出
model = init_chat_model("deepseek:deepseek-v4-flash", seed=42, temperature=0)
resp1 = model.invoke("介绍菜鸟教程")
resp2 = model.invoke("介绍菜鸟教程")
print(f"seed=42, 结果相同: {resp1.content == resp2.content}")


"""
小结：
    temperature	    float	    因模型而异	    任务需要稳定性时设为 0~0.3，需要创造性时设为 0.7~1.0
    max_tokens	    int     	模型上限	        输出长度需要控制时
    timeout	        int/float	None	        生产环境建议始终设置
    max_retries	    int     	因模型而异	    网络不稳定时建议 2~3
    base_url	    str     	官方地址	        使用代理、中转或本地服务时
    top_p	        float	    1.0	            需要核采样控制时（替代 temperature）
    stop	        list[str]	无	            需要精确控制输出结尾时
"""