from langchain.chat_models import init_chat_model

# 完整语法
# model = init_chat_model(
#     model,                    # str | None：模型名称（provider:model 格式）
#     *,
#     model_provider=None,      # str | None：单独的模型提供商
#     configurable_fields=None, # None | "any" | list[str]：可运行时修改的字段
#     config_prefix=None,       # str | None：配置键前缀
#     **kwargs,                 # 模型特定参数（temperature、max_tokens 等）
# )

# 可配置参数
from langchain.chat_models import init_chat_model

# 不指定 model，返回可配置模型
# 可以固定一些参数（如 temperature=0.7），其余运行时指定
configurable_model = init_chat_model(temperature=0.7)

# 运行时通过 config 指定模型
response = configurable_model.invoke(
    "介绍菜鸟教程 RUNOOB",
    config={"configurable": {"model": "deepseek-v4-flash"}}
)
print(response.content)

# 同一个模型实例，可以用不同的模型来执行
response = configurable_model.invoke(
    "介绍菜鸟教程 RUNOOB",
    config={"configurable": {"model": "claude-sonnet-4-5"}}
)
print(response.content)





# 常用 kwargs 参数
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "deepseek:deepseek-v4-flash",

    # 控制输出随机性（0~2），值越小输出越稳定
    temperature=0.3,

    # 限制输出最大 token 数（控制成本）
    max_tokens=200,

    # 请求超时时间（秒）
    timeout=30,

    # 失败重试次数
    max_retries=2,

    # 自定义 API 地址（代理/中转场景）
    # base_url="https://your-proxy.com/v1",

    # 速率限制器（控制请求频率）
    # rate_limiter=MyRateLimiter(requests_per_second=5),
)

response = model.invoke("菜鸟教程 RUNOOB 是什么？")
print(response.content)




#运行时切换模型
from langchain.chat_models import init_chat_model

# 创建可配置模型，并设置默认值
model = init_chat_model(
    "deepseek:deepseek-v4-flash",       # 默认模型
    configurable_fields="any",  # 所有参数都可在运行时修改，1.None：不可配置，返回普通的BaseChatModel。2.“any”。3.["model", "temperature"]	只有列表中指定的字段可配置
    config_prefix="my",         # 配置键前缀
    temperature=0.3,            # 默认温度
)

# 使用默认配置运行
response = model.invoke("介绍菜鸟教程")
print(f"默认配置: {response.content[:50]}...")

# 运行时覆盖模型和参数（注意 my_ 前缀）
response = model.invoke(
    "介绍菜鸟教程 RUNOOB",
    config={
        "configurable": {
            "my_model": "deepseek:deepseek-v4-pro",       # 切换模型
            "my_temperature": 0.9,             # 调整温度
        }
    }
)
print(f"覆盖配置: {response.content[:50]}...")