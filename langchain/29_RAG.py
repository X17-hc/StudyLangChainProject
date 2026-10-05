from dotenv import load_dotenv
load_dotenv()


# 环境准备
#  pip install langchain-deepseek langchain-chroma chromadb
"""
    包	                    用途
    langchain-deepseek	    提供 OpenAI Embedding 模型
    langchain-chroma	    Chroma 向量数据库的 LangChain 集成
    chromadb	            Chroma 向量数据库（轻量级，适合入门）
"""

# Embedding 模型初始化
from langchain_openai import OpenAIEmbeddings

# OpenAI 的文本嵌入模型
# 将文本转换为向量（一组浮点数）
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 测试：将一段文本转为向量
text = "菜鸟教程 RUNOOB 是一个编程学习平台"
vector = embeddings.embed_query(text)

print(f"文本: {text}")
print(f"向量维度: {len(vector)}")   # text-embedding-3-small 是 1536 维
print(f"向量前 5 个值: {vector[:5]}")




# 2.创建向量存储
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 初始化 Embedding 模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 创建 Chroma 向量存储（数据保存在本地目录）
vector_store = Chroma(
    collection_name="runoob_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",  # 持久化目录
)

# 添加文档（最简单的形式：文本列表）
texts = [
    "菜鸟教程（RUNOOB）是一个免费的编程学习网站，提供 HTML、CSS、JavaScript、Python 等教程。",
    "Python3 基础教程共 30 章，适合零基础入门，包含环境搭建、语法基础、面向对象等内容。",
    "HTML 基础教程共 25 章，覆盖 HTML 标签、表单、多媒体等基础知识。",
]

# add_texts 自动将文本转为向量并存储
vector_store.add_texts(texts)

print(f"已添加 {len(texts)} 个文档到向量存储")



# 3.创建向量存储
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 初始化 Embedding 模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 创建 Chroma 向量存储（数据保存在本地目录）
vector_store = Chroma(
    collection_name="runoob_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",  # 持久化目录
)

# 添加文档（最简单的形式：文本列表）
texts = [
    "菜鸟教程（RUNOOB）是一个免费的编程学习网站，提供 HTML、CSS、JavaScript、Python 等教程。",
    "Python3 基础教程共 30 章，适合零基础入门，包含环境搭建、语法基础、面向对象等内容。",
    "HTML 基础教程共 25 章，覆盖 HTML 标签、表单、多媒体等基础知识。",
]

# add_texts 自动将文本转为向量并存储
vector_store.add_texts(texts)
print(f"已添加 {len(texts)} 个文档到向量存储")



# 语义检索
# 语义搜索——不依赖关键词匹配，而是语义相似度
results = vector_store.similarity_search(
    "我想学 Python，有什么教程推荐？",
    k=2,  # 返回最相似的 2 个结果
)

print("搜索结果：")
for i, doc in enumerate(results):
    print(f"\n结果 {i+1}:")
    print(f"  内容: {doc.page_content}")
    print(f"  元数据: {doc.metadata}")

"""
    注意第一个搜索结果比第二个更相关——虽然第一个包含 "Python" 关键词，但它按 语义相似度 而非关键词匹配排序。这就是向量检索的优势。
"""



# 4.创建 Retriever 检索器。
# Retriever 是 Vector Store 的标准化接口：

# 从 vector_store 创建 retriever
retriever = vector_store.as_retriever(
    search_type="similarity",  # 相似度搜索
    search_kwargs={"k": 3},    # 返回前 3 个结果
)

# 使用 retriever
docs = retriever.invoke("Python 学习路线")
for doc in docs:
    print(f"- {doc.page_content[:60]}...")
