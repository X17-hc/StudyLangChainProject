from dotenv import load_dotenv
load_dotenv()


"""
之前的文章我们手动输入文本，但在实际项目中，文档可能来自 PDF、网页、Markdown 文件等。

本节介绍如何使用 Document Loader 加载各类文档，以及如何用 Text Splitter 将文档切分成适合检索的小块。
"""

"""
    Loader	                    来源	            安装包
    TextLoader	                .txt 文件	    langchain（内置）
    PyPDFLoader	                PDF 文件	        langchain-community + pypdf
    WebBaseLoader	            网页 URL	        langchain-community + beautifulsoup4
    CSVLoader	                CSV 文件	        langchain-community
    UnstructuredMarkdownLoader	Markdown 文件	langchain-community + unstructured
"""

# 1.Document Loader——加载文档.加载文本文件（内置，无需额外安装）
from langchain_community.document_loaders import TextLoader

loader = TextLoader("knowledge.txt", encoding="utf-8")
docs = loader.load()

print(f"加载了 {len(docs)} 个文档")
print(f"内容预览: {docs[0].page_content[:150]}...")

# 加载网页
# pip install langchain-community beautifulsoup4
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.runoob.com/python/python-tutorial.html")
docs = loader.load()
print(f"\n网页内容: {docs[0].page_content[:150]}...")



# 2.Text Splitter——文档切分
# 文档通常太长，需要切分成小块（chunk）才能有效检索。切分策略直接影响 RAG 效果
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 创建切分器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,         # 每块最多 500 个字符
    # chunk_overlap 很重要。如果块之间没有重叠，一个完整的句子可能被切成两半，导致检索时遗漏关键信息。50-100 字符的重叠是常见的设置。
    chunk_overlap=50,       # 块之间重叠 50 个字符
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    # 优先按段落分割，然后是句子，最后是字符
)

# 示例文档
long_text = """菜鸟教程（RUNOOB）是一个免费的编程学习平台。

平台提供了丰富的编程语言教程，包括但不限于：
- Python 教程：从基础语法到数据分析
- Java 教程：面向对象编程到 Spring 框架
- 前端教程：HTML、CSS、JavaScript 及其框架

所有教程都配有详细的代码示例和在线运行环境。
学习者可以通过边学边练的方式快速掌握编程技能。"""

# 切分文档
chunks = text_splitter.split_text(long_text)

print(f"原文长度: {len(long_text)} 字")
print(f"切分后: {len(chunks)} 块\n")

for i, chunk in enumerate(chunks):
    print(f"--- 块 {i+1} ({len(chunk)} 字) ---")
    print(chunk)
    print()

"""
    场景	        chunk_size	    chunk_overlap	原因
    FAQ 问答	    200~500     	20~50	        问答对较短，小块即可
    技术文档	    500~1000	    50~100	        技术内容需要更多上下文
    长文章/论文	1000~2000	    100~200	        需要保留段落完整性
    代码库	    500~1500	    0~50	        函数/类作为自然边界
"""




# 3.完整流程：加载 → 切分 → 向量化
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
# from langchain_community.document_loaders import TextLoader


# 流程 1：加载
# loader = TextLoader("runoob_knowledge.txt", encoding="utf-8")
# docs = loader.load()

# 为演示直接使用示例文本
docs = [
    "菜鸟教程（RUNOOB）是一个免费的编程学习网站。",
    "网站提供 Python、Java、HTML 等多种编程语言的教程。",
    "Python3 基础教程共 30 章，适合零基础入门学习。",
    "HTML 基础教程共 25 章，包含表单、多媒体等内容。",
    "菜鸟教程的所有基础教程都是免费的。",
]

# 流程 2：切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)
chunks = text_splitter.create_documents(docs)

# 流程 3：向量化存储
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./runoob_db",
)

print(f"已建立索引：{len(chunks)} 个文档块")

# 流程 4：检索
results = vector_store.similarity_search("Python 教程有多少章？", k=2)
for doc in results:
    print(f"检索结果: {doc.page_content}")


"""
    输出：
        已建立索引：5 个文档块
        检索结果: Python3 基础教程共 30 章，适合零基础入门学习。
        检索结果: 菜鸟教程（RUNOOB）是一个免费的编程学习网站。
"""