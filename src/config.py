import os
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 读取api-key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# 设置基准模型参数
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
LLM_MODEL_NAME = "deepseek-chat"

# Embedding模型
EMBEDDING_MODEL_NAME = 'BAAI/bge-large-zh-v1.5'
EMBEDDING_DIMENSION = 1024

# Reranker模型
RERANKER_MODEL_NAME = 'BAAI/bge-reranker-large'

# Milvus向量数据库参数
MILVUS_HOST = "localhost"  # 或者你的Milvus服务器IP
MILVUS_PORT = "19530"
MILVUS_COLLECTION_NAME = "legal_clauses_hybrid"
# 定义用于存储原文并建立BM25索引的字段名
TEXT_FIELD_NAME = "clause_text"
PRIMARY_FIELD_NAME = "id"
VECTOR_FIELD_NAME = "vector"

# 数据文件路径
DATA_FILE_PATH = r"..\data\civil_code.txt"

# 召回阶段返回的文档数
RETRIEVER_TOP_K = 20

# 重排后最终选择的文档数
RERANKER_TOP_N = 5

# Web_Search模块的Tavily API配置
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY