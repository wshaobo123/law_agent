from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool
from sentence_transformers import CrossEncoder

from config import (
    MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION_NAME, EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME, RETRIEVER_TOP_K, RERANKER_TOP_N,TEXT_FIELD_NAME, VECTOR_FIELD_NAME
)


# 初始化Embedding和Reranker模型
print("Initializing models for tools...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cuda'})
reranker = CrossEncoder(RERANKER_MODEL_NAME, max_length=512, device='cuda')
print("Models initialized.")


# 向量数据库检索
def retrieve_and_rerank(query: str) -> str:
    """
    完整RAG流程：
    1. 从Milvus进行混合检索 (召回)
    2. 使用BGE-Reranker模型进行重排 (精排)
    3. 格式化并返回最相关的法律条文
    """
    try:
        # 连接到已存在的Milvus集合
        vector_store = Milvus(
            embedding_function=embeddings,
            collection_name=MILVUS_COLLECTION_NAME,
            connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
            # 指定字段名
            text_field=TEXT_FIELD_NAME,
            vector_field=VECTOR_FIELD_NAME,
        )

        # 1.召回，使用混合检索
        retriever = vector_store.as_retriever(
            search_type="hybrid",
            search_kwargs={'k': RETRIEVER_TOP_K}
        )
        retrieved_docs = retriever.invoke(query)

        # 2.重排
        pairs = [(query, doc.page_content) for doc in retrieved_docs]
        scores = reranker.predict(pairs)
        scored_docs = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)

        # 3.筛选并格式化
        top_n_docs = [doc for score, doc in scored_docs[:RERANKER_TOP_N]]
        context = "\n\n---\n\n".join(
            [f"来源: {doc.metadata.get('metadata', {}).get('article', '未知条款')}\n内容: {doc.page_content}" for doc in
             top_n_docs])

        return f"《民法典》相关条款如下：\n{context}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"在检索法律条文时发生错误: {e}"


# 将RAG函数包装成LangChain的Tool, 告诉Agent什么时候应该使用这个工具
rag_tool = Tool(
    name="query_civil_code",
    func=retrieve_and_rerank,
    description='''当需要查询中华人民共和国民法典的具体法律条文、定义或规定时使用。
                《中华人民共和国民法典》共7编、1260条，各编依次为总则、物权、合同、人格权、婚姻家庭、继承、侵权责任，以及附则。
                例如，关于离婚财产分割、合同规定、继承权等基础法律问题。'''
)

# Web_Search工具
web_search_tool = TavilySearchResults(max_results=3)
web_search_tool.name = "web_search"
web_search_tool.description = ("""当需要获取实时信息、最新的法律新闻、司法解释或查找具体的、公开的法律案例时使用。
                               当问题涉及到具体日期、近期事件或民法典中没有明确规定的情况时，此工具非常有用。""")

# 将所有工具放入一个列表
agent_tools = [rag_tool, web_search_tool]