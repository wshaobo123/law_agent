import re
import warnings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from pymilvus import (
    utility, connections, Collection, CollectionSchema, FieldSchema, DataType
)

from config import (
    DATA_FILE_PATH, EMBEDDING_MODEL_NAME, MILVUS_HOST, MILVUS_PORT,
    MILVUS_COLLECTION_NAME, EMBEDDING_DIMENSION,
    TEXT_FIELD_NAME, PRIMARY_FIELD_NAME, VECTOR_FIELD_NAME
)


warnings.filterwarnings("ignore")
# 加载并切分文档
def load_and_split_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    pattern = re.compile(r'(第[一二三四五六七八九十百千]+条\s.*?(?=\n第[一二三四五六七八九十百千]+条|\Z))', re.S)
    clauses = pattern.findall(text)
    documents = []
    for i, clause in enumerate(clauses):
        article_match = re.match(r'(第[一二三四五六七八九十百千]+条)', clause)
        article_number = article_match.group(1) if article_match else f"未知条款_{i + 1}"
        doc = Document(
            page_content=clause.strip(),
            metadata={"source": file_path, "article": article_number}
        )
        documents.append(doc)
    print(f"成功将文本切分为 {len(documents)} 个法律条文。")
    return documents


# 定义并创建支持混合检索的Schema
def create_hybrid_collection_if_not_exists():
    """
    手动定义Schema，创建集合，并为向量和文本字段分别建立索引。
    """
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    if utility.has_collection(MILVUS_COLLECTION_NAME):
        print(f"集合 '{MILVUS_COLLECTION_NAME}' 已存在，将直接使用。")
        return Collection(MILVUS_COLLECTION_NAME)

    print(f"集合 '{MILVUS_COLLECTION_NAME}' 不存在，开始创建...")

    # 1. 定义字段 (Fields)
    # 主键字段
    pk_field = FieldSchema(
        name=PRIMARY_FIELD_NAME, dtype=DataType.INT64, is_primary=True, auto_id=True
    )
    # 用于BM25的文本字段
    text_field = FieldSchema(
        name=TEXT_FIELD_NAME, dtype=DataType.VARCHAR, max_length=65535  # max_length很重要
    )
    # 向量字段
    vector_field = FieldSchema(
        name=VECTOR_FIELD_NAME, dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION
    )

    # 2. 定义集合结构
    schema = CollectionSchema(
        fields=[pk_field, text_field, vector_field],
        description="法律条文混合检索集合",
        enable_dynamic_field=True  # 允许添加其他metadata
    )

    # 3. 创建集合
    collection = Collection(
        name=MILVUS_COLLECTION_NAME,
        schema=schema,
        using='default'
    )
    print(f"集合 '{MILVUS_COLLECTION_NAME}' 创建成功！")

    # 4. 创建索引
    # 为向量字段创建索引
    vector_index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",  # 或者 IVF_FLAT 等
        "params": {"M": 8, "efConstruction": 200}
    }
    collection.create_index(
        field_name=VECTOR_FIELD_NAME,
        index_params=vector_index_params
    )
    print(f"为字段 '{VECTOR_FIELD_NAME}' 创建向量索引成功。")

    # 为文本字段创建BM25索引
    collection.create_index(
        field_name=TEXT_FIELD_NAME,
        index_type="INVERTED"  # BM25使用的索引类型
    )
    print(f"为字段 '{TEXT_FIELD_NAME}' 创建BM25索引成功。")

    return collection


def ingest_data(collection, documents, embeddings):
    """
    将数据处理并插入到指定的集合中。
    """
    if collection.num_entities > 0:
        print("集合中已有数据，跳过插入过程。如需重新插入，请先清空集合。")
        return

    print("开始处理并插入数据...")
    data_to_insert = []
    for doc in documents:
        # 手动生成embedding
        vector = embeddings.embed_query(doc.page_content)
        # 准备实体数据
        entity = {
            TEXT_FIELD_NAME: doc.page_content,
            VECTOR_FIELD_NAME: vector,
            "metadata": doc.metadata  # 动态字段
        }
        data_to_insert.append(entity)

    collection.insert(data_to_insert)
    # 刷新集合以确保数据可被搜索
    collection.flush()
    print(f"成功插入 {len(data_to_insert)} 条数据。")
    # 加载集合到内存以备搜索
    collection.load()
    print("集合已加载到内存。")


if __name__ == "__main__":
    # 1. 创建或获取集合
    collection = create_hybrid_collection_if_not_exists()

    # 2. 加载并切分文档
    docs = load_and_split_text(DATA_FILE_PATH)

    if docs:
        # 3. 初始化Embedding模型
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})

        # 4. 插入数据
        ingest_data(collection, docs, embeddings)
    else:
        print("没有加载到任何文档。")