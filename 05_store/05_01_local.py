"""
    Llamalndex提供了非常多的数据存储组件，允许用户定制外部数据存储。
    1.Document stores:用来存储Document和Node.
    2.Index stores: 用来存储index相关的元数据、
    3.Vector stores: 用来存储向量数据。
    4.Chat stores:用来存储聊天记录。
    5.Property stores:用来存储知识图谱相关的数据。

    StorageContext：LLamaIndex所提供的配置和管理存储器的核心对象，可以看作是以上存储组件的“容器”或“管理器”

    创建方式：
    - StorageContext.from_defaults(): 这是最常用的创建方式。您可以不带任何参数调用它，此时它会使用所有存储组件的默认内存实现 (SimpleDocumentStore, SimpleIndexStore, SimpleVectorStore)。
    - 您也可以在 from_defaults() 中传入您自己实例化的特定存储对象来覆盖默认值，例如：StorageContext.from_defaults(vector_store=my_chroma_vector_store,docstore=my_mongo_doc_store)。
    - 当您需要从磁盘加载之前通过 persist() 保存的默认存储时，可以使用StorageContext.from_defaults(persist_dir="目录")。

"""
import os

from langchain_community.embeddings import DashScopeEmbeddings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels

# 初始化嵌入模型
embed_model = LangchainEmbedding(
    DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
)

# 初始化对话模型
dashscope_llm = DashScope(
    model_name=DashScopeGenerationModels.QWEN_MAX,
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 本地持久化
PERSIST_DIR = "./storage"

if not os.path.exists(PERSIST_DIR):
    # 如果目录不存在，则加载数据并创建索引
    print("创建新索引并持久化...")
    documents = SimpleDirectoryReader(input_dir="../resource", required_exts=[".txt"]).load_data()
    # 创建索引
    index = VectorStoreIndex.from_documents(documents=documents, embed_model=embed_model)
    # 持久化到磁盘
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # 如果存储目录已存在，则直接加载
    print("从磁盘加载已有索引...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    # 指定嵌入模型
    index = load_index_from_storage(storage_context=storage_context, embed_model=embed_model)

# 无论哪种方式，现在index都已准备好
query_engine = index.as_query_engine(llm=dashscope_llm)
response = query_engine.query("LlamaIndex是什么")
print(response)
