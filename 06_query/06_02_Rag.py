"""
    LlamaIndex与RAG
    基于LlamaIndex组件构建Rag应用：
    1、文档加载器
    2、构建索引
    3、存储
    4、查询

"""
import os

import chromadb
from langchain_community.embeddings import DashScopeEmbeddings
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.vector_stores.chroma import ChromaVectorStore

# 嵌入模型
embed_model = LangchainEmbedding(
    DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
)

# 对话模型
dashscope_llm = DashScope(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model_name=DashScopeGenerationModels.QWEN_MAX
)

# 1.加载参考数据
documents = SimpleDirectoryReader(input_dir="../resource", required_exts=[".pdf"]).load_data()

# 3. 创建ChromaVectorStore 实例
db = chromadb.PersistentClient("./chroma_db")
chroma_collection = db.get_or_create_collection("chroma_rag")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 创建一个StorageContext，并明确告诉它，我们的向量数据要存储在刚才配置好的vector_store(即ChromaDB)中
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 2.构建索引，数据会自动存储在ChromaDB
index = VectorStoreIndex(documents, embed_model=embed_model, storage_context=storage_context)

# 4.构建检索引擎
query_engine = index.as_query_engine(llm=dashscope_llm)
response = query_engine.query("怎么晋升？")
print(response)
