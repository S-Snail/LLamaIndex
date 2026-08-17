"""
    用的最多是把Index存储到向量数据库

"""
import os

import chromadb
from langchain_community.embeddings import DashScopeEmbeddings
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.vector_stores.chroma import ChromaVectorStore

# 初始化向量模型
embed_model = LangchainEmbedding(
    DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
)

# 初始化对话模型
dashscope_ll = DashScope(
    model_name=DashScopeGenerationModels.QWEN_MAX,
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

documents = SimpleDirectoryReader(input_dir="../resource", required_exts=[".txt"]).load_data()

# 1.初始化ChromaDB
db = chromadb.PersistentClient("./chroma_db")

# 2.创建collection
chroma_collection = db.get_or_create_collection(name="quickstart")

# 3.创建ChromaVectorStore实例
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 4. 创建一个StorageContext，并明确告诉它，我们的向量数据要存储在刚才配置好的vector_store(即ChromaDB)中
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 5. 构建索引，数据会自动存入ChromaDB
# LlamaIndex 在生成向量嵌入后，会自动将这些向量和相关的文本数据通过vector_store适配器存入ChromaDB中
index = VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context, embed_model=embed_model)

query_engine = index.as_query_engine(llm=dashscope_ll)
response = query_engine.query("LlamaIndex是什么？")
print(response)
