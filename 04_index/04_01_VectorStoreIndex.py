"""
    索引（Indexing）
    一种数据结构，它组织你的Node，以便能被LLM更高效的查询
"""


# import os
# import ssl
#
# # 解决mac nltk ssl证书报错
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
#
# import nltk
# # 下载需要的两个资源
# nltk.download('stopwords')
# nltk.download('punkt_tab')

import os

from langchain_community.embeddings import DashScopeEmbeddings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.langchain import LangchainEmbedding

"""
        VectorStoreIndex：最常用、最强大的索引类型
        将每个Node的文本及其向量嵌入存储在一个向量存储（VectorStore）中
"""
# 嵌入模型
embed_model = LangchainEmbedding(
    DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
)

documents = SimpleDirectoryReader("../resource").load_data()
node_parser = SentenceSplitter(chunk_size=128, chunk_overlap=10)
nodes = node_parser.get_nodes_from_documents(documents)

# 使用默认的 Embedding Model 生成嵌入，默认是使用OpenAI相关模型
# show_progress = True 会显示一个进度条，对于大数据量处理时很有用

# 方式一：直接从documents创建，需要指定嵌入模型
# index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, show_progress=True)

# 方式二：从 nodes 创建，需要指定嵌入模型
index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True)
