"""
    SummaryIndex：将所有Node简单的按顺序存储，查询时，它会按顺序 或 递归的对所有Node进行总结。
    适用于对整个文档集进行摘要的任务。
"""
import os

from langchain_community.embeddings import DashScopeEmbeddings
from llama_index.core import SimpleDirectoryReader, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels

# 嵌入模型
embed_model = LangchainEmbedding(
    DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
)

# 对话模型
dashscope_llm = DashScope(
    model_name=DashScopeGenerationModels.QWEN_MAX,
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

documents = SimpleDirectoryReader("../resource", required_exts=[".pdf"]).load_data()
node_parser = SentenceSplitter(chunk_size=128, chunk_overlap=10)
nodes = node_parser.get_nodes_from_documents(documents)

# SummaryIndex 通常用于摘要任务
summary_index = SummaryIndex(nodes, embed_model=embed_model)
# 创建一个用于摘要的查询索引
# response_mode = "tree_summarize" # 是SummaryIndex 常用的响应模式，它会递归的总结节点
summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize", llm=dashscope_llm)
summary_response = summary_query_engine.query("请用一句话总结所有文档的内容。")
print(summary_response)
