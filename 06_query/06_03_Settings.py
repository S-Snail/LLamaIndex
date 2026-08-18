"""
    Setings是Llamalndex中的全局设置管理器。
    它允许用户设置全局的配置，如LLM、索引、查询引擎、Chat Engine等。有了Settings，用户就不用在每个组价中单独配置这些参数了
"""
import os

from langchain_community.embeddings import DashScopeEmbeddings
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels

# 对话模型
dashscope_llm = DashScope(
    model_name=DashScopeGenerationModels.QWEN_MAX,
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 嵌入模型
embed_model = LangchainEmbedding(
    DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
)

#  不设置，默认使用的是openai的模型
Settings.embed_model = embed_model
Settings.llm = dashscope_llm

# 加载参考数据
documents = SimpleDirectoryReader(input_dir="../resource", required_exts=[".pdf"]).load_data()
index = VectorStoreIndex(documents)
query_engine = index.as_query_engine()
response = query_engine.query("怎么晋升？")
print(response)
