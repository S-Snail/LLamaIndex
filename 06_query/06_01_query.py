"""
    QueryEngine：查询引擎
    是一个通用的接口，用自然语言对你的数据查询。
    它是一个无状态 (stateless) 的组件，每次查询都是独立的，不记得之前的对话。
"""
import os

from langchain_community.embeddings import DashScopeEmbeddings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels

# 初始化嵌入模型
embed_model = LangchainEmbedding(
    DashScopeEmbeddings(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="text-embedding-v1"
    )
)

# 初始化对话模型
dashscope_llm = DashScope(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model_name=DashScopeGenerationModels.QWEN_MAX
)

documents = SimpleDirectoryReader(input_dir="../resource", required_exts=[".txt"]).load_data()
index = VectorStoreIndex.from_documents(documents=documents, embed_model=embed_model)

# 配置查询引擎，指定 top_k 和 response_mode
# Response Mode (响应模式):
#   - compact (默认): 尽可能地将 Node 文本打包进一个 LLM 调用中，以节省成本。
#   - refine: 逐个处理 Node，先基于第一个 Node 生成初步答案，然后用后续 Node 的逐步“精炼”和完善这个答案。LLM 调用次数较多，但答案可能更详细。
#   - tree_summarize: 递归地对 Node 进行总结，像一棵树一样，最终汇总成一个根答案。非常适合摘要任务。
#   - no_text: 只执行检索，不调用 LLM 生成答案。用于调试，检查检索器返回了哪些节点。
query_engine = index.as_query_engine(
    llm=dashscope_llm,
    similarity_top_k=2,  # 检索最相关的 2 个节点
    response_mode="refine"  # 使用 refine 模式合成答案
)
response = query_engine.query("LlamaIndex是什么？")
print(response)
