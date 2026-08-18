"""
    评估响应忠实度
"""
import os

from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels

load_dotenv()

# 对话模型
dashscope_llm = DashScope(
    model_name=DashScopeGenerationModels.QWEN_MAX,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)
# 嵌入模型
embed_model = LangchainEmbedding(
    DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )
)
#  不设置，默认使用的是openai的模型
Settings.llm = dashscope_llm
Settings.embed_model = embed_model

'''加载数据'''
reader = SimpleDirectoryReader(input_dir="./data/")
documents = reader.load_data()

'''创建索引'''
index = VectorStoreIndex.from_documents(documents, show_progress=True)

'''创建查询引擎'''
query_engine = index.as_query_engine()
response = query_engine.query("deepseek的发展历程")
print(f"响应内容: {response.response}")
print(f"源节点数量: {len(response.source_nodes)}")

'''实例化 FaithfulnessEvaluator'''
evaluator = FaithfulnessEvaluator()

'''评估查询'''
eval_result = evaluator.evaluate_response(response=response)
print("\n忠实度评估结果:")
print(f"  是否通过 (Passing): {eval_result.passing}")
print(f"  分数 (Score): {eval_result.score}")
print(f"  反馈 (Feedback): \n{eval_result.feedback}")

# 演示如何针对每个源上下文单独评估 (更细粒度)
print("\n--- 针对每个源上下文单独进行忠实度评估 ---")
response_str = response.response  # 获取响应的文本字符串
for i, source_node in enumerate(response.source_nodes):
    print(f"\n评估响应对于源节点 {i + 1} 的忠实度:")
    # contexts 参数需要一个字符串列表
    context_content = source_node.get_content()
    eval_result_node_faith = evaluator.evaluate(
        response=response_str,
        contexts=[context_content]  # 将单个节点内容作为上下文列表
    )
    print(f"  源节点 {i + 1} - 是否通过: {eval_result_node_faith.passing}, 分数: {eval_result_node_faith.score}")
    # print(f"  源节点 {i+1} - 反馈: {eval_result_node_faith.feedback}") # 可以取消注释查看详细反馈
