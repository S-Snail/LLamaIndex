"""
    基于LlamaIndex的RAG系统评估
    1.Retrieval Evaluation (检索评估)：衡量检索器（Retriever）的性能。
  常用指标：
    - Hit Rate (命中率): 检索到的节点中，有多少是正确的（预先标注的）？
    - MRR (Mean Reciprocal Rank, 平均倒数排名): 正确答案的排名有多靠前？排名越靠前，得分越高。

    2.Response Evaluation (响应评估)：衡量 LLM 生成的最终答案的质量。
    - 忠实度评估：答案与上下文的相关性评估；
    - 相关性评估：答案与原始问题的相关性评估；

"""
import os

from langchain_community.embeddings import DashScopeEmbeddings
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import generate_question_context_pairs, RetrieverEvaluator
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels

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

documents = SimpleDirectoryReader(input_dir="./data/").load_data()
node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
nodes = node_parser.get_nodes_from_documents(documents)

# 从 nodes 创建索引
index = VectorStoreIndex(nodes, show_progress=True)

# 1. 自动生成一个评估数据集 (问题, 相关文档ID)
# 在真实场景中，你可能需要手动标注更高质量的数据集
qa_dataset = generate_question_context_pairs(nodes)

# 2. 定义一个检索器
retriever = index.as_retriever(similarity_top_k=2)

# 3. 创建检索评估器
retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever
)

# 4. 在数据集上运行评估
# async def main_eval():
#     return await retriever_evaluator.aevaluate_dataset(qa_dataset)
# eval_results = asyncio.run(main_eval())
#
# # 5. 打印评估结果
# print("\n检索评估结果:")
# print(eval_results)

import asyncio
from tabulate import tabulate  # 安装依赖: pip install tabulate

# 评估结果结构解析，数据统计
def format_eval_results(eval_results):
    results_table = []
    total_mrr = 0.0
    total_hit_rate = 0.0
    total_queries = len(eval_results)

    for idx, result in enumerate(eval_results):
        query = result.query
        expected_ids = result.expected_ids
        retrieved_ids = result.retrieved_ids
        mrr_score = result.metric_dict["mrr"].score
        hit_rate_score = result.metric_dict["hit_rate"].score

        # 标记成功/失败
        status = "✅ Success" if hit_rate_score > 0 else "❌ Failed"

        # 提取前3个检索ID（避免过长）
        retrieved_ids_str = ", ".join(retrieved_ids[:3]) if retrieved_ids else "N/A"
        expected_ids_str = ", ".join(expected_ids) if expected_ids else "N/A"

        results_table.append([
            idx + 1,
            query[:50] + "...",  # 截断长查询
            expected_ids_str,
            retrieved_ids_str,
            f"{mrr_score:.2f}",
            f"{hit_rate_score:.2f}",
            status
        ])

        total_mrr += mrr_score
        total_hit_rate += hit_rate_score

    # 计算平均指标
    avg_mrr = total_mrr / total_queries
    avg_hit_rate = total_hit_rate / total_queries

    # 打印表格
    headers = ["#", "Query (truncated)", "Expected IDs", "Retrieved IDs", "MRR", "Hit Rate", "Status"]
    print(tabulate(results_table, headers=headers, tablefmt="rounded_grid"))

    # 打印总体统计
    print("\n📊 Overall Statistics:")
    print(f"Average MRR: {avg_mrr:.4f}")
    print(f"Average Hit Rate: {avg_hit_rate:.4f}")
    print(f"Total Queries: {total_queries}")
    print(f"Success Rate: {sum(1 for r in eval_results if r.metric_dict['hit_rate'].score > 0)/total_queries:.2%}")

# 在 main_eval 中调用
async def main_eval():
    results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
    return results

eval_results = asyncio.run(main_eval())
format_eval_results(eval_results)