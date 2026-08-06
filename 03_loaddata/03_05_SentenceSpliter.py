"""
    Node
# 3. SentenceSpliter - 智能文本分割器(重点) ：
    推荐的默认分割器之一。它会按句子来分割文本，同时尽量保持每个块的大小 (chunk_size) 和块之间的重叠 (chunk_overlap)
"""
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

# 创建一个解析器实例
sentence_spliter = SentenceSplitter(
    chunk_size=200,  # 增加 chunk_size
    chunk_overlap=20  # 增加重叠部分
)

# 使用更长的文本
documents = [Document(text="这是一个很长很长的示例文本，它需要被分割成多个小块。LlamaIndex 使得这个过程非常简单。")]
text_nodes = sentence_spliter.get_nodes_from_documents(documents)

print(f"原始文档数: {len(documents)}")
print(f"分割后的节点数: {len(text_nodes)}")

# 打印分割后的详细内容
for i, node in enumerate(text_nodes):
    print(f"\n=== 节点 {i+1} ===")
    print(f"内容: {node.get_content()}")
    print(f"元数据: {node.metadata}")
    print(f"节点 ID: {node.node_id}")
    print("-" * 100)
