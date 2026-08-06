"""
    Node
# 2. MarkdownNodeParse：专门用于解析 Markdown 格式的文本，能够理解 Markdown 的结构
"""
from pathlib import Path

from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.file import FlatReader

md_content = """
# LlamaIndex 简介

LlamaIndex 是一个强大的工具。

## 核心功能
- 数据加载
- 数据索引
- 数据查询
"""
md_file_path = Path("./data/test.md")
with open(md_file_path, "w", encoding="utf-8") as f:
    f.write(md_content)

# 使用 FlatReader 加载Markdown文件作为Document
# FlatReader 简单的将整个文件内容读入 Document
md_docs_list = FlatReader().load_data(md_file_path)

markdown_parser = MarkdownNodeParser()
# MarkdownNodeParser 会尝试根据Markdown的语义结构（如标题、列表、代码块等）来创建节点
structured_md_nodes = markdown_parser.get_nodes_from_documents(md_docs_list)

print(f"\n通过 MarkdownNodeParser 解析出 {len(structured_md_nodes)} 个结构化 Node。")

for i, node in enumerate(structured_md_nodes):
    print(f"i = {i}\n", node.get_content())
    print(f"Metadata: {node.metadata}")  # MarkdownNodeParser 会添加一些结构相关的元数据
    print("-" * 100)
