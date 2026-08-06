"""
    Node
# 1. SimpleFileNodeParser: 一个智能的调度器，可以与 FlatReader 结合使用，根据文件扩展名自动选择最合适的特定文件解析器（如 MarkdownNodeParser, JSONNodeParser）。
    - HTMLNodeParser: 使用 beautifulsoup 库解析原始 HTML 内容，可以指定要提取文本的 HTML 标签。
    - JSONNodeParser: 用于解析原始 JSON 数据，将其中的文本内容转换为节点。
    - MarkdownNodeParser: 专门用于解析 Markdown 格式的文本，能够理解 Markdown 的结构
"""
from pathlib import Path

from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.readers.file import FlatReader

# 假设我们有一个 Markdown 文件 'test.md'
# 先创建一个示例 Markdown 文件

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

# 使用FlatReader 加载Markdown文件作为一个Document
# FlatReader 简单的将整个文件内容读入一个Document
md_docs_list = FlatReader().load_data(md_file_path)
print(f"通过 FlatReader 加载了 {len(md_docs_list)} 个 Markdown 文档。")

# 实例化 SimpleFileNodeParser
# 它内部会根据文档类型（通过元数据判断 或 直接处理文本）选择合适的解析逻辑
# 对于已加载的Document对象，它主要依赖 MarkdownNodeParser（如果内容是Markdown）
file_parser = SimpleFileNodeParser()

# 从加载的 Document 对象中解析出 Node
# 注意：SimpleFileNodeParser 内部的Markdown解析逻辑，可能倾向于将整个Document视为一个大Node，
# 或者按照Markdown结构（如标题）分割。具体行为取决于其内部实现。
# 如果想更细粒度大分割Markdown内容，通常会先用MarkdownNodeParser获得结构化节点，
# 然后再对这些节点的文本内容使用SentenceSpliter 等文本分割器。
# 为了掩饰SimpleFileNodeParser 本身，我们直接传入Document
md_nodes_from_file_parser = file_parser.get_nodes_from_documents(md_docs_list)
print(f"通过 SimpleFileNodeParser 从Markdown 文档解析出{len(md_nodes_from_file_parser)} 个Node")

if md_nodes_from_file_parser:
    print("第一个 Node 的内容示例 (SimpleFileNodeParser):")
    # 获取第一个Node的内容，并只显示前200个字符（避免输出过长）
    print(md_nodes_from_file_parser[0].get_content()[:200] + "...")
