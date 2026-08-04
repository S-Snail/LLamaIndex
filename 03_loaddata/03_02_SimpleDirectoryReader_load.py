"""
    3. 文档加载
    从文件夹读取文件创建Document

"""
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_dir="../resource", required_exts=[".txt"])
documents = reader.load_data()
print(documents)
print(len(documents))
