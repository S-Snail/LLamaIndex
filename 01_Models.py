"""
    LlamaIndex 调用各大模型示例

"""

import os

from langchain_community.chat_models import ChatTongyi
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.llms.deepseek import DeepSeek
from llama_index.llms.langchain import LangChainLLM
from llama_index.llms.openai import OpenAI

# 调用OpenAI模型
# llm = OpenAI(
#     model="gpt-3.5-turbo",
#     api_key=os.getenv("OPENAI_API_KEY")
# )
# res = llm.complete("你是谁？")
# print(res.text)


# 调用阿里百炼
# dashscope_llm = DashScope(
#     model_name=DashScopeGenerationModels.QWEN_MAX,
#     api_key=os.getenv("DASHSCOPE_API_KEY")
# )
# res = dashscope_llm.complete("你是谁？")
# print(res)

# 调用DeepSeek
# llm = DeepSeek(
#     model="deepseek-chat",
#     api_key=os.getenv("DEEPSEEK_API_KEY")
# )
# res = llm.complete("你是谁？")
# print(res)

# 整合Langchain
tongyi = ChatTongyi(
    model="qwen-max"
)
llm = LangChainLLM(llm=tongyi)
res = llm.complete("你是谁？")
print(res)
# 流式输出
res = llm.stream_complete("你是谁？")
for data in res:
    print(data.delta, end="")
