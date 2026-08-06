"""
    2.提示词模版

    在提示词模板中，通过{{ 关键字 }} 作为占位符，模板对象.format()进行赋值，从而动态生成提示词，常用的模板：
    1.ChatPromptTemplate
    2.LangchainPromptTemplate
    3.RichPromptTemplate

    优势：
    1. 清晰易懂的提示    2. 增强可重用性   3. 维护成本更低
"""
from langchain_core.prompts import PromptTemplate
from llama_index.core import ChatPromptTemplate
from llama_index.core.prompts import LangchainPromptTemplate, RichPromptTemplate

"""
    1. 使用聊天提示词模版 - ChatPromptTemplate
"""
# # 定义消息模版，使用元组（角色，内容）的形式
# messages_template = [
#     ("system", "以下是给你的问题, answer the question: \n\n"),
#     ("user", "{question}")
# ]
#
# template = ChatPromptTemplate.from_messages(messages_template)
# # 格式化并打印模版
# print(template.format_messages(question="How many params does llama 2 have"))


"""
    2.LangchainPromptTemplate --- 对LangChain中提示词模版的支持
"""
# # 定义一个LangChain样式的提示模版
# template_str = PromptTemplate(
#     template="""
#      Answer the following question based on the provided context:
#      Context:
#      {context}
#
#      Question:
#      {question}
#
#      Answer:
# """,
#     input_variables=["context", "question"]
# )
# # 创建LangChainPromptTemplate对象
# prompt_template = LangchainPromptTemplate(template=template_str)
#
# # 准备输入变量
# context = "Llama 2 is a large language model developed by Meta with 7B, 13B, and 70B parameter versions."
# question = "How many different parameter sizes does Llama 2 have?"
#
# # 格式化模板，生成完整的提示文案
# formatted_prompt = prompt_template.format(context=context, question=question)
# print("Formatted Prompt:\n", formatted_prompt)

"""
    3.RichPromptTemplate - 使用复杂的模版化的语言
"""
qa_prompt_tmpl_str = """\
Context information is below.
---------------------
{{ context_str }}
---------------------
Given the context information and not prior knowledge, answer the query.
Please write the answer in the style of {{ tone_name }}
Query: {{ query_str }}
Answer: \
"""
prompt_temp = RichPromptTemplate(template_str=qa_prompt_tmpl_str)
# partial_format 预先填充模板中部分内容，这种方法在需要动态调整部分模板内容（如语气、风格、上下文等）时非常有用
partial_prompt_tmpl = prompt_temp.partial_format(tone_name="xxx")
# 一次性填充所有变量
fmt_prompt = partial_prompt_tmpl.format(
    context_str="In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters",
    query_str="How many params does llama 2 have",
)
print(fmt_prompt)

print("-" * 100)
# ----------------- 另一种语言：JinJa2
prompt = RichPromptTemplate(
    """
    {% chat role = "system" %}
    You are now chatting with {{ user }}
    {% endchat %}
    
    {% chat role = "user" %}
    {{ user_msg}}
    {% endchat %}
    """
)
print(prompt.format(user="John", user_msg="Hello!"))
