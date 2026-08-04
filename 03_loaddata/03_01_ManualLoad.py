"""
    3. 文档加载
    手动创建Document

"""
from llama_index.core import Document

# 手动创建Document
doc = Document(
    doc_id="manual_id",
    text="这是第一份手动创建的文档。",
    metadata={"source": "manual", "version": 1.0}
)
print(doc)

text_list = ["Q:在线支付取消订单后钱怎么返还?订单取消后，款项会在一个工作日内，直接返还到您的美团账户余额。",
             "Q:怎么查看退款是否成功?退款会在一个工作日之内到美团账户余额，可在“账号管理一我的账号”中查看是否到账。",
             "Q:余额提现到账时间是多久?1-7个工作日内可退回您的支付账户。由于银行处理可能有延迟，具体以账户的到账时间为准。",
             "Q:申请退款后，商家拒绝了怎么办?申请退款后，如果商家拒绝，此时回到订单页面点击“退款申诉”，美团客服介入处理。"
             ]
docs = [Document(text=s) for s in text_list]
print(docs)
print(len(docs))
