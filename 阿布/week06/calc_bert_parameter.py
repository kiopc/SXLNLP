import torch
from transformers import BertModel


# 计算bert参数数量
bert = BertModel.from_pretrained(r"bert-base-chinese")
parmers = bert.state_dict()
w_sum = []
for k, v in parmers.items():
    size = v.shape
    res = torch.tensor(size).prod()
    w_sum.append(res)

print(sum(w_sum))  # tensor(24301056)
