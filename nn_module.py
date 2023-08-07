import torch
from torch import nn


# 自定义模型
class my_module(nn.Module):  # 从父类继承为子类
    def __init__(self):
        super().__init__()  # 初始化父类

    def forward(self, input): # 处理模型前向路径
        output = input + 1
        return output


# 例化模型类
module = my_module()
x = torch.tensor(1.0)
output = module(x)  # 通过模型对数据进行处理
print(output)
