# 加载训练好的模型,并导出模型参数(weight & bias)到文件,方便FPGA/ASIC进行参考工作
import numpy
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d
# import pandas as pd
import struct
from model import cifar10_net


# 搭建神经网络
class cifar10_net(nn.Module):
    def __init__(self):
        super(cifar10_net, self).__init__()
        # self.net = Sequential(Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
        #                       MaxPool2d(kernel_size=2),
        #                       Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
        #                       MaxPool2d(kernel_size=2),
        #                       Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
        #                       MaxPool2d(kernel_size=2),
        #                       nn.Flatten(),
        #                       nn.Linear(in_features=1024, out_features=64),
        #                       nn.Linear(in_features=64, out_features=10)
        #                       )
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.maxpool1 = MaxPool2d(kernel_size=2)
        self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.maxpool2 = MaxPool2d(kernel_size=2)
        self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.maxpool3 = MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=1024, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # y = self.net(x)
        # return y
        y = self.conv1(x)
        y = self.maxpool1(y)
        y = self.conv2(y)
        y = self.maxpool2(y)
        y = self.conv3(y)
        y = self.maxpool3(y)
        y = self.flatten(y)
        y = self.linear1(y)
        y = self.linear2(y)
        return y


# 模型加载
model = torch.load("model_200.pth", map_location=torch.device("cpu"))
print(model)


# 参数导出
i = 0
max_param = 0
min_param = 0
for name, param in model.named_parameters():
    print("name: {}\t\t\tshape: {}".format(name, param.shape))
    param = param.detach()  # 转换为requires_grad为false的tensor，得到的tensor不需要计算其梯度
    # 量化&数据转换
    param_q = param << 7  # 定点数量化, 7位小数位, 1位符号位
    param_q = torch.round(param_q)  # 截断为最近的整数
    param_q = torch.reshape(param_q, (-1, 1))  # 转换为1列tensor
    param_np = param_q.numpy()  # 转换为numpy数组
    param_int = param_np.astype(int)  # 转换为int类型
    i = 0
    for data in param_int:
        # print(data)
        if data < 0:  # 求补码, 正数原码和补码相同不处理
            param_int[i] = data + 2**8
        i = i + 1
    # 文件写入
    np.savetxt("{}.txt".format(name), param_int, fmt="%02x")  # 保存np数组到txt文件,从FPGA中读取即可
    np.savetxt("{}_d.txt".format(name), param_int, fmt="%d")  # 以整数形式保存,方便np.loadtxt()调用


#     # 寻找权重中的最大值, 确定量化方案
#     if torch.max(param) > max_param:
#         max_param = torch.max(param)
#     if torch.min(param) < min_param:
#         min_param = torch.min(param)
#     i = i + 1
#
# # 关于量化和进制转换的小测试
# max_q = max_param << 7  # 定点数量化, 7位小数位, 1位符号位
# min_q = min_param << 7
# max_q = torch.round(max_q)  # 调整到最接近的整数
# min_q = torch.round(min_q)
# max_q_int = int(max_q.item())  # 得到整型结果
# min_q_int = int(min_q.item())
#
# # 打印结果
# print("Max: {}, Quant_max: {}, Quant_Hex: {:x}".format(max_param, max_q_int, max_q_int))
# print("Min: {}, Quant_min: {}, Quant_Hex: {:x}".format(min_param, min_q_int, min_q_int))
