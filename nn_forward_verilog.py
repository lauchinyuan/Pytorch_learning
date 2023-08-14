# 使用经过定点化后的模型参数构建神经网络前向路径
# 作为Verilog/FPGA/ASIC计算结果的参考
# 与pytorch自带的模型算子不同之处在于,本工程模型算子并不直接返回结果,而是将其保存在文件中
# 模型算子的输入均是1列经过定点量化后的numpy array(按照宽、高、通道的顺序排列)
# 这一过程亦可在MATLAB中进行
import numpy as np
import torch
# 导入自定义的算子
from nn_basic import NNConv2d, NNLinear, NNMaxPool2d


# 网络结构
# cifar10_net(
#   (net): Sequential(
#     (0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#     (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#     (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (4): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#     (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (6): Flatten(start_dim=1, end_dim=-1)
#     (7): Linear(in_features=1024, out_features=64, bias=True)
#     (8): Linear(in_features=64, out_features=10, bias=True)
#   )
# )




# (0): Conv2d
# 三通道卷积核,一共有out_channel(64)个
kernel_1 = torch.arange(1, 5 * 5 * 3 * 64 + 1)
kernel_1 = torch.reshape(kernel_1, (-1, 1))
kernel_1 = kernel_1.numpy()
# 三通道输入特征图
img_1 = torch.arange(0, 32 * 32 * 3)
img_1 = torch.reshape(img_1, (-1, 1))
img_1 = img_1.numpy()
# 偏置
bias_1 = torch.arange(1, 64 + 1)
bias_1 = torch.reshape(bias_1, (-1, 1))
bias_1 = bias_1.numpy()

# 例化卷积类
NNConv2d_1 = NNConv2d(name="NNConv2d_1", kernel_size_w=5, kernel_size_h=5, in_size_w=32, in_size_h=32,
                      stride_h=1, stride_w=1, in_channel=3, out_channel=64)
NNConv2d_1.conv_2d(kernel=kernel_1, img=img_1, bias=bias_1)

# 池化
img_2 = torch.arange(1, 28 * 28 * 64 + 1)  # 用作测试的img_2
img_2 = torch.reshape(img_2, (-1, 1))  # 默认的数据组织格式为1列的
img_2 = img_2.numpy()  # 转为numpy array
# 池化类
NNMaxPool2d_1 = NNMaxPool2d(in_channel=64, kernel_size=2, in_size_h=28, in_size_w=28, stride=2, name="NNMaxPool2d_1")
NNMaxPool2d_1.maxpool2d(img=img_2)


# 线性化
img = torch.arange(1, 1024 + 1)  # 用作测试的img_2
img = torch.reshape(img, (-1, 1))  # 默认的数据组织格式为1列的
img = img.numpy()  # 转为numpy array

weight = torch.arange(0, 1024 * 64)
weight = torch.reshape(weight, (-1, 1))
weight = weight.numpy()

bias = torch.arange(0, 64)
bias = torch.reshape(bias, (-1, 1))
bias = bias.numpy()

# 例化线性类
NNLinear_1 = NNLinear(in_channel=1024, out_channel=64, name="NNLinear_1")
NNLinear_1.linear(img=img, weight=weight, bias=bias)

