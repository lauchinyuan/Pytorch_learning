# 使用经过定点化后的模型参数构建神经网络前向路径
# 作为Verilog/FPGA/ASIC计算结果的参考
# 与pytorch自带的模型算子不同之处在于,本工程模型算子并不直接返回结果,而是将其保存在文件中
# 模型算子的输入均是1列经过定点量化后的numpy array(按照宽、高、通道的顺序排列)
# 这一过程亦可在MATLAB中进行
import PIL.Image
import numpy as np
import torch
import torchvision
from numpy import size

# 导入自定义的算子
from nn_basic import NNConv2d, NNLinear, NNMaxPool2d, NNPadding2d


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




# # (0): Conv2d
# # 从txt文件中读取三通道卷积核(8bit量化已完成)
# kernel_1 = np.loadtxt("net.0.weight_d.txt", dtype=np.int8)  # 工程中量化后的权重和偏置都使用8bit有符号数
# # 读取bias
# bias_1 = np.loadtxt("net.0.bias_d.txt", dtype=np.int8)
# # # 三通道输入特征图
# # img = PIL.Image.open("val/airplane/3.png")
#
# img_1 = torch.arange(0, 32 * 32 * 3)
# img_1 = torch.reshape(img_1, (-1, 1))
# img_1 = img_1.numpy()
#
# # 第一层卷积
# NNConv2d_1 = NNConv2d(name="NNConv2d_1", kernel_size_w=5, kernel_size_h=5, in_size_w=32, in_size_h=32,
#                       stride_h=1, stride_w=1, in_channel=3, out_channel=32)
# NNConv2d_1.conv_2d(kernel=kernel_1, img=img_1, bias=bias_1)
#
# # 读取卷积计算结果,并进行池化
# img_2 = np.loadtxt("NNConv2d_1_d.txt", dtype=np.int16)
#
# # 池化
# # img_2 = torch.arange(1, 28 * 28 * 64 + 1)  # 用作测试的img_2
# # img_2 = torch.reshape(img_2, (-1, 1))  # 默认的数据组织格式为1列的
# # img_2 = img_2.numpy()  # 转为numpy array
# # 池化类
# NNMaxPool2d_1 = NNMaxPool2d(in_channel=32, kernel_size=2, in_size_h=28, in_size_w=28, stride=2, name="NNMaxPool2d_1")
# NNMaxPool2d_1.maxpool2d(img=img_2)
#
# # 读取池化结果
# img_3 = np.loadtxt("NNMaxPool2d_1_d.txt", dtype=np.int16)

# # 线性化
# img = torch.arange(1, 1024 + 1)  # 用作测试的img_2
# img = torch.reshape(img, (-1, 1))  # 默认的数据组织格式为1列的
# img = img.numpy()  # 转为numpy array
#
# weight = torch.arange(0, 1024 * 64)
# weight = torch.reshape(weight, (-1, 1))
# weight = weight.numpy()
#
# bias = torch.arange(0, 64)
# bias = torch.reshape(bias, (-1, 1))
# bias = bias.numpy()
#
# # 例化线性类
# NNLinear_1 = NNLinear(in_channel=1024, out_channel=64, name="NNLinear_1")
# NNLinear_1.linear(img=img, weight=weight, bias=bias)
#







# 三通道输入特征图
img = PIL.Image.open("./dataset/val/airplane/90.png")
trans = torchvision.transforms.ToTensor()
img = trans(img)
img_2 = torch.round(img * 2**7)
img = torch.reshape(img, (-1, 1)).numpy()
img_2 = torch.reshape(img_2, (-1, 1)).numpy()
np.savetxt("img.txt", img)
np.savetxt("img_2.txt", img_2, fmt="%d")

# NNPadding2d_1 = NNPadding2d(in_size_w=32, in_size_h=32, in_channel=3, padding_h=2, padding_w=2, name="NNPadding2d")
# NNPadding2d_1.padding(img)







