# 使用经过定点化后的模型参数构建神经网络前向路径
# 作为Verilog/FPGA/ASIC计算结果的参考
# 这一过程亦可在MATLAB中进行,但MATLAB不方便进行量化模型的模拟
import numpy as np
import torch
from torch import nn, functional
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter

from model import cifar10_net
from torch.nn.quantized import Conv2d as QConv2d

# 量化后模型的加载, 需要与量化过程中保持一致的处理
model_fp32 = cifar10_net(is_quant=True)
state_dict = torch.load("./model_pth/model_int8.pth")
model_fp32.qconfig = torch.quantization.get_default_qconfig('x86')
load_model_prepared = torch.quantization.prepare(model_fp32)
model_int8 = torch.quantization.convert(load_model_prepared)
model_int8.load_state_dict(state_dict)

# 浮点模型加载, 作为对照
model_fp32 = torch.load("./model_pth/model_900.pth")

# 给定一个小的测试数据
x = torch.arange(0, 3 * 32 * 32).reshape((1, 3, 32, 32))
x = x.type(torch.FloatTensor)
x = x * 0.00001
# print("x: {}".format(x))

# 给模型一个输入
model_int8(x)

x_q = torch.quantize_per_tensor(x, zero_point=0, scale=0.007874015718698502, dtype=torch.quint8)
x_q_dq = x_q.dequantize()
model_fp32(x_q_dq)

# 得到模型处理的中间结果
# for key, value in model_int8.__dict__.items():
#     # if "quant_res" in key:  # 中间结果均含有res关键词
#     # if key == "quant_res":
#     #     print(key, value.int_repr())

quant_res = model_int8.quant_res  # scale=0.007874015718698502, zp=0
# conv1_weight: scale=0.004655691795051098, zero_point=0
conv1_res = model_int8.conv1_res  # scale=0.11349091678857803,zero_point=75
maxpool1_res = model_int8.maxpool1_res  # scale=0.11349091678857803, zero_point=75
conv2_res = model_int8.conv2_res  # scale=0.6031823754310608,zero_point=68
maxpool2_res = model_int8.maxpool2_res  # scale=0.6031823754310608, zero_point=68
conv3_res = model_int8.conv3_res  # scale=1.0444879531860352,zero_point=62
maxpool3_res = model_int8.maxpool3_res  # scale=1.0444879531860352, zero_point=62
flatten_res = model_int8.flatten_res  # scale=1.0444879531860352, zero_point=62
linear1_res = model_int8.linear1_res  # scale=0.4158819019794464, zero_point=67
linear2_res = model_int8.linear2_res  # scale=2.0874075889587402, zero_point=52, torch.quint8
dequant_res = model_int8.dequant_res

# fp32模型的中间结果
fp32_conv1_res = model_fp32.conv1_res
fp32_maxpool1_res = model_fp32.maxpool1_res
fp32_conv2_res = model_fp32.conv2_res
fp32_maxpool2_res = model_fp32.maxpool2_res
fp32_conv3_res = model_fp32.conv3_res
fp32_maxpool3_res = model_fp32.maxpool3_res
fp32_flatten_res = model_fp32.flatten_res
fp32_linear1_res = model_fp32.linear1_res
fp32_linear2_res = model_fp32.linear2_res

# 中间结果打印
# print("quant_res: {}".format(quant_res))
# print("conv1_res: {}".format(conv1_res))
# print("maxpool1_res: {}".format(maxpool1_res))
# print("conv2_res: {}".format(conv2_res))
# print("maxpool2_res: {}".format(maxpool2_res))
# print("conv3_res: {}".format(conv3_res))
# print("maxpool3_res: {}".format(maxpool3_res))
# print("flatten_res: {}".format(flatten_res))
# print("linear1_res: {}".format(linear1_res))
# print("linear2_res: {}".format(linear2_res))
# print("dequant_res: {}".format(dequant_res))


# 模拟量化模型的整数卷积过程
# 参数获取
weight_int = 0
bias_fp32 = 0
for key in state_dict:
    # print(key)
    if "conv1.weight" in key:  # 提取量化后权重到txt文件,文件保存为1列,
        weight_int = state_dict[key]  # int8 ([32, 3, 5, 5])
    if "conv1.bias" in key:
        bias_fp32 = state_dict[key]  # float32 torch.Size([32])

# 将定点转换为浮点数来进行卷积,则卷积后的结果与未量化模型保持一致
# weight_int_dequant = weight_int.dequantize()
# print(weight_int_dequant)
# print(x_q_dq)
# net = Conv2d(in_channels=3, out_channels=32, padding=2,  kernel_size=5)
# net.weight.data = weight_int_dequant
# net.bias.data = bias_fp32
# net_out = net(x_q_dq)
# net_out = torch.quantize_per_tensor(net_out, scale=0.11349091678857803, zero_point=75, dtype=torch.quint8)
# # net_out = net_out.dequantize()
# print("_______________________net____________________")
# print(net_out)
# print("_______________________conv____________________")
# print(conv1_res)
#
# error = (conv1_res.int_repr() - net_out.int_repr()).sum()
# print(error)

# # 使用量化的卷积op尝试对量化后的参数进行卷积运算, 结果一致
# net = QConv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
# s_dict = {"bias": bias_fp32, "weight": weight_int, "scale": 0.11349091678857803, "zero_point": 75}
# net.load_state_dict(s_dict)
# conv_res = net(quant_res)
# print("_______________________net____________________")
# print(conv_res)
# print("_______________________conv____________________")
# print(conv1_res)

# 对量化后的整数数据减去其量化零点, 再卷积, 并进行scale调整, 结果与量化后一致
# in_int = quant_res.int_repr()
# weight_int = weight_int.int_repr()
# bias_int = torch.quantize_per_tensor(bias_fp32, zero_point=0, scale=0.004655691795051098 * 0.007874015718698502,
#                                      dtype=torch.qint32).int_repr()
# print(bias_int)
# net = Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
# net.weight.data = weight_int.to(torch.float32)
# net.bias.data = bias_int.to(torch.float32)
# net_out = net(in_int.to(torch.float32))
# M = 0.004655691795051098*0.007874015718698502/0.11349091678857803
# net_q = np.round((M * net_out + 75).detach()).to(torch.uint8)
# print("_______________________net____________________")
# print(net_q)
# print("_______________________conv____________________")
# conv1_res_int = conv1_res.int_repr()
# print(conv1_res.int_repr())
# error = (conv1_res_int != net_q).sum()
# print(error)


# 探究不同scale量化位宽的影响, 经过测试, scale小数位为63位时可以实现零误差计算
# 但硬件开销过于大,因此选择16bit小数位+0位整数位作为scale因子的定点量化方案
s1 = 0.007874015718698502
s2 = 0.004655691795051098
s3 = 0.113490916788578031412
M = s1*s2/s3
print(M)
# writer = SummaryWriter("logs")
# step = 0
# for n in range(64):
#     M_int = round(M * 2**n)
#     M0 = M_int * (2**(-n))
#     error = abs(M0 - M)
#     writer.add_scalar("error", error, step)
#     writer.add_scalar("fixed vs. float", M, step)
#     writer.add_scalar("fixed vs. float", M0, step)
#     print(error)
#     print("n: {}, error:{}".format(n, (error/M)*100))
#     step = step + 1
# writer.close()


# 实际的scale量化方案下的卷积过程模拟
# in_int = quant_res.int_repr()
# weight_int = weight_int.int_repr()
# bias_int = torch.quantize_per_tensor(bias_fp32, zero_point=0, scale=0.004655691795051098 * 0.007874015718698502,
#                                      dtype=torch.qint32).int_repr()
# net = Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
# net.weight.data = weight_int.to(torch.float32)
# net.bias.data = bias_int.to(torch.float32)
# net_out = net(in_int.to(torch.float32))
# s1 = 0.007874015718698502
# s2 = 0.004655691795051098
# s3 = 0.11349091678857803
# M = s1 * s2 / s3
# n = 16  # 量化小数位数
# M_fix = int(M * 2 ** n)
# M0 = M_fix * 2**(-n)  # 量化后实际表示的scale值
# net_q = (((M_fix * net_out).to(torch.int64) >> n) + 75).to(torch.uint8)
# # net_q = np.round((M * net_out + 75).detach()).to(torch.uint8)
# print("_______________________net____________________")
# print(net_q)
# print("_______________________conv____________________")
# conv1_res_int = conv1_res.int_repr()
# print(conv1_res.int_repr())
# error = (conv1_res_int != net_q).sum()
# print("n: {}, error: {}, error rate:{}".format(n, error, error / (32 * 32 * 32)))





# 数据类型
# zero_point: int64
# scale: float32
# weight: qint8
# bias: float32

# 量化数据
# quant.scale:tensor([0.0079])
# quant.zero_point:tensor([0])
# conv1.scale:0.11349091678857803
# conv1.zero_point:75
# conv2.scale:0.6031823754310608
# conv2.zero_point:68
# conv3.scale:1.0444879531860352
# conv3.zero_point:62
# linear1.scale:0.4158819019794464
# linear1.zero_point:67
# linear2.scale:2.0874075889587402
# linear2.zero_point:52


# 量化模型结构
# conv1.weight
# conv1.bias
# conv1.scale
# conv1.zero_point
# conv2.weight
# conv2.bias
# conv2.scale
# conv2.zero_point
# conv3.weight
# conv3.bias
# conv3.scale
# conv3.zero_point
# linear1.scale
# linear1.zero_point
# linear1._packed_params.dtype
# linear1._packed_params._packed_params
# linear2.scale
# linear2.zero_point
# linear2._packed_params.dtype
# linear2._packed_params._packed_params
# quant.scale
# quant.zero_poin
