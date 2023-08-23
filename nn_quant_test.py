# 使用经过定点化后的模型进行相关实验测试
# 作为Verilog/FPGA/ASIC计算结果的参考
# 这一过程亦可在MATLAB中进行,但MATLAB不方便进行量化模型的模拟
import numpy as np
import torch
import torchvision.datasets
from torch import nn, functional
from torch.nn import Conv2d, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import cifar10_net
from torch.nn.quantized import Conv2d as QConv2d

# 从nn_quant_export.py导入相关参数
from nn_quant_export import scale_dict, quant_dict, bias_dict, zero_point_dict

# 量化后模型的加载, 需要与量化过程中保持一致的处理
model_fp32 = cifar10_net(is_quant=True)
state_dict = torch.load("./model_pth/model_int8.pth")
model_fp32.qconfig = torch.quantization.get_default_qconfig('x86')
load_model_prepared = torch.quantization.prepare(model_fp32)
model_int8 = torch.quantization.convert(load_model_prepared)
model_int8.load_state_dict(state_dict)
# 浮点模型加载, 作为对照
model_fp32 = torch.load("./model_pth/model_900.pth")

model_int8.eval()
model_fp32.eval()

# CIFAR10测试集
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=12)

# 给定一个小的测试数据
x, tar = test_data[7692]
x = x.reshape((1, 3, 32, 32))

# 给模型一个输入
y_int8 = model_int8(x)
# tensor([[ 48.0104,  12.5244,   2.0874, -10.4370,   2.0874, -12.5244, -18.7867,
#          -16.6993,  14.6119, -20.8741]])

x_q = torch.quantize_per_tensor(x, zero_point=quant_dict["quant.out.zero_point"],
                                scale=quant_dict["quant.out.scale"], dtype=torch.quint8)
x_q_dq = x_q.dequantize()
y_fp32 = model_fp32(x).detach()
# print(y_fp32)


# 得到模型处理的中间结果
quant_res = model_int8.quant_res  # scale=0.007874015718698502, zp=0
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
    if "conv1.weight" in key:
        weight_int = state_dict[key]  # int8 ([32, 3, 5, 5])
    if "conv1.bias" in key:
        bias_fp32 = state_dict[key]  # float32 torch.Size([32])

# # 将定点转换为浮点数来进行卷积,则卷积后的结果与未量化模型保持一致
# weight_int_dequant = weight_int.dequantize()
# # print(weight_int_dequant)
# # print(x_q_dq)
# net = Conv2d(in_channels=3, out_channels=32, padding=2, kernel_size=5)
# net.weight.data = weight_int_dequant
# net.bias.data = bias_fp32
# net_out = net(x_q_dq)
# net_out = torch.quantize_per_tensor(net_out, scale=quant_dict["conv1.out.scale"],
#                                     zero_point=quant_dict["conv1.out.zero_point"], dtype=torch.quint8)
# # net_out = net_out.dequantize()
# # print("_______________________net____________________")
# # print(net_out)
# # print("_______________________conv____________________")
# # print(conv1_res)
# error = (conv1_res.int_repr() - net_out.int_repr()).sum()
# print("_____________int_float_int_error_________________")
# print(error)

# # 使用量化的卷积op尝试对量化后的参数进行卷积运算, 结果一致
# net = QConv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
# s_dict = {"bias": bias_fp32, "weight": weight_int, "scale": quant_dict["conv1.out.scale"],
#           "zero_point": quant_dict["conv1.out.zero_point"]}
# net.load_state_dict(s_dict)
# conv_res = net(quant_res)
# # print("_______________________net____________________")
# # print(conv_res)
# # print("_______________________conv____________________")
# # print(conv1_res)
# conv_res_int = conv_res.int_repr()
# net_q = conv1_res.int_repr()
# error = (conv_res_int != net_q).sum()
# print("___________________quantized_op_error____________")
# print(error)

# # # 对量化后的整数数据减去其量化零点, 再卷积, 并进行scale调整, 结果与量化后基本上都是一致的
# # # 在FPGA中适合使用这个方案
# n = 16
# in_int = quant_res.int_repr()
# weight_int = weight_int.int_repr()
# s1 = scale_dict["conv1.in.scale"]
# s2 = quant_dict["conv1.weight.scale"]
# s3 = quant_dict["conv1.out.scale"]
# zp = quant_dict["conv1.out.zero_point"]
# # 0.004655691795051098 * 0.007874015718698502,
# bias_int = torch.quantize_per_tensor(bias_fp32, zero_point=0, scale=s1 * s2,
#                                      dtype=torch.qint32).int_repr()
# # print(bias_int)
# net = Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
# net.weight.data = weight_int.to(torch.float32)
# net.bias.data = bias_int.to(torch.float32)
# net_out = net(in_int.to(torch.float32))
# M = s1 * s2 / s3  # 未量化的真实M
# M0_fix = round(M * 2**n)
# M0 = M0_fix * 2**(-n)  # 经过量化后的M0
# net_q = np.round((M0 * net_out + zp).detach()).to(torch.uint8)
# # print("_______________________net____________________")
# # print(net_q)
# # print("_______________________conv____________________")
# conv1_res_int = conv1_res.int_repr()
# # print(conv1_res.int_repr())
# error = (conv1_res_int != net_q).sum()
# print("___________________scale_zero_int_error___________")
# print(error)


# 模拟量化模型的整数卷积conv2过程
# 参数获取
# weight_int = 0
# bias_fp32 = 0
# for key in state_dict:
#     if "conv2.weight" in key:
#         weight_int = state_dict[key]  # int8 ([32, 3, 5, 5])
#     if "conv2.bias" in key:
#         bias_fp32 = state_dict[key]  # float32 torch.Size([32])
# # # 对量化后的整数数据减去其量化零点, 再卷积, 并进行scale调整, 结果与量化后基本上都是一致的
# # # 在FPGA中适合使用这个方案
# in_int = maxpool1_res.int_repr() - zero_point_dict["conv2.in.zero_point"]
# in_int = in_int.to(torch.int8)
# weight_int = weight_int.int_repr()
# s1 = scale_dict["conv2.in.scale"]
# s2 = quant_dict["conv2.weight.scale"]
# s3 = quant_dict["conv2.out.scale"]
# zp = quant_dict["conv2.out.zero_point"]
# bias_int = torch.quantize_per_tensor(bias_fp32, zero_point=0, scale=s1 * s2,
#                                      dtype=torch.qint32).int_repr()
# # print(bias_int)
# net = Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
# net.weight.data = weight_int.to(torch.float32)
# net.bias.data = bias_int.to(torch.float32)
# net_out = net(in_int.to(torch.float32))
# M = s1 * s2 / s3
# net_q = np.round((M * net_out + zp).detach()).to(torch.uint8)
# print("_______________________net____________________")
# print(net_q)
# print("_______________________conv____________________")
# conv2_res_int = conv2_res.int_repr()
# # print(conv1_res.int_repr())
# error = (conv2_res_int != net_q).sum()
# print("___________________scale_zero_int_error___________")
# print(error)

# 对量化后的整数数据减去其量化零点, 进行第二次线性化Linear2的模拟, 结果基本一致
# 在FPGA中适合使用这个方案
# weight_int = 0
# bias_fp32 = 0
# for key in state_dict:
#     if "linear2._packed_params._packed_params" in key:
#         weight_int, bias_fp32 = state_dict[key]
# in_int = (linear1_res.int_repr() - 67).to(torch.int8)
# print(in_int)
# weight_int = weight_int.int_repr() - 0
# s1 = scale_dict["linear2.in.scale"]
# s2 = quant_dict["linear2.weight.scale"]
# s3 = quant_dict["linear2.out.scale"]
# zp = quant_dict["linear2.out.zero_point"]
# bias_int = torch.quantize_per_tensor(bias_fp32, zero_point=0, scale=s1 * s2,
#                                      dtype=torch.qint32).int_repr()
# net = Linear(in_features=64, out_features=10)
# # net = Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
# net.weight.data = weight_int.to(torch.float32)
# net.bias.data = bias_int.to(torch.float32)
# net_out = net(in_int.to(torch.float32))
# M = s1 * s2 / s3
# net_q = np.round((M * net_out + zp).detach()).to(torch.uint8)
# print("_______________________net____________________")
# print(net_q)
# print("_______________________conv____________________")
# linear2_res_int = linear2_res.int_repr()
# print(linear2_res_int)
# error = (linear2_res_int - net_q).sum()
# print("___________________scale_zero_int_error___________")
# print(error)






# # 探究不同scale量化位宽的影响, 经过测试, scale小数位为63位时可以实现零误差计算
# # 但硬件开销过于大,因此选择16bit小数位+0位整数位作为scale因子的定点量化方案
# s1 = 0.007874015718698502
# s2 = 0.004655691795051098
# s3 = 0.113490916788578031412
# M = s1*s2/s3
# print(M)
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

# # 实际scale量化效果测试, 未完成
# n = 16
# name = "conv1"
# s1 = scale_dict["{}.in.scale".format(name)]
# s2 = quant_dict["{}.weight.scale".format(name)]
# s3 = quant_dict["{}.out.scale".format(name)]
# M = s1*s2/s3  # 真实M_scale值
#
# # 权重及bias提取
# w = None
# b = None
# for key in state_dict:
#     if "{}.weight".format(name) in key:
#         w = state_dict[key].int_repr()
#     if "{}.bias".format(name) in key:
#         b = state_dict[key].detach()
# print(b)
# conv = Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
# conv.weight.data = w
# conv.bias.data = b.to(torch.float)







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

# 测试量化模型对于测试数据集少量图片的表现,获取模型最终输出
# test_loader = DataLoader(test_data, batch_size=12)
# i = 0
# for data in test_loader:
#     imgs, targets = data
#     fp32_res = model_fp32(imgs)
#     int8_res = model_int8(imgs)
#     fp32_prd_target = fp32_res.argmax(1)
#     int8_prd_target = int8_res.argmax(1)
#     if (int8_prd_target.equal(fp32_prd_target)) and (fp32_prd_target.equal(targets)):
#         print(fp32_prd_target)
#         print(int8_prd_target)
#         print(targets)
#         print(i)
#     i=i+1 # batch_size=12, i=641时连续12个预测值相同

