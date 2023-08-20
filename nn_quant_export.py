#  量化后模型参数导出, 以便在FPGA上实现推理
import numpy as np
import torch
from torch import nn

from model import cifar10_net

# 量化后模型的加载
model_fp32 = cifar10_net(is_quant=True)
state_dict = torch.load("./model_pth/model_int8.pth")
model_fp32.qconfig = torch.quantization.get_default_qconfig('x86')
load_model_prepared = torch.quantization.prepare(model_fp32)
model_int8 = torch.quantization.convert(load_model_prepared)
model_int8.load_state_dict(state_dict)
# print(model_int8)


# 量化后模型参数导出
for key in state_dict:
    # print(key)
    # if "weight" in key:  # 提取量化后权重到txt文件,文件保存为1列,
    #     weight_int = state_dict[key].int_repr()  # 提取整数表达
    #     # print(key)
    #     # print(weight_int)
    #     weight_int = torch.reshape(weight_int, (-1, 1))  # 重排为1列格式
    #
    #     # 转成uint8是因为np.savetxt只支持无符号16进制保存,实际上二进制值并不改变
    #     weight_int_np = weight_int.numpy().astype("uint8")
    #     np.savetxt("./txt/{}_qint8.txt".format(key), weight_int_np, fmt="%02x")

    # 获取量化参数 Zero_point & scale, 参数量小, print并记录到字典中
    if "scale" in key:
        print("{}:{}".format(key, state_dict[key]))
    if "zero_point" in key:
        print("{}:{}".format(key, state_dict[key]))

    # # 某些层的模型参数形式是元组
    # if ("_packed_params" in key) and ("dtype" not in key):
    #     w, b = state_dict[key]
    #     # print("shape of weights: {}".format(w.shape))
    #     # print("shape of bias: {}".format(b.shape))
    #     # print("type of weights: {}".format(w.dtype))
    #     # print("type of bias: {}".format(b.dtype))
    #
    #     # 对权重进行处理, 保存为1列txt
    #     weight_int = torch.reshape(w.int_repr(), (-1, 1))
    #     weight_int_np = weight_int.numpy().astype("uint8")
    #     # print(weight_int_np)
    #     np.savetxt("./txt/" + key.replace("_packed_params._packed_params", "") + "weight_qint8.txt",
    #                weight_int_np, fmt="%02x")



# 对于bias的处理需要在其他层的scale参数提取完成后进行
# for key in state_dict:
#     # bias依然是float32类型的,但在实际计算时会转换为int32类型
#     if "bias" in key:
#         print("{}:{}".format(key, state_dict[key]))
#
#     if key == "conv1.bias":
#         print(state_dict[key])
#         bias_float32 = state_dict[key]
#         S1 = 0.0079
#         S2 = 0.04326891526579857
#         bias_int32 = torch.quantize_per_tensor(bias_float32, scale=S1 * S2, zero_point=0, dtype=torch.qint32)
#         print(bias_int32.int_repr())
#     # 某些层的模型参数需要解压缩才能查看
#     if ("_packed_params" in key) and ("dtype" not in key):
#         w, b = state_dict[key]
#
#         # 对权重进行处理, 保存为1列txt
#         weight_int = torch.reshape(w.int_repr(), (-1, 1))
#         weight_int_np = weight_int.numpy().astype("uint8")
#         # print(weight_int_np)
#         np.savetxt("./txt/" + key.replace("_packed_params._packed_params", "") + "weight_qint8.txt",
#                    weight_int_np, fmt="%02x")


# for name, module in model_int8.named_modules():
#     print(name)



