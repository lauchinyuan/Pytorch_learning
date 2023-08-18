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
    if "weight" in key:  # 提取量化后权重到txt文件,文件保存为1列,
        weight_int = state_dict[key].int_repr()  # 提取整数表达
        # print(key)
        # print(weight_int)
        weight_int = torch.reshape(weight_int, (-1, 1))  # 重排为1列格式

        # 转成uint8是因为np.savetxt只支持无符号16进制保存,实际上二进制值并不改变
        weight_int_np = weight_int.numpy().astype("uint8")
        np.savetxt("./txt/{}_qint8.txt".format(key), weight_int_np, fmt="%02x")

    # 获取量化参数 Zero_point & scale, 参数量小, print记录即可
    if "scale" in key:
        print("{}:{}".format(key, state_dict[key]))
    if "zero_point" in key:
        print("{}:{}".format(key, state_dict[key]))

    # bias依然是float32类型的,但在实际计算时会转换为int32类型
    if "bias" in key:
        print("{}:{}".format(key, state_dict[key]))

    # 某些层的模型参数需要解压缩才能查看
    if ("_packed_params" in key) and ("dtype" not in key):
        w, b = state_dict[key]
        # print("shape of weights: {}".format(w.shape))
        # print("shape of bias: {}".format(b.shape))
        # print("type of weights: {}".format(w.dtype))
        # print("type of bias: {}".format(b.dtype))

        # 对权重进行处理, 保存为1列txt
        weight_int = torch.reshape(w.int_repr(), (-1, 1))
        weight_int_np = weight_int.numpy().astype("uint8")
        # print(weight_int_np)
        np.savetxt("./txt/" + key.replace("_packed_params._packed_params", "") + "weight_qint8.txt",
                   weight_int_np, fmt="%02x")







# zero_point: int64
# scale: float32
# weight: qint8
# bias: float32


# Zero_point & scale
# conv1.scale:0.04326891526579857
# conv1.zero_point:75
# conv2.scale:0.08344896882772446
# conv2.zero_point:53
# conv3.scale:0.10633488744497299
# conv3.zero_point:61
# linear1.scale:0.08185768127441406
# linear1.zero_point:56
# linear2.scale:0.09722229838371277
# linear2.zero_point:61
# quant.scale:tensor([0.0079])
# quant.zero_point:tensor([0])


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
