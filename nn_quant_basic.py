# 模拟量化神经网络中的各种OP(卷积、池化等)，利用量化后的定点数方案
import numpy as np
import torch
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from nn_quant_export import zero_point_dict, bias_dict, fix_point_dict, quant_dict, scale_dict


# 从txt读取量化后的激活,暂时只读取第n_img张图片
def read_img(file_path, n_img, img_channel, img_size_h, img_size_w):
    # 索引特定的文件位置
    idx_start = (n_img - 1) * img_channel * img_size_h * img_size_w
    idx_end = n_img * img_channel * img_size_h * img_size_w
    with open(file_path, "r") as file:
        hex_strings = file.readlines()
    hex_strings = [s.replace("\n", "") for s in hex_strings]  # 去掉列表中字符串的\n
    hex_int = [int(s, 16) for s in hex_strings]
    img = np.array(hex_int[idx_start:idx_end], dtype="uint8")
    return torch.from_numpy(img).reshape((1, img_channel, img_size_h, img_size_w))


# 从txt文件读取量化后的权重, 返回相应大小的tensor
def read_conv_weight(file_path, kernel_size, in_channels, out_channels):
    with open(file_path, 'r') as file:
        hex_strings = file.read().splitlines()

    # 将16进制字符串转换为整数(uint8)
    hex_values = [int(hex_str, 16) for hex_str in hex_strings]
    # 创建numpy数组
    weight = np.array(hex_values, dtype="int8")
    return torch.from_numpy(weight).reshape((out_channels, in_channels,
                                             kernel_size, kernel_size))


def read_bias(file_path):
    with open(file_path, "r") as file:
        hex_string = file.read().splitlines()
    hex_value = [int(s, 16) for s in hex_string]
    hex_np = np.array(hex_value, dtype=np.uint32)  # 直接使用np.int32报错,故先转为uint32,再变为int32
    hex_np = hex_np.astype(np.int32)
    hex_tensor = torch.from_numpy(hex_np)
    return hex_tensor


# 从txt文件读取线性层量化后的权重, 返回相应大小的tensor
def read_linear_weight(file_path, in_channels, out_channels):
    with open(file_path, 'r') as file:
        hex_strings = file.read().splitlines()

    # 将16进制字符串转换为整数(uint8)
    hex_values = [int(hex_str, 16) for hex_str in hex_strings]
    # 创建numpy数组
    weight_conv1 = np.array(hex_values, dtype="int8")
    return torch.from_numpy(weight_conv1).reshape((out_channels, in_channels))


# 定点量化卷积模拟function,会依据name的值自动访问相关的字典获得zero_point、scale和bias
# 要求这些字典的名称和key值符合本程序设计的规范
# n是s1*s2/s3的定点量化小数点数,默认为16
def q_conv(img, w, b, name, in_channels, out_channels, kernel_size, padding, n):
    # 权重和激活分别减去对应的zero_point, 进行2D卷积
    # 注意激活转换完成后要转为(int8)类型,因为有负数的情况
    img_sub_zp = (img - zero_point_dict["{}.in.zero_point".format(name)])
    img_sub_zp = img_sub_zp.to(torch.int8)
    w_sub_zp = w - quant_dict["{}.weight.zero_point".format(name)]
    img_sub_zp = img_sub_zp.to(torch.float)

    conv = Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=kernel_size, padding=padding)
    conv.weight.data = w_sub_zp.to(torch.float)  # 设置卷积权重
    conv.bias.data = b.to(torch.float)  # 设置bias
    conv_res = conv(img_sub_zp).to(torch.int32)  # 卷积
    # 对卷积后的结果乘以量化后的(s1*s2/s3)定点数,并舍去小数部分
    scale_conv_res = ((2**(-n) * fix_point_dict["{}.fix.scale".format(name)]) * conv_res).to(torch.int8)
    # 加上输出对应的zero_point, 即为量化卷积计算的结果
    return (scale_conv_res + quant_dict["{}.out.zero_point".format(name)]).to(torch.uint8)


# 最大池化,量化版本和非量化版本一致
def q_max_pooling(kernel_size, img):
    max_pooling = MaxPool2d(kernel_size=2)
    max_pool_res = max_pooling(img.to(torch.float))
    return max_pool_res.to(torch.uint8)


# 平铺
def q_flatten(img):
    flatten = Flatten()
    return flatten(img)


# 量化线性层
def q_linear(w, b, img, name, in_features, out_features, n):
    w_sub_zp = w - quant_dict["{}.weight.zero_point".format(name)]
    in_sub_zp = img - zero_point_dict["{}.in.zero_point".format(name)]
    in_sub_zp = in_sub_zp.to(torch.int8)
    linear = Linear(in_features=in_features, out_features=out_features)
    linear.bias.data = b.to(torch.float)
    linear.weight.data = w_sub_zp.to(torch.float)
    out = linear(in_sub_zp.to(torch.float)).to(torch.int32).detach()
    # 乘以对应的量化后的(s1*s2/s3)scale,并保留整数位,加上输出的zero_point即为最终数据
    scale_out = ((2**(-n) * fix_point_dict["{}.fix.scale".format(name)]) * out) + quant_dict[
        "{}.out.zero_point".format(name)]
    return scale_out.to(torch.uint8)
