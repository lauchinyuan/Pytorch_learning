# 模拟前向网络时用到的基本算子
# 模型算子的输入均是1列经过定点量化后的numpy array(按照宽、高、通道的顺序排列)
import torch
import numpy as np
# 定义一个计算2维卷积的类,其输入的kernel和img都是1列的,输出结果也是1列,文件名可自定义
class NNConv2d:
    def __init__(self, kernel_size_w, kernel_size_h, in_size_w, in_size_h, stride_w, stride_h, in_channel, out_channel,
                 name):
        self.kernel_size_w = kernel_size_w  # 卷积核高度
        self.kernel_size_h = kernel_size_h  # 卷积核宽度
        self.in_size_w = in_size_w  # 输出特征图宽度
        self.in_size_h = in_size_h  # 输出特征图高度
        self.stride_w = stride_w  # 横向步长
        self.stride_h = stride_h  # 纵向步长
        self.in_channel = in_channel  # 输入通道数
        self.out_channel = out_channel  # 输出通道数
        self.out_size_w = int(((in_size_w - kernel_size_w) / stride_w) + 1)  # 输出特征图宽度
        self.out_size_h = int(((in_size_h - kernel_size_h) / stride_h) + 1)  # 输出特征图高度
        self.name = name  # 保存数据的文件名

    def conv_2d(self, kernel, img, bias):
        # 初始化输出特征图,原本的输出特征图应该是三维tensor(out_channel(通道)*out_size_h(高度)*out_size_w(宽度))
        # 数据展开为1列,与FPGA中的结果保持一致
        ofm = torch.zeros(self.out_channel * self.out_size_h * self.out_size_w, 1)
        ofm = ofm.numpy().astype(int)
        for oc in range(self.out_channel):  # 遍历输出特征图通道,也是更换不同的卷积核
            for oh in range(self.out_size_h):  # 遍历输出特征图的每一行
                for ow in range(self.out_size_w):  # 输出特征图的一行内的数据
                    sum_conv = 0  # 卷积累加值清零
                    for ic in range(self.in_channel):  # 遍历输入通道
                        for kh in range(self.kernel_size_h):  # 遍历卷积核的每一行
                            for kw in range(self.kernel_size_w):  # 卷积核遍历行内数据
                                # 乘积累加
                                sum_conv = sum_conv + kernel[oc * self.in_channel * self.kernel_size_w *
                                                             self.kernel_size_h + ic * self.kernel_size_h *
                                                             self.kernel_size_w + (kh * self.kernel_size_w) + kw] * \
                                           img[ic * self.in_size_w * self.in_size_h + (oh * self.in_size_w) +
                                               ow + (kh * self.in_size_w) + kw]
                    # 一个多通道卷积窗口计算完成,将卷积值加上偏置保存到ofm(输出特征图)
                    ofm[oc * self.out_size_w * self.out_size_h + (oh * self.out_size_w) + ow] = sum_conv + bias[oc]
        # 卷积计算完成, 写入文件
        np.savetxt("{}.txt".format(self.name), ofm, fmt="%d")


# 定义最大池化类
class NNMaxPool2d:
    def __init__(self, in_channel, in_size_w, in_size_h, stride, kernel_size, name):
        self.in_channel = in_channel
        self.in_size_w = in_size_w
        self.in_size_h = in_size_h
        self.stride = stride
        self.kernel_size = kernel_size
        self.out_size_w = int(((in_size_w - kernel_size) / stride) + 1)
        self.out_size_h = int(((in_size_h - kernel_size) / stride) + 1)
        self.name = name

    def maxpool2d(self, img):
        ofm = torch.zeros((self.out_size_w * self.out_size_h * self.in_channel, 1))
        ofm = ofm.numpy().astype(int)
        for ic in range(self.in_channel):  # 遍历输入特征图的通道
            for oh in range(self.out_size_h):  # 遍历输出特征图的行
                for ow in range(self.out_size_w):  # 遍历输出特征图行内的每一个数据
                    win_max = 0
                    for kh in range(self.kernel_size):  # 遍历窗口的行
                        for kw in range(self.kernel_size):  # 遍历行内数据
                            if kh == 0 and kw == 0:  # 窗口第一次遍历时,令最先遍历到的值为最大值
                                win_max = img[ic * self.in_size_w * self.in_size_h + oh * self.stride * self.in_size_w +
                                              ow * self.stride + kh * self.in_size_w + kw]
                            elif img[ic * self.in_size_w * self.in_size_h + oh * self.stride * self.in_size_w + ow *
                                     self.stride + kh * self.in_size_w + kw] > win_max:
                                win_max = img[ic * self.in_size_w * self.in_size_h + oh * self.stride * self.in_size_w +
                                              ow * self.stride + kh * self.in_size_w + kw]
                            else:
                                win_max = win_max
                    # 一个窗口寻找完成,保存结果
                    ofm[ic * self.out_size_h * self.out_size_w + oh * self.out_size_w + ow] = win_max
        np.savetxt("{}.txt".format(self.name), ofm, fmt="%02x")


class NNLinear:
    def __init__(self, in_channel, out_channel, name):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.name = name

    def linear(self, img, weight, bias):
        # 开辟输出特征图空间
        ofm = torch.zeros((self.out_channel, 1))
        ofm = ofm.numpy().astype(np.int64)
        for oc in range(self.out_channel):
            linear_sum = 0
            for ic in range(self.in_channel):
                linear_sum = linear_sum + weight[oc * self.in_channel + ic] * img[ic]
            # 累加完成,加上偏置并保存到ofm
            ofm[oc] = linear_sum + bias[oc]
        np.savetxt("{}.txt".format(self.name), ofm, fmt="%d")
