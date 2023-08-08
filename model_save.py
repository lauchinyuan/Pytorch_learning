import torch
import torchvision.models
from torch import nn
from torch.nn import Conv2d, Sequential, MaxPool2d

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1,保存模型结构和参数
torch.save(vgg16, "vgg16.pth")

# 保存方式2,保存为字典,只保留参数
torch.save(vgg16.state_dict(), "vgg16_dict.pth")


# 自定义模型保存1
class my_module(nn.Module):
    def __init__(self):
        super(my_module, self).__init__()
        self.module1 = Sequential(Conv2d(in_channels=3, out_channels=12, kernel_size=3),
                                   MaxPool2d(kernel_size=3))

    def forward(self, x):
        y = self.module1(x)
        return y

# 保存自定义模型结构和参数
# 注意：在加载时仍无法直接加载模型结构,需要在加载时定义模型,详见model_load.py
module = my_module()
torch.save(module, "module.pth")
