import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d

# 对应方法1读取
vgg16 = torch.load("vgg16.pth")
print(vgg16)

#对应方法2读取
vgg16 = torchvision.models.vgg16(pretrained=False) 
vgg16_dict = torch.load("vgg16_dict.pth")
vgg16.load_state_dict(vgg16_dict)
print(vgg16_dict)
print(vgg16)

# 自定义模型加载
# 定义模型结构,但暂时无需例化
class my_module(nn.Module):
    def __init__(self):
        super(my_module, self).__init__()
        self.module1 = Sequential(Conv2d(in_channels=3, out_channels=12, kernel_size=3),
                                   MaxPool2d(kernel_size=3))

    def forward(self, x):
        y = self.module1(x)
        return y

model  = torch.load("module.pth")
print(model)