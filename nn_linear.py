import torch
import torchvision.datasets
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset", download=True, transform=torchvision.transforms.ToTensor(),
                                       train=False)
loader = DataLoader(dataset, batch_size=64)

# 定义网络
class my_module(nn.Module):
    def __init__(self):
        super(my_module, self).__init__()
        self.linear = Linear(196608, 1024)
    def forward(self, input):
        output = self.linear(input)
        return output

# 例化网络
linear = my_module()

for data in loader:
    imgs, targets = data
    imgs = torch.flatten(imgs)  # 将图像摊平
    lin_res = linear(imgs)
    print(lin_res.shape)