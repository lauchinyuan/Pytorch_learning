import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d


# 搭建神经网络
class cifar10_net(nn.Module):
    def __init__(self):
        super(cifar10_net, self).__init__()
        self.net = Sequential(Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
                              MaxPool2d(kernel_size=2),
                              Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
                              MaxPool2d(kernel_size=2),
                              Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
                              MaxPool2d(kernel_size=2),
                              nn.Flatten(),
                              nn.Linear(in_features=1024, out_features=64),
                              nn.Linear(in_features=64, out_features=10)
                              )

    def forward(self, x):
        y = self.net(x)
        return y


# 模型正确性的简单测试
if __name__ == "__main__":
    model = cifar10_net()
    input = torch.ones(64, 3, 32, 32)  # batch_size为64, 3通道
    output = model(input)
    print(output.shape)