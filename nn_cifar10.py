import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备数据集
train_data = torchvision.datasets.CIFAR10("./dataset", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=64)
train_loader = DataLoader(train_data, batch_size=64)


# 构建模型
class my_module(nn.Module):
    def __init__(self):
        super(my_module, self).__init__()
        # self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        # self.maxpool1 = MaxPool2d(kernel_size=2)
        # self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        # self.maxpool2 = MaxPool2d(kernel_size=2)
        # self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        # self.maxpool3 = MaxPool2d(kernel_size=2)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024, 64)
        # self.linear2 = Linear(64, 10)

        # (可选)将模型直接拼接起来
        self.module1 = Sequential(Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
                                  MaxPool2d(kernel_size=2),
                                  Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
                                  MaxPool2d(kernel_size=2),
                                  Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
                                  MaxPool2d(kernel_size=2),
                                  Flatten(),
                                  Linear(1024, 64),
                                  Linear(64, 10)
                                  )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.module1(x)
        return x


# 例化网络
cifar10_nn = my_module()
print(cifar10_nn)

# 处理数据
i = 0
for data in test_loader:
    if i==0:
        img, target = data
    imgs, targets = data
    nn_out = cifar10_nn(imgs)
    i = i+1
    # print(nn_out)


# 查看网络结构
writer = SummaryWriter("logs")
writer.add_graph(cifar10_nn, img)
writer.close()

