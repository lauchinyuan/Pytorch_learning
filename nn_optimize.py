import torch
import torchvision.datasets
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

test_data = torchvision.datasets.CIFAR10("./dataset", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(test_data, batch_size=64)


class my_module(nn.Module):
    def __init__(self):
        super(my_module, self).__init__()

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
        x = self.module1(x)
        return x


network = my_module()
loss = nn.CrossEntropyLoss()  # 损失函数
optim = torch.optim.SGD(network.parameters(), lr=0.01)  # 例化优化器

# 对每组数据进行20轮学习
for epoch in range(20):
    total_loss = 0
    for data in dataloader:
        imgs, targets = data
        output = network(imgs)
        result_loss = loss(output, targets)
        optim.zero_grad()  # 将优化器梯度设置为0
        result_loss.backward()  # 反向传播, 可以计算出梯度
        optim.step()  # 调用优化器对参数进行调优
        total_loss = total_loss + result_loss
    print(total_loss)