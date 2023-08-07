import torch
import torchvision.datasets
from torch import conv2d, nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 基本卷积过程
# 输入特征
my_input = torch.tensor([[1, 2, 3, 4, 1],
                         [2, 5, 2, 1, 3],
                         [1, 0, 0, 1, 0],
                         [1, 0, 0, 8, 6],
                         [1, 0, 1, 0, 0]])

# 卷积核
kernel = torch.tensor([[1, 2, 3],
                       [1, 1, 0],
                       [1, 1, 2]])

# 尺度变换
my_input = torch.reshape(my_input, (1, 1, 5, 5))  # (1, 1)代表(batch_size, channel)
kernel = torch.reshape(kernel, (1, 1, 3, 3))

# 查看尺寸
print(my_input.shape)
print(kernel.shape)

# 卷积变换
output = conv2d(my_input, kernel, stride=1)
print(output)

# 对实际图像进行卷积
# 数据集
dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
# dataloader
loader = DataLoader(dataset, batch_size=64)


# 搭建神经网络
class my_conv2d(nn.Module):
    def __init__(self):
        super(my_conv2d, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1)

    def forward(self, x):
        y = self.conv1(x)
        return y


# 实例化模型
my_module = my_conv2d()






writer = SummaryWriter("logs")
index = 0
for data in loader:
    imgs, targets = data
    output = my_module(imgs)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("input", imgs, index)
    writer.add_images("output", output, index)
    index = index + 1

writer.close()