import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 自定义数据的池化例子
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)  # dtype代表输入的数据类型
input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)
class my_module(nn.Module):
    def __init__(self):
        super(my_module, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)  # ceil模式下窗口有效数据不足也会继续进行最大池化
    def forward(self, input):
        output = self.maxpool1(input)
        return output

pool_module = my_module()
output = pool_module(input)
print(output)

# 对数据集图片的池化例子
dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())
loader = DataLoader(dataset, batch_size=64)
writer = SummaryWriter("logs")
writer_ind = 0
for data in loader:
    img, target = data
    out = pool_module(img)
    writer.add_images("input", img, writer_ind)
    writer.add_images("output", out, writer_ind)
    writer_ind = writer_ind + 1
writer.close()