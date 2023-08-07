import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1.0, -0.2],
                      [5.2, -5.2]])
input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)


# 定义模型类,从父类中继承
class my_relu(nn.Module):
    def __init__(self):
        super(my_relu, self).__init__()
        self.relu = ReLU(inplace=False)  # inplace设置为False,代表不对input进行替换,而是返回ReLU的结果

    def forward(self, input):
        relu_res = self.relu(input)
        return relu_res

# sigmoid类
class my_sigmoid(nn.Module):
    def __init__(self):
        super(my_sigmoid, self).__init__()
        self.sigmoid = Sigmoid()  # inplace设置为False,代表不对input进行替换,而是返回ReLU的结果

    def forward(self, input):
        sigmoid_res = self.sigmoid(input)
        return sigmoid_res

# 实例化模型
relu = my_relu()
sigmoid = my_sigmoid()
output = relu(input)  # 处理数据
print(input)
print(output)


# 处理数据集数据
dataset = torchvision.datasets.CIFAR10("./dataset", download=True, train=False,
                                       transform=torchvision.transforms.ToTensor())
loader = DataLoader(dataset, batch_size=64)
writer = SummaryWriter("logs")
writer_ind = 0
for data in loader:
    imgs, targets = data
    relu_res = relu(imgs)
    sigmoid_res = sigmoid(imgs)
    writer.add_images("input", imgs, writer_ind)
    writer.add_images("relu_output", relu_res, writer_ind)
    writer.add_images("sigmoid_output", sigmoid_res, writer_ind)
    writer_ind = writer_ind + 1
writer.close()
