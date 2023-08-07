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


loss = nn.CrossEntropyLoss()

network = my_module()
for data in dataloader:
    imgs, targets = data
    output = network(imgs)
    # print(output)
    # print(targets)
    result_loss = loss(output, targets)
    print(result_loss)
    result_loss.backward()  # 反向传播, 可以计算出梯度
