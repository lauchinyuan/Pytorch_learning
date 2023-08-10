# 用外部图片对训练好的模型进行功能测试
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d

# 读取测试图片
image_path = "./image/horse.jpg"
img = Image.open(image_path)
img = img.convert("RGB")

# 对图像大小进行变换并转换其类型
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
img = transform(img)


# 定义网络模型结构
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


# 加载已经训练好的模型
model = torch.load("model_19.pth", map_location=torch.device("cpu"))
print(model)

# 将数据输入模型进行处理
img = torch.reshape(img, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(img)
print(output)
print(output.argmax(1))
