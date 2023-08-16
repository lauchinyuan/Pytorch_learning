# 用外部图片对训练好的模型进行功能测试
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d
from model import cifar10_net

# 读取测试图片
# image_path = "./image/cat2.jpg"
image_path = "./dataset/val/airplane/90.png"
img = Image.open(image_path)
img = img.convert("RGB")

# 对图像大小进行变换并转换其类型
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
img = transform(img)


# 加载已经训练好的模型
model = torch.load("model_200.pth", map_location=torch.device("cpu"))
# model = cifar10_net()
print(model)
# 将数据输入模型进行处理
img = torch.reshape(img, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(img)
print(output)
print(output.argmax(1))
