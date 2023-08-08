import torchvision
from torch import nn

# train_set = torchvision.datasets.ImageNet("./datasets", split="train", download=True,
#                                           transform=torchvision.transforms.ToTensor())

train_set = torchvision.datasets.CIFAR10("./dataset", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())

# 导入内建模型
# 没有进行预训练的模型
vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print(vgg16_true)

# VGG16默认分类是1000类,将其改为分为10类的网络,以便分析CIFAR10数据集
# # 在后面添加线性层
# vgg16_true.add_module("add_linear", nn.Linear(in_features=1000, out_features=10))
# print(vgg16_true)

# # 在标签为classifier的组添加模型
# vgg16_true.classifier.add_module("linear_after_classifier", nn.Linear(in_features=1000, out_features=10))
# print(vgg16_true)

# 修改特定位置的模型
vgg16_true.classifier[6] = nn.Linear(in_features=4096, out_features=10)
print(vgg16_true)


