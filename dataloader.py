import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 导入数据集
test_set = torchvision.datasets.CIFAR10("./dataset", train=False, download=True, transform=torchvision.transforms.\
                                        ToTensor())
train_set = torchvision.datasets.CIFAR10("./dataset", train=True, download=True, transform=torchvision.transforms.\
                                         ToTensor())
# 实例化加载器对象,每次获取4个数据,对其img和target进行打包
loader = DataLoader(dataset=test_set, batch_size=4, shuffle=True, num_workers=0)

# 第一张图片及其target
img, target = test_set[0]
print(img.shape)
print(test_set.classes[target])

# 使用tensorboard进行图像显示
writer = SummaryWriter("logs")

# 使用循环取得loader里面的数据
step = 0
for data in loader:  # 每个data都是batch_size张数据的组合
    images, targets = data
    print(images.shape)
    print(targets)
    writer.add_images("loader_image", images, step)
    step = step + 1
writer.close()







