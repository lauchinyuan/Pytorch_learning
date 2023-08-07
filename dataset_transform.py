# 内建dataset相关操作
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

# 例化进行类型变换的变换对象
dataset_tran = transforms.Compose([transforms.ToTensor()])

#  导入数据集,对数据集中的每一个图片都进行dataset_tran变换
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True, transform=dataset_tran)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True, transform=dataset_tran)
print(test_set.classes)  # 打印数据集类别标签
img, target = test_set[0]  # 分别获取第一张图片及其对应的target
print(test_set.classes[target])  # 查看test_set对应的类别

# tensorboard用于显示
writer = SummaryWriter("logs")
for i in range(1, 10):
    img, target = test_set[i]
    writer.add_image("testset", img, i)
writer.close()


