from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import cv2

img_path = "hymenoptera_data/hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path)  # 以 Jpegimagefile格式打开图片
writer = SummaryWriter("logs")

# To tensor
tensor_trans = transforms.ToTensor()  # 实例化ToTensor工具
tensor_img = tensor_trans(img)  # 转换为tensor格式
cv_img = cv2.imread(img_path)  # 读取numpy.array格式的image

# Normalize
# 需要指定三个通道的均值和方差
normal_trans = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 输入的图像必须是tensor格式的
img_normal = normal_trans(tensor_img)

# Resize
resize_trans = transforms.Resize(128)
resize_img = resize_trans(tensor_img)

# 随机裁剪
crop_trans = transforms.RandomCrop(100)
# 分10次查看裁剪结果
for i in range(1, 10):
    crop_img = crop_trans(tensor_img)
    writer.add_image("Random Crop", crop_img, i)

# 多种变换结合起来
# 先resize后变为tensor
resize_tensor_trans = transforms.Compose([resize_trans, tensor_trans])
resize_tensor_img = resize_tensor_trans(img)



# 使用tensorboard观察数据

writer.add_image("tensor_image", tensor_img, 1)
writer.add_image("normal_image", img_normal, 1)
writer.add_image("resize_image", resize_img, 1)
writer.add_image("resize_tensor_img", resize_tensor_img, 1)
writer.close()


print(type(tensor_img))