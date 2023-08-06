from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self, root_dir, label_dir):  # 类初始化函数
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)  # 将root_dir和label_dir进行拼接,生成新的地址
        self.image_path = os.listdir(self.path)  # 将path中的文件名转换为列表img_path

    def __getitem__(self, idx):
        image_name = self.image_path[idx]  # 获取图片名称
        image_item_path = os.path.join(self.path, image_name)
        # 读取图片
        img = Image.open(image_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):  # 数据长度
        return len(self.image_path)


train_root_dir = "hymenoptera_data/hymenoptera_data/train"  # 根目录
train_label_ants = "ants"  # label目录，目录名称也是label名称
train_label_bees = "bees"

# 实例化蚂蚁和蜜蜂数据集
ant_dataset = MyData(train_root_dir, train_label_ants)
bee_dataset = MyData(train_root_dir, train_label_bees)

# 将两个数据集相加得到训练集
train_dataset = ant_dataset + bee_dataset


