# 导入tensorboard数据可视化工具
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter类
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")  # 实例化SummaryWriter类，参数代表并将事件文件保存到logs文件中

# 使用例子，绘制y=x图像
for i in range(100):
    writer.add_scalar("y=2x", 2 * i, i)  # 添加标量(title, y, x),注意这里是一个标题对应一个图

img_path = "hymenoptera_data/hymenoptera_data/train/ants/0013035.jpg"
image_PIL = Image.open(img_path)  # 注意这里打开的图片类型是PIL.jpegImagePlugin,而SummaryWriter的add_image不支持
image_array = np.array(image_PIL)  # 转换为numpy array形式

print(image_array.shape)  # 查看图片的形状大小

# 可以将step改为2,从而在test这一标签的图片中有迭代过程
writer.add_image("test", image_array, 1, dataformats='HWC')  # 添加图像,指定图片数据格式,step=1,H为高,W为宽,C为通道

writer.close()
