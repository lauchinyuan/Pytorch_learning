# 对模型进行"训练后静态量化(Post Training Static Quantization)",并保存模型

import torch
import torchvision.datasets
from PIL import Image
from torch.utils.data import DataLoader

from model import cifar10_net

# 导入测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=64)

# 加载已经训练好的float32模型,此时需要将模型的输入输出端插入量化、反量化节点
dict_path = "./model_pth/model_dict_900.pth"
fp32_pth_path = "./model_pth/model_900.pth"
model_32fp_q = cifar10_net(is_quant=True)
state_dict = torch.load(dict_path)
model_32fp_q.load_state_dict(state_dict)
model_32fp = model_32fp_q
# model_32fp = torch.load("./model_pth/model_200.pth", map_location=torch.device("cpu"))

# 将模型设置为eval模式,才能进行静态量化
model_32fp.eval()

# 量化配置, 使用x86设备上的默认配置
model_32fp.qconfig = torch.quantization.get_default_qconfig('x86')

# 由于模型中没有出现常见的融合层,因此不进行模型融合
# 准备模型,插入observers
model_32fp_prepared = torch.quantization.prepare(model_32fp)

# 输入有代表的测试数据, 对activation的量化参数进行校准
for data in test_loader:
    imgs, targets = data
    model_32fp_prepared(imgs)

# 转换为int8模型
model_int8 = torch.quantization.convert(model_32fp_prepared, inplace=False)

# 保存量化后模型, 模型加载参考quant_load.py文件
torch.save(model_int8.state_dict(), "./model_pth/model_int8.pth")

# 量化后模型准确性测试, 并与未量化的模型进行对比
# 加载没有插入量化节点的同等参数的浮点模型
# model_fp32 = cifar10_net(is_quant=False)
# model_fp32.load_state_dict(state_dict)
model_fp32 = torch.load(fp32_pth_path, map_location=torch.device("cpu"))

# 模式转换
model_int8.eval()
model_fp32.eval()
total_test_size = len(test_data)  # 总的图片个数
with torch.no_grad():
    total_accuracy_int8 = 0  # 总的预测正确个数
    total_accuracy_fp32 = 0
    for data in test_loader:
        imgs, targets = data
        result_int8 = model_int8(imgs)  # 量化模型处理结果
        result_fp32 = model_fp32(imgs)  # 浮点模型处理结果
        pred_target_int8 = result_int8.argmax(1)
        pred_target_fp32 = result_fp32.argmax(1)
        accuracy_int8 = (pred_target_int8 == targets).sum()  # 单个batch预测正确的个数
        accuracy_fp32 = (pred_target_fp32 == targets).sum()  # 单个batch预测正确的个数
        total_accuracy_int8 = total_accuracy_int8 + accuracy_int8  # 总正确个数累计
        total_accuracy_fp32 = total_accuracy_fp32 + accuracy_fp32  # 总正确个数累计

# 结果分析
print("量化模型预测正确率为: {}".format(total_accuracy_int8/total_test_size))
print("浮点模型测正确率为: {}".format(total_accuracy_fp32/total_test_size))
print(model_int8)

