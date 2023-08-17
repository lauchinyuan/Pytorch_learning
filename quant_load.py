# 加载模型
import torch

from model import cifar10_net

state_dict = torch.load("./model_pth/model_int8.pth")

# 对fp32模型进行相同的量化处理,才能正常加载模型参数
load_model = cifar10_net(is_quant=True)
load_model.qconfig = torch.quantization.get_default_qconfig('x86')
load_model_prepared = torch.quantization.prepare(load_model)
load_model_int8 = torch.quantization.convert(load_model_prepared)
load_model_int8.load_state_dict(state_dict)
load_model_int = load_model_int8
print(load_model_int)