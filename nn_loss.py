import torch
from torch import nn
from torch.nn import L1Loss, MSELoss

# 自定义数据测试
inputs = torch.tensor([1, 2, 3],dtype=torch.float32)
output = torch.tensor([1, 2, 5],dtype=torch.float32)
inputs = torch.reshape(inputs, (1, 1, 1, 3))
output = torch.reshape(output, (1, 1, 1, 3))

loss = L1Loss()
mse = MSELoss()
L1loss_res = loss(inputs, output)
mse_loss = mse(inputs, output)
print(L1loss_res)
print(mse_loss)

# 交叉熵
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss()
result = loss_cross(x, y)  # (-0.2) + ln(exp(0.1) + exp(0.2) + exp(0.3))
print(result)