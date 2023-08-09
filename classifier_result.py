# 对分类器分类结果进行简单分析处理
import torch

# 自设定分类器输出
output = torch.Tensor([[0.2, 0.1],   # 索引为0处出现最大值
                       [0.3, 0.4]])  # 索引为1处出现最大值
prd_target = output.argmax(1)
print(prd_target)  # 求横向最大值所在的索引,索引从0开始, 1代表横向比较
# tensor([0, 1])

# 自设定分类target
target = torch.tensor([1, 1])
print(prd_target == target)   # 匹配情况
true_result = (prd_target == target).sum()
print("正确个数： {}".format(true_result))
