import torch

# 给定一个随机数
x = torch.randn(2, 2, 2, dtype=torch.float32)
print(x)
# tensor([[ 1.3408,  0.7691],
#         [ 0.1121, -0.0851]])

# 公式1(量化)：xq = round(x / scale + zero_point)
# 使用给定的scale和 zero_point 来把一个float tensor转化为 quantized tensor
xq = torch.quantize_per_tensor(x, scale=0.5, zero_point=8, dtype=torch.quint8)
print(xq)
# tensor([[1.5000, 1.0000],
#         [0.0000, 0.0000]], size=(2, 2), dtype=torch.quint8,
#        quantization_scheme=torch.per_tensor_affine, scale=0.5, zero_point=8)

print(xq.int_repr())  # 给定一个量化的张量，返回一个以 uint8_t 作为数据类型的张量
# tensor([[11, 10],
#         [ 8,  8]], dtype=torch.uint8)
xq.q_per_channel_scales()

# 公式2(反量化)：xdq = (xq - zero_point) * scale
# 使用给定的scale和 zero_point 来把一个 quantized tensor 转化为 float tensor
xdq = xq.dequantize()
print(xdq)
# tensor([[1.5000, 1.0000],
#         [0.0000, 0.0000]])
