# -*- coding: utf-8 -*-
import torch

# 构造一个5*3矩阵，不初始化
x1 = torch.empty(5, 3)
# print(x1)

# 构造一个随机初始化的矩阵
x2 = torch.rand(5, 3)
# print(x2)

# 构造一个矩阵全为0，而且数据类型是long
x3 = torch.zeros(5, 3, dtype=torch.long)
# print(x3)

# 构造一个张量，直接使用数据
x4 = torch.tensor([5.5, 3])
# print(x4)

# 创建一个tensor基于已经存在的tensor
x5 = x1.new_ones(5, 3, dtype=torch.double)
# print(x5)
x6 = torch.randn_like(x5, dtype=torch.float)
# print(x6)

# 获取维度信息
# print(x6.size())

# 注意：torch.Size是一个元组，所以它支持左右的元组操作

# 操作
# 在接下来的例子中，将会看到加法操作
# 加法：方式一
x = torch.rand(5, 3)
y = torch.rand(5, 3)
# print(x + y)

# 加法：方式二
# print(torch.add(x, y))

# 加法：提供一个输出tensor作为参数
result = torch.empty(5, 3)
torch.add(x, y, out=result)
# print(result)

# 加法：in-place
# add x into y
y.add_(x)
# print(y)

# 改变大小：使用torch.view改变tensor的大小
x7 = torch.randn(4, 4)
y7 = x7.view(16)
z7 = x7.view(-1, 8)  # -1表示从其他维度推断，由于后面是8，所以这里是2
# print(x7.size(), y7.size(), z7.size())  # torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])

# 如果是仅包含一个元素的tensor，可以使用.item()来得到对应的python数值
x8 = torch.randn(1)
print(x8)
print(x8.item())
