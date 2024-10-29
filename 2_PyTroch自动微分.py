import torch

# 创建一个张量并设置requires_grad=True用来追踪其计算历史
x = torch.ones(2, 2, requires_grad=True)
print(x)

# 对张量进行操作
y = x + 2
print(y)

# y是计算的结果，所以它有grad_fn
print(y.grad_fn)

# 对y进行更多操作
z = y * y * 3
out = z.mean()

print(z, out)

# .requires_grad_( ... ) 可以改变现有张量的 requires_grad属性。如果没有指定的话，默认输入的flag是False。
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# 梯度
# 现在我们开始进行反向传播，因为out是一个标量，所以out.backward()等于out.backward(torch.tensor(1))。
out.backward()
# 打印梯度 d(out)/dx
print(x.grad)
