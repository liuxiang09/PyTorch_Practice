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

# 可以使用autograd做更多的操作
# 现在来看一个雅可比向量积的例子：
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

# 现在在这种情况下，y不再是一个标量。torch.autograd不能直接计算整个雅可比，但是如果我们只想要雅可比向量积，只需要简单的传递向量给backward作为参数。
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

# 也可以通过将代码块包装在with torch.no_grad()，来停止对从跟踪历史中的.requires_grad=True的张量自动求导。
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)

# 或者使用.detach()来获取一个新的tensor，该tensor不再跟踪其计算历史。
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)


