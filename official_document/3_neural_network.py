"""
一个典型的神经网络训练过程包括以下几点：

1.定义一个包含可训练参数的神经网络

2.迭代整个输入

3.通过神经网络处理输入

4.计算损失(loss)

5.反向传播梯度到神经网络的参数

6.更新网络的参数，典型的用一个简单的更新方法：weight = weight - learning_rate *gradient
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 定义了两个卷积层conv1和conv2，接着是三个全连接层fc1、fc2和fc3
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 一个线性连接（仿射变换：y = xA^T + b）
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 向前传播函数
    def forward(self, x):
        # 2x2的池化窗口
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果大小是正方形，可以只使用一个数字
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # 计算展平后的特征数量
    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批大小维度的其余所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

# 尝试随机生成一个 32x32 的输入
# 为了使用这个网络在 MNIST 数据及上，你需要把数据集中的图片维度修改为 32x32。
input = torch.randn(1, 1, 32, 32)
print("input:", input)
out = net(input)
print("output:", out)

# 把所有参数梯度缓存器置零，用随机的梯度来反向传播
net.zero_grad()
out.backward(torch.randn(1, 10))

output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print("loss", loss)
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


# 反向传播
# 为了实现反向传播损失，我们所有需要做的事情仅仅是使用 loss.backward()。你需要清空现存的梯度，要不然帝都将会和现存的梯度累计到一起。
# 现在我们调用 loss.backward() ，然后看一下 con1 的偏置项在反向传播之前和之后的变化。
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# 现在我们看到了，如何使用损失函数。
# 唯一剩下的事情就是更新神经网络的参数。
# 更新神经网络参数：
# 最简单的更新规则就是随机梯度下降。
# weight = weight - learning_rate * gradient
# 我们可以使用 python 来实现这个规则：
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# 然而，在使用神经网络时，你想要使用各种不同的更新规则，比如 SGD, Nesterov-SGD, Adam, RMSProp 等。
# 为了能够做到这一点，我们构建了一个较小的包：torch.optim，它实现了所有的这些方法。
# 使用它非常的简单：
import torch
import torch.optim as optim

# 创建你的优化器
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 在你的训练循环中：
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
