import matplotlib.pyplot as plt

# 准备数据集，线性关系
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 随机的初始化权重 
w = 1.0


# 找线性模型
def forward(x):
    return x * w


# 损失函数MSE
def loss(xs, ys):
    cost = 0 # 储存loss ^ 2的和
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs) # MSE


# 批量梯度下降：选取所有的样本做梯度下降
# 获取当前的梯度是多少
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


epoch_list = []
loss_list = []
print('predict (before training)', 4, forward(4))
for epoch in range(100):
    cost_val = loss(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val  # 0.01 学习率
    print('epoch:', epoch, 'w=', w, 'loss=', cost_val)
    epoch_list.append(epoch)
    loss_list.append(cost_val)

print('predict (after training)', 4, forward(4))
plt.plot(epoch_list, loss_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()
