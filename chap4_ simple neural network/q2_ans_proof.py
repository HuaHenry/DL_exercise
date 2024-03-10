# 使用两层relu网络拟合任何函数
import torch
import torch.nn as nn
import torch.optim as optim

# 模型结构定义
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 拟合的目标函数
def target_function(x):
    return torch.sin(x)+x*torch.cos(x)+x

# 生成训练数据
x_train = torch.unsqueeze(torch.linspace(-2 * 3.1416, 2 * 3.1416, 100), dim=1)
y_train = target_function(x_train)

# 初始化神经网络
input_size = 1
hidden_size = 100
output_size = 1
model = TwoLayerNet(input_size, hidden_size, output_size)

# 损失函数和优化器定义
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练100000轮
num_epochs = 100000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试
x_test = torch.unsqueeze(torch.linspace(-2 * 3.1416, 2 * 3.1416, 200), dim=1)
y_test = target_function(x_test)
with torch.no_grad():
    model.eval()
    y_pred = model(x_test)

# 绘制图像
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(x_train, y_train, c='b', label='Training data')
plt.plot(x_test, y_test, c='g', label='True function')
plt.plot(x_test, y_pred, c='r', label='Predicted function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()