# 在 PyTorch 中构建单层神经网络

> 原文：[`machinelearningmastery.com/building-a-single-layer-neural-network-in-pytorch/`](https://machinelearningmastery.com/building-a-single-layer-neural-network-in-pytorch/)

神经网络是一组彼此相连的神经元节点。这些神经元不仅与相邻的神经元连接，还与距离更远的神经元连接。

神经网络的主要思想是每个层中的神经元有一个或多个输入值，并通过对输入应用某些数学函数来生成输出值。一层中的神经元的输出成为下一层的输入。

单层神经网络是一种人工神经网络，其中输入层和输出层之间只有一个隐藏层。这是深度学习流行之前的经典架构。在本教程中，你将有机会构建一个仅具有一个隐藏层的神经网络。特别地，你将学习：

+   如何在 PyTorch 中构建单层神经网络。

+   如何使用 PyTorch 训练单层神经网络。

+   如何使用单层神经网络对一维数据进行分类。

**快速启动你的项目**，参考我的书籍 [Deep Learning with PyTorch](https://machinelearningmastery.com/deep-learning-with-pytorch/)。它提供了**自学教程**和**有效代码**。

让我们开始吧。![](img/5d222a4a9a81586d10e46bcea04b481b.png)

在 PyTorch 中构建单层神经网络。

图片由 [Tim Cheung](https://unsplash.com/photos/He3wMrz8c7k) 提供。保留部分权利。

## 概述

本教程分为三个部分，分别是

+   +   准备数据集

    +   构建模型

    +   训练模型

## 准备数据

神经网络简单来说是一个用某些参数近似其他函数的函数。让我们生成一些数据，看看我们的单层神经网络如何将函数近似化以使数据线性可分。稍后在本教程中，你将可视化训练过程中函数的重叠情况。

```py
import torch
import matplotlib.pyplot as plt

# generate synthetic the data
X = torch.arange(-30, 30, 1).view(-1, 1).type(torch.FloatTensor)
Y = torch.zeros(X.shape[0])
Y[(X[:, 0] <= -10)] = 1.0
Y[(X[:, 0] > -10) & (X[:, 0] < 10)] = 0.5
Y[(X[:, 0] > 10)] = 0
```

数据使用 matplotlib 绘制后，呈现如下图。

```py
...
plt.plot(X, Y)
plt.show()
```

![](img/75aa284a267d06f9beca5af59e392dbc.png)

### 想要开始使用 PyTorch 进行深度学习吗？

立即参加我的免费电子邮件速成课程（附示例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## 使用 `nn.Module` 构建模型

接下来，让我们使用 `nn.Module` 构建自定义的单层神经网络模块。如果需要更多关于 `nn.Module` 的信息，请查看之前的教程。

该神经网络包括一个输入层、一个具有两个神经元的隐藏层和一个输出层。在每一层之后，应用一个 sigmoid 激活函数。PyTorch 中还提供了其他激活函数，但该网络的经典设计是使用 sigmoid 函数。

这是你的单层神经网络的代码示例。

```py
...

# Define the class for single layer NN
class one_layer_net(torch.nn.Module):    
    # Constructor
    def __init__(self, input_size, hidden_neurons, output_size):
        super(one_layer_net, self).__init__()
        # hidden layer 
        self.linear_one = torch.nn.Linear(input_size, hidden_neurons)
        self.linear_two = torch.nn.Linear(hidden_neurons, output_size) 
        # defining layers as attributes
        self.layer_in = None
        self.act = None
        self.layer_out = None
    # prediction function
    def forward(self, x):
        self.layer_in = self.linear_one(x)
        self.act = torch.sigmoid(self.layer_in)
        self.layer_out = self.linear_two(self.act)
        y_pred = torch.sigmoid(self.linear_two(self.act))
        return y_pred
```

让我们也实例化一个模型对象。

```py
# create the model 
model = one_layer_net(1, 2, 1)  # 2 represents two neurons in one hidden layer
```

## 训练模型

在开始训练循环之前，让我们为模型定义损失函数和优化器。您将编写一个用于交叉熵损失的损失函数，并使用随机梯度下降进行参数优化。

```py
def criterion(y_pred, y):
    out = -1 * torch.mean(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred))
    return out
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

现在你已经有了所有组件来训练模型。让我们训练 5000 个 epochs。您将看到神经网络在每 1000 个 epochs 后如何逼近函数的图表。

```py
# Define the training loop
epochs=5000
cost = []
total=0
for epoch in range(epochs):
    total=0
    epoch = epoch + 1
    for x, y in zip(X, Y):
        yhat = model(x)
        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # get total loss 
        total+=loss.item() 
    cost.append(total)
    if epoch % 1000 == 0:
        print(str(epoch)+ " " + "epochs done!") # visualze results after every 1000 epochs   
        # plot the result of function approximator
        plt.plot(X.numpy(), model(X).detach().numpy())
        plt.plot(X.numpy(), Y.numpy(), 'm')
        plt.xlabel('x')
        plt.show()
```

在 1000 个 epochs 之后，模型近似了以下函数：![](img/abcc65969416e141fc5bbaa935035a0b.png)

但是在 5000 个 epochs 之后，它改进到以下结果：![](img/14b59102632200a0307b78e58395314c.png)

从中可以看出，蓝色的近似值更接近紫色的数据。正如你所见，神经网络相当好地近似了这些函数。如果函数更复杂，你可能需要更多的隐藏层或更多的隐藏层神经元，即一个更复杂的模型。

让我们也绘制图表，看看训练过程中损失是如何减少的。

```py
# plot the cost
plt.plot(cost)
plt.xlabel('epochs')
plt.title('cross entropy loss')
plt.show()
```

你应该会看到：![](img/423642b2e038969698998972e2b0655f.png)

将所有内容整合在一起，以下是完整的代码：

```py
import torch
import matplotlib.pyplot as plt

# generate synthetic the data
X = torch.arange(-30, 30, 1).view(-1, 1).type(torch.FloatTensor)
Y = torch.zeros(X.shape[0])
Y[(X[:, 0] <= -10)] = 1.0
Y[(X[:, 0] > -10) & (X[:, 0] < 10)] = 0.5
Y[(X[:, 0] > 10)] = 0

plt.plot(X, Y)
plt.show()

# Define the class for single layer NN
class one_layer_net(torch.nn.Module):    
    # Constructor
    def __init__(self, input_size, hidden_neurons, output_size):
        super(one_layer_net, self).__init__()
        # hidden layer 
        self.linear_one = torch.nn.Linear(input_size, hidden_neurons)
        self.linear_two = torch.nn.Linear(hidden_neurons, output_size) 
        # defining layers as attributes
        self.layer_in = None
        self.act = None
        self.layer_out = None
    # prediction function
    def forward(self, x):
        self.layer_in = self.linear_one(x)
        self.act = torch.sigmoid(self.layer_in)
        self.layer_out = self.linear_two(self.act)
        y_pred = torch.sigmoid(self.linear_two(self.act))
        return y_pred

# create the model 
model = one_layer_net(1, 2, 1)  # 2 represents two neurons in one hidden layer

def criterion(y_pred, y):
    out = -1 * torch.mean(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred))
    return out
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Define the training loop
epochs=5000
cost = []
total=0
for epoch in range(epochs):
    total=0
    epoch = epoch + 1
    for x, y in zip(X, Y):
        yhat = model(x)
        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # get total loss 
        total+=loss.item() 
    cost.append(total)
    if epoch % 1000 == 0:
        print(str(epoch)+ " " + "epochs done!") # visualze results after every 1000 epochs   
        # plot the result of function approximator
        plt.plot(X.numpy(), model(X).detach().numpy())
        plt.plot(X.numpy(), Y.numpy(), 'm')
        plt.xlabel('x')
        plt.show()

# plot the cost
plt.plot(cost)
plt.xlabel('epochs')
plt.title('cross entropy loss')
plt.show()
```

## 总结

在本教程中，您学习了如何构建和训练神经网络并估计函数。特别是，您学到了：

+   如何在 PyTorch 中构建一个单层神经网络。

+   如何使用 PyTorch 训练单层神经网络。

+   如何使用单层神经网络对一维数据进行分类。
