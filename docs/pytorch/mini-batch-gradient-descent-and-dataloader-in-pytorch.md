# PyTorch 中的小批量梯度下降和 DataLoader

> 原文：[`machinelearningmastery.com/mini-batch-gradient-descent-and-dataloader-in-pytorch/`](https://machinelearningmastery.com/mini-batch-gradient-descent-and-dataloader-in-pytorch/)

小批量梯度下降是一种用于训练深度学习模型的梯度下降算法变体。该算法的核心思想是将训练数据分成批次，然后逐批次进行处理。在每次迭代中，我们同时更新属于特定批次的所有训练样本的权重。这个过程在不同的批次上重复，直到整个训练数据集被处理完毕。与批量梯度下降相比，这种方法的主要优势在于它可以显著减少计算时间和内存使用，因为它不是一次性处理所有训练样本。

`DataLoader`是 PyTorch 中加载和预处理数据的模块。它可用于从文件加载数据，或生成合成数据。

在本教程中，我们将向您介绍小批量梯度下降的概念。您还将了解如何使用 PyTorch 的`DataLoader`来实现它。具体来说，我们将涵盖以下内容：

+   在 PyTorch 中实现小批量梯度下降。

+   PyTorch 中 DataLoader 的概念以及如何使用它加载数据。

+   随机梯度下降与小批量梯度下降的区别。

+   如何使用 PyTorch DataLoader 实现随机梯度下降。

+   如何使用 PyTorch DataLoader 实现小批量梯度下降。

**通过我的书籍[《PyTorch 深度学习》](https://machinelearningmastery.com/deep-learning-with-pytorch/)来启动您的项目**。它提供了带有**工作代码**的**自学教程**。

让我们开始吧！[](../Images/3f2641e03ddc35049caa853f237fb63f.png)

PyTorch 中的小批量梯度下降和 DataLoader。

图片由[Yannis Papanastasopoulos](https://unsplash.com/photos/kKzbyDeb62M)拍摄。部分权利保留。

## 概述

本教程分为六个部分，它们分别是：

+   PyTorch 中的 DataLoader

+   准备数据和线性回归模型

+   构建数据集和 DataLoader 类

+   使用随机梯度下降和 DataLoader 进行训练

+   使用小批量梯度下降和 DataLoader 进行训练

+   绘制图表进行比较

## PyTorch 中的 DataLoader

当您计划构建深度学习管道来训练模型时，一切都始于数据加载。数据越复杂，加载到管道中就越困难。PyTorch 的`DataLoader`是一个方便的工具，不仅能够轻松加载数据，还可以帮助应用数据增强策略，并在较大数据集中迭代样本。您可以从`torch.utils.data`中导入`DataLoader`类，如下所示。

```py
from torch.utils.data import DataLoader
```

`DataLoader`类中有几个参数，我们只讨论`dataset`和`batch_size`。`dataset`是你在`DataLoader`类中找到的第一个参数，它将数据加载到管道中。第二个参数是`batch_size`，表示在一次迭代中处理的训练样本数。

```py
DataLoader(dataset, batch_size=n)
```

## 准备数据和线性回归模型

我们将重用在之前教程中生成的线性回归数据：

```py
import torch
import numpy as np
import matplotlib.pyplot as plt

# Creating a function f(X) with a slope of -5
X = torch.arange(-5, 5, 0.1).view(-1, 1)
func = -5 * X

# Adding Gaussian noise to the function f(X) and saving it in Y
Y = func + 0.4 * torch.randn(X.size())
```

和之前的教程一样，我们初始化了一个变量`X`，其值范围从$-5$到$5$，并创建了一个斜率为$-5$的线性函数。然后，加入高斯噪声以生成变量`Y`。

我们可以使用 matplotlib 绘制数据以可视化模式：

```py
...
# Plot and visualizing the data points in blue
plt.plot(X.numpy(), Y.numpy(), 'b+', label='Y')
plt.plot(X.numpy(), func.numpy(), 'r', label='func')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid('True', color='y')
plt.show()
```

![](img/c7d2910711d5a9cf82e04f012cd344f1.png)

回归模型的数据点

接下来，我们将基于简单的线性回归方程构建一个前向函数。我们将训练模型以获取两个参数（$w$和$b$）。所以，让我们定义一个模型的前向传播函数以及一个损失标准函数（MSE 损失）。参数变量`w`和`b`将定义在函数外部：

```py
...
# defining the function for forward pass for prediction
def forward(x):
    return w * x + b

# evaluating data points with Mean Square Error (MSE)
def criterion(y_pred, y):
    return torch.mean((y_pred - y) ** 2)
```

### 想开始使用 PyTorch 进行深度学习吗？

立即获取我的免费电子邮件速成课程（包含示例代码）。

点击注册并获取课程的免费 PDF 电子书版本。

## 构建数据集和`DataLoader`类

让我们构建我们的`Dataset`和`DataLoader`类。`Dataset`类允许我们构建自定义数据集并对其应用各种变换。`DataLoader`类则用于将数据集加载到模型训练的管道中。它们的创建方式如下。

```py
# Creating our dataset class
class Build_Data(Dataset):    
    # Constructor
    def __init__(self):
        self.x = torch.arange(-5, 5, 0.1).view(-1, 1)
        self.y = -5 * X
        self.len = self.x.shape[0]        
    # Getting the data
    def __getitem__(self, index):    
        return self.x[index], self.y[index]    
    # Getting length of the data
    def __len__(self):
        return self.len

# Creating DataLoader object
dataset = Build_Data()
train_loader = DataLoader(dataset = dataset, batch_size = 1)
```

## 使用随机梯度下降和`DataLoader`进行训练

当批量大小设置为 1 时，训练算法称为**随机梯度下降**。类似地，当批量大小大于 1 但小于整个训练数据的大小时，训练算法称为**迷你批量梯度下降**。为了简便起见，我们将使用随机梯度下降和`DataLoader`进行训练。

如之前所示，我们将随机初始化可训练参数$w$和$b$，定义其他参数如学习率或步长，创建一个空列表来存储损失，并设置训练的轮数。

```py
w = torch.tensor(-10.0, requires_grad = True)
b = torch.tensor(-20.0, requires_grad = True)

step_size = 0.1
loss_SGD = []
n_iter = 20
```

在 SGD 中，我们只需在每次训练迭代中从数据集中选择一个样本。因此，一个简单的 for 循环加上前向和反向传播就是我们所需要的：

```py
for i in range (n_iter):    
    # calculating loss as in the beginning of an epoch and storing it
    y_pred = forward(X)
    loss_SGD.append(criterion(y_pred, Y).tolist())
    for x, y in train_loader:
        # making a prediction in forward pass
        y_hat = forward(x)
        # calculating the loss between original and predicted data points
        loss = criterion(y_hat, y)    
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updating the parameters after each iteration
        w.data = w.data - step_size * w.grad.data
        b.data = b.data - step_size * b.grad.data    
        # zeroing gradients after each iteration
        w.grad.data.zero_()
        b.grad.data.zero_()
```

将所有内容结合起来，下面是训练模型的完整代码，即`w`和`b`：

```py
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
torch.manual_seed(42)

# Creating a function f(X) with a slope of -5
X = torch.arange(-5, 5, 0.1).view(-1, 1)
func = -5 * X
# Adding Gaussian noise to the function f(X) and saving it in Y
Y = func + 0.4 * torch.randn(X.size())

w = torch.tensor(-10.0, requires_grad = True)
b = torch.tensor(-20.0, requires_grad = True)

# defining the function for forward pass for prediction
def forward(x):
    return w * x + b

# evaluating data points with Mean Square Error (MSE)
def criterion(y_pred, y):
    return torch.mean((y_pred - y) ** 2)

# Creating our dataset class
class Build_Data(Dataset):
    # Constructor
    def __init__(self):
        self.x = torch.arange(-5, 5, 0.1).view(-1, 1)
        self.y = -5 * X
        self.len = self.x.shape[0]
    # Getting the data
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    # Getting length of the data
    def __len__(self):
        return self.len

# Creating DataLoader object
dataset = Build_Data()
train_loader = DataLoader(dataset=dataset, batch_size=1)

step_size = 0.1
loss_SGD = []
n_iter = 20

for i in range (n_iter):
    # calculating loss as in the beginning of an epoch and storing it
    y_pred = forward(X)
    loss_SGD.append(criterion(y_pred, Y).tolist())
    for x, y in train_loader:
        # making a prediction in forward pass
        y_hat = forward(x)
        # calculating the loss between original and predicted data points
        loss = criterion(y_hat, y)
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updating the parameters after each iteration
        w.data = w.data - step_size * w.grad.data
        b.data = b.data - step_size * b.grad.data
        # zeroing gradients after each iteration
        w.grad.data.zero_()
        b.grad.data.zero_()
```

## 使用迷你批量梯度下降和`DataLoader`进行训练

更进一步，我们将使用迷你批量梯度下降和`DataLoader`训练我们的模型。我们将设置不同的批量大小进行训练，即 10 和 20。批量大小为 10 的训练如下：

```py
...
train_loader_10 = DataLoader(dataset=dataset, batch_size=10)

w = torch.tensor(-10.0, requires_grad=True)
b = torch.tensor(-20.0, requires_grad=True)

step_size = 0.1
loss_MBGD_10 = []
iter = 20

for i in range (iter):    
    # calculating loss as in the beginning of an epoch and storing it
    y_pred = forward(X)
    loss_MBGD_10.append(criterion(y_pred, Y).tolist())
    for x, y in train_loader_10:
        # making a prediction in forward pass
        y_hat = forward(x)
        # calculating the loss between original and predicted data points
        loss = criterion(y_hat, y)
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updating the parameters after each iteration
        w.data = w.data - step_size * w.grad.data
        b.data = b.data - step_size * b.grad.data    
        # zeroing gradients after each iteration
        w.grad.data.zero_()
        b.grad.data.zero_()
```

下面是如何以 20 的批量大小实现相同功能：

```py
...
train_loader_20 = DataLoader(dataset=dataset, batch_size=20)

w = torch.tensor(-10.0, requires_grad=True)
b = torch.tensor(-20.0, requires_grad=True)

step_size = 0.1
loss_MBGD_20 = []
iter = 20

for i in range(iter):    
    # calculating loss as in the beginning of an epoch and storing it
    y_pred = forward(X)
    loss_MBGD_20.append(criterion(y_pred, Y).tolist())
    for x, y in train_loader_20:
        # making a prediction in forward pass
        y_hat = forward(x)
        # calculating the loss between original and predicted data points
        loss = criterion(y_hat, y)    
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updating the parameters after each iteration
        w.data = w.data - step_size * w.grad.data
        b.data = b.data - step_size * b.grad.data    
        # zeroing gradients after each iteration
        w.grad.data.zero_()
        b.grad.data.zero_()
```

将所有内容结合起来，以下是完整的代码：

```py
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
torch.manual_seed(42)

# Creating a function f(X) with a slope of -5
X = torch.arange(-5, 5, 0.1).view(-1, 1)
func = -5 * X
# Adding Gaussian noise to the function f(X) and saving it in Y
Y = func + 0.4 * torch.randn(X.size())

w = torch.tensor(-10.0, requires_grad=True)
b = torch.tensor(-20.0, requires_grad=True)

# defining the function for forward pass for prediction
def forward(x):
    return w * x + b

# evaluating data points with Mean Square Error (MSE)
def criterion(y_pred, y):
    return torch.mean((y_pred - y) ** 2)

# Creating our dataset class
class Build_Data(Dataset):
    # Constructor
    def __init__(self):
        self.x = torch.arange(-5, 5, 0.1).view(-1, 1)
        self.y = -5 * X
        self.len = self.x.shape[0]
    # Getting the data
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    # Getting length of the data
    def __len__(self):
        return self.len

# Creating DataLoader object
dataset = Build_Data()
train_loader_10 = DataLoader(dataset=dataset, batch_size=10)

step_size = 0.1
loss_MBGD_10 = []
iter = 20

for i in range(n_iter):
    # calculating loss as in the beginning of an epoch and storing it
    y_pred = forward(X)
    loss_MBGD_10.append(criterion(y_pred, Y).tolist())
    for x, y in train_loader_10:
        # making a prediction in forward pass
        y_hat = forward(x)
        # calculating the loss between original and predicted data points
        loss = criterion(y_hat, y)
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updateing the parameters after each iteration
        w.data = w.data - step_size * w.grad.data
        b.data = b.data - step_size * b.grad.data
        # zeroing gradients after each iteration
        w.grad.data.zero_()
        b.grad.data.zero_()

train_loader_20 = DataLoader(dataset=dataset, batch_size=20)

# Reset w and b
w = torch.tensor(-10.0, requires_grad=True)
b = torch.tensor(-20.0, requires_grad=True)

loss_MBGD_20 = []

for i in range(n_iter):
    # calculating loss as in the beginning of an epoch and storing it
    y_pred = forward(X)
    loss_MBGD_20.append(criterion(y_pred, Y).tolist())
    for x, y in train_loader_20:
        # making a prediction in forward pass
        y_hat = forward(x)
        # calculating the loss between original and predicted data points
        loss = criterion(y_hat, y)
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updating the parameters after each iteration
        w.data = w.data - step_size * w.grad.data
        b.data = b.data - step_size * b.grad.data
        # zeroing gradients after each iteration
        w.grad.data.zero_()
        b.grad.data.zero_()
```

## 绘制比较图表

最后，让我们可视化所有三种算法（即随机梯度下降、小批量梯度下降（批量大小为 10）和批量大小为 20）在训练期间损失的减少情况。

```py
plt.plot(loss_SGD,label = "Stochastic Gradient Descent")
plt.plot(loss_MBGD_10,label = "Mini-Batch-10 Gradient Descent")
plt.plot(loss_MBGD_20,label = "Mini-Batch-20 Gradient Descent")
plt.xlabel('epoch')
plt.ylabel('Cost/total loss')
plt.legend()
plt.show()
```

![](img/3cdffead736887fa03b6fcfd99f9086e.png)

从图中可以看出，小批量梯度下降可能会更快收敛，因为我们可以通过计算每一步的平均损失来对参数进行更精确的更新。

综合来看，以下是完整代码：

```py
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
torch.manual_seed(42)

# Creating a function f(X) with a slope of -5
X = torch.arange(-5, 5, 0.1).view(-1, 1)
func = -5 * X
# Adding Gaussian noise to the function f(X) and saving it in Y
Y = func + 0.4 * torch.randn(X.size())

w = torch.tensor(-10.0, requires_grad=True)
b = torch.tensor(-20.0, requires_grad=True)

# defining the function for forward pass for prediction
def forward(x):
    return w * x + b

# evaluating data points with Mean Square Error (MSE)
def criterion(y_pred, y):
    return torch.mean((y_pred - y) ** 2)

# Creating our dataset class
class Build_Data(Dataset):
    # Constructor
    def __init__(self):
        self.x = torch.arange(-5, 5, 0.1).view(-1, 1)
        self.y = -5 * X
        self.len = self.x.shape[0]
    # Getting the data
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    # Getting length of the data
    def __len__(self):
        return self.len

# Creating DataLoader object
dataset = Build_Data()
train_loader = DataLoader(dataset=dataset, batch_size=1)

step_size = 0.1
loss_SGD = []
n_iter = 20

for i in range(n_iter):
    # calculating loss as in the beginning of an epoch and storing it
    y_pred = forward(X)
    loss_SGD.append(criterion(y_pred, Y).tolist())
    for x, y in train_loader:
        # making a prediction in forward pass
        y_hat = forward(x)
        # calculating the loss between original and predicted data points
        loss = criterion(y_hat, y)
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updating the parameters after each iteration
        w.data = w.data - step_size * w.grad.data
        b.data = b.data - step_size * b.grad.data
        # zeroing gradients after each iteration
        w.grad.data.zero_()
        b.grad.data.zero_()

train_loader_10 = DataLoader(dataset=dataset, batch_size=10)

# Reset w and b
w = torch.tensor(-10.0, requires_grad=True)
b = torch.tensor(-20.0, requires_grad=True)

loss_MBGD_10 = []

for i in range(n_iter):
    # calculating loss as in the beginning of an epoch and storing it
    y_pred = forward(X)
    loss_MBGD_10.append(criterion(y_pred, Y).tolist())
    for x, y in train_loader_10:
        # making a prediction in forward pass
        y_hat = forward(x)
        # calculating the loss between original and predicted data points
        loss = criterion(y_hat, y)
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updating the parameters after each iteration
        w.data = w.data - step_size * w.grad.data
        b.data = b.data - step_size * b.grad.data
        # zeroing gradients after each iteration
        w.grad.data.zero_()
        b.grad.data.zero_()

train_loader_20 = DataLoader(dataset=dataset, batch_size=20)

# Reset w and b
w = torch.tensor(-10.0, requires_grad=True)
b = torch.tensor(-20.0, requires_grad=True)

loss_MBGD_20 = []

for i in range(n_iter):
    # calculating loss as in the beginning of an epoch and storing it
    y_pred = forward(X)
    loss_MBGD_20.append(criterion(y_pred, Y).tolist())
    for x, y in train_loader_20:
        # making a prediction in forward pass
        y_hat = forward(x)
        # calculating the loss between original and predicted data points
        loss = criterion(y_hat, y)
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updating the parameters after each iteration
        w.data = w.data - step_size * w.grad.data
        b.data = b.data - step_size * b.grad.data
        # zeroing gradients after each iteration
        w.grad.data.zero_()
        b.grad.data.zero_()

plt.plot(loss_SGD,label="Stochastic Gradient Descent")
plt.plot(loss_MBGD_10,label="Mini-Batch-10 Gradient Descent")
plt.plot(loss_MBGD_20,label="Mini-Batch-20 Gradient Descent")
plt.xlabel('epoch')
plt.ylabel('Cost/total loss')
plt.legend()
plt.show()
```

## 总结

在本教程中，你了解了小批量梯度下降、`DataLoader`及其在 PyTorch 中的实现。特别是，你学到了：

+   在 PyTorch 中实现小批量梯度下降。

+   PyTorch 中的 `DataLoader` 概念以及如何使用它加载数据。

+   随机梯度下降与小批量梯度下降之间的区别。

+   如何使用 PyTorch `DataLoader` 实现随机梯度下降。

+   如何使用 PyTorch `DataLoader` 实现小批量梯度下降。
