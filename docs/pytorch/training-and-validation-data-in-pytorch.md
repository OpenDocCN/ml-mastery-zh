# PyTorch 中的训练和验证数据

> 原文：[`machinelearningmastery.com/training-and-validation-data-in-pytorch/`](https://machinelearningmastery.com/training-and-validation-data-in-pytorch/)

训练数据是机器学习算法用来学习的数据集。它也称为训练集。验证数据是机器学习算法用来测试其准确性的一组数据。验证算法性能就是将预测输出与验证数据中的已知真实值进行比较。

训练数据通常很大且复杂，而验证数据通常较小。训练样本越多，模型的表现就会越好。例如，在垃圾邮件检测任务中，如果训练集中有 10 封垃圾邮件和 10 封非垃圾邮件，那么机器学习模型可能难以检测到新邮件中的垃圾邮件，因为没有足够的信息来判断垃圾邮件的样子。然而，如果我们有 1000 万封垃圾邮件和 1000 万封非垃圾邮件，那么我们的模型检测新垃圾邮件会容易得多，因为它已经见过了许多垃圾邮件的样子。

在本教程中，你将学习 PyTorch 中训练和验证数据的内容。我们还将演示训练和验证数据对机器学习模型的重要性，特别是神经网络。特别地，你将学习到：

+   PyTorch 中训练和验证数据的概念。

+   数据如何在 PyTorch 中划分为训练集和验证集。

+   如何使用 PyTorch 内置函数构建一个简单的线性回归模型。

+   如何使用不同的学习率来训练我们的模型以获得期望的准确性。

+   如何调整超参数以获得最佳的数据模型。

**快速启动你的项目**，请参阅我的书籍[《用 PyTorch 深度学习》](https://machinelearningmastery.com/deep-learning-with-pytorch/)。它提供了**自学教程**和**工作代码**。

让我们开始吧！[](../Images/5917059615dafe61e0e6e0ee4f6ceac6.png)

使用 PyTorch 中的优化器。

图片由[Markus Krisetya](https://unsplash.com/photos/Vkp9wg-VAsQ)提供。部分版权保留。

## 概述

本教程分为三部分；它们是：

+   为训练和验证集构建数据类

+   构建和训练模型

+   可视化结果

## 为训练和验证集构建数据类

首先，我们加载一些本教程中需要的库。

```py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
```

我们将从构建一个自定义数据集类开始，以生成足够的合成数据。这将允许我们将数据拆分为训练集和验证集。此外，我们还将添加一些步骤将异常值包含到数据中。

```py
# Creating our dataset class
class Build_Data(Dataset):
    # Constructor
    def __init__(self, train = True):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.func = -5 * self.x + 1
        self.y = self.func + 0.4 * torch.randn(self.x.size())
        self.len = self.x.shape[0]
        # adding some outliers
        if train == True:
            self.y[10:12] = 0
            self.y[30:35] = 25
        else:
            pass                
    # Getting the data
    def __getitem__(self, index):    
        return self.x[index], self.y[index]    
    # Getting length of the data
    def __len__(self):
        return self.len

train_set = Build_Data()
val_set = Build_Data(train=False)
```

对于训练集，我们默认将`train`参数设置为`True`。如果设置为`False`，则会生成验证数据。我们将训练集和验证集创建为不同的对象。

现在，让我们可视化我们的数据。你会看到在`$x=-2$`和`$x=0$`的异常值。

```py
# Plotting and visualizing the data points
plt.plot(train_set.x.numpy(), train_set.y.numpy(), 'b+', label='y')
plt.plot(train_set.x.numpy(), train_set.func.numpy(), 'r', label='func')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid('True', color='y')
plt.show()
```

![](img/74d19cb54ba44ff680cd75ebd2ea7369.png)

训练和验证数据集

生成上述图表的完整代码如下。

```py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

# Creating our dataset class
class Build_Data(Dataset):
    # Constructor
    def __init__(self, train = True):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.func = -5 * self.x + 1
        self.y = self.func + 0.4 * torch.randn(self.x.size())
        self.len = self.x.shape[0]
        # adding some outliers
        if train == True:
            self.y[10:12] = 0
            self.y[30:35] = 25
        else:
            pass
    # Getting the data
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    # Getting length of the data
    def __len__(self):
        return self.len

train_set = Build_Data()
val_set = Build_Data(train=False)

# Plotting and visualizing the data points
plt.plot(train_set.x.numpy(), train_set.y.numpy(), 'b+', label='y')
plt.plot(train_set.x.numpy(), train_set.func.numpy(), 'r', label='func')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid('True', color='y')
plt.show()
```

## 构建和训练模型

PyTorch 中的 `nn` 包为我们提供了许多有用的函数。我们将从 `nn` 包中导入线性回归模型和损失准则。此外，我们还将从 `torch.utils.data` 包中导入 `DataLoader`。

```py
...
model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
trainloader = DataLoader(dataset=train_set, batch_size=1)
```

我们将创建一个包含各种学习率的列表，以一次训练多个模型。这是深度学习从业者中的一种常见做法，他们调整不同的超参数以获得最佳模型。我们将训练和验证损失存储在张量中，并创建一个空列表 `Models` 来存储我们的模型。之后，我们将绘制图表来评估我们的模型。

```py
...
learning_rates = [0.1, 0.01, 0.001, 0.0001]
train_err = torch.zeros(len(learning_rates))
val_err = torch.zeros(len(learning_rates))
Models = []
```

为了训练模型，我们将使用各种学习率与随机梯度下降（SGD）优化器。训练和验证数据的结果将与模型一起保存在列表中。我们将训练所有模型 20 个周期。

```py
...
epochs = 20

# iterate through the list of various learning rates 
for i, learning_rate in enumerate(learning_rates):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    for epoch in range(epochs):
        for x, y in trainloader:
            y_hat = model(x)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # training data
    Y_hat = model(train_set.x)
    train_loss = criterion(Y_hat, train_set.y)
    train_err[i] = train_loss.item()

    # validation data
    Y_hat = model(val_set.x)
    val_loss = criterion(Y_hat, val_set.y)
    val_err[i] = val_loss.item()
    Models.append(model)
```

上述代码分别收集训练和验证的损失。这帮助我们理解训练效果如何，例如是否过拟合。如果我们发现验证集的损失与训练集的损失差异很大，那么我们的训练模型未能对未见过的数据进行泛化，即验证集。

### 想要开始使用 PyTorch 进行深度学习吗？

立即获取我的免费电子邮件速成课程（附示例代码）。

点击注册并获取课程的免费 PDF Ebook 版本。

## 可视化结果

在上述代码中，我们使用相同的模型（线性回归）并在固定的训练周期下进行训练。唯一的变化是学习率。然后我们可以比较哪一个学习率在收敛速度上表现最佳。

让我们可视化每个学习率的训练和验证数据的损失图。通过查看图表，你可以观察到在学习率为 0.001 时损失最小，这意味着我们的模型在这个学习率下更快地收敛。

```py
plt.semilogx(np.array(learning_rates), train_err.numpy(), label = 'total training loss')
plt.semilogx(np.array(learning_rates), val_err.numpy(), label = 'total validation loss')
plt.ylabel('Total Loss')
plt.xlabel('learning rate')
plt.legend()
plt.show()
```

![](img/245eb7e1d10344f52b605a55369db77f.png)

损失 vs 学习率

让我们也绘制每个模型在验证数据上的预测结果。一个完全收敛的模型应能完美拟合数据，而一个尚未收敛的模型则会产生偏离数据的预测结果。

```py
# plotting the predictions on validation data
for model, learning_rate in zip(Models, learning_rates):
    yhat = model(val_set.x)
    plt.plot(val_set.x.numpy(), yhat.detach().numpy(), label = 'learning rate:' + str(learning_rate))
plt.plot(val_set.x.numpy(), val_set.func.numpy(), 'or', label = 'validation data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```

我们看到的预测结果可视化如下：![](img/27f946929f6b40cab47a48b13292e3fd.png)

正如你所见，绿色线更接近验证数据点。这是具有最佳学习率（0.001）的线。

以下是从创建数据到可视化训练和验证损失的完整代码。

```py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

# Creating our dataset class
class Build_Data(Dataset):
    # Constructor
    def __init__(self, train=True):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.func = -5 * self.x + 1
        self.y = self.func + 0.4 * torch.randn(self.x.size())
        self.len = self.x.shape[0]
        # adding some outliers
        if train == True:
            self.y[10:12] = 0
            self.y[30:35] = 25
        else:
            pass
    # Getting the data
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    # Getting length of the data
    def __len__(self):
        return self.len

train_set = Build_Data()
val_set = Build_Data(train=False)

criterion = torch.nn.MSELoss()
trainloader = DataLoader(dataset=train_set, batch_size=1)

learning_rates = [0.1, 0.01, 0.001, 0.0001]
train_err = torch.zeros(len(learning_rates))
val_err = torch.zeros(len(learning_rates))
Models = []

epochs = 20

# iterate through the list of various learning rates 
for i, learning_rate in enumerate(learning_rates):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    for epoch in range(epochs):
        for x, y in trainloader:
            y_hat = model(x)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # training data
    Y_hat = model(train_set.x)
    train_loss = criterion(Y_hat, train_set.y)
    train_err[i] = train_loss.item()

    # validation data
    Y_hat = model(val_set.x)
    val_loss = criterion(Y_hat, val_set.y)
    val_err[i] = val_loss.item()
    Models.append(model)

plt.semilogx(np.array(learning_rates), train_err.numpy(), label = 'total training loss')
plt.semilogx(np.array(learning_rates), val_err.numpy(), label = 'total validation loss')
plt.ylabel('Total Loss')
plt.xlabel('learning rate')
plt.legend()
plt.show()

# plotting the predictions on validation data
for model, learning_rate in zip(Models, learning_rates):
    yhat = model(val_set.x)
    plt.plot(val_set.x.numpy(), yhat.detach().numpy(), label = 'learning rate:' + str(learning_rate))
plt.plot(val_set.x.numpy(), val_set.func.numpy(), 'or', label = 'validation data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```

## 总结

在本教程中，你学习了 PyTorch 中训练数据和验证数据的概念。特别是，你了解了：

+   PyTorch 中训练和验证数据的概念。

+   数据如何在 PyTorch 中被拆分为训练集和验证集。

+   如何使用 PyTorch 中的内置函数构建一个简单的线性回归模型。

+   如何使用不同的学习率来训练我们的模型，以获得期望的准确性。

+   如何调整超参数，以便为你的数据获得最佳模型。
