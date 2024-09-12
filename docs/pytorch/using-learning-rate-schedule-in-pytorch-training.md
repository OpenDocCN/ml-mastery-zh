# 在 PyTorch 训练中使用学习率调度

> 原文：[`machinelearningmastery.com/using-learning-rate-schedule-in-pytorch-training/`](https://machinelearningmastery.com/using-learning-rate-schedule-in-pytorch-training/)

训练神经网络或大型深度学习模型是一项困难的优化任务。

训练神经网络的经典算法称为 [随机梯度下降](https://machinelearningmastery.com/gradient-descent-for-machine-learning/)。已经很好地证明，通过在训练过程中使用会变化的 [学习率](https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/)，你可以在某些问题上实现性能提升和更快的训练。

在这篇文章中，你将了解什么是学习率调度以及如何在 PyTorch 中为你的神经网络模型使用不同的学习率调度。

阅读本文后，你将了解到：

+   学习率调度在模型训练中的作用

+   如何在 PyTorch 训练循环中使用学习率调度

+   如何设置自己的学习率调度

### 想开始使用 PyTorch 深度学习吗？

现在就参加我的免费电子邮件速成课程（包含示例代码）。

点击注册，并获取课程的免费 PDF 电子书版本。

让我们开始吧！[](../Images/7a13d9fa39a1b8fc273f193425bc5a11.png)

在 PyTorch 训练中使用学习率调度

图片由 [Cheung Yin](https://unsplash.com/photos/A_lVW8yIQM0) 提供。保留部分权利。

## 概述

本文分为三个部分；它们是

+   训练模型的学习率调度

+   在 PyTorch 训练中应用学习率调度

+   自定义学习率调度

## 训练模型的学习率调度

梯度下降是一种数值优化算法。它的作用是使用公式更新参数：

$$

w := w – \alpha \dfrac{dy}{dw}

$$

在这个公式中，$w$ 是参数，例如神经网络中的权重，而 $y$ 是目标，例如损失函数。它的作用是将 $w$ 移动到可以最小化 $y$ 的方向。这个方向由微分提供，即 $\dfrac{dy}{dw}$，但你应该移动 $w$ 的多少则由**学习率** $\alpha$ 控制。

一个简单的开始是使用在梯度下降算法中的恒定学习率。但使用**学习率调度**你可以做得更好。调度是使学习率适应梯度下降优化过程，从而提高性能并减少训练时间。

在神经网络训练过程中，数据以批次的形式输入网络，一个时期内有多个批次。每个批次触发一个训练步骤，其中梯度下降算法更新一次参数。然而，通常学习率调度只在每个 [训练时期](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/) 更新一次。

你可以像每一步那样频繁地更新学习率，但通常它会在每个 epoch 更新一次，因为你需要了解网络的表现，以便决定学习率应该如何更新。通常，模型会在每个 epoch 使用验证数据集进行评估。

调整学习率的方式有多种。在训练开始时，你可能会倾向于使用较大的学习率，以便粗略地改进网络，从而加快进度。在非常复杂的神经网络模型中，你也可能会倾向于在开始时逐渐增加学习率，因为你需要网络在不同的预测维度上进行探索。然而，在训练结束时，你总是希望将学习率调整得更小。因为那时你即将获得模型的最佳性能，如果学习率过大会容易超调。

因此，在训练过程中，最简单且可能最常用的学习率适应方式是逐渐减少学习率的技术。这些技术的好处在于，在训练程序开始时使用较大的学习率值时，可以做出较大的更改，并在训练程序后期将学习率降低，从而使更新权重时的学习率较小，训练更新也较小。

这会在早期快速学习到好的权重，并在之后进行微调。

接下来，让我们看看如何在 PyTorch 中设置学习率调度。

**通过我的书籍** [《深度学习与 PyTorch》](https://machinelearningmastery.com/deep-learning-with-pytorch/) **来启动你的项目**。它提供了 **自学教程** 和 **可运行的代码**。

## 在 PyTorch 训练中应用学习率调度

在 PyTorch 中，一个模型通过优化器进行更新，学习率是优化器的一个参数。学习率调度是一种算法，用于更新优化器中的学习率。

以下是创建学习率调度的示例：

```py
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=10)
```

PyTorch 在 `torch.optim.lr_scheduler` 子模块中提供了许多学习率调度器。所有的调度器都需要优化器作为第一个参数。根据调度器的不同，你可能需要提供更多的参数来进行设置。

我们从一个示例模型开始。下面的模型旨在解决 [电离层二分类问题](http://archive.ics.uci.edu/ml/datasets/Ionosphere)。这是一个小型数据集，你可以 [从 UCI 机器学习库下载](http://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data)。将数据文件放置在你的工作目录中，文件名为 `ionosphere.csv`。

电离层数据集适合用于神经网络的练习，因为所有输入值都是相同量级的小数值。

一个小型神经网络模型构建了一个具有 34 个神经元的单隐藏层，使用 ReLU 激活函数。输出层有一个神经元，并使用 sigmoid 激活函数来输出类似概率的值。

使用的是普通随机梯度下降算法，固定学习率为 0.1。模型训练了 50 个周期。优化器的状态参数可以在`optimizer.param_groups`中找到；其中学习率是`optimizer.param_groups[0]["lr"]`的浮点值。在每个周期结束时，打印出优化器的学习率。

完整示例如下。

```py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# load dataset, split into input (X) and output (y) variables
dataframe = pd.read_csv("ionosphere.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:34].astype(float)
y = dataset[:,34]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# convert into PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# train-test split for evaluation of the model
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# create model
model = nn.Sequential(
    nn.Linear(34, 34),
    nn.ReLU(),
    nn.Linear(34, 1),
    nn.Sigmoid()
)

# Train the model
n_epochs = 50
batch_size = 24
batch_start = torch.arange(0, len(X_train), batch_size)
lr = 0.1
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)
model.train()
for epoch in range(n_epochs):
    for start in batch_start:
        X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch %d: SGD lr=%.4f" % (epoch, optimizer.param_groups[0]["lr"]))

# evaluate accuracy after training
model.eval()
y_pred = model(X_test)
acc = (y_pred.round() == y_test).float().mean()
acc = float(acc)
print("Model accuracy: %.2f%%" % (acc*100))
```

运行此模型产生：

```py
Epoch 0: SGD lr=0.1000
Epoch 1: SGD lr=0.1000
Epoch 2: SGD lr=0.1000
Epoch 3: SGD lr=0.1000
Epoch 4: SGD lr=0.1000
...
Epoch 45: SGD lr=0.1000
Epoch 46: SGD lr=0.1000
Epoch 47: SGD lr=0.1000
Epoch 48: SGD lr=0.1000
Epoch 49: SGD lr=0.1000
Model accuracy: 86.79%
```

你可以确认学习率在整个训练过程中没有变化。让我们让训练过程以较大的学习率开始，以较小的学习率结束。为了引入学习率调度器，你需要在训练循环中运行其`step()`函数。上述代码修改为以下内容：

```py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# load dataset, split into input (X) and output (y) variables
dataframe = pd.read_csv("ionosphere.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:34].astype(float)
y = dataset[:,34]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# convert into PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# train-test split for evaluation of the model
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# create model
model = nn.Sequential(
    nn.Linear(34, 34),
    nn.ReLU(),
    nn.Linear(34, 1),
    nn.Sigmoid()
)

# Train the model
n_epochs = 50
batch_size = 24
batch_start = torch.arange(0, len(X_train), batch_size)
lr = 0.1
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)
model.train()
for epoch in range(n_epochs):
    for start in batch_start:
        X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]
    print("Epoch %d: SGD lr %.4f -> %.4f" % (epoch, before_lr, after_lr))

# evaluate accuracy after training
model.eval()
y_pred = model(X_test)
acc = (y_pred.round() == y_test).float().mean()
acc = float(acc)
print("Model accuracy: %.2f%%" % (acc*100))
```

它打印出：

```py
Epoch 0: SGD lr 0.1000 -> 0.0983
Epoch 1: SGD lr 0.0983 -> 0.0967
Epoch 2: SGD lr 0.0967 -> 0.0950
Epoch 3: SGD lr 0.0950 -> 0.0933
Epoch 4: SGD lr 0.0933 -> 0.0917
...
Epoch 28: SGD lr 0.0533 -> 0.0517
Epoch 29: SGD lr 0.0517 -> 0.0500
Epoch 30: SGD lr 0.0500 -> 0.0500
Epoch 31: SGD lr 0.0500 -> 0.0500
...
Epoch 48: SGD lr 0.0500 -> 0.0500
Epoch 49: SGD lr 0.0500 -> 0.0500
Model accuracy: 88.68%
```

上述代码使用了`LinearLR()`。它是一个线性率调度器，并且需要三个附加参数，`start_factor`、`end_factor`和`total_iters`。你将`start_factor`设置为 1.0，`end_factor`设置为 0.5，`total_iters`设置为 30，因此它将在 10 个相等步骤中将乘法因子从 1.0 减少到 0.5。经过 10 步后，因子将保持在 0.5。这一因子随后会与优化器中的原始学习率相乘。因此，你将看到学习率从$0.1\times 1.0 = 0.1$减少到$0.1\times 0.5 = 0.05$。

除了`LinearLR()`，你还可以使用`ExponentialLR()`，其语法为：

```py
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
```

如果你将`LinearLR()`替换为此，你将看到学习率更新如下：

```py
Epoch 0: SGD lr 0.1000 -> 0.0990
Epoch 1: SGD lr 0.0990 -> 0.0980
Epoch 2: SGD lr 0.0980 -> 0.0970
Epoch 3: SGD lr 0.0970 -> 0.0961
Epoch 4: SGD lr 0.0961 -> 0.0951
...
Epoch 45: SGD lr 0.0636 -> 0.0630
Epoch 46: SGD lr 0.0630 -> 0.0624
Epoch 47: SGD lr 0.0624 -> 0.0617
Epoch 48: SGD lr 0.0617 -> 0.0611
Epoch 49: SGD lr 0.0611 -> 0.0605
```

在每次调度器更新时，学习率通过与常量因子`gamma`相乘来更新。

## 自定义学习率调度

没有普遍适用的规则表明特定的学习率调度是最有效的。有时，你可能希望拥有 PyTorch 未提供的特殊学习率调度。可以使用自定义函数定义一个自定义学习率调度。例如，你希望有一个学习率为：

$$

lr_n = \dfrac{lr_0}{1 + \alpha n}

$$

在第$n$个周期，其中$lr_0$是第 0 个周期的初始学习率，$\alpha$是常量。你可以实现一个函数，给定周期$n$计算学习率$lr_n$：

```py
def lr_lambda(epoch):
    # LR to be 0.1 * (1/1+0.01*epoch)
    base_lr = 0.1
    factor = 0.01
    return base_lr/(1+factor*epoch)
```

然后，你可以设置`LambdaLR()`以根据以下函数更新学习率：

```py
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

修改之前的示例以使用`LambdaLR()`，你将得到以下内容：

```py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# load dataset, split into input (X) and output (y) variables
dataframe = pd.read_csv("ionosphere.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:34].astype(float)
y = dataset[:,34]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# convert into PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# train-test split for evaluation of the model
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# create model
model = nn.Sequential(
    nn.Linear(34, 34),
    nn.ReLU(),
    nn.Linear(34, 1),
    nn.Sigmoid()
)

def lr_lambda(epoch):
    # LR to be 0.1 * (1/1+0.01*epoch)
    base_lr = 0.1
    factor = 0.01
    return base_lr/(1+factor*epoch)

# Train the model
n_epochs = 50
batch_size = 24
batch_start = torch.arange(0, len(X_train), batch_size)
lr = 0.1
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
model.train()
for epoch in range(n_epochs):
    for start in batch_start:
        X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]
    print("Epoch %d: SGD lr %.4f -> %.4f" % (epoch, before_lr, after_lr))

# evaluate accuracy after training
model.eval()
y_pred = model(X_test)
acc = (y_pred.round() == y_test).float().mean()
acc = float(acc)
print("Model accuracy: %.2f%%" % (acc*100))
```

其结果为：

```py
Epoch 0: SGD lr 0.0100 -> 0.0099
Epoch 1: SGD lr 0.0099 -> 0.0098
Epoch 2: SGD lr 0.0098 -> 0.0097
Epoch 3: SGD lr 0.0097 -> 0.0096
Epoch 4: SGD lr 0.0096 -> 0.0095
...
Epoch 45: SGD lr 0.0069 -> 0.0068
Epoch 46: SGD lr 0.0068 -> 0.0068
Epoch 47: SGD lr 0.0068 -> 0.0068
Epoch 48: SGD lr 0.0068 -> 0.0067
Epoch 49: SGD lr 0.0067 -> 0.0067
```

注意，虽然提供给`LambdaLR()`的函数假设有一个参数`epoch`，但它并不与训练循环中的周期绑定，而只是计数你调用了多少次`scheduler.step()`。

## 使用学习率调度的技巧

本节列出了一些在使用神经网络的学习率调度时需要考虑的技巧和窍门。

+   **增加初始学习率**。因为学习率很可能会减小，所以从较大的值开始减小。较大的学习率将导致权重产生更大的变化，至少在开始阶段是这样，这样可以使您后续的微调更加有效。

+   **使用较大的动量**。许多优化器可以考虑动量。使用较大的动量值将有助于优化算法在学习率减小到较小值时继续朝正确方向进行更新。

+   **尝试不同的调度**。不清楚要使用哪种学习率调度，因此尝试几种不同的配置选项，看看哪种在解决您的问题时效果最好。还可以尝试指数变化的调度，甚至可以根据模型在训练或测试数据集上的准确性响应的调度。

## 进一步阅读

以下是有关在 PyTorch 中使用学习率的更多详细文档：

+   [如何调整学习率](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)，来自 PyTorch 文档

## 摘要

在本文中，您发现了用于训练神经网络模型的学习率调度。

阅读本文后，您学到了：

+   学习率如何影响您的模型训练

+   如何在 PyTorch 中设置学习率调度

+   如何创建自定义学习率调度
