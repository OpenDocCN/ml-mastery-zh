# 为 PyTorch 模型创建训练循环

> 原文：[`machinelearningmastery.com/creating-a-training-loop-for-pytorch-models/`](https://machinelearningmastery.com/creating-a-training-loop-for-pytorch-models/)

PyTorch 提供了许多深度学习模型的构建模块，但训练循环并不包括在其中。这种灵活性允许你在训练过程中做任何你想做的事情，但某些基本结构在大多数使用场景中是通用的。

在本文中，你将看到如何创建一个训练循环，为你的模型训练提供必要的信息，并可以选择显示任何信息。完成本文后，你将了解：

+   训练循环的基本构建块

+   如何使用 tqdm 显示训练进度

**用我的书 [Deep Learning with PyTorch](https://machinelearningmastery.com/deep-learning-with-pytorch/) 启动你的项目**。它提供了 **自学教程** 和 **实用代码**。

让我们开始吧。![](img/0672407873f4aff7bd5553961f40e664.png)

为 PyTorch 模型创建训练循环

图片由 [pat pat](https://unsplash.com/photos/4DE9h3fpLiI) 提供。版权所有。

## 概述

本文分为三部分，分别是：

+   深度学习模型的训练要素

+   在训练期间收集统计数据

+   使用 tqdm 报告训练进度

## 深度学习模型的训练要素

与所有机器学习模型一样，模型设计指定了操作输入并生成输出的算法。但在模型中，有些参数需要调整以实现这一目标。这些模型参数也被称为权重、偏差、内核或其他名称，具体取决于特定模型和层。训练是将样本数据输入模型，以便优化器可以调整这些参数。

当你训练一个模型时，你通常从一个数据集开始。每个数据集包含大量的数据样本。当你获得数据集时，建议将其分为两个部分：训练集和测试集。训练集进一步分为批次，并在训练循环中使用，以驱动梯度下降算法。然而，测试集用作基准，以判断你的模型表现如何。通常，你不会将训练集作为度量，而是使用测试集，因为测试集没有被梯度下降算法看到，从而可以判断你的模型是否对未见过的数据适应良好。

过拟合是指模型在训练集上表现得过于好（即，非常高的准确率），但在测试集上的表现显著下降。欠拟合是指模型甚至无法在训练集上表现良好。自然，你不希望在一个好的模型中看到这两种情况。

神经网络模型的训练是以周期为单位的。通常，一个周期意味着你遍历整个训练集一次，尽管你一次只输入一个批次。在每个周期结束时，通常会做一些例行任务，如使用测试集对部分训练好的模型进行基准测试、检查点模型、决定是否提前停止训练、收集训练统计数据等。

在每个周期中，你将数据样本以批次的形式输入模型，并运行梯度下降算法。这是训练循环中的一步，因为你在一次前向传递（即，提供输入并捕获输出）和一次反向传递（从输出评估损失指标并将每个参数的梯度反向到输入层）中运行模型。反向传递使用自动微分来计算梯度。然后，这些梯度由梯度下降算法用于调整模型参数。一个周期包含多个步骤。

复用[之前教程](https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/)中的示例，你可以下载[数据集](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)并将数据集拆分为两部分，如下所示：

```py
import numpy as np
import torch

# load the dataset
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# split the dataset into training and test sets
Xtrain = X[:700]
ytrain = y[:700]
Xtest = X[700:]
ytest = y[700:]
```

这个数据集很小——只有 768 个样本。在这里，它将前 700 个样本作为训练集，其余的作为测试集。

这不是本文的重点，但你可以复用[之前文章](https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/)中的模型、损失函数和优化器：

```py
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)
print(model)

# loss function and optimizer
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

有了数据和模型，这就是最简训练循环，每一步都有前向和反向传递：

```py
n_epochs = 50    # number of epochs to run
batch_size = 10  # size of each batch
batches_per_epoch = len(Xtrain) // batch_size

for epoch in range(n_epochs):
    for i in range(batches_per_epoch):
        start = i * batch_size
        # take a batch
        Xbatch = Xtrain[start:start+batch_size]
        ybatch = ytrain[start:start+batch_size]
        # forward pass
        y_pred = model(Xbatch)
        loss = loss_fn(y_pred, ybatch)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
```

在内部的 for 循环中，你取数据集中的每一个批次并评估损失。损失是一个 PyTorch 张量，它记住了如何得出其值。然后你将优化器管理的所有梯度清零，并调用`loss.backward()`来运行反向传播算法。结果设置了所有张量的梯度，这些张量直接或间接地依赖于张量`loss`。随后，调用`step()`时，优化器将检查其管理的每个参数并更新它们。

完成所有步骤后，你可以使用测试集运行模型以评估其性能。评估可以基于不同于损失函数的函数。例如，这个分类问题使用准确率：

```py
...

# evaluate trained model with test set
with torch.no_grad():
    y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print("Accuracy {:.2f}".format(accuracy * 100))
```

将一切整合在一起，这就是完整代码：

```py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# load the dataset
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# split the dataset into training and test sets
Xtrain = X[:700]
ytrain = y[:700]
Xtest = X[700:]
ytest = y[700:]

model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)
print(model)

# loss function and optimizer
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 50    # number of epochs to run
batch_size = 10  # size of each batch
batches_per_epoch = len(Xtrain) // batch_size

for epoch in range(n_epochs):
    for i in range(batches_per_epoch):
        start = i * batch_size
        # take a batch
        Xbatch = Xtrain[start:start+batch_size]
        ybatch = ytrain[start:start+batch_size]
        # forward pass
        y_pred = model(Xbatch)
        loss = loss_fn(y_pred, ybatch)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()

# evaluate trained model with test set
with torch.no_grad():
    y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print("Accuracy {:.2f}".format(accuracy * 100))
```

## 训练期间收集统计数据

上述训练循环应该适用于可以在几秒钟内完成训练的小模型。但对于较大的模型或较大的数据集，你会发现训练所需的时间显著增加。在等待训练完成的同时，你可能希望查看进度，以便在出现任何错误时中断训练。

通常，在训练过程中，你希望看到以下内容：

+   在每一步中，你想要知道损失指标，并期望损失降低。

+   在每一步中，你想要了解其他指标，例如训练集上的准确率，这些指标是感兴趣的但不参与梯度下降。

+   在每个 epoch 结束时，你想要用测试集评估部分训练的模型并报告评估指标。

+   在训练结束时，你希望能够可视化以上指标。

这些都是可能的，但是你需要在训练循环中添加更多代码，如下所示：

```py
n_epochs = 50    # number of epochs to run
batch_size = 10  # size of each batch
batches_per_epoch = len(Xtrain) // batch_size

# collect statistics
train_loss = []
train_acc = []
test_acc = []

for epoch in range(n_epochs):
    for i in range(batches_per_epoch):
        start = i * batch_size
        # take a batch
        Xbatch = Xtrain[start:start+batch_size]
        ybatch = ytrain[start:start+batch_size]
        # forward pass
        y_pred = model(Xbatch)
        loss = loss_fn(y_pred, ybatch)
        acc = (y_pred.round() == ybatch).float().mean()
        # store metrics
        train_loss.append(float(loss))
        train_acc.append(float(acc))
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
        # print progress
        print(f"epoch {epoch} step {i} loss {loss} accuracy {acc}")
    # evaluate model at end of epoch
    y_pred = model(Xtest)
    acc = (y_pred.round() == ytest).float().mean()
    test_acc.append(float(acc))
    print(f"End of {epoch}, accuracy {acc}")
```

当你收集损失和准确率到列表中时，你可以使用 matplotlib 将它们绘制出来。但要小心，你在每一步收集了训练集的统计数据，但测试集的准确率只在每个 epoch 结束时。因此，你希望在每个 epoch 中显示训练循环中的平均准确率，以便它们可以相互比较。

```py
import matplotlib.pyplot as plt

# Plot the loss metrics, set the y-axis to start from 0
plt.plot(train_loss)
plt.xlabel("steps")
plt.ylabel("loss")
plt.ylim(0)
plt.show()

# plot the accuracy metrics
avg_train_acc = []
for i in range(n_epochs):
    start = i * batch_size
    average = sum(train_acc[start:start+batches_per_epoch]) / batches_per_epoch
    avg_train_acc.append(average)

plt.plot(avg_train_acc, label="train")
plt.plot(test_acc, label="test")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0)
plt.show()
```

将所有内容整合在一起，以下是完整的代码：

```py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# load the dataset
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',') # split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# split the dataset into training and test sets
Xtrain = X[:700]
ytrain = y[:700]
Xtest = X[700:]
ytest = y[700:]

model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)
print(model)

# loss function and optimizer
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 50    # number of epochs to run
batch_size = 10  # size of each batch
batches_per_epoch = len(Xtrain) // batch_size

# collect statistics
train_loss = []
train_acc = []
test_acc = []

for epoch in range(n_epochs):
    for i in range(batches_per_epoch):
        # take a batch
        start = i * batch_size
        Xbatch = Xtrain[start:start+batch_size]
        ybatch = ytrain[start:start+batch_size]
        # forward pass
        y_pred = model(Xbatch)
        loss = loss_fn(y_pred, ybatch)
        acc = (y_pred.round() == ybatch).float().mean()
        # store metrics
        train_loss.append(float(loss))
        train_acc.append(float(acc))
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
        # print progress
        print(f"epoch {epoch} step {i} loss {loss} accuracy {acc}")
    # evaluate model at end of epoch
    y_pred = model(Xtest)
    acc = (y_pred.round() == ytest).float().mean()
    test_acc.append(float(acc))
    print(f"End of {epoch}, accuracy {acc}")

import matplotlib.pyplot as plt

# Plot the loss metrics
plt.plot(train_loss)
plt.xlabel("steps")
plt.ylabel("loss")
plt.ylim(0)
plt.show()

# plot the accuracy metrics
avg_train_acc = []
for i in range(n_epochs):
    start = i * batch_size
    average = sum(train_acc[start:start+batches_per_epoch]) / batches_per_epoch
    avg_train_acc.append(average)

plt.plot(avg_train_acc, label="train")
plt.plot(test_acc, label="test")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0)
plt.show()
```

故事还没有结束。事实上，你可以在训练循环中添加更多代码，特别是在处理更复杂的模型时。一个例子是检查点。你可能想要保存你的模型（例如使用 pickle），这样，如果出于任何原因你的程序停止，你可以从中间重新启动训练循环。另一个例子是早停，它允许你在每个 epoch 结束时监视测试集的准确率，并在一段时间内看不到模型改进时中断训练。这是因为你可能不能进一步进行，考虑到模型的设计，而且你不想过拟合。

### 想要开始使用 PyTorch 进行深度学习吗？

现在就参加我的免费电子邮件快速课程（附有示例代码）。

点击注册并获取课程的免费 PDF 电子书版本。

## 使用 tqdm 报告训练进度

如果你运行以上代码，你会发现在训练循环运行时屏幕上打印了很多行。你的屏幕可能会很杂乱。而且你可能还想看到一个动画进度条，以更好地告诉你训练进度到了哪一步。`tqdm`库是创建进度条的流行工具。将以上代码转换为使用 tqdm 可以更加简单：

```py
for epoch in range(n_epochs):
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            # take a batch
            start = i * batch_size
            Xbatch = Xtrain[start:start+batch_size]
            ybatch = ytrain[start:start+batch_size]
            # forward pass
            y_pred = model(Xbatch)
            loss = loss_fn(y_pred, ybatch)
            acc = (y_pred.round() == ybatch).float().mean()
            # store metrics
            train_loss.append(float(loss))
            train_acc.append(float(acc))
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            bar.set_postfix(
                loss=float(loss),
                acc=f"{float(acc)*100:.2f}%"
            )
    # evaluate model at end of epoch
    y_pred = model(Xtest)
    acc = (y_pred.round() == ytest).float().mean()
    test_acc.append(float(acc))
    print(f"End of {epoch}, accuracy {acc}")
```

使用`tqdm`创建一个迭代器，使用`trange()`就像 Python 的`range()`函数一样，并且你可以在循环中读取数字。你可以通过更新其描述或“后缀”数据访问进度条，但你必须在其内容耗尽之前这样做。`set_postfix()`函数非常强大，因为它可以显示任何内容。

实际上，除了`trange()`之外还有一个`tqdm()`函数，它迭代现有列表。你可能会发现它更容易使用，并且你可以重写以上循环如下：

```py
starts = [i*batch_size for i in range(batches_per_epoch)]

for epoch in range(n_epochs):
    with tqdm.tqdm(starts, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # take a batch
            Xbatch = Xtrain[start:start+batch_size]
            ybatch = ytrain[start:start+batch_size]
            # forward pass
            y_pred = model(Xbatch)
            loss = loss_fn(y_pred, ybatch)
            acc = (y_pred.round() == ybatch).float().mean()
            # store metrics
            train_loss.append(float(loss))
            train_acc.append(float(acc))
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            bar.set_postfix(
                loss=float(loss),
                acc=f"{float(acc)*100:.2f}%"
            )
    # evaluate model at end of epoch
    y_pred = model(Xtest)
    acc = (y_pred.round() == ytest).float().mean()
    test_acc.append(float(acc))
    print(f"End of {epoch}, accuracy {acc}")
```

以下是完整的代码（不包括 matplotlib 绘图）：

```py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

# load the dataset
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',') # split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# split the dataset into training and test sets
Xtrain = X[:700]
ytrain = y[:700]
Xtest = X[700:]
ytest = y[700:]

model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)
print(model)

# loss function and optimizer
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 50    # number of epochs to run
batch_size = 10  # size of each batch
batches_per_epoch = len(Xtrain) // batch_size

# collect statistics
train_loss = []
train_acc = []
test_acc = []

for epoch in range(n_epochs):
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            # take a batch
            start = i * batch_size
            Xbatch = Xtrain[start:start+batch_size]
            ybatch = ytrain[start:start+batch_size]
            # forward pass
            y_pred = model(Xbatch)
            loss = loss_fn(y_pred, ybatch)
            acc = (y_pred.round() == ybatch).float().mean()
            # store metrics
            train_loss.append(float(loss))
            train_acc.append(float(acc))
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            bar.set_postfix(
                loss=float(loss),
                acc=f"{float(acc)*100:.2f}%"
            )
    # evaluate model at end of epoch
    y_pred = model(Xtest)
    acc = (y_pred.round() == ytest).float().mean()
    test_acc.append(float(acc))
    print(f"End of {epoch}, accuracy {acc}")
```

## 总结

在本文中，你详细了解了如何为 PyTorch 模型正确设置训练循环。具体来说，你看到了：

+   实现训练循环所需的元素是什么。

+   训练循环如何将训练数据与梯度下降优化器连接起来

+   如何在训练循环中收集信息并展示这些信息
