# 使用 PyTorch 进行深度学习（9 天迷你课程）

> 原文：[`machinelearningmastery.com/deep-learning-with-pytorch-9-day-mini-course/`](https://machinelearningmastery.com/deep-learning-with-pytorch-9-day-mini-course/)

深度学习是一个迷人的研究领域，其技术在一系列具有挑战性的机器学习问题上取得了世界级的成果。开始深入学习深度学习可能有些困难。

**你应该使用哪个库，以及应该专注于哪些技术？**

在这个由 9 部分组成的速成课程中，你将会使用易于使用且强大的 PyTorch 库发现 Python 中的应用深度学习。这个迷你课程旨在为已经熟悉 Python 编程且了解基本机器学习概念的实践者提供帮助。让我们开始吧。

这是一篇长而实用的文章。你可能想要打印出来。

让我们开始吧。

![](img/9471c948e144cb14d6168d40fad1bcbb.png)

使用 PyTorch 进行深度学习（9 天迷你课程）

照片由[Cosmin Georgian](https://unsplash.com/photos/people-near-pagoda-under-white-and-blue-sky-gd3ysFyrsTQ)拍摄。部分权利保留。

## 这个迷你课程适合谁？

在我们开始之前，让我们确保你在正确的地方。下面的列表提供了一些关于这门课程设计对象的一般指导方针。如果你不完全符合这些点，不要惊慌，你可能只需要在某个领域或另一个领域进行一些复习以跟上节奏。

+   **知道如何写一点代码的开发者**。这意味着对于你来说，用 Python 完成任务并在工作站上设置生态系统并不是什么大问题（这是一个先决条件）。这并不意味着你是编程巫师，但确实意味着你不怕安装软件包和编写脚本。

+   **了解一点机器学习的开发者**。这意味着你了解机器学习的基础知识，如交叉验证、一些算法和偏差-方差权衡。这并不意味着你是机器学习博士，只是说你知道里程碑或知道在哪里查找它们。

这个迷你课程不是一本深度学习的教科书。

它将带领你从一个在 Python 中略懂机器学习的开发者，变成一个能够产生结果并将深度学习的力量引入自己项目的开发者。

## 迷你课程概述

这个迷你课程分为 9 部分。

每一课设计为一般开发者约 30 分钟完成。有些课可能会更快完成，有些你可能会选择深入学习，花更多时间。

您可以根据自己的节奏完成每一部分。一个舒适的时间表可能是每天完成一课，共九天。强烈推荐。

在接下来的 9 课中，您将学习以下主题：

+   **第 1 课**: PyTorch 简介

+   **第 2 课**: 构建你的第一个多层感知器模型

+   **第 3 课**: 训练一个 PyTorch 模型

+   **第 4 课**: 使用 PyTorch 模型进行推断

+   **第 5 课**: 从 Torchvision 加载数据

+   **第 6 课**: 使用 PyTorch DataLoader

+   **第 7 课**：卷积神经网络

+   **第 8 课**：训练图像分类器

+   **第 9 课**：使用 GPU 训练

这将会非常有趣。

你需要做一些工作，包括一点阅读、一些研究和一点编程。你想学习深度学习，对吧？

**在评论中发布你的结果**；我会为你加油！

坚持下去，别放弃。

## 第 01 课：PyTorch 简介

PyTorch 是由 Facebook 创建和发布的一个用于深度学习计算的 Python 库。它源自早期的库 Torch 7，但完全重写了。

这是最受欢迎的两个深度学习库之一。PyTorch 是一个完整的库，具备训练深度学习模型的能力，同时支持在推理模式下运行模型，并支持使用 GPU 以加速训练和推理。它是一个我们不能忽视的平台。

在本课中，你的目标是安装 PyTorch，并熟悉 PyTorch 程序中使用的符号表达式的语法。

例如，你可以使用`pip`安装 PyTorch。在撰写本文时，PyTorch 的最新版本是 2.0。PyTorch 为每个平台提供了预构建版本，包括 Windows、Linux 和 macOS。只要有一个有效的 Python 环境，`pip`会为你处理这些，以提供你平台上的最新版本。

除了 PyTorch，还有`torchvision`库，它通常与 PyTorch 一起使用。它提供了许多有用的函数来帮助计算机视觉项目。

```py
sudo pip install torch torchvision
```

一个可以作为起点的小示例 PyTorch 程序如下所示：

```py
# Example of PyTorch library
import torch
# declare two symbolic floating-point scalars
a = torch.tensor(1.5)
b = torch.tensor(2.5)
# create a simple symbolic expression using the add function
c = torch.add(a, b)
print(c)
```

在[PyTorch 主页](https://www.pytorch.org/)上了解更多关于 PyTorch 的信息。

### 你的任务

重复以上代码以确保你已正确安装 PyTorch。你还可以通过运行以下 Python 代码行来检查你的 PyTorch 版本：

```py
import torch
print(torch.__version__)
```

在下一课中，你将使用 PyTorch 构建一个神经网络模型。

## 第 02 课：构建你的第一个多层感知器模型

深度学习是构建大规模神经网络的过程。神经网络的最简单形式称为多层感知器模型。神经网络的构建块是人工神经元或感知器。这些是简单的计算单元，具有加权输入信号，并使用激活函数产生输出信号。

感知器被排列成网络。一排感知器被称为一个层，一个网络可以有多个层。网络中感知器的架构通常称为网络拓扑。一旦配置完成，神经网络需要在你的数据集上进行训练。经典且仍然首选的神经网络训练算法称为随机梯度下降。

![简单神经元模型](img/c8f8094f52ec2ce1580e555f70538bf9.png)

简单神经元模型

PyTorch 允许你用极少的代码行开发和评估深度学习模型。

在接下来的内容中，你的目标是使用 PyTorch 开发你的第一个神经网络。使用来自 UCI 机器学习库的标准二分类数据集，如[Pima Indians 数据集](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)。

为了保持简单，网络模型仅由几层**全连接**感知机组成。在这个特定模型中，数据集有 12 个输入或**预测变量**，输出是一个 0 或 1 的单一值。因此，网络模型应有 12 个输入（在第一层）和 1 个输出（在最后一层）。你的第一个模型将按如下方式构建：

```py
import torch.nn as nn

model = nn.Sequential(
  nn.Linear(8, 12),
  nn.ReLU(),
  nn.Linear(12, 8),
  nn.ReLU(),
  nn.Linear(8, 1),
  nn.Sigmoid()
)
print(model)
```

这是一个包含 3 层全连接层的网络。每一层都是使用`nn.Linear(x, y)`语法在 PyTorch 中创建的，其中第一个参数是输入到该层的数量，第二个参数是输出的数量。在每一层之间，使用了修正线性激活函数，但在输出层，应用了 sigmoid 激活函数，使得输出值介于 0 和 1 之间。这是一个典型的网络。深度学习模型通常会在模型中包含许多这样的层。

### 你的任务

重复上述代码并观察打印的模型输出。尝试在上述第一个`Linear`层之后添加另一个输出 20 个值的层。你应该如何修改`nn.Linear(12, 8)`这一行以适应这个添加的层？

在下一课中，你将看到如何训练这个模型。

## 课程 03: 训练一个 PyTorch 模型

在 PyTorch 中构建神经网络并没有说明你应该如何为特定任务训练模型。实际上，在这方面有很多变种，这些变种由**超参数**描述。在 PyTorch 或所有深度学习模型中，你需要决定以下内容来训练模型：

+   数据集是什么，特别是输入和目标的样子如何？

+   什么是评估模型拟合数据优度的损失函数？

+   用于训练模型的优化算法是什么，以及优化算法的参数如学习率和次数是什么？

    训练的迭代次数

在上一课中，使用了 Pima Indians 数据集，并且所有输入都是数字。这将是最简单的情况，因为你不需要对数据进行任何预处理，因为神经网络可以直接处理数字。

由于这是一个二分类问题，因此损失函数应该是二元交叉熵。这意味着模型输出的目标值是 0 或 1，用于分类结果。但在实际中，模型可能输出介于两者之间的任何值。离目标值越近越好（即，**损失**越低）。

梯度下降是优化神经网络的算法。梯度下降有许多变种，而 Adam 是最常用的算法之一。

实现上述所有内容，加上在上一课中构建的模型，以下是训练过程的代码：

```py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

loss_fn = nn.BCELoss() # binary cross-entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 10
for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')
```

上述的 for 循环用于获取一个**批次**的数据并将其输入到模型中。然后观察模型的输出并计算损失函数。根据损失函数，优化器将对模型进行一步微调，以便更好地匹配训练数据。经过若干次更新步骤后，模型应该足够接近训练数据，以便能够以较高的准确率预测目标。

### 你的任务

运行上述训练循环，并观察随着训练循环的进行，损失如何减少。

在下一课中，你将看到如何使用训练好的模型。

## 课程 04：使用 PyTorch 模型进行推断

一个训练好的神经网络模型是一个记住了输入和目标之间关系的模型。然后，该模型可以在给定另一个输入的情况下预测目标。

在 PyTorch 中，一个训练好的模型可以像函数一样运行。假设你已经在前一课中训练了这个模型，你可以简单地如下使用它：

```py
i = 5
X_sample = X[i:i+1]
y_pred = model(X_sample)
print(f"{X_sample[0]} -> {y_pred[0]}")
```

但实际上，更好的推断方法是如下：

```py
i = 5
X_sample = X[i:i+1]
model.eval()
with torch.no_grad():
    y_pred = model(X_sample)
print(f"{X_sample[0]} -> {y_pred[0]}")
```

一些模型在训练和推断之间表现不同。`model.eval()` 这一行是为了告诉模型意图是进行推断。`with torch.no_grad()` 这一行是为了创建一个运行模型的上下文，以便 PyTorch 知道计算梯度是不必要的。这可以减少资源消耗。

这也是你可以评估模型的方式。模型输出一个 sigmoid 值，该值在 0 和 1 之间。你可以通过将值四舍五入到最接近的整数（即布尔标签）来解释这个值。通过比较四舍五入后的预测与目标匹配的频率，你可以为模型分配一个准确率百分比，如下：

```py
model.eval()
with torch.no_grad():
    y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")
```

### 你的任务

运行上述代码，看看你得到的准确率是多少。你应该大致达到 75%。

在下一课中，你将学习关于 torchvision 的内容。

## 课程 05：从 Torchvision 加载数据

Torchvision 是 PyTorch 的姊妹库。在这个库中，有专门用于图像和计算机视觉的函数。正如你所期望的，有帮助你读取图像或调整对比度的函数。但可能最重要的是提供一个易于获取一些图像数据集的接口。

在下一课中，你将构建一个深度学习模型来分类小图像。这是一个使计算机能够识别图像内容的模型。正如你在前面的课程中看到的，拥有数据集来训练模型是非常重要的。你将要使用的数据集是 CIFAR-10。它是一个包含 10 种不同物体的数据集。还有一个更大的数据集叫做 CIFAR-100。

CIFAR-10 数据集可以从互联网下载。但是如果你已经安装了 torchvision，你只需执行以下操作：

```py
import matplotlib.pyplot as plt
import torchvision

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

fig, ax = plt.subplots(4, 6, sharex=True, sharey=True, figsize=(12,8))
for i in range(0, 24):
    row, col = i//6, i%6
    ax[row][col].imshow(trainset.data[i])
plt.show()
```

`torchvision.datasets.CIFAR10` 函数帮助你将 CIFAR-10 数据集下载到本地目录。数据集分为训练集和测试集。因此，上面的两行代码是为了获取它们。然后你绘制从下载的数据集中获得的前 24 张图像。数据集中的每张图像是 32×32 像素的以下任意一种：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船或卡车。

### 你的任务

根据上述代码，你能找到一种方法来分别计算训练集和测试集中总共有多少张图像吗？

在下一课中，你将学习如何使用 PyTorch DataLoader。

## 课程 06：使用 PyTorch DataLoader

上一课中的 CIFAR-10 图像确实是 numpy 数组格式。但为了供 PyTorch 模型使用，它需要是 PyTorch 张量。将 numpy 数组转换为 PyTorch 张量并不难，但在训练循环中，你仍然需要将数据集划分为批次。PyTorch DataLoader 可以帮助你使这个过程更加顺畅。

返回到上一课中加载的 CIFAR-10 数据集，你可以做以下操作以实现相同的效果：

```py
import matplotlib.pyplot as plt
import torchvision
import torch
from torchvision.datasets import CIFAR10

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = CIFAR10(root='./data', train=False, download=True, transform=transform)

batch_size = 24
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

fig, ax = plt.subplots(4, 6, sharex=True, sharey=True, figsize=(12,8))
for images, labels in trainloader:
    for i in range(batch_size):
        row, col = i//6, i%6
        ax[row][col].imshow(images[i].numpy().transpose([1,2,0]))
    break  # take only the first batch
plt.show()
```

在这段代码中，`trainset` 是通过 `transform` 参数创建的，这样数据在提取时会转换为 PyTorch 张量。这是在 `DataLoader` 后续的行中执行的。`DataLoader` 对象是一个 Python 可迭代对象，你可以提取输入（即图像）和目标（即整数类别标签）。在这种情况下，你将批量大小设置为 24，并迭代第一个批次。然后你展示批次中的每张图像。

### 你的任务

运行上面的代码，并与你在上一课中生成的 matplotlib 输出进行比较。你应该会看到输出不同。为什么？在`DataLoader`行中有一个参数导致了这个差异。你能找出是哪一个吗？

在下一课中，你将学习如何构建深度学习模型来分类 CIFAR-10 数据集中的图像。

## 课程 07：卷积神经网络

图像是二维结构。你可以通过将其展开为一维向量来轻松地转换它们，并构建神经网络模型对其进行分类。但已知保留二维结构更为合适，因为分类涉及的是图像中的内容，这具有**平移不变性**。

处理图像的神经网络标准方法是使用卷积层。使用卷积层的神经网络称为卷积神经网络。示例如下：

```py
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2)),
    nn.Flatten(),
    nn.Linear(8192, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 10)
)
print(model)
```

在上述内容中，我们使用了多次 `Conv2d` 层以及 `ReLU` 激活。卷积层用于学习和提取图像的 **特征**。你添加的卷积层越多，网络可以学习到更多的高级特征。最终，你会使用一个池化层（上面的 `MaxPool2d`）来对提取的特征进行分组，将它们展平为一个向量，然后传递给一个多层感知机网络进行最终分类。这是图像分类模型的常见结构。

### 你的任务

运行上述代码以确保你可以正确创建一个模型。你没有在模型中指定输入图像的大小，但它实际上被固定为 32×32 像素的 RGB（即 3 个颜色通道）。这一固定设置在网络中在哪里？

在下一课中，你将使用上一课中的 DataLoader 来训练上述模型。

## 课程 08：训练图像分类器

配合为 CIFAR-10 数据集创建的 DataLoader，你可以使用以下训练循环来训练前一课中的卷积神经网络：

```py
import torch.nn as nn
import torch.optim as optim

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

n_epochs = 20
for epoch in range(n_epochs):
    model.train()
    for inputs, labels in trainloader:
        y_pred = model(inputs)
        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            y_pred = model(inputs)
            acc += (torch.argmax(y_pred, 1) == labels).float().sum()
            count += len(labels)
    acc /= count
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))
```

这将需要一些时间运行，你应该看到模型能够达到至少 70% 的准确率。

这个模型是一个多类别分类网络。输出不是一个，而是多个分数，每个类别一个。我们认为分数越高，模型越有信心图像属于某个类别。因此，使用的损失函数是 **交叉熵**，即多类别版本的二元交叉熵。

在上述训练循环中，你应该会看到许多你在前面课程中学到的元素，包括在模型中切换训练模式和推理模式，使用 `torch.no_grad()` 上下文，以及准确率的计算。

### 你的任务

阅读上述代码以确保你理解它的作用。运行此代码以观察随着训练的进行准确率的提高。你最终达到了什么准确率？

在下一课中，你将学习如何使用 GPU 加速同一模型的训练。

## 课程 09：使用 GPU 进行训练

你在上一课中进行的模型训练应该需要一段时间。如果你有支持的 GPU，你可以大大加快训练速度。

在 PyTorch 中使用 GPU 的方法是先将模型和数据发送到 GPU，然后可以选择将结果从 GPU 发送回 CPU，或者直接在 GPU 上进行评估。

修改上一课的代码以使用 GPU 并不困难。下面是需要做的内容：

```py
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

n_epochs = 20
for epoch in range(n_epochs):
    model.train()
    for inputs, labels in trainloader:
        y_pred = model(inputs.to(device))
        loss = loss_fn(y_pred, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            y_pred = model(inputs.to(device))
            acc += (torch.argmax(y_pred, 1) == labels.to(device)).float().sum()
            count += len(labels)
    acc /= count
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))
```

所做的更改如下：你检查 GPU 是否可用并相应地设置 `device`。然后将模型发送到该设备。当输入（即一批图像）传递到模型时，它也应该首先发送到相应的设备。由于模型输出也会在那里，因此损失计算或准确率计算也应将目标首先发送到 GPU。

### 你的任务

你可以看到，在 CPU 和 GPU 上运行 PyTorch 的方式大致相同。如果你可以访问到 GPU，尝试比较这两者的速度。你能观察到快了多少？

这是最后一课。

## 结束！（*看看你走了多远*）

你做到了。做得好！

花点时间回顾一下你走过的路程。

+   你发现了 PyTorch 作为一个 Python 中的深度学习库。

+   你使用 PyTorch 构建了你的第一个神经网络，并学习了如何用神经网络进行分类。

+   你学习了深度学习的关键组成部分，包括损失函数、优化器、训练循环和评估。

+   最后，你迈出了下一步，学习了关于卷积神经网络以及如何用于计算机视觉任务。

## 总结

**你在迷你课程中表现如何？**

你喜欢这个速成课程吗？

**你有任何问题吗？有没有什么难点？**

让我知道。请在下面留言。
