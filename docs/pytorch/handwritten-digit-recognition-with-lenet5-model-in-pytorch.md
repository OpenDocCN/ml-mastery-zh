# 使用 PyTorch 中的 LeNet5 模型进行手写数字识别

> 原文：[`machinelearningmastery.com/handwritten-digit-recognition-with-lenet5-model-in-pytorch/`](https://machinelearningmastery.com/handwritten-digit-recognition-with-lenet5-model-in-pytorch/)

深度学习技术的一个流行演示是图像数据中的对象识别。机器学习和深度学习中的“hello world”是用于手写数字识别的 MNIST 数据集。在本帖中，你将发现如何开发一个深度学习模型，以在 MNIST 手写数字识别任务中达到接近最先进的性能。完成本章后，你将了解：

+   如何使用 torchvision 加载 MNIST 数据集

+   如何为 MNIST 问题开发和评估基线神经网络模型

+   如何实现和评估一个简单的卷积神经网络用于 MNIST

+   如何为 MNIST 实现最先进的深度学习模型

**通过我的书籍** [《PyTorch 深度学习》](https://machinelearningmastery.com/deep-learning-with-pytorch/) **启动你的项目**。它提供了**自学教程**和**实用代码**。

让我们开始吧！[](../Images/72c4d28201df8cb70f7682ca6ed5aeb1.png)

使用 PyTorch 中的 LeNet5 模型进行手写数字识别

照片由 [Johnny Wong](https://unsplash.com/photos/la0WP7U3-AM) 提供。部分权利保留。

## 概述

本帖分为五个部分，它们是：

+   MNIST 手写数字识别问题

+   在 PyTorch 中加载 MNIST 数据集

+   使用多层感知机的基线模型

+   用于 MNIST 的简单卷积神经网络

+   LeNet5 用于 MNIST

## MNIST 手写数字识别问题

MNIST 问题是一个经典问题，可以展示卷积神经网络的强大。MNIST 数据集由 Yann LeCun、Corinna Cortes 和 Christopher Burges 开发，用于评估机器学习模型在手写数字分类问题上的表现。该数据集由来自国家标准与技术研究院（NIST）的多个扫描文档数据集构成。这也是数据集名称的来源，称为 Modified NIST 或 MNIST 数据集。

数字图像来自各种扫描文档，经过大小标准化和居中处理。这使得该数据集非常适合评估模型，开发人员可以专注于机器学习，数据清理或准备工作最小化。每个图像是一个 28×28 像素的灰度方块（总共 784 像素）。数据集的标准拆分用于评估和比较模型，其中 60,000 张图像用于训练模型，另有 10,000 张图像用于测试。

这个问题的目标是识别图像上的数字。需要预测十个数字（0 到 9）或十个类别。当前最先进的预测准确率达到 99.8%，这是通过大型卷积神经网络实现的。

### 想要开始使用 PyTorch 进行深度学习？

现在就参加我的免费电子邮件速成课程（包含示例代码）。

点击注册并获取课程的免费 PDF 电子书版本。

## 在 PyTorch 中加载 MNIST 数据集

`torchvision`库是 PyTorch 的一个姊妹项目，提供用于计算机视觉任务的专门功能。`torchvision`中有一个函数可以下载 MNIST 数据集以供 PyTorch 使用。第一次调用此函数时，数据集会被下载并存储在本地，因此以后不需要再次下载。下面是一个小脚本，用于下载和可视化 MNIST 数据集训练子集中的前 16 张图像。

```py
import matplotlib.pyplot as plt
import torchvision

train = torchvision.datasets.MNIST('./data', train=True, download=True)

fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)
for i in range(4):
    for j in range(4):
        ax[i][j].imshow(train.data[4*i+j], cmap="gray")
plt.show()
```

![](img/34ab7ddbba87120d8b138ee58f69f538.png)

## 多层感知器的基准模型

你真的需要像卷积神经网络这样的复杂模型来获得 MNIST 的最佳结果吗？使用一个非常简单的神经网络模型（具有单隐藏层）也可以获得良好的结果。在本节中，你将创建一个简单的多层感知器模型，其准确率达到 99.81%。你将用这个模型作为与更复杂的卷积神经网络模型比较的基准。首先，让我们检查一下数据的样子：

```py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

# Load MNIST data
train = torchvision.datasets.MNIST('data', train=True, download=True)
test = torchvision.datasets.MNIST('data', train=True, download=True)
print(train.data.shape, train.targets.shape)
print(test.data.shape, test.targets.shape)
```

你应该会看到：

```py
torch.Size([60000, 28, 28]) torch.Size([60000])
torch.Size([10000, 28, 28]) torch.Size([10000])
```

训练数据集的结构是实例、高度和宽度的三维数组。对于多层感知器模型，你必须将图像降维为像素向量。在这种情况下，28×28 大小的图像将成为 784 个像素输入向量。你可以使用`reshape()`函数轻松完成此转换。

像素值为 0 到 255 之间的灰度值。使用神经网络模型时，几乎总是一个好主意对输入值进行一些缩放。因为尺度是已知且行为良好的，你可以通过将每个值除以 255 的最大值来非常快速地将像素值归一化到 0 到 1 的范围内。

在接下来的步骤中，你将转换数据集，将其转换为浮点数，并通过缩放浮点值来归一化它们，你可以在下一步轻松完成归一化。

```py
# each sample becomes a vector of values 0-1
X_train = train.data.reshape(-1, 784).float() / 255.0
y_train = train.targets
X_test = test.data.reshape(-1, 784).float() / 255.0
y_test = test.targets
```

输出目标`y_train`和`y_test`是形式为 0 到 9 的整数标签。这是一个多类别分类问题。你可以将这些标签转换为独热编码（one-hot encoding），或者像本例一样保持为整数标签。你将使用交叉熵函数来评估模型的性能，PyTorch 实现的交叉熵函数可以应用于独热编码的目标或整数标签的目标。

现在你可以创建你的简单神经网络模型了。你将通过 PyTorch 的`Module`类来定义你的模型。

```py
class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 784)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(784, 10)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.layer2(x)
        return x
```

该模型是一个简单的神经网络，具有一个隐藏层，隐藏层的神经元数量与输入数量（784）相同。隐藏层的神经元使用了 rectifier 激活函数。该模型的输出是**logits**，意味着它们是实数，可以通过 softmax 函数转换为类似概率的值。你不需要显式地应用 softmax 函数，因为交叉熵函数会为你完成这项工作。

你将使用随机梯度下降算法（学习率设置为 0.01）来优化这个模型。训练循环如下：

```py
model = Baseline()

optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=100)

n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    y_pred = model(X_test)
    acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))
```

MNIST 数据集很小。这个例子应该在一分钟内完成，结果如下。这个简单的网络可以达到 92% 的准确率。

```py
Epoch 0: model accuracy 84.11%
Epoch 1: model accuracy 87.53%
Epoch 2: model accuracy 89.01%
Epoch 3: model accuracy 89.76%
Epoch 4: model accuracy 90.29%
Epoch 5: model accuracy 90.69%
Epoch 6: model accuracy 91.10%
Epoch 7: model accuracy 91.48%
Epoch 8: model accuracy 91.74%
Epoch 9: model accuracy 91.96%
```

下面是上述 MNIST 数据集多层感知机分类的完整代码。

```py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

# Load MNIST data
train = torchvision.datasets.MNIST('data', train=True, download=True)
test = torchvision.datasets.MNIST('data', train=True, download=True)

# each sample becomes a vector of values 0-1
X_train = train.data.reshape(-1, 784).float() / 255.0
y_train = train.targets
X_test = test.data.reshape(-1, 784).float() / 255.0
y_test = test.targets

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 784)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(784, 10)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.layer2(x)
        return x

model = Baseline()

optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=100)

n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    y_pred = model(X_test)
    acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))
```

## 简单的卷积神经网络用于 MNIST

现在你已经了解了如何使用多层感知机模型对 MNIST 数据集进行分类。接下来，让我们尝试一个卷积神经网络模型。在这一部分，你将创建一个简单的 CNN，用于 MNIST，展示如何使用现代 CNN 实现的所有方面，包括卷积层、池化层和 dropout 层。

在 PyTorch 中，卷积层应该处理图像。图像的张量应该是像素值，维度为 (sample, channel, height, width)，但当你使用 PIL 等库加载图像时，像素通常以 (height, width, channel) 的维度呈现。可以使用 `torchvision` 库中的转换将其转换为适当的张量格式。

```py
...
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,), (128,)),
])
train = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
test = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=100)
testloader = torch.utils.data.DataLoader(test, shuffle=True, batch_size=100)
```

你需要使用 `DataLoader`，因为在从 `DataLoader` 读取数据时会应用转换。

接下来，定义你的神经网络模型。卷积神经网络比标准的多层感知机更复杂，因此你将从使用简单结构开始，这些结构利用了所有元素以实现最先进的结果。下面总结了网络架构。

1.  第一个隐藏层是一个卷积层，`nn.Conv2d()`。该层将灰度图像转换为 10 个特征图，滤波器大小为 5×5，并使用 ReLU 激活函数。这是一个输入层，期望输入的图像结构如上所述。

1.  接下来是一个池化层，取最大值，`nn.MaxPool2d()`。它配置为 2×2 的池化大小，步幅为 1。它的作用是在每个通道的 2×2 像素块中取最大值，并将该值分配给输出像素。结果是每个通道的特征图为 27×27 像素。

1.  下一个层是使用 dropout 的正则化层，`nn.Dropout()`。它配置为随机排除 20% 的神经元，以减少过拟合。

1.  接下来是一个将 2D 矩阵数据转换为向量的层，使用 `nn.Flatten`。输入有 10 个通道，每个通道的特征图大小为 27×27。此层允许输出由标准的全连接层处理。

1.  接下来是一个具有 128 个神经元的全连接层。使用 ReLU 激活函数。

1.  最后，输出层有十个神经元，用于十个类别。您可以通过在其上应用 softmax 函数将输出转换为类似概率的预测。

此模型使用交叉熵损失和 Adam 优化算法进行训练。实现如下：

```py
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.dropout = nn.Dropout(0.2)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(27*27*10, 128)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu1(self.conv(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu2(self.fc(self.flat(x)))
        x = self.output(x)
        return x

model = CNN()

optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in trainloader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    acc = 0
    count = 0
    for X_batch, y_batch in testloader:
        y_pred = model(X_batch)
        acc += (torch.argmax(y_pred, 1) == y_batch).float().sum()
        count += len(y_batch)
    acc = acc / count
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))
```

运行上述操作需要几分钟，并产生以下结果：

```py
Epoch 0: model accuracy 81.74%
Epoch 1: model accuracy 85.38%
Epoch 2: model accuracy 86.37%
Epoch 3: model accuracy 87.75%
Epoch 4: model accuracy 88.00%
Epoch 5: model accuracy 88.17%
Epoch 6: model accuracy 88.81%
Epoch 7: model accuracy 88.34%
Epoch 8: model accuracy 88.86%
Epoch 9: model accuracy 88.75%
```

不是最佳结果，但这展示了卷积层如何工作。

下面是使用简单卷积网络的完整代码。

```py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

# Load MNIST data
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,), (128,)),
])
train = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
test = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=100)
testloader = torch.utils.data.DataLoader(test, shuffle=True, batch_size=100)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.dropout = nn.Dropout(0.2)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(27*27*10, 128)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu1(self.conv(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu2(self.fc(self.flat(x)))
        x = self.output(x)
        return x

model = CNN()

optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in trainloader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    acc = 0
    count = 0
    for X_batch, y_batch in testloader:
        y_pred = model(X_batch)
        acc += (torch.argmax(y_pred, 1) == y_batch).float().sum()
        count += len(y_batch)
    acc = acc / count
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))
```

## LeNet5 用于 MNIST

前一模型仅具有一个卷积层。当然，您可以添加更多层以构建更深的模型。卷积层在神经网络中的有效性最早的演示之一是“LeNet5”模型。该模型旨在解决 MNIST 分类问题。它有三个卷积层和两个全连接层，共五个可训练层。

在其开发时期，使用双曲正切函数作为激活函数很常见。因此在这里使用它。该模型实现如下：

```py
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.act2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.act3 = nn.Tanh()

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(1*1*120, 84)
        self.act4 = nn.Tanh()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        # input 1x28x28, output 6x28x28
        x = self.act1(self.conv1(x))
        # input 6x28x28, output 6x14x14
        x = self.pool1(x)
        # input 6x14x14, output 16x10x10
        x = self.act2(self.conv2(x))
        # input 16x10x10, output 16x5x5
        x = self.pool2(x)
        # input 16x5x5, output 120x1x1
        x = self.act3(self.conv3(x))
        # input 120x1x1, output 84
        x = self.act4(self.fc1(self.flat(x)))
        # input 84, output 10
        x = self.fc2(x)
        return x
```

与前一模型相比，LeNet5 没有 Dropout 层（因为 Dropout 层是在 LeNet5 几年后才被发明的），而是使用平均池化代替最大池化（即对 2×2 像素的区域取像素值的平均值而不是最大值）。但 LeNet5 模型最显著的特征是使用步长和填充来将图像尺寸从 28×28 像素减小到 1×1 像素，并将通道数从一个（灰度）增加到 120。

填充意味着在图像边界添加值为 0 的像素，使其稍微变大。没有填充时，卷积层的输出将比其输入小。步幅参数控制滤波器移动以生成输出中的下一个像素。通常为 1 以保持相同大小。如果大于 1，则输出是输入的**下采样**。因此在 LeNet5 模型中，池化层中使用步幅 2，例如将 28×28 像素图像变为 14×14。

训练该模型与训练之前的卷积网络模型相同，如下所示：

```py
...
model = LeNet5()

optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in trainloader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    acc = 0
    count = 0
    for X_batch, y_batch in testloader:
        y_pred = model(X_batch)
        acc += (torch.argmax(y_pred, 1) == y_batch).float().sum()
        count += len(y_batch)
    acc = acc / count
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))
```

运行此代码可能会看到：

```py
Epoch 0: model accuracy 89.46%
Epoch 1: model accuracy 93.14%
Epoch 2: model accuracy 94.69%
Epoch 3: model accuracy 95.84%
Epoch 4: model accuracy 96.43%
Epoch 5: model accuracy 96.99%
Epoch 6: model accuracy 97.14%
Epoch 7: model accuracy 97.66%
Epoch 8: model accuracy 98.05%
Epoch 9: model accuracy 98.22%
```

在这里，我们实现了超过 98%的准确率。

以下是完整的代码。

```py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

# Load MNIST data
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,), (128,)),
])
train = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
test = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=100)
testloader = torch.utils.data.DataLoader(test, shuffle=True, batch_size=100)

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.act2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.act3 = nn.Tanh()

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(1*1*120, 84)
        self.act4 = nn.Tanh()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        # input 1x28x28, output 6x28x28
        x = self.act1(self.conv1(x))
        # input 6x28x28, output 6x14x14
        x = self.pool1(x)
        # input 6x14x14, output 16x10x10
        x = self.act2(self.conv2(x))
        # input 16x10x10, output 16x5x5
        x = self.pool2(x)
        # input 16x5x5, output 120x1x1
        x = self.act3(self.conv3(x))
        # input 120x1x1, output 84
        x = self.act4(self.fc1(self.flat(x)))
        # input 84, output 10
        x = self.fc2(x)
        return x

model = LeNet5()

optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in trainloader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    acc = 0
    count = 0
    for X_batch, y_batch in testloader:
        y_pred = model(X_batch)
        acc += (torch.argmax(y_pred, 1) == y_batch).float().sum()
        count += len(y_batch)
    acc = acc / count
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))
```

## MNIST 资源

MNIST 数据集已经非常研究。以下是您可能想要查看的一些额外资源。

+   Yann LeCun，Corinna Cortes 和 Christopher J.C. Burges。[手写数字 MNIST 数据库。](http://yann.lecun.com/exdb/mnist/)

+   Rodrigo Benenson。[这张图像属于哪个类？分类数据集结果](https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.htm)，2016 年。

+   [数字识别器](https://www.kaggle.com/c/digit-recognizer)：使用著名的 MNIST 数据学习计算机视觉基础知识。Kaggle。

+   Hubert Eichner。[JavaScript 中手写数字识别的神经网络。](http://myselph.de/neuralNet.html)

## 总结

在这篇文章中，你了解了 MNIST 手写数字识别问题以及使用 Python 和 Keras 库开发的深度学习模型，这些模型能够取得出色的结果。通过这一章节的学习，你学到了：

+   如何在 PyTorch 中使用 torchvision 加载 MNIST 数据集

+   如何将 MNIST 数据集转换为 PyTorch 张量，以便卷积神经网络消费

+   如何使用 PyTorch 创建用于 MNIST 的卷积神经网络模型

+   如何为 MNIST 分类实现 LeNet5 模型
