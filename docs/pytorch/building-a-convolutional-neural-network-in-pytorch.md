# 在 PyTorch 中构建卷积神经网络

> 原文：[`machinelearningmastery.com/building-a-convolutional-neural-network-in-pytorch/`](https://machinelearningmastery.com/building-a-convolutional-neural-network-in-pytorch/)

神经网络由相互连接的层构成。有许多不同类型的层。对于图像相关的应用，你总是可以找到卷积层。这是一种参数非常少但应用于大尺寸输入的层。它之所以强大，是因为它可以保留图像的空间结构。因此，它被用于在计算机视觉神经网络中产生最先进的结果。在本文中，你将了解卷积层及其构建的网络。完成本文后，你将了解：

+   什么是卷积层和池化层

+   它们在神经网络中的适配方式

+   如何设计使用卷积层的神经网络

**启动你的项目**，请参阅我的书 [Deep Learning with PyTorch](https://machinelearningmastery.com/deep-learning-with-pytorch/)。它提供了**自学教程**和**可运行的代码**。

让我们开始吧！[](../Images/e26c3bb8e90bc698643284c6ebc8b725.png)

在 PyTorch 中构建卷积神经网络

图片由 [Donna Elliot](https://unsplash.com/photos/O0yASWUhAgQ) 提供。部分权利保留。

## 概述

本文分为四部分；它们是

+   卷积神经网络的理由

+   卷积神经网络的构建模块

+   卷积神经网络的一个示例

+   特征图中包含什么？

## 卷积神经网络的理由

让我们考虑构建一个神经网络来处理灰度图像作为输入，这是深度学习在计算机视觉中的最简单用例。

灰度图像是一个像素数组。每个像素的值通常在 0 到 255 的范围内。一个 32×32 的图像将有 1024 个像素。将其作为神经网络的输入意味着第一层将至少有 1024 个输入权重。

查看像素值对理解图片几乎没有用，因为数据隐藏在空间结构中（例如，图片上是否有水平线或垂直线）。因此，传统神经网络将难以从图像输入中提取信息。

卷积神经网络使用卷积层来保留像素的空间信息。它学习相邻像素的相似度，并生成**特征表示**。卷积层从图片中看到的内容在某种程度上对扭曲是不变的。例如，即使输入图像的颜色发生偏移、旋转或缩放，卷积神经网络也能预测相同的结果。此外，卷积层具有较少的权重，因此更容易训练。

## 卷积神经网络的构建模块

卷积神经网络的最简单用例是分类。你会发现它包含三种类型的层：

1.  卷积层

1.  池化层

1.  全连接层

卷积层上的神经元称为滤波器。在图像应用中通常是一个二维卷积层。滤波器是一个 2D 补丁（例如 3×3 像素），应用在输入图像像素上。这个 2D 补丁的大小也称为感受野，表示它一次可以看到图像的多大部分。

卷积层的滤波器是与输入像素相乘，然后将结果求和。这个结果是输出的一个像素值。滤波器会在输入图像周围移动，填充所有输出的像素值。通常会对同一个输入应用多个滤波器，产生多个输出张量。这些输出张量称为这一层生成的**特征图**，它们被堆叠在一起作为一个张量，作为下一层的输入传递。

![](img/8baa5476210e37181192a8952af8022d.png)

将一个二维输入应用滤波器生成特征图的示例

卷积层的输出称为特征图，因为通常它学到了输入图像的特征。例如，在特定位置是否有垂直线条。从像素学习特征有助于在更高层次理解图像。多个卷积层堆叠在一起，以从低级细节推断出更高级别的特征。

池化层用于**降采样**前一层的特征图。通常在卷积层后使用以整合学习到的特征。它可以压缩和泛化特征表示。池化层也有一个感受野，通常是在感受野上取平均值（平均池化）或最大值（最大池化）。

全连接层通常是网络中的最后一层。它将前面卷积和池化层整合的特征作为输入，产生预测结果。可能会有多个全连接层堆叠在一起。在分类的情况下，通常看到最终全连接层的输出应用 softmax 函数，产生类似概率的分类结果。

### 想要开始使用 PyTorch 进行深度学习吗？

现在开始免费的电子邮件快速入门课程（含示例代码）。

点击注册并免费获得课程的 PDF 电子书版本。

## 一个卷积神经网络的例子

以下是一个在 CIFAR-10 数据集上进行图像分类的程序。

```py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

batch_size = 32
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(8192, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, 10)

    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x32x32, output 32x32x32
        x = self.act2(self.conv2(x))
        # input 32x32x32, output 32x16x16
        x = self.pool2(x)
        # input 32x16x16, output 8192
        x = self.flat(x)
        # input 8192, output 512
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # input 512, output 10
        x = self.fc4(x)
        return x

model = CIFAR10Model()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

n_epochs = 20
for epoch in range(n_epochs):
    for inputs, labels in trainloader:
        # forward, backward, and then weight update
        y_pred = model(inputs)
        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = 0
    count = 0
    for inputs, labels in testloader:
        y_pred = model(inputs)
        acc += (torch.argmax(y_pred, 1) == labels).float().sum()
        count += len(labels)
    acc /= count
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))

torch.save(model.state_dict(), "cifar10model.pth")
```

CIFAR-10 数据集提供的图像为 32×32 像素的 RGB 彩色图（即 3 个颜色通道）。有 10 类，用整数 0 到 9 标记。当你在 PyTorch 神经网络模型上处理图像时，你会发现姐妹库 `torchvision` 很有用。在上面的例子中，你使用它从互联网下载 CIFAR-10 数据集，并将其转换为 PyTorch 张量：

```py
...
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
```

你还在 PyTorch 中使用了`DataLoader`来帮助创建训练批次。训练是优化模型的交叉熵损失，使用随机梯度下降。它是一个分类模型，因此分类的准确性比交叉熵更直观，它在每个 epoch 结束时通过比较输出 logit 中的最大值与数据集的标签来计算：

```py
...
acc += (torch.argmax(y_pred, 1) == labels).float().sum()
```

运行上述程序来训练网络需要一些时间。这个网络应该能够在分类中达到 70%以上的准确率。

在图像分类网络中，典型的是在早期阶段由卷积层、dropout 和池化层交错组成。然后，在后期阶段，卷积层的输出被展平并由一些全连接层处理。

## 特征图中包含什么？

上述定义的网络中有两个卷积层。它们都定义了 3×3 的核大小，因此每次看 9 个像素以产生一个输出像素。注意第一个卷积层将 RGB 图像作为输入。因此，每个像素有三个通道。第二个卷积层将具有 32 个通道的特征图作为输入。因此，它看到的每个“像素”将有 32 个值。因此，尽管它们具有相同的感受野，第二个卷积层具有更多的参数。

让我们看看特征图中有什么。假设我们从训练集中选择了一个输入样本：

```py
import matplotlib.pyplot as plt

plt.imshow(trainset.data[7])
plt.show()
```

你应该能看到这是一张马的图像，32×32 像素，带有 RGB 通道：![](img/f44ca8cebde0bb87ce531335393a68ed.png)

首先，你需要将其转换为 PyTorch 张量，并将其转换为一个图像的批次。PyTorch 模型期望每个图像以(channel, height, width)的格式作为张量，但你读取的数据是(height, width, channel)的格式。如果你使用`torchvision`来将图像转换为 PyTorch 张量，则此格式转换会自动完成。否则，在使用之前需要 **重新排列** 维度。

然后，将其通过模型的第一个卷积层，并捕获输出。你需要告诉 PyTorch 在这个计算中不需要梯度，因为你不打算优化模型权重：

```py
X = torch.tensor([trainset.data[7]], dtype=torch.float32).permute(0,3,1,2)
model.eval()
with torch.no_grad():
    feature_maps = model.conv1(X)
```

特征图存储在一个张量中。你可以使用 matplotlib 来可视化它们：

```py
fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
for i in range(0, 32):
    row, col = i//8, i%8
    ax[row][col].imshow(feature_maps[0][i])
plt.show()
```

之后，你可能会看到以下内容：![](img/2ccc6c2ed9cbee1c1f6f19960bd9586c.png)

特征图之所以被称为特征图，是因为它们突出显示了输入图像中的某些特征。使用一个小窗口来识别特征（在本例中是一个 3×3 像素的滤波器）。输入图像有三个色彩通道。每个通道应用了不同的滤波器，它们的结果被合并为一个输出特征。

类似地，你可以显示第二个卷积层输出的特征图，如下所示：

```py
X = torch.tensor([trainset.data[7]], dtype=torch.float32).permute(0,3,1,2)

model.eval()
with torch.no_grad():
    feature_maps = model.act1(model.conv1(X))
    feature_maps = model.drop1(feature_maps)
    feature_maps = model.conv2(feature_maps)

fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
for i in range(0, 32):
    row, col = i//8, i%8
    ax[row][col].imshow(feature_maps[0][i])
plt.show()
```

显示如下：![](img/8b83c74f12d624e59d75488949f550ac.png)

相对于第一个卷积层的输出，第二个卷积层的特征图看起来更模糊、更抽象。但这些对模型来识别对象更有用。

将所有内容整合在一起，下面的代码加载了前一节保存的模型并生成了特征图：

```py
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(8192, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, 10)

    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x32x32, output 32x32x32
        x = self.act2(self.conv2(x))
        # input 32x32x32, output 32x16x16
        x = self.pool2(x)
        # input 32x16x16, output 8192
        x = self.flat(x)
        # input 8192, output 512
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # input 512, output 10
        x = self.fc4(x)
        return x

model = CIFAR10Model()
model.load_state_dict(torch.load("cifar10model.pth"))

plt.imshow(trainset.data[7])
plt.show()

X = torch.tensor([trainset.data[7]], dtype=torch.float32).permute(0,3,1,2)
model.eval()
with torch.no_grad():
    feature_maps = model.conv1(X)
fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
for i in range(0, 32):
    row, col = i//8, i%8
    ax[row][col].imshow(feature_maps[0][i])
plt.show()

with torch.no_grad():
    feature_maps = model.act1(model.conv1(X))
    feature_maps = model.drop1(feature_maps)
    feature_maps = model.conv2(feature_maps)
fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
for i in range(0, 32):
    row, col = i//8, i%8
    ax[row][col].imshow(feature_maps[0][i])
plt.show()
```

## 进一步阅读

如果你想深入了解这个主题，本节提供了更多资源。

### 文章

+   [卷积层在深度学习神经网络中的工作原理](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/)

+   [分类器训练](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)，来自 PyTorch 教程

### 书籍

+   第九章：卷积网络，《深度学习》（https://amzn.to/2Dl124s），2016 年。

### API

+   [nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) 层在 PyTorch 中的应用

## 总结

在本文中，你学会了如何使用卷积神经网络处理图像输入，并如何可视化特征图。

具体来说，你学到了：

+   典型卷积神经网络的结构

+   滤波器大小对卷积层的影响是什么

+   在网络中堆叠卷积层的效果是什么

+   如何提取和可视化卷积神经网络的特征图
