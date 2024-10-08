# 在 PyTorch 中构建多层感知器模型

> 原文：[`machinelearningmastery.com/building-multilayer-perceptron-models-in-pytorch/`](https://machinelearningmastery.com/building-multilayer-perceptron-models-in-pytorch/)

PyTorch 库用于深度学习。深度学习确实只是大规模神经网络或多层感知器网络的另一种名称。在其最简单的形式中，多层感知器是串联在一起的一系列层。在这篇文章中，你将发现可以用来创建神经网络和简单深度学习模型的简单组件。

用我的书籍[《PyTorch 深度学习》](https://machinelearningmastery.com/deep-learning-with-pytorch/)**启动你的项目**。它提供了**自学教程**和**可运行的代码**。

让我们开始吧！[](../Images/13e5c97fd1dae9155df5f747a1c211bf.png)

在 PyTorch 中构建多层感知器模型

图片来源：[Sharon Cho](https://unsplash.com/photos/fc7Kplqt9mk)。部分权利保留。

## 概述

本文分为六部分，它们是：

+   PyTorch 中的神经网络模型

+   模型输入

+   层、激活和层属性

+   损失函数和模型优化器

+   模型训练与推理

+   模型检查

## PyTorch 中的神经网络模型

PyTorch 可以做很多事情，但最常见的用例是构建深度学习模型。最简单的模型可以使用`Sequential`类来定义，它只是一个线性堆叠的层串联在一起。你可以创建一个`Sequential`模型，并一次性定义所有层，例如：

```py
import torch
import torch.nn as nn

model = nn.Sequential(...)
```

你应该在处理顺序中将所有层定义在括号内，从输入到输出。例如：

```py
model = nn.Sequential(
    nn.Linear(764, 100),
    nn.ReLU(),
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10),
    nn.Sigmoid()
)
```

使用`Sequential`的另一种方式是传入一个有序字典，你可以为每一层分配名称：

```py
from collections import OrderedDict
import torch.nn as nn

model = nn.Sequential(OrderedDict([
    ('dense1', nn.Linear(764, 100)),
    ('act1', nn.ReLU()),
    ('dense2', nn.Linear(100, 50)),
    ('act2', nn.ReLU()),
    ('output', nn.Linear(50, 10)),
    ('outact', nn.Sigmoid()),
]))
```

如果你想逐层构建，而不是一次性完成所有工作，你可以按照以下方式进行：

```py
model = nn.Sequential()
model.add_module("dense1", nn.Linear(8, 12))
model.add_module("act1", nn.ReLU())
model.add_module("dense2", nn.Linear(12, 8))
model.add_module("act2", nn.ReLU())
model.add_module("output", nn.Linear(8, 1))
model.add_module("outact", nn.Sigmoid())
```

在需要根据某些条件构建模型的复杂情况下，你会发现这些内容非常有帮助。

## 模型输入

模型中的第一层提示了输入的形状。在上面的示例中，你有`nn.Linear(764, 100)`作为第一层。根据你使用的不同层类型，参数可能有不同的含义。但在这个例子中，它是一个`Linear`层（也称为密集层或全连接层），这两个参数告诉**该层**的输入和输出维度。

请注意，批次的大小是隐式的。在这个示例中，你应该将形状为`(n, 764)`的 PyTorch 张量传入该层，并期望返回形状为`(n, 100)`的张量，其中`n`是批次的大小。

### 想开始使用 PyTorch 进行深度学习吗？

现在就参加我的免费电子邮件速成课程（含示例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## 层、激活和层属性

在 PyTorch 中定义了许多种类的神经网络层。实际上，如果你愿意，定义自己的层也很简单。以下是一些你可能经常看到的常见层：

+   `nn.Linear(input, output)`：全连接层

+   `nn.Conv2d(in_channel, out_channel, kernel_size)`：二维卷积层，在图像处理网络中很受欢迎。

+   `nn.Dropout(probability)`：Dropout 层，通常添加到网络中以引入正则化。

+   `nn.Flatten()`：将高维输入张量重塑为 1 维（每个批次中的每个样本）。

除了层，还有激活函数。这些是应用于张量每个元素的函数。通常，你会将层的输出传递给激活函数，然后再作为输入传递给后续层。一些常见的激活函数包括：

+   `nn.ReLU()`：整流线性单元，现在最常用的激活函数。

+   `nn.Sigmoid()` 和 `nn.Tanh()`：Sigmoid 和双曲正切函数，这些是旧文献中常用的选择。

+   `nn.Softmax()`：将向量转换为类似概率的值；在分类网络中很受欢迎。

你可以在 PyTorch 的文档中找到所有不同层和激活函数的列表。

PyTorch 的设计非常模块化。因此，你不需要在每个组件中进行太多调整。以 `Linear` 层为例，你只需指定输入和输出的形状，而不是其他细节，如如何初始化权重。然而，几乎所有组件都可以接受两个额外的参数：设备和数据类型。

PyTorch 设备指定了此层将在哪个位置执行。通常，你可以选择 CPU 或 GPU，或者省略它，让 PyTorch 决定。要指定设备，你可以这样做（CUDA 意味着支持的 nVidia GPU）：

```py
nn.Linear(764, 100, device="cpu")
```

或

```py
nn.Linear(764, 100, device="cuda:0")
```

数据类型参数 (`dtype`) 指定了此层应操作的数据类型。通常，这是一个 32 位浮点数，通常你不想更改它。但如果需要指定不同的类型，必须使用 PyTorch 类型，例如：

```py
nn.Linear(764, 100, dtype=torch.float16)
```

## 损失函数和模型优化器

神经网络模型是矩阵操作的序列。与输入无关并保存在模型中的矩阵称为权重。训练神经网络将**优化**这些权重，以便它们生成你想要的输出。在深度学习中，优化这些权重的算法是梯度下降。

梯度下降有很多变体。你可以通过为模型准备一个优化器来选择适合你的优化器。它不是模型的一部分，但你会在训练过程中与模型一起使用它。使用方式包括定义一个**损失函数**并使用优化器最小化损失函数。损失函数会给出一个**距离分数**，以告诉模型输出距离你期望的输出有多远。它将模型的输出张量与期望的张量进行比较，在不同的上下文中，期望的张量被称为**标签**或**真实值**。因为它作为训练数据集的一部分提供，所以神经网络模型是一个监督学习模型。

在 PyTorch 中，你可以简单地取模型的输出张量并对其进行操作以计算损失。但你也可以利用 PyTorch 提供的函数，例如，

```py
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(output, label)
```

在这个例子中，`loss_fn` 是一个函数，而 `loss` 是一个支持自动微分的张量。你可以通过调用 `loss.backward()` 来触发微分。

以下是 PyTorch 中一些常见的损失函数：

+   `nn.MSELoss()`：均方误差，适用于回归问题

+   `nn.CrossEntropyLoss()`：交叉熵损失，适用于分类问题

+   `nn.BCELoss()`：二元交叉熵损失，适用于二分类问题

创建优化器类似：

```py
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

所有优化器都需要一个包含所有需要优化的参数的列表。这是因为优化器是在模型之外创建的，你需要告诉它在哪里查找参数（即模型权重）。然后，优化器会根据 `backward()` 函数调用计算的梯度，并根据优化算法将其应用于参数。

这是一些常见优化器的列表：

+   `torch.optim.Adam()`：Adam 算法（自适应矩估计）

+   `torch.optim.NAdam()`：具有 Nesterov 动量的 Adam 算法

+   `torch.optim.SGD()`：随机梯度下降

+   `torch.optim.RMSprop()`：RMSprop 算法

你可以在 PyTorch 的文档中找到所有提供的损失函数和优化器的列表。你可以在文档中相应优化器的页面上了解每个优化算法的数学公式。

## 模型训练和推理

PyTorch 没有专门的模型训练和评估函数。一个定义好的模型本身就像一个函数。你传入一个输入张量，并返回一个输出张量。因此，编写训练循环是你的责任。一个最简单的训练循环如下：

```py
for n in range(num_epochs):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

如果你已经有一个模型，你可以简单地使用 `y_pred = model(X)` 并利用输出张量 `y_pred` 进行其他用途。这就是如何使用模型进行预测或推断。然而，模型不期望单个输入样本，而是一个包含多个输入样本的张量。如果模型要处理一个输入向量（即一维），你应当向模型提供一个二维张量。通常，在推断的情况下，你会故意创建一个包含一个样本的批次。

## 模型检查

一旦你有了模型，你可以通过打印模型来检查它：

```py
print(model)
```

这将为你提供例如以下内容：

```py
Sequential(
  (0): Linear(in_features=8, out_features=12, bias=True)
  (1): ReLU()
  (2): Linear(in_features=12, out_features=8, bias=True)
  (3): ReLU()
  (4): Linear(in_features=8, out_features=1, bias=True)
  (5): Sigmoid()
)
```

如果你想保存模型，你可以使用 Python 的 `pickle` 库。但你也可以使用 PyTorch 来访问它：

```py
torch.save(model, "my_model.pickle")
```

这样，你就将整个模型对象保存到 pickle 文件中。你可以通过以下方式检索模型：

```py
model = torch.load("my_model.pickle")
```

但推荐的保存模型的方式是将模型设计留在代码中，只保存权重。你可以这样做：

```py
torch.save(model.state_dict(), "my_model.pickle")
```

`state_dict()` 函数仅提取状态（即模型中的权重）。要检索它，你需要从头开始重建模型，然后像这样加载权重：

```py
model = nn.Sequential(...)
model.load_state_dict(torch.load("my_model.pickle"))
```

## 资源

你可以通过以下资源进一步了解如何在 PyTorch 中创建简单的神经网络和深度学习模型：

### 在线资源

+   [`torch.nn` 文档](https://pytorch.org/docs/stable/nn.html)

+   [`torch.optim` 文档](https://pytorch.org/docs/stable/optim.html)

+   [PyTorch 教程](https://pytorch.org/tutorials/)

## 总结

在这篇文章中，你了解了可以用来创建人工神经网络和深度学习模型的 PyTorch API。具体来说，你学习了 PyTorch 模型的生命周期，包括：

+   构建模型

+   创建和添加层及激活函数

+   为训练和推断准备模型
