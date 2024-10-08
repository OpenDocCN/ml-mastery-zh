# 可视化 PyTorch 模型

> 原文：[`machinelearningmastery.com/visualizing-a-pytorch-model/`](https://machinelearningmastery.com/visualizing-a-pytorch-model/)

PyTorch 是一个深度学习库。你可以用 PyTorch 构建非常复杂的深度学习模型。然而，有时你可能需要模型架构的图形化表示。在这篇文章中，你将学习：

+   如何将你的 PyTorch 模型保存为交换格式

+   如何使用 Netron 创建图形表示。

**启动你的项目**，请参考我的书籍 [《深度学习与 PyTorch》](https://machinelearningmastery.com/deep-learning-with-pytorch/)。它提供了 **自学教程** 和 **可运行代码**。

让我们开始吧。![](img/efea4b2258dc7e93cdff3b26bcbc6afc.png)

可视化 PyTorch 模型

照片由 [Ken Cheung](https://unsplash.com/photos/10py7Mvmf1g) 拍摄。版权所有。

## 概述

本文分为两部分；它们是

+   为什么 PyTorch 模型的图形化表示很困难

+   如何使用 Netron 创建模型图

## 为什么 PyTorch 模型的图形化表示很困难

PyTorch 是一个非常灵活的深度学习库。严格来说，它从不强制规定你应如何构建模型，只要模型能像一个函数那样将输入张量转换为输出张量即可。这是一个问题：使用一个模型，你永远无法知道它是如何工作的，除非你跟踪输入张量并收集轨迹直到得到输出张量。因此，将 PyTorch 模型转换为图片并非易事。

解决这个问题有多个库。但一般来说，只有两种方法可以解决：你可以在前向传递中跟踪一个张量，看看应用了什么操作（即，层），或者在反向传递中跟踪一个张量，查看梯度是如何传播到输入的。你只能以这种方式找到关于模型内部结构的线索。

### 想开始使用 PyTorch 进行深度学习吗？

立即获取我的免费电子邮件速成课程（包含示例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## 如何使用 Netron 创建模型图

当你保存一个 PyTorch 模型时，你是在保存它的状态。你可以使用 `model.state_dict()` 获取模型状态。虽然权重张量有名称，这有助于你将它们恢复到模型中，但你无法获得权重之间如何连接的线索。你唯一能够连接张量并找出它们关系的方法是获取张量梯度：当你运行一个模型并获得输出时，包括对其他张量的依赖在内的计算会被每个中间张量记住，以便进行自动微分。

实际上，如果你想了解 PyTorch 模型背后的算法，这也是一种方法。只有少数工具可以从 PyTorch 模型创建图形。下面，你将了解工具 Netron。它是一个“深度学习模型查看器”。这是一个可以在 macOS、Linux 和 Windows 上安装和运行的软件。你可以访问以下页面并下载适用于你的平台的软件：

+   [`github.com/lutzroeder/netron/releases`](https://github.com/lutzroeder/netron/releases)

还有一个 [在线版本](https://netron.app/)，你可以通过上传模型文件来查看你的模型。

Netron 不能从保存的状态中可视化 PyTorch 模型，因为没有足够的线索来说明模型的结构。然而，PyTorch 允许你将模型转换为 Netron 可以理解的交换格式 ONNX。

我们从一个例子开始。在下面，你创建了一个简单的模型来对鸢尾花数据集进行分类。这是一个有三个类别的分类问题。因此，模型应该输出一个包含三个值的向量。你为这个问题创建的完整代码如下，其中数据集来自 scikit-learn：

```py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X = data['data']
y = data['target']
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

class IrisModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 8)
        self.act = nn.ReLU()
        self.output = nn.Linear(8, 3)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x

# loss metric and optimizer
model = IrisModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# prepare model and training parameters
n_epochs = 100
batch_size = 10
batch_start = torch.arange(0, len(X_train), batch_size)

# training loop
for epoch in range(n_epochs):
    for start in batch_start:
        # take a batch
        X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
        # forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()

# validating model
y_pred = model(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
acc = float(acc)*100
print("Model accuracy: %.2f%%" % acc)
```

运行上述代码将生成以下结果，例如：

```py
Model accuracy: 97.78%
```

所以你知道 `model` 是一个可以接受张量并返回张量的 PyTorch 模型。你可以使用 `torch.onnx.export()` 函数将此模型**转换**为 ONNX 格式：

```py
torch.onnx.export(model, X_test, 'iris.onnx', input_names=["features"], output_names=["logits"])
```

运行此操作将会在本地目录创建一个文件 `iris.onnx`。你需要提供一个与模型兼容的**示例张量**作为输入（上例中的`X_test`）。这是因为在转换过程中，需要跟随这个示例张量来理解应应用哪些操作，从而可以一步步将算法转换为 ONNX 格式。PyTorch 模型中的每个权重都是一个张量，并且每个张量都有一个名称。但输入和输出张量通常没有命名，因此你需要在运行 `export()` 时为它们提供一个名称。这些名称应作为字符串列表提供，因为通常情况下，一个模型可以接受多个张量并返回多个张量。

通常，你应该在训练循环之后运行 `export()`。这是因为创建的 ONNX 模型包含一个完整的模型，你可以在没有 PyTorch 库的情况下运行它。你希望将优化后的权重保存到其中。然而，为了在 Netron 中可视化模型，模型的质量并不是问题。你可以在创建 PyTorch 模型后立即运行 `export()`。

启动 Netron 后，你可以打开保存的 ONNX 文件。在这个例子中，你应该会看到以下屏幕：

![](img/c42c718ae5cd2d16240ae1be9ef59569.png)

它展示了输入张量如何通过不同的操作连接到深度学习模型的输出张量。你提供给 `export()` 函数的输入和输出张量的名称会在可视化中使用。点击一个框会给你更多关于该张量或操作的详细信息。然而，你在 Netron 中看到的操作名称可能与 PyTorch 中的名称不同。例如，在上面的屏幕中，`nn.Linear()` 层变成了“Gemm”，代表“通用矩阵乘法”操作。你甚至可以通过 Netron 对层的权重进行检查，方法是点击几下。

如果你希望保存这个可视化的副本，你可以在 Netron 中将其导出为 PNG 格式。

## 进一步阅读

Netron 是一个开源项目，你可以在 GitHub 上找到它的源代码：

+   [`github.com/lutzroeder/netron`](https://github.com/lutzroeder/netron)

Netron 的在线版本如下：

+   [`netron.app/`](https://netron.app/)

另一个可视化库是 torchviz，但与上面你看到的例子不同，它跟踪模型的反向传递：

+   [`github.com/szagoruyko/pytorchviz`](https://github.com/szagoruyko/pytorchviz)

## 总结

在这篇文章中，你学会了如何可视化一个模型。特别是，你学到了：

+   为什么可视化 PyTorch 模型很困难

+   如何将 PyTorch 模型转换为 ONNX 格式

+   如何使用 Netron 可视化 ONNX 模型
