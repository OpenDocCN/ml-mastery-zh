# 使用 PyTorch 中的多线性回归进行多目标预测

> 原文：[`machinelearningmastery.com/multi-target-predictions-with-multilinear-regression-in-pytorch/`](https://machinelearningmastery.com/multi-target-predictions-with-multilinear-regression-in-pytorch/)

在前几个教程中，我们使用了单输出的多线性回归，而在这里，我们将探讨如何利用多线性回归进行多目标预测。复杂的神经网络架构本质上是让每个神经元单元独立执行线性回归，然后将其结果传递给另一个神经元。因此，了解这种回归的工作原理对于理解神经网络如何执行多目标预测是很有用的。

本文的目标是提供 PyTorch 中多目标预测实现的逐步指南。我们将通过使用一个线性回归模型的框架来实现，该模型接受多个特征作为输入并生成多个结果。

我们将从导入模型所需的包开始。然后，我们将定义我们的输入数据点以及我们希望通过模型实现的目标。特别是，我们将演示：

+   如何理解多维度的多线性回归。

+   如何使用 PyTorch 中的多线性回归进行多目标预测。

+   如何使用 PyTorch 中的 `nn.Module` 构建线性类。

+   如何使用单个输入数据样本进行多目标预测。

+   如何使用多个输入数据样本进行多目标预测。

注意，本教程中我们不会训练 MLR 模型，我们将仅查看它如何进行简单的预测。在 PyTorch 系列的后续教程中，我们将学习如何在数据集上训练这个模型。

使用我的书 [Deep Learning with PyTorch](https://machinelearningmastery.com/deep-learning-with-pytorch/) **快速启动你的项目**。它提供了**自学教程**和**可运行代码**。

让我们开始吧！[](../Images/cdd32cc247e36c58378d342bdfd55393.png)

使用 PyTorch 中的多线性回归进行多目标预测。

图片由 [Dan Gold](https://unsplash.com/photos/yhQhvK04QPc) 提供。保留所有权利。

## 概述

本教程分为三个部分；它们是

+   创建模块

+   使用简单输入样本进行预测

+   使用多个输入样本进行预测

## 创建模块

我们将为多线性回归模型构建一个自定义线性类。我们将定义一个线性类，并使其成为 PyTorch 包 `nn.Module` 的子类。这个类继承了包中的所有方法和属性，例如 `nn.Linear`。

```py
import torch
torch.manual_seed(42)

# define the class for multilinear regression
class MLR(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred
```

现在，让我们创建模型对象并相应地定义参数。由于我们计划进行多目标预测，让我们首先检查模型在单个输入样本上的工作情况。之后，我们将对多个输入样本进行预测。

## 使用单输入样本进行预测

我们将创建我们的模型对象，它接受一个输入样本并进行五次预测。

```py
...
# building the model object
model = MLR(1, 5)
```

现在，让我们定义模型的输入张量`x`并进行预测。

```py
...
# define the single input sample 'x' and make predictions
x = torch.tensor([[2.0]])
y_pred = model(x)
print(y_pred)
```

下面是输出的样子。

```py
tensor([[ 1.7309,  1.1732,  0.1187,  2.7188, -1.1718]],
       grad_fn=<AddmmBackward0>)
```

如你所见，我们的模型从仅一个输入样本中进行了多次预测。下面是我们如何列出模型参数的方法。

```py
...
print(list(model.parameters()))
```

输出如下：

```py
[Parameter containing:
 tensor([[ 0.7645],
         [ 0.8300],
         [-0.2343],
         [ 0.9186],
         [-0.2191]], requires_grad=True),
 Parameter containing:
 tensor([ 0.2018, -0.4869,  0.5873,  0.8815, -0.7336], requires_grad=True)]
```

你可能会得到不同的数字结果，因为这些是随机权重，但权重张量的形状会与我们设计的一致，即一个输入得到五个输出。

### 想要开始使用 PyTorch 进行深度学习吗？

现在就来我的免费电子邮件速成课程（包含示例代码）。

点击注册，同时获得免费的 PDF 电子书版本课程。

## 使用多个输入样本进行预测

类似地，我们定义一个张量`X`用于多个输入样本，其中每一行代表一个数据样本。

```py
# define the multiple input tensor 'x' and make predictions
X = torch.tensor([[2.0],[4.0],[6.0]])
```

我们可以使用多个输入样本进行多目标预测。

```py
...
Y_pred = model(X)
print(Y_pred)
```

由于我们有三个输入样本，我们应该看到三个输出样本，如下所示：

```py
tensor([[ 1.7309,  1.1732,  0.1187,  2.7188, -1.1718],
        [ 3.2599,  2.8332, -0.3498,  4.5560, -1.6100],
        [ 4.7890,  4.4932, -0.8184,  6.3932, -2.0482]],
       grad_fn=<AddmmBackward0>)
```

将所有内容整合在一起，以下是完整代码：

```py
import torch
torch.manual_seed(42)

# define the class for multilinear regression
class MLR(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred

# building the model object
model = MLR(1, 5)

# define the single input sample 'x' and make predictions
x = torch.tensor([[2.0]])
y_pred = model(x)
print(y_pred)
print(list(model.parameters()))

# define the multiple input tensor 'x' and make predictions
X = torch.tensor([[2.0],[4.0],[6.0]])
Y_pred = model(X)
print(Y_pred)
```

## 总结

在本教程中，你学习了如何使用多元线性回归模型进行多目标预测。特别是，你学到了：

+   如何理解多维度的多元线性回归。

+   如何使用 PyTorch 中的多元线性回归进行多目标预测。

+   如何使用 PyTorch 中的‘nn.Module’构建线性分类。

+   如何使用单个输入数据样本进行多目标预测。

+   如何使用多个输入数据样本进行多目标预测。
