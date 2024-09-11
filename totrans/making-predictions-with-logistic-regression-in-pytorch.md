# 使用 PyTorch 中的逻辑回归进行预测

> 原文：[`machinelearningmastery.com/making-predictions-with-logistic-regression-in-pytorch/`](https://machinelearningmastery.com/making-predictions-with-logistic-regression-in-pytorch/)

逻辑回归是一种用于建模事件概率的统计技术。它常用于机器学习中进行预测。当需要预测分类结果时，我们应用逻辑回归。

在 PyTorch 中，逻辑回归的构建类似于线性回归。它们都应用于线性输入。但逻辑回归专门用于分类问题，例如将数据分类为两种结果之一（0 或 1）。

在本教程中，我们将重点介绍如何使用逻辑回归进行预测。我们将学习如何利用 PyTorch 库中的一些有用包轻松创建逻辑回归模型。特别是，我们将学习：

+   如何使用 PyTorch 中的逻辑回归进行预测。

+   逻辑函数及其在张量上的实现。

+   如何通过`nn.Sequential`构建逻辑回归模型。

+   如何构建用于逻辑回归的自定义模块。

**启动你的项目**，参考我的书籍 [《深度学习与 PyTorch》](https://machinelearningmastery.com/deep-learning-with-pytorch/)。它提供了**自学教程**和**可运行代码**。

让我们开始吧。![](img/457c0f3b051171ebdac81b5c3ccf40da.png)

使用 PyTorch 中的逻辑回归进行预测。

图片由 [Manson Yim](https://unsplash.com/photos/O-hXklfVxOo) 提供。版权所有。

## 概述

本教程分为四部分；它们是

+   创建数据类

+   使用`nn.Module`构建模型

+   使用小批量梯度下降进行训练

+   绘制进度图

## 什么是逻辑函数？

当数据集中某一点的类别使用线性函数计算时，我们得到一个正数或负数，如$-3$、$2$、$4$等。当我们构建分类器，尤其是二分类器时，我们希望它能返回 0 或 1。在这种情况下，可以使用 sigmoid 或逻辑函数，因为该函数总是返回 0 到 1 之间的值。通常，我们会设置一个阈值，如 0.5，将结果四舍五入以确定输出类别。

在 PyTorch 中，逻辑函数由`nn.Sigmoid()`方法实现。让我们使用 PyTorch 中的`range()`方法定义一个张量，并应用逻辑函数以观察输出。

```py
import torch
torch.manual_seed(42)

xrange = torch.range(-50, 50, 0.5)
sig_func = torch.nn.Sigmoid()
y_pred = sig_func(xrange)
```

让我们看看图像的样子。

```py
import matplotlib.pyplot as plt

plt.plot(xrange.numpy(), y_pred.numpy())
plt.xlabel('range')
plt.ylabel('y_pred')
plt.show()
```

![](img/4c9e9cf96e0d95b2e9e4ea434eb7aefa.png)

逻辑函数

如图所示，逻辑函数的值范围在 0 和 1 之间，过渡发生在 0 附近。

### 想开始使用 PyTorch 进行深度学习？

立即参加我的免费电子邮件速成课程（包含示例代码）。

点击注册，并免费获得课程的 PDF 电子书版。

## 通过`nn.Sequential`构建逻辑回归模型

PyTorch 中的`nn.Sequential`包使我们能够构建逻辑回归模型，就像我们可以构建线性回归模型一样。我们只需定义一个输入张量并通过模型处理它。

让我们定义一个逻辑回归模型对象，该对象接受一维张量作为输入。

```py
...
log_regr = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Sigmoid())
```

这个模型包含一个线性函数层。线性函数的输出传递给逻辑函数进行预测。

我们可以使用`parameters()`方法检查模型参数的列表。这些参数在此情况下应被随机初始化，但我们可以看到它们的形状与我们在模型中指定的形状一致。

```py
...
print(list(log_regr.parameters()))
```

输出结果如下所示。

```py
[Parameter containing:
tensor([[0.7645]], requires_grad=True), Parameter containing:
tensor([0.8300], requires_grad=True)]
```

现在，让我们定义一个一维张量`x`，并使用我们的逻辑回归模型进行预测。

```py
x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
```

我们将张量强制设置为`float32`类型，因为这是我们的模型所期望的。将这些数据样本输入模型后，我们将得到以下预测结果。

```py
y_pred = log_regr(x)
print("here is model prediction: ", y_pred)
```

它的输出如下：

```py
here is model prediction:  tensor([[0.8313],
        [0.9137],
        [0.9579],
        [0.9799]], grad_fn=<SigmoidBackward0>)
```

把所有内容整合在一起，以下是完整的代码：

```py
import matplotlib.pyplot as plt
import torch
torch.manual_seed(42)

xrange = torch.range(-50, 50, 0.5)
sig_func = torch.nn.Sigmoid()
y_pred = sig_func(xrange)
plt.plot(xrange.numpy(), y_pred.numpy())
plt.xlabel('range')
plt.ylabel('y_pred')
plt.show()

log_regr = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Sigmoid())
print(list(log_regr.parameters()))

x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y_pred = log_regr(x)
print("here is model prediction: ", y_pred)
```

## 自定义逻辑回归模块

了解如何构建自定义模块在处理高级深度学习解决方案时是必要的。我们可以尝试语法并构建我们自定义的逻辑回归模块。它应与上面的`nn.Sequential`模型完全相同。

我们将定义类并继承`nn.Module`包中的所有方法和属性。在类的`forward()`函数中，我们将使用`sigmoid()`方法，该方法接受来自类的线性函数的输出并进行预测。

```py
# build custom module for logistic regression
class LogisticRegression(torch.nn.Module):    
    # build the constructor
    def __init__(self, n_inputs):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_inputs, 1)

    # make predictions
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
```

我们可以实例化这个类对象。

```py
...
log_regr_cus = LogisticRegression(1)
```

现在，让我们对我们上面定义的张量`x`进行预测。

```py
...
y_pred = log_regr_cus(x)
print("here is model prediction: ", y_pred)
```

输出将是：

```py
here is model prediction:  tensor([[0.6647],
        [0.6107],
        [0.5537],
        [0.4954]], grad_fn=<SigmoidBackward0>)
```

如你所见，我们自定义的逻辑回归模型的工作方式与上面的`nn.Sequential`版本完全相同。

把所有内容整合在一起，以下是完整的代码：

```py
import torch
torch.manual_seed(42)

# build custom module for logistic regression
class LogisticRegression(torch.nn.Module):    
    # build the constructor
    def __init__(self, n_inputs):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_inputs, 1)

    # make predictions
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
log_regr_cus = LogisticRegression(1)
y_pred = log_regr_cus(x)
print("here is model prediction: ", y_pred)
```

## 总结

在本教程中，你学习了逻辑回归的一些基础知识以及如何在 PyTorch 中实现它。特别是，你学习了：

+   如何在 PyTorch 中使用逻辑回归进行预测。

+   关于逻辑函数及其在张量上的实现。

+   如何使用`nn.Sequential`构建逻辑回归模型。

+   如何构建自定义的逻辑回归模块。
