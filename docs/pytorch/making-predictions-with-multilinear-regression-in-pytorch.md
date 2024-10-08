# 使用 PyTorch 进行多元线性回归预测

> 原文：[`machinelearningmastery.com/making-predictions-with-multilinear-regression-in-pytorch/`](https://machinelearningmastery.com/making-predictions-with-multilinear-regression-in-pytorch/)

多元线性回归模型是一种监督学习算法，可用于在给定多个输入变量 `x` 的情况下预测目标变量 `y`。它是一个线性回归问题，其中使用了多个输入变量 `x` 或特征来预测目标变量 `y`。该算法的一个典型使用案例是根据房屋的大小、房间数量和年龄预测房价。

在之前的教程中，我们关注了简单线性回归，其中我们使用了单个变量 `x` 来预测目标变量 `y`。从现在开始，我们将使用多个输入变量进行预测。虽然本教程仅关注从多个输入变量 `x` 中进行单个输出预测 `y`，但在后续教程中，我们将介绍多个输入-多个输出回归问题。通常，实际场景中会采用相同的实践来构建更复杂的神经网络架构。

本教程将展示如何实现一个多元

PyTorch 中的线性回归模型。特别地，您将学习：

+   如何在多维中回顾线性回归。

+   如何使用 PyTorch 进行多元线性回归模型预测。

+   如何在 PyTorch 中使用 `Linear` 类进行多元线性回归。

+   如何使用 PyTorch 中的 `nn.Module` 构建自定义模块。

**启动您的项目**，请参阅我的书 [Deep Learning with PyTorch](https://machinelearningmastery.com/deep-learning-with-pytorch/)。它提供了 **自学教程** 和 **可运行的代码**。

让我们开始吧！[](../Images/cfb414ae8bd3f39d9577ea2dcbbf4ef5.png)

使用 PyTorch 的优化器。

图片由 [Mark Boss](https://unsplash.com/photos/W0zGOsdNFaE) 提供。保留所有权利。

## 概述

本教程分为三个部分；它们是

+   准备预测数据

+   使用 `Linear` 类进行多元线性回归

+   可视化结果

## 准备预测数据

与简单线性回归模型的情况类似，我们来初始化模型的权重和偏差。请注意，由于我们将处理多个输入变量，因此我们使用了多维张量来表示权重和偏差。

```py
import torch
torch.manual_seed(42)

# Setting weights and bias
w = torch.tensor([[3.0], 
                  [4.0]], requires_grad=True)
b = torch.tensor([[1.0]], requires_grad=True)
```

接下来，我们将定义用于预测的前向函数。之前我们使用了标量乘法，但在这里我们使用 PyTorch 的 `mm` 函数进行 **矩阵乘法**。这个函数实现了一个具有多个输入变量的线性方程。请注意，多维张量是矩阵，并且需要遵循一些规则，例如矩阵乘法。我们将在后面进一步讨论这些规则。

```py
# Defining our forward function for prediction
def forward(x):
    # using mm module for matrix multiplication 
    y_pred = torch.mm(x, w) + b
    return y_pred
```

既然我们已经初始化了权重和偏差，并构建了用于预测的前向函数，我们来定义一个用于输入变量的张量 `x`。

```py
# define a tensor 'x'
x = torch.tensor([[2.0, 4.0]])
# predict the value with forward function
y_pred = forward(x)
# show the result
print("Printing Prediction: ", y_pred)
```

这将打印

```py
Printing Prediction:  tensor([[23.]], grad_fn=<AddBackward0>)
```

注意，在矩阵乘法`torch.mm(x, w)`中，矩阵`x`中的**列数**必须等于`w`中的**行数**。在这种情况下，我们有一个$1\times 2$的张量`x`和一个$2\times 1$的张量`w`，矩阵乘法后得到一个$1\times 1$的张量。

类似地，我们可以对多个样本应用线性方程。例如，让我们创建一个张量`X`，其中每一行代表一个样本。

```py
# define a tensor 'X' with multiple rows
X = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0], 
                  [5.0, 6.0]])
```

对于预测，我们将使用上面相同的函数。

```py
# Making predictions for Multi-Dimensional tensor "X"
y_pred = forward(X)
print("Predictions for 'X': ", y_pred)
```

其输出为

```py
Predictions for 'X':  tensor([[12.],
        [26.],
        [40.]], grad_fn=<AddBackward0>)
```

如你所见，我们已经得到了多个输入变量的结果。

## 使用`Linear`类进行多线性回归

与其从头编写函数，我们可以使用 PyTorch 的内置`Linear`类进行预测。这在构建复杂而强大的模型架构时更为有用。

让我们创建一个`Linear`模型，并对上面定义的相同张量`X`进行预测。这里我们将定义两个参数：

+   `in_features`：表示输入变量`X`的数量以及模型权重的数量，在这种情况下为 2。

+   `out_features`：表示输出/预测值的数量，在这种情况下为 1。

```py
# using Pytorch's own built-in fuction to define the LR model
lr_model = torch.nn.Linear(in_features=2, out_features=1)
```

现在，让我们使用随机初始化的权重和偏差，通过`lr_model`对象对`X`进行预测。

```py
# Making predictions for X
y_pred = lr_model(X)
print("Predictions for 'X': ", y_pred)
```

在这种情况下，输出如下：

```py
Predictions for 'X':  tensor([[-0.5754],
        [-1.2430],
        [-1.9106]], grad_fn=<AddmmBackward0>)
```

注意输出的形状，而不是值。这与我们使用矩阵乘法的先前情况相同。

### 想要开始使用 PyTorch 进行深度学习吗？

立即获取我的免费电子邮件速成课程（包括示例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## 使用`nn.Module`创建自定义模块

另外，我们还可以为线性模型创建自定义模块。虽然目前看来这可能有些多余，但当我们构建最先进的神经网络时，这可能是必要的。

注意，自定义模块是对象和类。在这种情况下，我们将定义一个线性回归类`LR`，并将其设置为`nn.Module`包的子类。因此，`nn.Module`包中的所有方法和属性都将被继承。

我们将在构造函数的参数中定义输入和输出的大小，即`input_features`和`output_features`。此外，我们将在对象构造函数中调用`super()`，这使我们能够使用父类`nn.Module`中的方法和属性。现在我们可以使用`torch.nn.Linear`对象，并在其中定义参数`input_features`和`output_features`。

最后，为了进行预测，我们将定义`forward`函数。

```py
...
# creating custom modules with package 'nn.Module'
class LR(torch.nn.Module):
    # Object Constructor
    def __init__(self, input_features, output_features):
        super().__init__()
        self.linear = torch.nn.Linear(input_features, output_features)
    # define the forward function for prediction
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
```

我们将按照以下方式构建具有两个输入和一个输出的线性回归模型。

```py
# build the model object
LR_model = LR(2, 1)
```

现在，让我们再次使用自定义模块对包含多个输入样本的张量`X`进行预测。

```py
# make predictions for multiple input samples of 'X'
y_pred  = LR_model(X)
print("Predictions for 'X': ", y_pred)
```

其输出为

```py
Predictions for 'X':  tensor([[0.3405],
        [0.5596],
        [0.7787]], grad_fn=<AddmmBackward0>)
```

使用`parameters()`方法，我们可以获得随机初始化参数的列表。

```py
print(list(LR_model.parameters()))
```

其输出为

```py
[Parameter containing:
tensor([[ 0.6496, -0.1549]], requires_grad=True), Parameter containing:
tensor([0.1427], requires_grad=True)]
```

另外，我们还可以使用`state_dict()`方法检查模型的参数。

将所有内容结合起来，以下是用不同方式创建多元线性回归模型的完整代码：

```py
import torch

# Setting weights and bias
w = torch.tensor([[3.0], 
                  [4.0]], requires_grad=True)
b = torch.tensor([[1.0]], requires_grad=True)

# Defining our forward function for prediction
def forward(x):
    # using .mm module for matrix multiplication 
    y_pred = torch.mm(x, w) + b
    return y_pred

# define a tensor 'x'
x = torch.tensor([[2.0, 4.0]])
# predict the value with forward function
y_pred = forward(x)
# show the result
print("Printing Prediction: ", y_pred)

# define a tensor 'X' with multiple rows
X = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0], 
                  [5.0, 6.0]])

# Making predictions for Multi-Dimensional tensor "X"
y_pred = forward(X)
print("Predictions for 'X': ", y_pred)

# using Pytorch's own built-in fuction to define the LR model
lr_model = torch.nn.Linear(in_features=2, out_features=1)

# Making predictions for X
y_pred = lr_model(X)
print("Predictions for 'X': ", y_pred)

# creating custom modules with package 'nn.Module'
class LR(torch.nn.Module):
    # Object Constructor
    def __init__(self, input_features, output_features):
        super().__init__()
        self.linear = torch.nn.Linear(input_features, output_features)
    # define the forward function for prediction
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# build the model object
LR_model = LR(2, 1)

# make predictions for multiple input samples of 'X'
y_pred  = LR_model(X)
print("Predictions for 'X': ", y_pred)

print(list(LR_model.parameters()))
```

## 总结

在本教程中，你学习了如何使用多元线性回归模型进行预测。特别是，你学习了：

+   如何回顾多维线性回归。

+   如何使用 PyTorch 进行多元线性回归模型预测。

+   如何在 PyTorch 中使用类 `Linear` 进行多元线性回归。

+   如何在 PyTorch 中使用 `nn.Module` 构建自定义模块。
