# 在 PyTorch 中进行线性预测

> 原文：[`machinelearningmastery.com/making-linear-predictions-in-pytorch/`](https://machinelearningmastery.com/making-linear-predictions-in-pytorch/)

线性回归是一种统计技术，用于估计两个变量之间的关系。线性回归的一个简单示例是根据某人的体重的平方根预测其身高（这也是 BMI 的基础）。为此，我们需要找到直线的斜率和截距。斜率是一个变量随另一个变量变化一个单位时的变化量。截距是我们直线与$y$轴的交点。

让我们以简单的线性方程 $y=wx+b$ 为例。输出变量是 $y$，输入变量是 $x$。方程的斜率和 $y$ 截距由字母 $w$ 和 $b$ 表示，因此称它们为方程的参数。知道这些参数可以让你预测任何给定 $x$ 值的结果 $y$。

既然你已经学会了简单线性回归的一些基础知识，让我们尝试在 PyTorch 框架中实现这个有用的算法。在这里，我们将重点关注以下几点：

+   什么是线性回归，它如何在 PyTorch 中实现。

+   如何在 PyTorch 中导入线性类并使用它进行预测。

+   我们如何为线性回归问题构建自定义模块，或者为未来更复杂的模型构建模块。

**通过我的书** [《深度学习与 PyTorch》](https://machinelearningmastery.com/deep-learning-with-pytorch/) **启动你的项目**。它提供了 **自学教程** 和 **可运行的代码**。

那么让我们开始吧！[](../Images/823dae920371197bb77b9912d74465ff.png)

在 PyTorch 中进行线性预测。

图片由 [Daryan Shamkhali](https://unsplash.com/photos/pMCbPPPBSkA) 提供。保留所有权利。

## 概述

本教程分为三部分，它们是

+   准备张量

+   使用 PyTorch 的线性类

+   构建自定义线性类

## 准备张量

请注意，在本教程中，我们将涵盖只有两个参数的一维线性回归。我们将创建这个线性表达式：

$$y=3x+1$$

我们将在 PyTorch 中将参数 $w$ 和 $b$ 定义为张量。我们将 `requires_grad` 参数设置为 `True`，表示我们的模型需要学习这些参数：

```py
import torch

# defining the parameters 'w' and 'b'
w = torch.tensor(3.0, requires_grad = True)
b = torch.tensor(1.0, requires_grad = True)
```

在 PyTorch 中，预测步骤称为前向步骤。因此，我们将编写一个函数，使我们能够在任何给定的 $x$ 值下进行 $y$ 的预测。

```py
# function of the linear equation for making predictions
def forward(x):
    y_pred = w * x + b
    return y_pred
```

现在我们已经定义了线性回归函数，让我们在 $x=2$ 处做一个预测。

```py
# let's predict y_pred at x = 2
x = torch.tensor([[2.0]])
y_pred = forward(x)
print("prediction of y at 'x = 2' is: ", y_pred)
```

这会输出

```py
prediction of y at 'x = 2' is:  tensor([[7.]], grad_fn=<AddBackward0>)
```

让我们也用多个 $x$ 输入来评估方程。

```py
# making predictions at multiple values of x
x = torch.tensor([[3.0], [4.0]])
y_pred = forward(x)
print("prediction of y at 'x = 3 & 4' is: ", y_pred)
```

这会输出

```py
prediction of y at 'x = 3 & 4' is:  tensor([[10.],
        [13.]], grad_fn=<AddBackward0>)
```

正如你所见，线性方程的函数成功地预测了多个 $x$ 值的结果。

总结来说，这就是完整的代码。

```py
import torch

# defining the parameters 'w' and 'b'
w = torch.tensor(3.0, requires_grad = True)
b = torch.tensor(1.0, requires_grad = True)

# function of the linear equation for making predictions
def forward(x):
    y_pred = w * x + b
    return y_pred

# let's predict y_pred at x = 2
x = torch.tensor([[2.0]])
y_pred = forward(x)
print("prediction of y at 'x = 2' is: ", y_pred)

# making predictions at multiple values of x
x = torch.tensor([[3.0], [4.0]])
y_pred = forward(x)
print("prediction of y at 'x = 3 & 4' is: ", y_pred)
```

### 想开始使用 PyTorch 进行深度学习吗？

现在就报名参加我的免费电子邮件速成课程（包括示例代码）。

点击注册并免费获取课程的 PDF 电子书版本。

## 使用 PyTorch 中的线性类

要解决实际问题，你需要构建更复杂的模型，为此，PyTorch 带来了许多有用的包，包括线性类，允许我们进行预测。以下是我们如何从 PyTorch 导入线性类模块。我们还将随机初始化参数。

```py
from torch.nn import Linear
torch.manual_seed(42)
```

请注意，之前我们定义了$w$和$b$的值，但在实践中，它们在启动机器学习算法之前是随机初始化的。

让我们创建一个线性对象模型，并使用`parameters()`方法访问模型的参数（$w$和$b$）。`Linear`类使用以下参数初始化：

+   `in_features`：反映每个输入样本的大小

+   `out_features`：反映每个输出样本的大小

```py
linear_regression = Linear(in_features=1, out_features=1)
print("displaying parameters w and b: ",
      list(linear_regression.parameters()))
```

这打印

```py
displaying parameters w and b:  [Parameter containing:
tensor([[0.5153]], requires_grad=True), Parameter containing:
tensor([-0.4414], requires_grad=True)]
```

同样地，你可以使用`state_dict()`方法获取包含参数的字典。

```py
print("getting python dictionary: ",linear_regression.state_dict())
print("dictionary keys: ",linear_regression.state_dict().keys())
print("dictionary values: ",linear_regression.state_dict().values())
```

这打印

```py
getting python dictionary:  OrderedDict([('weight', tensor([[0.5153]])), ('bias', tensor([-0.4414]))])
dictionary keys:  odict_keys(['weight', 'bias'])
dictionary values:  odict_values([tensor([[0.5153]]), tensor([-0.4414])])
```

现在我们可以重复之前的操作。让我们使用单个$x$值进行预测。

```py
# make predictions at x = 2
x = torch.tensor([[2.0]])
y_pred = linear_regression(x)
print("getting the prediction for x: ", y_pred)
```

这给出了

```py
getting the prediction for x:  tensor([[0.5891]], grad_fn=<AddmmBackward0>)
```

这对应于$0.5153 \times 2 - 0.4414 = 0.5891$。同样地，我们将为多个$x$值进行预测。

```py
# making predictions at multiple values of x
x = torch.tensor([[3.0], [4.0]])
y_pred = linear_regression(x)
print("prediction of y at 'x = 3 & 4' is: ", y_pred)
```

这打印

```py
prediction of y at 'x = 3 & 4' is:  tensor([[1.1044],
        [1.6197]], grad_fn=<AddmmBackward0>)
```

将所有内容放在一起，完整的代码如下所示

```py
import torch
from torch.nn import Linear

torch.manual_seed(1)

linear_regression = Linear(in_features=1, out_features=1)
print("displaying parameters w and b: ", list(linear_regression.parameters()))
print("getting python dictionary: ",linear_regression.state_dict())
print("dictionary keys: ",linear_regression.state_dict().keys())
print("dictionary values: ",linear_regression.state_dict().values())

# make predictions at x = 2
x = torch.tensor([[2.0]])
y_pred = linear_regression(x)
print("getting the prediction for x: ", y_pred)

# making predictions at multiple values of x
x = torch.tensor([[3.0], [4.0]])
y_pred = linear_regression(x)
print("prediction of y at 'x = 3 & 4' is: ", y_pred)
```

## 构建自定义线性类

PyTorch 提供了构建自定义线性类的可能性。在后续教程中，我们将使用这种方法构建更复杂的模型。让我们从 PyTorch 中导入`nn`模块，以构建自定义线性类。

```py
from torch import nn
```

PyTorch 中的自定义模块是从`nn.Module`派生的类。我们将构建一个简单线性回归的类，并命名为`Linear_Regression`。这将使它成为`nn.Module`的子类。因此，所有方法和属性将继承到这个类中。在对象构造函数中，我们将声明输入和输出参数。此外，我们通过调用`nn.Module`中的线性类来创建一个超级构造函数。最后，在定义类中的前向函数时，我们将从输入样本生成预测。

```py
class Linear_Regression(nn.Module):
    def __init__(self, input_sample, output_sample):        
        # Inheriting properties from the parent calss
        super(Linear_Regression, self).__init__()
        self.linear = nn.Linear(input_sample, output_sample)

    # define function to make predictions
    def forward(self, x):
        output = self.linear(x)
        return output
```

现在，让我们创建一个简单的线性回归模型。在这种情况下，它将仅是一条线的方程。为了检查，让我们也打印出模型参数。

```py
model = Linear_Regression(input_sample=1, output_sample=1)
print("printing the model parameters: ", list(model.parameters()))
```

这打印

```py
printing the model parameters:  [Parameter containing:
tensor([[-0.1939]], requires_grad=True), Parameter containing:
tensor([0.4694], requires_grad=True)]
```

就像我们在教程的早期会话中所做的那样，我们将评估我们的自定义线性回归模型，并尝试为单个和多个输入$x$进行预测。

```py
x = torch.tensor([[2.0]])
y_pred = model(x)
print("getting the prediction for x: ", y_pred)
```

这打印

```py
getting the prediction for x:  tensor([[0.0816]], grad_fn=<AddmmBackward0>)
```

这对应于$-0.1939 \times 2 + 0.4694 = 0.0816$。正如你所看到的，我们的模型能够预测结果，并且结果是一个张量对象。同样地，让我们尝试为多个$x$值获取预测。

```py
x = torch.tensor([[3.0], [4.0]])
y_pred = model(x)
print("prediction of y at 'x = 3 & 4' is: ", y_pred)
```

这打印

```py
prediction of y at 'x = 3 & 4' is:  tensor([[-0.1122],
        [-0.3061]], grad_fn=<AddmmBackward0>)
```

因此，该模型也适用于多个$x$值。

将所有内容放在一起，以下是完整的代码

```py
import torch
from torch import nn

torch.manual_seed(42)

class Linear_Regression(nn.Module):
    def __init__(self, input_sample, output_sample):
        # Inheriting properties from the parent calss
        super(Linear_Regression, self).__init__()
        self.linear = nn.Linear(input_sample, output_sample)

    # define function to make predictions
    def forward(self, x):
        output = self.linear(x)
        return output

model = Linear_Regression(input_sample=1, output_sample=1)
print("printing the model parameters: ", list(model.parameters()))

x = torch.tensor([[2.0]])
y_pred = model(x)
print("getting the prediction for x: ", y_pred)

x = torch.tensor([[3.0], [4.0]])
y_pred = model(x)
print("prediction of y at 'x = 3 & 4' is: ", y_pred)
```

## 总结

在本教程中，我们讨论了如何从头开始构建神经网络，从一个简单的线性回归模型开始。我们探索了在 PyTorch 中实现简单线性回归的多种方法。特别是，我们学到了：

+   什么是线性回归，以及如何在 PyTorch 中实现它。

+   如何在 PyTorch 中导入线性类并用它进行预测。

+   如何为线性回归问题构建自定义模块，或者为将来更复杂的模型构建准备。
