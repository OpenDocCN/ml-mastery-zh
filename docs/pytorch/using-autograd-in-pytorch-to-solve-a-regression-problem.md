# 使用 PyTorch 的 Autograd 解决回归问题

> 原文：[`machinelearningmastery.com/using-autograd-in-pytorch-to-solve-a-regression-problem/`](https://machinelearningmastery.com/using-autograd-in-pytorch-to-solve-a-regression-problem/)

我们通常使用 PyTorch 来构建神经网络。然而，PyTorch 不仅仅能做到这些。由于 PyTorch 还是一个具有自动微分能力的张量库，你可以轻松使用它来解决梯度下降的数值优化问题。在这篇文章中，你将学习 PyTorch 的自动微分引擎 autograd 是如何工作的。

完成此教程后，你将学到：

+   PyTorch 中的 autograd 是什么

+   如何利用 autograd 和优化器解决优化问题

**快速启动你的项目**，参考我的书籍 [《PyTorch 深度学习》](https://machinelearningmastery.com/deep-learning-with-pytorch/)。它提供了 **自学教程** 和 **可运行的代码**。

让我们开始吧！[](../Images/2e0828a473ddb5c2989609d73a615dac.png)

使用 PyTorch 中的 autograd 来解决回归问题。

图片由 [Billy Kwok](https://unsplash.com/photos/eCzKRT7svdc) 提供。版权所有。

## 概述

本教程分为三个部分：

+   PyTorch 中的 Autograd

+   使用 Autograd 进行多项式回归

+   使用 Autograd 解决数学难题

## PyTorch 中的 Autograd

在 PyTorch 中，你可以将张量创建为变量或常量，并用它们构建表达式。这个表达式本质上是变量张量的函数。因此，你可以推导出其导数函数，即微分或梯度。这是深度学习模型训练循环的基础。PyTorch 核心中包含了这一特性。

用一个示例来解释 autograd 更容易。在 PyTorch 中，你可以如下创建一个常量矩阵：

```py
import torch

x = torch.tensor([1, 2, 3])
print(x)
print(x.shape)
print(x.dtype)
```

上述打印：

```py
tensor([1, 2, 3])
torch.Size([3])
torch.int64
```

这将创建一个整数向量（以 PyTorch 张量的形式）。这个向量在大多数情况下可以像 NumPy 向量一样工作。例如，你可以进行 `x+x` 或 `2*x`，结果正是你所期望的。PyTorch 配有许多与 NumPy 匹配的数组操作函数，如 `torch.transpose` 或 `torch.concatenate`。

但这个张量不被视为函数的变量，因为不支持对其进行微分。你可以通过一个额外的选项创建像变量一样工作的张量：

```py
import torch

x = torch.tensor([1., 2., 3.], requires_grad=True)
print(x)
print(x.shape)
print(x.dtype)
```

这将打印：

```py
tensor([1., 2., 3.], requires_grad=True)
torch.Size([3])
torch.float32
```

请注意，上述创建了一个浮点值的张量。这是必要的，因为微分需要浮点数，而不是整数。

操作（如 `x+x` 和 `2*x`）仍然可以应用，但在这种情况下，张量将记住它如何获得其值。你可以在以下示例中演示这一特性：

```py
import torch

x = torch.tensor(3.6, requires_grad=True)
y = x * x
y.backward()
print(x.grad)
```

这将打印：

```py
tensor(7.2000)
```

它的作用如下：这定义了一个变量 `x`（值为 3.6），然后计算 `y=x*x` 或 $y=x²$。然后你请求 $y$ 的微分。由于 $y$ 的值来源于 $x$，你可以在运行 `y.backward()` 之后立即在 `x.grad` 中以张量形式找到 $\dfrac{dy}{dx}$。你知道 $y=x²$ 意味着 $y’=2x$。因此输出会给你 $3.6\times 2=7.2$ 的值。

### 想要开始使用 PyTorch 进行深度学习？

现在立即报名我的免费电子邮件速成课程（包括示例代码）。

点击报名，还可以获得免费 PDF 电子书版本的课程。

## 使用 Autograd 进行多项式回归

PyTorch 中这个特性有什么帮助？假设你有一个形式为 $y=f(x)$ 的多项式，并且你得到了一些 $(x,y)$ 样本。你如何恢复多项式 $f(x)$？一种方法是对多项式假设一个随机系数，并将样本 $(x,y)$ 输入进去。如果多项式被找到，你应该看到 $y$ 的值与 $f(x)$ 匹配。它们越接近，你的估计就越接近正确的多项式。

这确实是一个数值优化问题，你想最小化 $y$ 和 $f(x)$ 之间的差异。你可以使用梯度下降来解决它。

让我们考虑一个例子。你可以按照如下方式在 NumPy 中构建一个多项式 $f(x)=x² + 2x + 3$：

```py
import numpy as np

polynomial = np.poly1d([1, 2, 3])
print(polynomial)
```

这将输出：

```py
   2
1 x + 2 x + 3
```

你可以将多项式用作函数，例如：

```py
print(polynomial(1.5))
```

这将输出 `8.25`，因为 $(1.5)²+2\times(1.5)+3 = 8.25$。

现在你可以使用 NumPy 从这个函数生成大量样本：

```py
N = 20   # number of samples

# Generate random samples roughly between -10 to +10
X = np.random.randn(N,1) * 5
Y = polynomial(X)
```

上述内容中，`X` 和 `Y` 都是形状为 `(20,1)` 的 NumPy 数组，它们与多项式 $f(x)$ 的 $y=f(x)$ 相关。

现在，假设你不知道这个多项式是什么，只知道它是二次的。你想恢复系数。由于二次多项式的形式为 $Ax²+Bx+C$，你有三个未知数需要找出。你可以使用你实现的梯度下降算法或现有的梯度下降优化器来找到它们。以下展示了它是如何工作的：

```py
import torch

# Assume samples X and Y are prepared elsewhere

XX = np.hstack([X*X, X, np.ones_like(X)])

w = torch.randn(3, 1, requires_grad=True)  # the 3 coefficients
x = torch.tensor(XX, dtype=torch.float32)  # input sample
y = torch.tensor(Y, dtype=torch.float32)   # output sample
optimizer = torch.optim.NAdam([w], lr=0.01)
print(w)

for _ in range(1000):
    optimizer.zero_grad()
    y_pred = x @ w
    mse = torch.mean(torch.square(y - y_pred))
    mse.backward()
    optimizer.step()

print(w)
```

循环之前的 `print` 语句给出了三个随机数字，例如：

```py
tensor([[1.3827],
        [0.8629],
        [0.2357]], requires_grad=True)
```

但在循环之后的结果会给你非常接近多项式中的系数：

```py
tensor([[1.0004],
        [1.9924],
        [2.9159]], requires_grad=True)
```

上述代码的作用如下：首先，它创建了一个包含 3 个值的变量向量 `w`，即系数 $A,B,C$。然后，你创建了一个形状为 $(N,3)$ 的数组，其中 $N$ 是数组 `X` 中样本的数量。这个数组有 3 列：分别是 $x²$、$x$ 和 1。这样的数组是通过 `np.hstack()` 函数从向量 `X` 构建的。类似地，你可以从 NumPy 数组 `Y` 构建 TensorFlow 常量 `y`。

随后，你使用 for 循环在 1,000 次迭代中运行梯度下降。在每次迭代中，你以矩阵形式计算$x \times w$以找到$Ax²+Bx+C$并将其分配给变量`y_pred`。然后，比较`y`和`y_pred`并计算均方误差。接下来，使用`backward()`函数导出梯度，即均方误差相对于系数`w`的变化率。根据这个梯度，你通过优化器使用梯度下降更新`w`。

本质上，上述代码将找到最小化均方误差的系数`w`。

综合以上，以下是完整的代码：

```py
import numpy as np
import torch

polynomial = np.poly1d([1, 2, 3])
N = 20   # number of samples

# Generate random samples roughly between -10 to +10
X = np.random.randn(N,1) * 5
Y = polynomial(X)

# Prepare input as an array of shape (N,3)
XX = np.hstack([X*X, X, np.ones_like(X)])

# Prepare tensors
w = torch.randn(3, 1, requires_grad=True)  # the 3 coefficients
x = torch.tensor(XX, dtype=torch.float32)  # input sample
y = torch.tensor(Y, dtype=torch.float32)   # output sample
optimizer = torch.optim.NAdam([w], lr=0.01)
print(w)

# Run optimizer
for _ in range(1000):
    optimizer.zero_grad()
    y_pred = x @ w
    mse = torch.mean(torch.square(y - y_pred))
    mse.backward()
    optimizer.step()

print(w)
```

## 使用自动微分解决数学难题

在上述中，使用了 20 个样本，这足以拟合一个二次方程。你也可以使用梯度下降来解决一些数学难题。例如，以下问题：

```py
[ A ]  +  [ B ]  =  9
  +         -
[ C ]  -  [ D ]  =  1
  =         =
  8         2
```

换句话说，要找到$A,B,C,D$的值，使得：

$$\begin{aligned}

A + B &= 9 \\

C – D &= 1 \\

A + C &= 8 \\

B – D &= 2

\end{aligned}$$

这也可以使用自动微分来解决，如下所示：

```py
import random
import torch

A = torch.tensor(random.random(), requires_grad=True)
B = torch.tensor(random.random(), requires_grad=True)
C = torch.tensor(random.random(), requires_grad=True)
D = torch.tensor(random.random(), requires_grad=True)

# Gradient descent loop
EPOCHS = 2000
optimizer = torch.optim.NAdam([A, B, C, D], lr=0.01)
for _ in range(EPOCHS):
    y1 = A + B - 9
    y2 = C - D - 1
    y3 = A + C - 8
    y4 = B - D - 2
    sqerr = y1*y1 + y2*y2 + y3*y3 + y4*y4
    optimizer.zero_grad()
    sqerr.backward()
    optimizer.step()

print(A)
print(B)
print(C)
print(D)
```

这个问题可能有多个解决方案。一个解决方案如下：

```py
tensor(4.7191, requires_grad=True)
tensor(4.2808, requires_grad=True)
tensor(3.2808, requires_grad=True)
tensor(2.2808, requires_grad=True)
```

这意味着$A=4.72$，$B=4.28$，$C=3.28$，$D=2.28$。你可以验证这个解是否符合问题要求。

上述代码将四个未知数定义为具有随机初始值的变量。然后你计算四个方程的结果并与期望答案进行比较。接着，你将平方误差求和，并要求 PyTorch 的优化器最小化它。最小的平方误差是零，当我们的解完全符合问题时实现。

注意 PyTorch 生成梯度的方式：你要求`sqerr`的梯度，它注意到，除了其他内容外，只有`A`、`B`、`C`和`D`是其依赖项，且`requires_grad=True`。因此找到四个梯度。然后，你通过优化器在每次迭代中将每个梯度应用到相应的变量上。

## 进一步阅读

如果你想深入了解这个主题，本节提供了更多资源。

**文章：**

+   [自动微分机制](https://pytorch.org/docs/stable/notes/autograd.html)

+   [自动微分包 – torch.autograd](https://pytorch.org/docs/stable/autograd.html)

## 总结

在这篇文章中，我们展示了 PyTorch 的自动微分是如何工作的。这是进行深度学习训练的基础。具体来说，你学到了：

+   PyTorch 中的自动微分是什么

+   如何使用梯度记录来进行自动微分

+   如何使用自动微分来解决优化问题
