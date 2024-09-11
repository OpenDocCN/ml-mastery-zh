# PyTorch 中的二维张量

> 原文：[`machinelearningmastery.com/two-dimensional-tensors-in-pytorch/`](https://machinelearningmastery.com/two-dimensional-tensors-in-pytorch/)

二维张量类似于二维度量。就像二维度量一样，二维张量也有 $n$ 行和列。

让我们以灰度图像为例，这是一个由数值构成的二维矩阵，通常称为像素。从 '0' 到 '255'，每个数字代表像素强度值。这里，最小强度数（即 '0'）代表图像中的黑色区域，而最高强度数（即 '255'）代表图像中的白色区域。使用 PyTorch 框架，这种二维图像或矩阵可以转换为二维张量。

在上一篇文章中，我们了解了 [PyTorch 中的一维张量](https://machinelearningmastery.com/one-dimensional-tensors-in-pytorch/) 并应用了一些有用的张量操作。在本教程中，我们将使用 PyTorch 库将这些操作应用于二维张量。具体来说，我们将学习：

+   如何在 PyTorch 中创建二维张量并探索它们的类型和形状。

+   关于二维张量的切片和索引操作的详细信息。

+   要对张量应用多种方法，如张量加法、乘法等。

**启动您的项目**，使用我的书籍 [深度学习与 PyTorch](https://machinelearningmastery.com/deep-learning-with-pytorch/)。它提供了带有 **工作代码** 的 **自学教程**。

让我们开始吧！![](img/9660d9ef15ab9f0b20162f27d5ced700.png)

PyTorch 中的二维张量

图片由 [dylan dolte](https://unsplash.com/photos/NIrgENd0sAY) 拍摄。部分权利保留。

## 教程概览

本教程分为几部分，它们是：

+   二维张量的类型和形状

+   将二维张量转换为 NumPy 数组

+   将 pandas series 转换为二维张量

+   二维张量上的索引和切片操作

+   二维张量的操作

## **二维张量的类型和形状**

让我们首先导入本教程中将要使用的几个必要库。

```py
import torch
import numpy as np 
import pandas as pd
```

要检查二维张量的类型和形状，我们将使用来自 PyTorch 的相同方法，[之前介绍过用于一维张量](https://machinelearningmastery.com/one-dimensional-tensors-in-pytorch/) 的方法。但是，它对于二维张量的工作方式应该是一样的吗？

让我们演示通过将整数的二维列表转换为二维张量对象。作为示例，我们将创建一个二维列表，并应用 `torch.tensor()` 进行转换。

```py
example_2D_list = [[5, 10, 15, 20],
                   [25, 30, 35, 40],
                   [45, 50, 55, 60]]
list_to_tensor = torch.tensor(example_2D_list)
print("Our New 2D Tensor from 2D List is: ", list_to_tensor)
```

```py
Our New 2D Tensor from 2D List is:  tensor([[ 5, 10, 15, 20],
        [25, 30, 35, 40],
        [45, 50, 55, 60]])
```

正如您所见，`torch.tensor()` 方法对于二维张量也非常有效。现在，让我们使用 `shape()`、`size()` 和 `ndimension()` 方法来返回张量对象的形状、大小和维度。

```py
print("Getting the shape of tensor object: ", list_to_tensor.shape)
print("Getting the size of tensor object: ", list_to_tensor.size())
print("Getting the dimensions of tensor object: ", list_to_tensor.ndimension())
```

```py
print("Getting the shape of tensor object: ", list_to_tensor.shape)
print("Getting the size of tensor object: ", list_to_tensor.size())
print("Getting the dimensions of tensor object: ", list_to_tensor.ndimension())
```

### 想要开始使用 PyTorch 进行深度学习吗？

立即注册我的免费电子邮件崩溃课程（附有示例代码）。

点击注册，还可以获取课程的免费 PDF 电子书版本。

## **将二维张量转换为 NumPy 数组**

PyTorch 允许我们将二维张量转换为 NumPy 数组，然后再转换回张量。让我们看看如何操作。

```py
# Converting two_D tensor to numpy array

twoD_tensor_to_numpy = list_to_tensor.numpy()
print("Converting two_Dimensional tensor to numpy array:")
print("Numpy array after conversion: ", twoD_tensor_to_numpy)
print("Data type after conversion: ", twoD_tensor_to_numpy.dtype)

print("***************************************************************")

# Converting numpy array back to a tensor

back_to_tensor = torch.from_numpy(twoD_tensor_to_numpy)
print("Converting numpy array back to two_Dimensional tensor:")
print("Tensor after conversion:", back_to_tensor)
print("Data type after conversion: ", back_to_tensor.dtype)
```

```py
Converting two_Dimensional tensor to numpy array:
Numpy array after conversion:  [[ 5 10 15 20]
 [25 30 35 40]
 [45 50 55 60]]
Data type after conversion:  int64
***************************************************************
Converting numpy array back to two_Dimensional tensor:
Tensor after conversion: tensor([[ 5, 10, 15, 20],
        [25, 30, 35, 40],
        [45, 50, 55, 60]])
Data type after conversion:  torch.int64
```

## **将 Pandas Series 转换为二维张量**

同样地，我们也可以将 pandas DataFrame 转换为张量。与一维张量类似，我们将使用相同的步骤进行转换。使用`values`属性获取 NumPy 数组，然后使用`torch.from_numpy`将 pandas DataFrame 转换为张量。

这是我们将如何执行此操作。

```py
# Converting Pandas Dataframe to a Tensor

dataframe = pd.DataFrame({'x':[22,24,26],'y':[42,52,62]})

print("Pandas to numpy conversion: ", dataframe.values)
print("Data type before tensor conversion: ", dataframe.values.dtype)

print("***********************************************")

pandas_to_tensor = torch.from_numpy(dataframe.values)
print("Getting new tensor: ", pandas_to_tensor)
print("Data type after conversion to tensor: ", pandas_to_tensor.dtype)
```

```py
Pandas to numpy conversion:  [[22 42]
 [24 52]
 [26 62]]
Data type before tensor conversion:  int64
***********************************************
Getting new tensor:  tensor([[22, 42],
        [24, 52],
        [26, 62]])
Data type after conversion to tensor:  torch.int64
```

## **二维张量的索引和切片操作**

对于索引操作，可以使用方括号访问张量对象中的不同元素。只需将对应的索引放入方括号中，即可访问张量中所需的元素。

在下面的例子中，我们将创建一个张量，并使用两种不同的方法访问某些元素。请注意，索引值应始终比二维张量中元素实际位置少一个。

```py
example_tensor = torch.tensor([[10, 20, 30, 40],
                               [50, 60, 70, 80],
                               [90, 100, 110, 120]])
print("Accessing element in 2nd row and 2nd column: ", example_tensor[1, 1])
print("Accessing element in 2nd row and 2nd column: ", example_tensor[1][1])

print("********************************************************")

print("Accessing element in 3rd row and 4th column: ", example_tensor[2, 3])
print("Accessing element in 3rd row and 4th column: ", example_tensor[2][3])
```

```py
Accessing element in 2nd row and 2nd column:  tensor(60)
Accessing element in 2nd row and 2nd column:  tensor(60)
********************************************************
Accessing element in 3rd row and 4th column:  tensor(120)
Accessing element in 3rd row and 4th column:  tensor(120)
```

当我们需要同时访问两个或更多元素时，我们需要使用张量切片。让我们使用之前的例子来访问第二行的前两个元素和第三行的前三个元素。

```py
example_tensor = torch.tensor([[10, 20, 30, 40],
                               [50, 60, 70, 80],
                               [90, 100, 110, 120]])
print("Accessing first two elements of the second row: ", example_tensor[1, 0:2])
print("Accessing first two elements of the second row: ", example_tensor[1][0:2])

print("********************************************************")

print("Accessing first three elements of the third row: ", example_tensor[2, 0:3])
print("Accessing first three elements of the third row: ", example_tensor[2][0:3])
```

```py
example_tensor = torch.tensor([[10, 20, 30, 40],
                               [50, 60, 70, 80],
                               [90, 100, 110, 120]])
print("Accessing first two elements of the second row: ", example_tensor[1, 0:2])
print("Accessing first two elements of the second row: ", example_tensor[1][0:2])

print("********************************************************")

print("Accessing first three elements of the third row: ", example_tensor[2, 0:3])
print("Accessing first three elements of the third row: ", example_tensor[2][0:3])
```

## **二维张量的操作**

在使用 PyTorch 框架处理二维张量时，有许多操作可以进行。在这里，我们将介绍张量加法、标量乘法和矩阵乘法。

### **二维张量的加法**

将两个张量相加类似于矩阵加法。这是一个非常直接的过程，您只需使用加号（+）运算符即可执行操作。让我们在下面的例子中相加两个张量。

```py
A = torch.tensor([[5, 10],
                  [50, 60], 
                  [100, 200]]) 
B = torch.tensor([[10, 20], 
                  [60, 70], 
                  [200, 300]])
add = A + B
print("Adding A and B to get: ", add)
```

```py
Adding A and B to get:  tensor([[ 15,  30],
        [110, 130],
        [300, 500]])
```

### **二维张量的标量和矩阵乘法**

二维张量的标量乘法与矩阵中的标量乘法相同。例如，通过与标量（例如 4）相乘，您将对张量中的每个元素乘以 4。

```py
new_tensor = torch.tensor([[1, 2, 3], 
                           [4, 5, 6]]) 
mul_scalar = 4 * new_tensor
print("result of scalar multiplication: ", mul_scalar)
```

```py
result of scalar multiplication:  tensor([[ 4,  8, 12],
        [16, 20, 24]])
```

关于二维张量的乘法，`torch.mm()`在 PyTorch 中为我们简化了操作。与线性代数中的矩阵乘法类似，张量对象 A（即 2×3）的列数必须等于张量对象 B（即 3×2）的行数。

```py
A = torch.tensor([[3, 2, 1], 
                  [1, 2, 1]])
B = torch.tensor([[3, 2], 
                  [1, 1], 
                  [2, 1]])
A_mult_B = torch.mm(A, B)
print("multiplying A with B: ", A_mult_B)
```

```py
multiplying A with B:  tensor([[13,  9],
        [ 7,  5]])
```

## 进一步阅读

PyTorch 与 TensorFlow 同时开发，直到 TensorFlow 在其 2.x 版本中采用了 Keras 之前，PyTorch 的语法更为简单。要学习 PyTorch 的基础知识，您可以阅读 PyTorch 教程：

+   [`pytorch.org/tutorials/`](https://pytorch.org/tutorials/)

特别是 PyTorch 张量的基础知识可以在张量教程页面找到：

+   [`pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html`](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)

对于 PyTorch 的入门者来说，也有不少适合的书籍。应推荐较新出版的书籍，因为工具和语法在积极地发展变化。一个例子是

+   《深度学习与 PyTorch》由 Eli Stevens、Luca Antiga 和 Thomas Viehmann 编写，2020 年出版。

    [《深度学习与 PyTorch》书籍链接](https://www.manning.com/books/deep-learning-with-pytorch)

## **总结**

在本教程中，您了解了 PyTorch 中的二维张量。

具体来说，您学到了：

+   如何在 PyTorch 中创建二维张量并探索它们的类型和形状。

+   关于二维张量的切片和索引操作的详细信息。

+   应用多种方法对张量进行操作，如张量的加法、乘法等。
