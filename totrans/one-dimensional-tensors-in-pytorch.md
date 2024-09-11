# Pytorch 中的一维张量

> 原文：[`machinelearningmastery.com/one-dimensional-tensors-in-pytorch/`](https://machinelearningmastery.com/one-dimensional-tensors-in-pytorch/)

PyTorch 是一个基于 Python 语言的开源深度学习框架。它允许你构建、训练和部署深度学习模型，提供了很多灵活性和效率。

PyTorch 主要集中在张量操作上，而张量可以是数字、矩阵或多维数组。

在这个教程中，我们将对一维张量执行一些基本操作，因为它们是复杂的数学对象，也是 PyTorch 库的重要组成部分。因此，在深入研究更高级的概念之前，我们应该先了解基础知识。

在完成本教程后，你将：

+   了解 PyTorch 中一维张量操作的基础知识。

+   了解张量类型和形状，并执行张量切片和索引操作。

+   能够对张量对象应用一些方法，如均值、标准差、加法、乘法等。

**启动你的项目**，使用我的书籍[《Deep Learning with PyTorch》](https://machinelearningmastery.com/deep-learning-with-pytorch/)。它提供了带有**工作代码**的**自学教程**。

让我们开始吧！![](img/a33f5fbb84b7bcd8c5bbde1f3c03eeca.png)

Pytorch 中的一维张量

照片由[Jo Szczepanska](https://unsplash.com/photos/9OKGEVJiTKk)提供。部分权利保留。

## **一维张量的类型和形状**

首先，让我们导入这个教程中将要使用的一些库。

```py
import torch
import numpy as np 
import pandas as pd
```

如果你有其他编程语言的经验，理解张量的最简单方法是将其视为多维数组。因此，一维张量就是一个一维数组，或者说是一个向量。为了将整数列表转换为张量，请应用`torch.tensor()`构造函数。例如，我们将取一个整数列表并将其转换为不同的张量对象。

```py
int_to_tensor = torch.tensor([10, 11, 12, 13])
print("Tensor object type after conversion: ", int_to_tensor.dtype)
print("Tensor object type after conversion: ", int_to_tensor.type())
```

```py
Tensor object type after conversion:  torch.int64
Tensor object type after conversion:  torch.LongTensor
```

同样，你也可以将相同的方法`torch.tensor()`应用于将浮点列表转换为浮点张量。

```py
float_to_tensor = torch.tensor([10.0, 11.0, 12.0, 13.0])
print("Tensor object type after conversion: ", float_to_tensor.dtype)
print("Tensor object type after conversion: ", float_to_tensor.type())
```

```py
Tensor object type after conversion:  torch.float32
Tensor object type after conversion:  torch.FloatTensor
```

注意，需要转换为张量的列表元素必须具有相同的类型。此外，如果你想将列表转换为特定的张量类型，torch 也允许你这样做。例如，下面的代码行将整数列表转换为浮点张量。

```py
int_list_to_float_tensor = torch.FloatTensor([10, 11, 12, 13])
int_list_to_float_tensor.type()
print("Tensor  type after conversion: ", int_list_to_float_tensor.type())
```

```py
Tensor  type after conversion:  torch.FloatTensor
```

类似地，`size()`和`ndimension()`方法允许你查找张量对象的大小和维度。

```py
print("Size of the int_list_to_float_tensor: ", int_list_to_float_tensor.size())
print("Dimensions of the int_list_to_float_tensor: ",int_list_to_float_tensor.ndimension())
```

```py
Size of the int_list_to_float_tensor:  torch.Size([4])
Dimensions of the int_list_to_float_tensor:  1
```

对于重塑张量对象，可以应用`view()`方法。它接受`rows`和`columns`作为参数。举个例子，让我们使用这个方法来重塑`int_list_to_float_tensor`。

```py
reshaped_tensor = int_list_to_float_tensor.view(4, 1)
print("Original Size of the tensor: ", reshaped_tensor)
print("New size of the tensor: ", reshaped_tensor)
```

```py
Original Size of the tensor:  tensor([[10.],
        [11.],
        [12.],
        [13.]])
New size of the tensor:  tensor([[10.],
        [11.],
        [12.],
        [13.]])
```

如你所见，`view()`方法已将张量的大小更改为`torch.Size([4, 1])`，其中有 4 行和 1 列。

在应用`view()`方法后，张量对象中的元素数量应保持不变，但你可以使用`-1`（例如`reshaped_tensor**.**view(-1, 1)`）来重塑一个动态大小的张量。

### **将 Numpy 数组转换为张量**

Pytorch 也允许你将 NumPy 数组转换为张量。你可以使用 `torch.from_numpy` 完成这个操作。让我们拿一个 NumPy 数组并应用这个操作。

```py
numpy_arr = np.array([10.0, 11.0, 12.0, 13.0])
from_numpy_to_tensor = torch.from_numpy(numpy_arr)

print("dtype of the tensor: ", from_numpy_to_tensor.dtype)
print("type of the tensor: ", from_numpy_to_tensor.type())
```

```py
dtype of the tensor:  torch.float64
type of the tensor:  torch.DoubleTensor
```

同样地，你可以将张量对象转换回 NumPy 数组。让我们用之前的例子展示如何做到这一点。

```py
tensor_to_numpy = from_numpy_to_tensor.numpy()
print("back to numpy from tensor: ", tensor_to_numpy)
print("dtype of converted numpy array: ", tensor_to_numpy.dtype)
```

```py
back to numpy from tensor:  [10\. 11\. 12\. 13.]
dtype of converted numpy array:  float64
```

### **将 Pandas Series 转换为张量**

你也可以将 Pandas Series 转换为张量。为此，首先需要使用 `values()` 函数将 Pandas Series 存储为 NumPy 数组。

```py
pandas_series=pd.Series([1, 0.2, 3, 13.1])
store_with_numpy=torch.from_numpy(pandas_series.values)
print("Stored tensor in numpy array: ", store_with_numpy)
print("dtype of stored tensor: ", store_with_numpy.dtype)
print("type of stored tensor: ", store_with_numpy.type())
```

```py
Stored tensor in numpy array:  tensor([ 1.0000,  0.2000,  3.0000, 13.1000], dtype=torch.float64)
dtype of stored tensor:  torch.float64
type of stored tensor:  torch.DoubleTensor
```

此外，Pytorch 框架允许我们对张量做很多事情，例如它的 `item()` 方法从张量返回一个 Python 数字，而 `tolist()` 方法则返回一个列表。

```py
new_tensor=torch.tensor([10, 11, 12, 13]) 
print("the second item is",new_tensor[1].item())
tensor_to_list=new_tensor.tolist()
print('tensor:', new_tensor,"\nlist:",tensor_to_list)
```

```py
the second item is 11
tensor: tensor([10, 11, 12, 13])
list: [10, 11, 12, 13]
```

## **一维张量中的索引和切片**

Pytorch 中的索引和切片操作与 Python 几乎相同。因此，第一个索引始终从 0 开始，最后一个索引小于张量的总长度。使用方括号访问张量中的任何数字。

```py
tensor_index = torch.tensor([0, 1, 2, 3])
print("Check value at index 0:",tensor_index[0])
print("Check value at index 3:",tensor_index[3])
```

```py
Check value at index 0: tensor(0)
Check value at index 3: tensor(3)
```

就像 Python 中的列表一样，你也可以在张量中的值上执行切片操作。此外，Pytorch 库还允许你更改张量中的某些值。

让我们举个例子来检查如何应用这些操作。

```py
example_tensor = torch.tensor([50, 11, 22, 33, 44])
sclicing_tensor = example_tensor[1:4]
print("example tensor : ", example_tensor)
print("subset of example tensor:", sclicing_tensor)
```

```py
example tensor :  tensor([50, 11, 22, 33, 44])
subset of example tensor: tensor([11, 22, 33])
```

现在，让我们改变 `example_tensor` 的索引 3 处的值。

```py
print("value at index 3 of example tensor:", example_tensor[3])
example_tensor[3] = 0
print("new tensor:", example_tensor)
```

```py
value at index 3 of example tensor: tensor(0)
new tensor: tensor([50, 11, 22,  0, 44])
```

### 想要开始使用 PyTorch 进行深度学习吗？

现在就开始我的免费电子邮件速成课程（附带示例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## **应用在一维张量上的一些函数**

在这一节中，我们将回顾一些可以应用在张量对象上的统计方法。

### **最小值和最大值函数**

这两种有用的方法用于在张量中找到最小值和最大值。以下是它们的工作原理。

我们将使用 `sample_tensor` 作为示例来应用这些方法。

```py
sample_tensor = torch.tensor([5, 4, 3, 2, 1])
min_value = sample_tensor.min()
max_value = sample_tensor.max()
print("check minimum value in the tensor: ", min_value)
print("check maximum value in the tensor: ", max_value)
```

```py
check minimum value in the tensor:  tensor(1)
check maximum value in the tensor:  tensor(5)
```

### **均值和标准差**

在张量进行统计操作时，常常使用均值和标准差。你可以使用 Pytorch 中的 `.mean()` 和 `.std()` 函数应用这两个指标。

让我们用一个例子来看看这两个指标是如何计算的。

```py
mean_std_tensor = torch.tensor([-1.0, 2.0, 1, -2])
Mean = mean_std_tensor.mean()
print("mean of mean_std_tensor: ", Mean)
std_dev = mean_std_tensor.std()
print("standard deviation of mean_std_tensor: ", std_dev)
```

```py
mean of mean_std_tensor:  tensor(0.)
standard deviation of mean_std_tensor:  tensor(1.8257)
```

## **一维张量上的简单加法和乘法运算**

在 Pytorch 中，可以轻松地对张量应用加法和乘法操作。在本节中，我们将创建两个一维张量来演示如何使用这些操作。

```py
a = torch.tensor([1, 1])
b = torch.tensor([2, 2])

add = a + b
multiply = a * b

print("addition of two tensors: ", add)
print("multiplication of two tensors: ", multiply)
```

```py
addition of two tensors:  tensor([3, 3])
multiplication of two tensors:  tensor([2, 2])
```

为了方便起见，以下是上述所有示例的综合，这样你可以一次尝试它们：

```py
import torch
import numpy as np
import pandas as pd

int_to_tensor = torch.tensor([10, 11, 12, 13])
print("Tensor object type after conversion: ", int_to_tensor.dtype)
print("Tensor object type after conversion: ", int_to_tensor.type())

float_to_tensor = torch.tensor([10.0, 11.0, 12.0, 13.0])
print("Tensor object type after conversion: ", float_to_tensor.dtype)
print("Tensor object type after conversion: ", float_to_tensor.type())

int_list_to_float_tensor = torch.FloatTensor([10, 11, 12, 13])
int_list_to_float_tensor.type()
print("Tensor  type after conversion: ", int_list_to_float_tensor.type())

print("Size of the int_list_to_float_tensor: ", int_list_to_float_tensor.size())
print("Dimensions of the int_list_to_float_tensor: ",int_list_to_float_tensor.ndimension())

reshaped_tensor = int_list_to_float_tensor.view(4, 1)
print("Original Size of the tensor: ", reshaped_tensor)
print("New size of the tensor: ", reshaped_tensor)

numpy_arr = np.array([10.0, 11.0, 12.0, 13.0])
from_numpy_to_tensor = torch.from_numpy(numpy_arr)
print("dtype of the tensor: ", from_numpy_to_tensor.dtype)
print("type of the tensor: ", from_numpy_to_tensor.type())

tensor_to_numpy = from_numpy_to_tensor.numpy()
print("back to numpy from tensor: ", tensor_to_numpy)
print("dtype of converted numpy array: ", tensor_to_numpy.dtype)

pandas_series=pd.Series([1, 0.2, 3, 13.1])
store_with_numpy=torch.from_numpy(pandas_series.values)
print("Stored tensor in numpy array: ", store_with_numpy)
print("dtype of stored tensor: ", store_with_numpy.dtype)
print("type of stored tensor: ", store_with_numpy.type())

new_tensor=torch.tensor([10, 11, 12, 13]) 
print("the second item is",new_tensor[1].item())
tensor_to_list=new_tensor.tolist()
print('tensor:', new_tensor,"\nlist:",tensor_to_list)

tensor_index = torch.tensor([0, 1, 2, 3])
print("Check value at index 0:",tensor_index[0])
print("Check value at index 3:",tensor_index[3])

example_tensor = torch.tensor([50, 11, 22, 33, 44])
sclicing_tensor = example_tensor[1:4]
print("example tensor : ", example_tensor)
print("subset of example tensor:", sclicing_tensor)

print("value at index 3 of example tensor:", example_tensor[3])
example_tensor[3] = 0
print("new tensor:", example_tensor)

sample_tensor = torch.tensor([5, 4, 3, 2, 1])
min_value = sample_tensor.min()
max_value = sample_tensor.max()
print("check minimum value in the tensor: ", min_value)
print("check maximum value in the tensor: ", max_value)

mean_std_tensor = torch.tensor([-1.0, 2.0, 1, -2])
Mean = mean_std_tensor.mean()
print("mean of mean_std_tensor: ", Mean)
std_dev = mean_std_tensor.std()
print("standard deviation of mean_std_tensor: ", std_dev)

a = torch.tensor([1, 1])
b = torch.tensor([2, 2])
add = a + b
multiply = a * b
print("addition of two tensors: ", add)
print("multiplication of two tensors: ", multiply)
```

## 进一步阅读

PyTorch 在 TensorFlow 发布 2.x 版本前曾有着更简单的语法，直到 TensorFlow 采用 Keras。要学习 PyTorch 的基础知识，你可能想阅读 PyTorch 的教程：

+   [`pytorch.org/tutorials/`](https://pytorch.org/tutorials/)

特别是可以在张量教程页面找到 PyTorch 张量的基础知识：

+   [`pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html`](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)

PyTorch 还有很多适合初学者的书籍。由于工具和语法在不断发展，推荐更近期出版的书籍。一个例子是

+   《深度学习与 PyTorch》（Eli Stevens, Luca Antiga, Thomas Viehmann，2020 年）

    [`www.manning.com/books/deep-learning-with-pytorch`](https://www.manning.com/books/deep-learning-with-pytorch)

## **摘要**

在本教程中，你学会了如何在 PyTorch 中使用一维张量。

具体来说，你学到了：

+   PyTorch 中一维张量操作的基础

+   关于张量类型和形状以及如何执行张量切片和索引操作

+   如何在张量对象上应用一些方法，例如平均值、标准差、加法和乘法
