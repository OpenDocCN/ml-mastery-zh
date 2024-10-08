# 在 PyTorch 中使用数据集类

> 原文：[`machinelearningmastery.com/using-dataset-classes-in-pytorch/`](https://machinelearningmastery.com/using-dataset-classes-in-pytorch/)

在机器学习和深度学习问题中，大量的工作都在于准备数据。数据通常是混乱的，需要在用于训练模型之前进行预处理。如果数据准备不正确，模型将无法很好地泛化。

数据预处理的一些常见步骤包括：

+   数据归一化：这包括将数据归一化到数据集中的一个值范围内。

+   数据增强：这包括通过添加噪声或特征的位移来生成新的样本，使它们更加多样化。

数据准备是任何机器学习流程中的关键步骤。PyTorch 带来了许多模块，例如 torchvision，它提供了数据集和数据集类，使数据准备变得容易。

在本教程中，我们将演示如何在 PyTorch 中使用数据集和变换，以便你可以创建自己的自定义数据集类，并以你希望的方式操作数据集。特别是，你将学习：

+   如何创建一个简单的数据集类并对其应用变换。

+   如何构建可调用的变换并将其应用于数据集对象。

+   如何在数据集对象上组合各种变换。

请注意，在这里你将使用简单的数据集来对概念有一个总体了解，而在本教程的下一部分，你将有机会使用图像的数据集对象。

**通过我的书籍[《深度学习与 PyTorch》](https://machinelearningmastery.com/deep-learning-with-pytorch/)** 来**启动你的项目**。它提供了**自学教程**和**可运行的代码**。

让我们开始吧！[](../Images/f1c2c2ed936adbba5fbf14c3624f34b9.png)

在 PyTorch 中使用数据集类

图片由[NASA](https://unsplash.com/photos/1lfI7wkGWZ4)提供。保留所有权利。

## 概述

本教程分为三部分，它们是：

+   创建一个简单的数据集类

+   创建可调用的变换

+   为数据集组合多个变换

## 创建一个简单的数据集类

在开始之前，我们需要导入一些包，然后创建数据集类。

```py
import torch
from torch.utils.data import Dataset
torch.manual_seed(42)
```

我们将从`torch.utils.data`中导入抽象类`Dataset`。因此，我们在数据集类中重写以下方法：

+   `__len__` 使得 `len(dataset)` 可以告诉我们数据集的大小。

+   `__getitem__` 用于通过支持索引操作来访问数据集中的数据样本。例如，`dataset[i]` 可用于检索第 i 个数据样本。

同样，`torch.manual_seed()` 强制随机函数每次重新编译时生成相同的数字。

现在，让我们定义数据集类。

```py
class SimpleDataset(Dataset):
    # defining values in the constructor
    def __init__(self, data_length = 20, transform = None):
        self.x = 3 * torch.eye(data_length, 2)
        self.y = torch.eye(data_length, 4)
        self.transform = transform
        self.len = data_length

    # Getting the data samples
    def __getitem__(self, idx):
        sample = self.x[idx], self.y[idx]
        if self.transform:
            sample = self.transform(sample)     
        return sample

    # Getting data size/length
    def __len__(self):
        return self.len
```

在对象构造函数中，我们创建了特征和目标的值，即`x`和`y`，并将其值分配给张量`self.x`和`self.y`。每个张量包含 20 个数据样本，而属性`data_length`存储数据样本的数量。我们将在教程的后面讨论转换。

`SimpleDataset`对象的行为类似于任何 Python 可迭代对象，如列表或元组。现在，让我们创建`SimpleDataset`对象，查看其总长度和索引 1 处的值。

```py
dataset = SimpleDataset()
print("length of the SimpleDataset object: ", len(dataset))
print("accessing value at index 1 of the simple_dataset object: ", dataset[1])
```

这将打印

```py
length of the SimpleDataset object:  20
accessing value at index 1 of the simple_dataset object:  (tensor([0., 3.]), tensor([0., 1., 0., 0.]))
```

由于我们的数据集是可迭代的，让我们使用循环打印出前四个元素：

```py
for i in range(4):
    x, y = dataset[i]
    print(x, y)
```

这将打印

```py
tensor([3., 0.]) tensor([1., 0., 0., 0.])
tensor([0., 3.]) tensor([0., 1., 0., 0.])
tensor([0., 0.]) tensor([0., 0., 1., 0.])
tensor([0., 0.]) tensor([0., 0., 0., 1.])
```

## 创建可调用的转换

在某些情况下，你需要创建可调用的转换，以便规范化或标准化数据。这些转换可以应用于张量。让我们创建一个可调用的转换，并将其应用于我们在本教程中早些时候创建的“简单数据集”对象。

```py
# Creating a callable tranform class mult_divide
class MultDivide:
    # Constructor
    def __init__(self, mult_x = 2, divide_y = 3):
        self.mult_x = mult_x
        self.divide_y = divide_y

    # caller
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x * self.mult_x
        y = y / self.divide_y
        sample = x, y
        return sample
```

我们创建了一个简单的自定义转换`MultDivide`，它将`x`乘以`2`并将`y`除以`3`。这没有实际用途，而是为了演示一个可调用的类如何作为我们数据集类的转换。记住，我们在`simple_dataset`中声明了一个参数`transform = None`。现在，我们可以用刚创建的自定义转换对象替换那个`None`。

那么，让我们展示一下如何操作，并在数据集上调用这个转换对象，以查看它如何转换数据集的前四个元素。

```py
# calling the transform object
mul_div = MultDivide()
custom_dataset = SimpleDataset(transform = mul_div)

for i in range(4):
    x, y = dataset[i]
    print('Idx: ', i, 'Original_x: ', x, 'Original_y: ', y)
    x_, y_ = custom_dataset[i]
    print('Idx: ', i, 'Transformed_x:', x_, 'Transformed_y:', y_)
```

这将打印

```py
Idx:  0 Original_x:  tensor([3., 0.]) Original_y:  tensor([1., 0., 0., 0.])
Idx:  0 Transformed_x: tensor([6., 0.]) Transformed_y: tensor([0.3333, 0.0000, 0.0000, 0.0000])
Idx:  1 Original_x:  tensor([0., 3.]) Original_y:  tensor([0., 1., 0., 0.])
Idx:  1 Transformed_x: tensor([0., 6.]) Transformed_y: tensor([0.0000, 0.3333, 0.0000, 0.0000])
Idx:  2 Original_x:  tensor([0., 0.]) Original_y:  tensor([0., 0., 1., 0.])
Idx:  2 Transformed_x: tensor([0., 0.]) Transformed_y: tensor([0.0000, 0.0000, 0.3333, 0.0000])
Idx:  3 Original_x:  tensor([0., 0.]) Original_y:  tensor([0., 0., 0., 1.])
Idx:  3 Transformed_x: tensor([0., 0.]) Transformed_y: tensor([0.0000, 0.0000, 0.0000, 0.3333])
```

如你所见，转换已成功应用于数据集的前四个元素。

### 想要开始使用 PyTorch 进行深度学习吗？

现在就获取我的免费电子邮件速成课程（附带示例代码）。

点击注册，还可以获得课程的免费 PDF 电子书版本。

## 为数据集组合多个转换

我们经常希望对数据集执行多个串行转换。这可以通过从 torchvision 中的 transforms 模块导入`Compose`类来实现。例如，假设我们构建了另一个转换`SubtractOne`，并将其应用于我们的数据集，除了我们之前创建的`MultDivide`转换。

一旦应用，新创建的转换将从数据集的每个元素中减去 1。

```py
from torchvision import transforms

# Creating subtract_one tranform
class SubtractOne:
    # Constructor
    def __init__(self, number = 1):
        self.number = number

    # caller
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x - self.number
        y = y - self.number
        sample = x, y
        return sample
```

如前所述，现在我们将用`Compose`方法组合这两个转换。

```py
# Composing multiple transforms
mult_transforms = transforms.Compose([MultDivide(), SubtractOne()])
```

请注意，首先将对数据集应用`MultDivide`转换，然后在转换后的数据集元素上应用`SubtractOne`转换。

我们将把包含两个转换（即`MultDivide()`和`SubtractOne()`）组合的`Compose`对象传递给我们的`SimpleDataset`对象。

```py
# Creating a new simple_dataset object with multiple transforms
new_dataset = SimpleDataset(transform = mult_transforms)
```

现在，多个转换的组合已应用于数据集，让我们打印出转换后的数据集的前四个元素。

```py
for i in range(4):
    x, y = dataset[i]
    print('Idx: ', i, 'Original_x: ', x, 'Original_y: ', y)
    x_, y_ = new_dataset[i]
    print('Idx: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
```

将所有内容放在一起，完整的代码如下：

```py
import torch
from torch.utils.data import Dataset
from torchvision import transforms

torch.manual_seed(2)

class SimpleDataset(Dataset):
    # defining values in the constructor
    def __init__(self, data_length = 20, transform = None):
        self.x = 3 * torch.eye(data_length, 2)
        self.y = torch.eye(data_length, 4)
        self.transform = transform
        self.len = data_length

    # Getting the data samples
    def __getitem__(self, idx):
        sample = self.x[idx], self.y[idx]
        if self.transform:
            sample = self.transform(sample)     
        return sample

    # Getting data size/length
    def __len__(self):
        return self.len

# Creating a callable tranform class mult_divide
class MultDivide:
    # Constructor
    def __init__(self, mult_x = 2, divide_y = 3):
        self.mult_x = mult_x
        self.divide_y = divide_y

    # caller
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x * self.mult_x
        y = y / self.divide_y
        sample = x, y
        return sample

# Creating subtract_one tranform
class SubtractOne:
    # Constructor
    def __init__(self, number = 1):
        self.number = number

    # caller
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x - self.number
        y = y - self.number
        sample = x, y
        return sample

# Composing multiple transforms
mult_transforms = transforms.Compose([MultDivide(), SubtractOne()])

# Creating a new simple_dataset object with multiple transforms
dataset = SimpleDataset()
new_dataset = SimpleDataset(transform = mult_transforms)

print("length of the simple_dataset object: ", len(dataset))
print("accessing value at index 1 of the simple_dataset object: ", dataset[1])

for i in range(4):
    x, y = dataset[i]
    print('Idx: ', i, 'Original_x: ', x, 'Original_y: ', y)
    x_, y_ = new_dataset[i]
    print('Idx: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
```

## 总结

在本教程中，你学会了如何在 PyTorch 中创建自定义数据集和转换。特别是，你学到了：

+   如何创建一个简单的数据集类并对其应用转换。

+   如何构建可调用的转换并将其应用于数据集对象。

+   如何在数据集对象上组合各种转换。
