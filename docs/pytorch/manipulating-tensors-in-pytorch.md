# 在 PyTorch 中操作张量

> 原文：[`machinelearningmastery.com/manipulating-tensors-in-pytorch/`](https://machinelearningmastery.com/manipulating-tensors-in-pytorch/)

PyTorch 是一个深度学习库。就像一些其他深度学习库一样，它对称为 **张量** 的数值数组执行操作。简而言之，张量就是多维数组。当我们处理张量时，某些操作使用得非常频繁。在 PyTorch 中，有一些函数专门用于处理张量。

在接下来的内容中，我们将简要概述 PyTorch 在张量方面的提供以及如何使用它们。完成本教程后，你将了解：

+   如何创建和操作 PyTorch 张量

+   PyTorch 的张量语法类似于 NumPy

+   你可以使用 PyTorch 中的常用函数来操作张量。

**启动你的项目**，请参阅我的书 [Deep Learning with PyTorch](https://machinelearningmastery.com/deep-learning-with-pytorch/)。它提供了 **自学教程** 和 **可运行的代码**。

让我们开始吧。![](img/468f6fbc5538c89afe0de5bd41bcfef4.png)

在 PyTorch 中操作张量。照片由 [Big Dodzy](https://unsplash.com/photos/n4BW2LPf7t8) 提供。版权所有。

## 概述

本教程分为四个部分；它们是：

+   创建张量

+   检查张量

+   操作张量

+   张量函数

## 创建张量

如果你对 NumPy 熟悉，你应该会记得有多种方式来创建数组。在 PyTorch 中创建张量也是如此。创建特定常量矩阵的最简单方法如下：

$$

\begin{bmatrix}

1 & 2 & 3 \\

4 & 5 & 6

\end{bmatrix}

$$

这是通过使用：

```py
import torch
a = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.int32)
print(a)
```

它打印：

```py
tensor([[1, 2, 3],
        [4, 5, 6]], dtype=torch.int32)
```

`dtype` 参数指定张量中值的数据类型。它是可选的。你还可以提供来自 NumPy 数组的值，并将其转换为 PyTorch 张量。

通常，你会为了某个特定目的创建张量。例如，如果你想要十个在 -1 和 1 之间均匀分布的值，你可以使用 `linspace()` 函数：

```py
a = torch.linspace(-1, 1, 10)
print(a)
```

它打印：

```py
tensor([-1.0000, -0.7778, -0.5556, -0.3333, -0.1111,  0.1111,  0.3333,  0.5556,
         0.7778,  1.0000])
```

不过，如果你想要一个包含随机值的张量（这在测试你的函数时非常有用），你可以创建一个如下的张量：

```py
a = torch.rand(3,4)
print(a)
```

例如，它打印：

```py
tensor([[0.4252, 0.1029, 0.9858, 0.7502],
        [0.1993, 0.6412, 0.2424, 0.6451],
        [0.7878, 0.7615, 0.9170, 0.8534]])
```

结果张量的维度为 $3\times 4$，每个值在 0 和 1 之间均匀分布。如果你想要值呈正态分布，只需将函数更改为 `randn()`：

```py
a = torch.randn(3,4)
```

如果你想要随机值为整数，例如，在 3 到 10 之间，你可以使用 `randint()` 函数：

```py
a = torch.randint(3, 10, size=(3,4))
print(a)
```

例如，这将产生：

```py
tensor([[4, 5, 7, 9],
        [3, 8, 8, 9],
        [4, 7, 7, 6]])
```

这些值的范围是 $3 \le x < 10$。默认情况下，下限是零，因此如果你希望值的范围是 $0 \le x < 10$，你可以使用：

```py
a = torch.randint(10, size=(3,4))
```

其他常用的张量包括零张量和所有值相同的张量。要创建一个零张量（例如，维度为 $2\times 3\times 4$），你可以使用：

```py
a = torch.zeros(2, 3, 4)
print(a)
```

它打印：

```py
tensor([[[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]],

        [[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]]])
```

如果你想创建一个所有值都是 5 的张量，你可以使用：

```py
a = torch.full((2,3,4), 5)
print(a)
```

它打印：

```py
tensor([[[5, 5, 5, 5],
         [5, 5, 5, 5],
         [5, 5, 5, 5]],

        [[5, 5, 5, 5],
         [5, 5, 5, 5],
         [5, 5, 5, 5]]])
```

但如果你想要所有值都为一，有一个更简单的函数：

```py
a = torch.ones(2,3,4)
```

最后，如果你想要一个单位矩阵，可以使用 `diag()` 或 `eye()` 来获得：

```py
a = torch.eye(4)
print(a)
```

它将打印出：

```py
tensor([[1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]])
```

### 想要开始使用 PyTorch 进行深度学习？

立即参加我的免费电子邮件速成课程（包括示例代码）。

点击注册，还可以获得课程的免费 PDF 电子书版本。

## 检查张量

一旦你有了一个张量，并且你想了解更多关于它的信息，你可以简单地使用 `print()` 将其打印到屏幕上。但如果张量太大，可以通过检查其形状来更容易地显示其维度：

```py
a = torch.zeros(2, 3, 4)
print(a.shape)
print(a.size())
```

它将打印出：

```py
torch.Size([2, 3, 4])
torch.Size([2, 3, 4])
```

张量的形状可以通过 `shape` 属性或 `size()` 函数来访问。如果你想查看有多少个维度（例如，$2\times 3\times 4$ 是 3，而 $3\times 4$ 是 2），你可以读取 `ndim` 属性：

```py
print(a.ndim)
```

这将给你“3”。如果你使用 `len()` 检查张量，它只会给你第一个维度的大小，例如：

```py
a = torch.zeros(2, 3, 4)
print(len(a))
```

它将打印出：

```py
2
```

另一个你可能想了解的张量属性是其数据类型。通常，在深度学习中你使用浮点数，但有时张量应为整数（例如，在图像作为像素值中）。要检查数据类型，你可以读取 `dtype` 属性：

```py
print(a.dtype)
```

它将打印出：

```py
torch.float32
```

如果你想更改数据类型，你可以使用新的类型重新创建张量：

```py
b = a.type(torch.int32)
print(a.dtype)
print(b.dtype)
```

上面的打印结果是：

```py
torch.float32
torch.int32
```

## 操作张量

深度学习中的一个常见操作是改变张量的形状。例如，你可能想要将 2D 张量转换为 1D 或向张量添加一个虚拟维度。你也可能想从较大的张量中提取子张量。

例如，你可以创建一个如下所示的张量：

```py
a = torch.randn(3,4,5)
print(a)
```

如果你得到：

```py
tensor([[[-1.1271e-01, -7.1124e-01,  1.1335e+00, -8.5644e-01, -1.4191e+00],
         [-1.9065e+00, -6.8386e-02,  5.8727e-01,  6.5890e-03, -2.6947e-01],
         [ 6.3194e-01, -7.7426e-01,  1.6546e+00,  1.2647e-01, -1.0944e+00],
         [ 3.7819e-01, -8.8670e-01,  5.3772e-01,  1.4985e+00,  5.8396e-01]],

        [[ 1.8704e+00,  2.0914e+00, -9.1604e-01,  1.2317e+00, -1.5722e-01],
         [ 2.4689e-01, -2.3157e-01, -3.3033e-01,  1.4021e+00, -6.9540e-01],
         [ 3.0298e-01, -1.4936e-01, -6.8863e-01,  1.6977e-01,  2.4682e+00],
         [-8.1375e-01,  4.8489e-01, -1.2024e+00, -4.9771e-01,  1.1728e-01]],

        [[-1.6011e+00,  1.5686e-03, -1.0560e-01, -1.2938e+00,  5.3077e-01],
         [-9.7636e-01, -9.1854e-01, -1.0002e+00,  1.1852e+00,  1.0328e+00],
         [ 9.6664e-01,  5.3752e-01, -3.1340e-02, -6.7852e-02, -7.2317e-01],
         [-5.5263e-01,  9.4754e-01, -5.4503e-01,  6.3850e-02,  1.2879e+00]]])
```

它允许你使用与 NumPy 相同的语法进行切片：

```py
print(a[1])
```

这将是：

```py
tensor([[ 1.8704,  2.0914, -0.9160,  1.2317, -0.1572],
        [ 0.2469, -0.2316, -0.3303,  1.4021, -0.6954],
        [ 0.3030, -0.1494, -0.6886,  0.1698,  2.4682],
        [-0.8137,  0.4849, -1.2024, -0.4977,  0.1173]])
```

或者如果你使用：

```py
print(a[1:, 2:4])
```

它将是：

```py
tensor([[[ 0.3030, -0.1494, -0.6886,  0.1698,  2.4682],
         [-0.8137,  0.4849, -1.2024, -0.4977,  0.1173]],

        [[ 0.9666,  0.5375, -0.0313, -0.0679, -0.7232],
         [-0.5526,  0.9475, -0.5450,  0.0638,  1.2879]]])
```

你还可以利用相同的切片语法来 **添加** 一个新维度。例如，

```py
print(a[:, None, :, None].shape)
```

你将看到：

```py
torch.Size([3, 1, 4, 1, 5])
```

在这里，你使用 `None` 在特定位置插入一个新维度。如果你需要将图像转换为只有一张图像的批次时，这很有用。如果你熟悉 NumPy，你可能会记得有一个 `expand_dims()` 函数用于此目的，但 PyTorch 没有提供。一个类似的函数是 `unsqueeze()`，如下所示：

```py
b = torch.unsqueeze(a, dim=2)
print(a.shape)
print(b.shape)
```

这将打印出：

```py
torch.Size([3, 4, 5])
torch.Size([3, 4, 1, 5])
```

NumPy 切片语法的一个强大特性是布尔索引。这在 PyTorch 张量中也受支持。例如：

```py
a = torch.randn(3,4)
print(a)
print(a[:, (a > -1).all(axis=0)])
```

你可能会看到：

```py
tensor([[ 1.2548,  0.4078,  0.5548, -0.7016],
        [-0.3720, -0.5682, -0.3430,  0.0886],
        [ 0.2151,  0.3626, -2.0275,  1.8121]])
tensor([[ 1.2548,  0.4078, -0.7016],
        [-0.3720, -0.5682,  0.0886],
        [ 0.2151,  0.3626,  1.8121]])
```

上面的代码选择了所有元素都大于 -1 的列。你也可以通过选择特定的列来操作张量：

```py
print(a[:, [1,0,0,1]])
```

这将导致：

```py
tensor([[ 0.4078,  1.2548,  1.2548,  0.4078],
        [-0.5682, -0.3720, -0.3720, -0.5682],
        [ 0.3626,  0.2151,  0.2151,  0.3626]])
```

要将 2D 张量转换为 1D，你可以使用：

```py
a = torch.randn(3,4)
print(a)
print(a.ravel())
```

结果将是：

```py
tensor([[-0.2718, -0.8309,  0.6263, -0.2499],
        [-0.1780,  1.1735, -1.3530, -1.2374],
        [-0.6050, -1.5524, -0.1008, -1.2782]])
tensor([-0.2718, -0.8309,  0.6263, -0.2499, -0.1780,  1.1735, -1.3530, -1.2374,
        -0.6050, -1.5524, -0.1008, -1.2782])
```

你也可以使用 `reshape()` 函数来实现相同的效果：

```py
print(a.reshape(-1))
```

结果应该与 `ravel()` 相同。但通常，`reshape()` 函数用于更复杂的目标形状：

```py
print(a.reshape(3,2,2))
```

这将打印出：

```py
tensor([[[-0.2718, -0.8309],
         [ 0.6263, -0.2499]],

        [[-0.1780,  1.1735],
         [-1.3530, -1.2374]],

        [[-0.6050, -1.5524],
         [-0.1008, -1.2782]]])
```

张量重塑的一个常见情况是矩阵转置。对于 2D 矩阵，它可以像 NumPy 一样轻松完成：

```py
print(a.T)
```

这将打印出：

```py
tensor([[-0.2718, -0.1780, -0.6050],
        [-0.8309,  1.1735, -1.5524],
        [ 0.6263, -1.3530, -0.1008],
        [-0.2499, -1.2374, -1.2782]])
```

但 PyTorch 中的 `transpose()` 函数要求你显式指定要交换的轴：

```py
print(a.transpose(0, 1))
```

这个结果与上面相同。如果你有多个张量，可以通过堆叠它们来组合（`vstack()` 用于垂直堆叠，`hstack()` 用于水平堆叠）。例如：

```py
a = torch.randn(3,3)
b = torch.randn(3,3)
print(a)
print(b)
print(torch.vstack([a,b]))
```

这可能会打印：

```py
tensor([[ 1.1739,  1.3546, -0.2886],
        [ 1.0444,  0.4437, -2.7933],
        [ 0.6805,  0.8401, -1.2527]])
tensor([[ 1.6273,  1.2622, -0.4362],
        [-1.6529,  0.6457, -0.1454],
        [-2.0960, -1.3024, -0.1033]])
tensor([[ 1.1739,  1.3546, -0.2886],
        [ 1.0444,  0.4437, -2.7933],
        [ 0.6805,  0.8401, -1.2527],
        [ 1.6273,  1.2622, -0.4362],
        [-1.6529,  0.6457, -0.1454],
        [-2.0960, -1.3024, -0.1033]])
```

连接函数类似：

```py
c = torch.concatenate([a, b])
print(c)
```

你将得到相同的张量：

```py
tensor([[ 1.1739,  1.3546, -0.2886],
        [ 1.0444,  0.4437, -2.7933],
        [ 0.6805,  0.8401, -1.2527],
        [ 1.6273,  1.2622, -0.4362],
        [-1.6529,  0.6457, -0.1454],
        [-2.0960, -1.3024, -0.1033]])
```

反向操作是分割，例如，

```py
print(torch.vsplit(c, 2))
```

它会打印

```py
(tensor([[ 1.1739,  1.3546, -0.2886],
        [ 1.0444,  0.4437, -2.7933],
        [ 0.6805,  0.8401, -1.2527]]), tensor([[ 1.6273,  1.2622, -0.4362],
        [-1.6529,  0.6457, -0.1454],
        [-2.0960, -1.3024, -0.1033]]))
```

这个函数告诉你要将张量分割成多少个，而不是每个张量的大小。后者在深度学习中确实更有用（例如，将一个大数据集的张量分割成许多小批量的张量）。等效的函数是：

```py
print(torch.split(c, 3, dim=0))
```

这应与你之前得到的结果相同。所以 `split(c, 3, dim=0)` 意味着在维度 0 上分割，使得每个结果张量的大小为 3。

## 张量函数

PyTorch 张量可以被视为数组。因此，你通常可以像使用 NumPy 数组一样使用它。例如，你可以使用常见数学函数的函数：

```py
a = torch.randn(2,3)
print(a)
print(torch.exp(a))
print(torch.log(a))
print(torch.sin(a))
print(torch.arctan(a))
print(torch.abs(a))
print(torch.square(a))
print(torch.sqrt(a))
print(torch.ceil(a))
print(torch.round(a))
print(torch.clip(a, 0.1, 0.9))
```

这会打印：

```py
tensor([[ 1.0567, -1.2609, -1.0856],
        [-0.9633,  1.3163, -0.4325]])
tensor([[2.8770, 0.2834, 0.3377],
        [0.3816, 3.7298, 0.6489]])
tensor([[0.0552,    nan,    nan],
        [   nan, 0.2749,    nan]])
tensor([[ 0.8708, -0.9524, -0.8846],
        [-0.8211,  0.9678, -0.4191]])
tensor([[ 0.8130, -0.9003, -0.8264],
        [-0.7667,  0.9211, -0.4082]])
tensor([[1.0567, 1.2609, 1.0856],
        [0.9633, 1.3163, 0.4325]])
tensor([[1.1167, 1.5898, 1.1785],
        [0.9280, 1.7328, 0.1871]])
tensor([[1.0280,    nan,    nan],
        [   nan, 1.1473,    nan]])
tensor([[ 2., -1., -1.],
        [-0.,  2., -0.]])
tensor([[ 1., -1., -1.],
        [-1.,  1., -0.]])
tensor([[0.9000, 0.1000, 0.1000],
        [0.1000, 0.9000, 0.1000]])
```

注意，如果函数未定义（例如，负数的平方根），`nan` 将是结果，但不会引发异常。在 PyTorch 中，你可以使用一个函数来检查张量的值是否为 `nan`：

```py
b = torch.sqrt(a)
print(b)
print(torch.isnan(b))
```

你将得到：

```py
tensor([[1.0280,    nan,    nan],
        [   nan, 1.1473,    nan]])
tensor([[False,  True,  True],
        [ True, False,  True]])
```

确实，除了这些定义的函数，Python 运算符也可以应用于张量：

```py
a = torch.randn(2, 3)
b = torch.randn(2, 3)
print(a)
print(b)
print(a+b)
print(a/b)
print(a ** 2)
```

你得到：

```py
tensor([[ 0.7378, -0.3469,  1.3089],
        [-1.9152,  0.3745, -0.7248]])
tensor([[-0.3650, -0.4768,  0.9331],
        [ 0.5095,  1.7169, -0.5463]])
tensor([[ 0.3729, -0.8237,  2.2421],
        [-1.4058,  2.0914, -1.2711]])
tensor([[-2.0216,  0.7275,  1.4027],
        [-3.7594,  0.2181,  1.3269]])
tensor([[0.5444, 0.1203, 1.7133],
        [3.6682, 0.1403, 0.5254]])
```

但在运算符中，矩阵乘法在深度学习中非常重要。你可以使用以下方法实现：

```py
print(torch.matmul(a, b.T))
print(a @ b.T)
```

这会打印

```py
tensor([[ 1.1176, -0.9347],
        [-0.1560,  0.0632]])
tensor([[ 1.1176, -0.9347],
        [-0.1560,  0.0632]])
```

这两者是相同的。实际上，Python 的 `@` 运算符也可以用于向量点积，例如：

```py
a = torch.randn(3)
b = torch.randn(3)
print(a)
print(b)
print(torch.dot(a, b))
print(a @ b)
```

它会打印：

```py
tensor([-0.8986, -0.6994,  1.1443])
tensor([-1.0666,  0.1455,  0.1322])
tensor(1.0081)
tensor(1.0081)
```

如果你把张量中的值视为样本，你可能还想找出一些关于它的统计数据。PyTorch 也提供了一些：

```py
a = torch.randn(3,4)
print(a)
print(torch.mean(a, dim=0))
print(torch.std(a, dim=0))
print(torch.cumsum(a, dim=0))
print(torch.cumprod(a, dim=0))
```

它会打印：

```py
tensor([[ 0.3331, -0.0190,  0.4814, -1.1484],
        [-0.5712,  0.8430, -1.6147, -1.1664],
        [ 1.7298, -1.7665, -0.5918,  0.3024]])
tensor([ 0.4972, -0.3142, -0.5750, -0.6708])
tensor([1.1593, 1.3295, 1.0482, 0.8429])
tensor([[ 0.3331, -0.0190,  0.4814, -1.1484],
        [-0.2381,  0.8240, -1.1333, -2.3148],
        [ 1.4917, -0.9425, -1.7251, -2.0124]])
tensor([[ 0.3331, -0.0190,  0.4814, -1.1484],
        [-0.1903, -0.0160, -0.7774,  1.3395],
        [-0.3291,  0.0283,  0.4601,  0.4051]])
```

但对于线性代数函数，你应在 PyTorch 的 linalg 子模块中找到它。例如：

```py
print(torch.linalg.svd(a))
```

你将看到：

```py
torch.return_types.linalg_svd(
U=tensor([[-0.0353,  0.1313,  0.9907],
        [-0.5576,  0.8201, -0.1286],
        [ 0.8294,  0.5569, -0.0443]]),
S=tensor([2.7956, 1.9465, 1.2715]),
Vh=tensor([[ 0.6229, -0.6919,  0.1404,  0.3369],
        [ 0.2767, -0.1515, -0.8172, -0.4824],
        [ 0.2570, -0.0385,  0.5590, -0.7874],
        [ 0.6851,  0.7048, -0.0073,  0.1840]]))
```

特别是对于卷积神经网络，填充张量可以使用以下方法：

```py
b = torch.nn.functional.pad(a, (1,1,0,2), value=0)
print(b)
```

这会打印：

```py
tensor([[ 0.0000,  0.3331, -0.0190,  0.4814, -1.1484,  0.0000],
        [ 0.0000, -0.5712,  0.8430, -1.6147, -1.1664,  0.0000],
        [ 0.0000,  1.7298, -1.7665, -0.5918,  0.3024,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]])
```

这个 `pad()` 函数的例子是创建维度 0 上的 (1,1) 填充和维度 1 上的 (0,2) 填充。换句话说，对于每个维度 0（行），我们在开始和结束各添加一个虚拟值（0）。对于每个维度 1（列），我们在开始处添加零个虚拟值，但在结束处添加两个虚拟值。

最后，由于 PyTorch 张量可以被视为数组，你可以直接将它们与其他工具如 matplotlib 一起使用。下面是一个使用 PyTorch 张量绘制表面的例子：

```py
import matplotlib.pyplot as plt
import torch

# create tensors
x = torch.linspace(-1, 1, 100)
y = torch.linspace(-2, 2, 100)
# create the surface
xx, yy = torch.meshgrid(x, y, indexing="xy")  # xy-indexing is matching numpy
z = torch.sqrt(1 - xx**2 - (yy/2)**2)
print(xx)

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection="3d")
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([0, 2])
ax.plot_surface(xx, yy, z, cmap="cividis")
ax.view_init(45, 35)
plt.show()
```

网格生成器产生的 `xx` 张量为：

```py
tensor([[-1.0000, -0.9798, -0.9596,  ...,  0.9596,  0.9798,  1.0000],
        [-1.0000, -0.9798, -0.9596,  ...,  0.9596,  0.9798,  1.0000],
        [-1.0000, -0.9798, -0.9596,  ...,  0.9596,  0.9798,  1.0000],
        ...,
        [-1.0000, -0.9798, -0.9596,  ...,  0.9596,  0.9798,  1.0000],
        [-1.0000, -0.9798, -0.9596,  ...,  0.9596,  0.9798,  1.0000],
        [-1.0000, -0.9798, -0.9596,  ...,  0.9596,  0.9798,  1.0000]])
```

创建的图像是：![](img/1598946c006f3d6132863705446d5e93.png)

## 总结

在本教程中，你发现了如何操作 PyTorch 张量。具体来说，你学到了：

+   什么是张量

+   如何在 PyTorch 中创建各种类型的张量

+   如何在 PyTorch 中重新形状、切片和操作张量

+   可以应用于 PyTorch 张量的常见函数