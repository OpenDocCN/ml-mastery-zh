# 变压器模型位置编码的温和介绍，第一部分

> 原文：[`machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/`](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/)

在语言中，单词的顺序及其在句子中的位置确实很重要。如果重新排列单词，整个句子的意义可能会发生变化。在实现自然语言处理解决方案时，递归神经网络有一个内置机制来处理序列的顺序。然而，变压器模型不使用递归或卷积，将每个数据点视为彼此独立。因此，模型中明确添加了位置编码，以保留句子中单词的顺序信息。位置编码是一种保持序列中对象顺序知识的方案。

在本教程中，我们将简化 Vaswani 等人那篇卓越论文中使用的符号，[Attention Is All You Need](https://arxiv.org/abs/1706.03762)。完成本教程后，你将了解：

+   什么是位置编码，为什么重要

+   变压器中的位置编码

+   使用 NumPy 在 Python 中编写并可视化位置编码矩阵

**用我的书** [Building Transformer Models with Attention](https://machinelearningmastery.com/transformer-models-with-attention/) **启动你的项目**。它提供了**自学教程**和**可运行的代码**，指导你构建一个完全可运行的变压器模型

*将句子从一种语言翻译成另一种语言*…

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2022/01/muhammad-murtaza-ghani-CIVbJZR8aAk-unsplash-scaled.jpg)

变压器模型中位置编码的温和介绍

照片由 [Muhammad Murtaza Ghani](https://unsplash.com/@murtaza327?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 提供，来自 [Unsplash](https://unsplash.com/s/photos/free-pakistan?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)，部分权利保留

## 教程概述

本教程分为四个部分，它们是：

1.  什么是位置编码

1.  变压器中位置编码背后的数学

1.  使用 NumPy 实现位置编码矩阵

1.  理解并可视化位置编码矩阵

## 什么是位置编码？

位置编码描述了实体在序列中的位置或位置，以便每个位置分配一个唯一的表示。许多原因导致在变压器模型中不使用单一数字（如索引值）来表示项的位置。对于长序列，索引可能会变得非常大。如果将索引值归一化到 0 和 1 之间，则可能会对变长序列造成问题，因为它们会被不同地归一化。

Transformers 使用一种智能的位置编码方案，其中每个位置/索引映射到一个向量。因此，位置编码层的输出是一个矩阵，其中矩阵的每一行表示序列中编码对象与其位置信息的和。下图展示了仅编码位置信息的矩阵示例。

![](https://machinelearningmastery.com/wp-content/uploads/2022/01/PE1.png)

## 三角函数正弦函数的快速回顾

这是对正弦函数的快速回顾；你也可以用余弦函数进行等效操作。该函数的范围是 [-1,+1]。该波形的频率是每秒完成的周期数。波长是波形重复自身的距离。不同波形的波长和频率如下所示：

![](https://machinelearningmastery.com/wp-content/uploads/2022/01/PE2.png)

### 想开始构建具有注意力机制的 Transformer 模型吗？

立即参加我的免费 12 天邮件速成课程（包括示例代码）。

点击注册，并且还可以获得课程的免费 PDF 电子书版本。

## Transformer 中的位置信息编码层

让我们直接进入正题。假设你有一个长度为 $L$ 的输入序列，并且需要该序列中第 $k^{th}$ 对象的位置。位置编码由具有不同频率的正弦和余弦函数给出：

\begin{eqnarray}

P(k, 2i) &=& \sin\Big(\frac{k}{n^{2i/d}}\Big)\\

P(k, 2i+1) &=& \cos\Big(\frac{k}{n^{2i/d}}\Big)

\end{eqnarray}

这里：

$k$: 输入序列中对象的位置，$0 \leq k < L/2$

$d$: 输出嵌入空间的维度

$P(k, j)$: 用于将输入序列中的位置 $k$ 映射到位置矩阵的索引 $(k,j)$ 的位置函数

$n$: 用户定义的标量，由 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 的作者设定为 10,000。

$i$: 用于映射到列索引 $0 \leq i < d/2$，一个单独的 $i$ 值同时映射到正弦和余弦函数

在上述表达式中，你可以看到偶数位置对应于正弦函数，而奇数位置对应于余弦函数。

### 示例

为了理解上述表达式，让我们以短语 “I am a robot” 为例，设定 n=100 和 d=4。下表显示了该短语的位置信息编码矩阵。实际上，对于任何四字母短语，位置信息编码矩阵在 n=100 和 d=4 的情况下都是相同的。

![](https://machinelearningmastery.com/wp-content/uploads/2022/01/PE3.png)

## 从头开始编码位置编码矩阵

这里是一个简短的 Python 代码示例，用于使用 NumPy 实现位置编码。代码经过简化，以便更容易理解位置编码。

Python

```py
import numpy as np
import matplotlib.pyplot as plt

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

P = getPositionEncoding(seq_len=4, d=4, n=100)
print(P)
```

输出

```py
[[ 0\.          1\.          0\.          1\.        ]
 [ 0.84147098  0.54030231  0.09983342  0.99500417]
 [ 0.90929743 -0.41614684  0.19866933  0.98006658]
 [ 0.14112001 -0.9899925   0.29552021  0.95533649]]
```

## 理解位置编码矩阵

为了理解位置编码，让我们先来看不同位置的正弦波，n=10,000 和 d=512。Python

```py
def plotSinusoid(k, d=512, n=10000):
    x = np.arange(0, 100, 1)
    denominator = np.power(n, 2*x/d)
    y = np.sin(k/denominator)
    plt.plot(x, y)
    plt.title('k = ' + str(k))

fig = plt.figure(figsize=(15, 4))    
for i in range(4):
    plt.subplot(141 + i)
    plotSinusoid(i*4)
```

下图是上述代码的输出：![](https://machinelearningmastery.com/wp-content/uploads/2022/01/PE4.png)

不同位置索引的正弦波

你可以看到每个位置 $k$ 对应一个不同的正弦波，它将单个位置编码成一个向量。如果你仔细查看位置编码函数，你会发现固定的 $i$ 的波长由下式给出：

$$

\lambda_{i} = 2 \pi n^{2i/d}

$$

因此，正弦波的波长形成了几何级数，并从 $2\pi$ 变化到 $2\pi n$。位置编码方案有许多优点。

1.  正弦和余弦函数的值在 [-1, 1] 范围内，这保持了位置编码矩阵值在规范化范围内。

1.  由于每个位置的正弦波不同，你有一种唯一的方式来编码每个位置。

1.  你有一种方法来衡量或量化不同位置之间的相似性，从而使你能够编码单词的相对位置。

## 可视化位置矩阵

让我们在更大的数值上可视化位置矩阵。使用 Python 的 `matshow()` 方法，来自 `matplotlib` 库。将 n=10,000 设置为原始论文中的值，你会得到如下结果：

Python

```py
P = getPositionEncoding(seq_len=100, d=512, n=10000)
cax = plt.matshow(P)
plt.gcf().colorbar(cax)
```

![](https://machinelearningmastery.com/wp-content/uploads/2022/01/PE5.png)

对于 n=10,000, d=512, 序列长度=100 的位置编码矩阵

## 位置编码层的最终输出是什么？

位置编码层将位置向量与单词编码相加，并输出该矩阵以供后续层使用。整个过程如下所示。

![](https://machinelearningmastery.com/wp-content/uploads/2022/01/PE6.png)

Transformer 中的位置编码层

## 进一步阅读

本节提供了更多关于该主题的资源，如果你希望深入了解。

### 书籍

+   [用于自然语言处理的 Transformers](https://www.amazon.com/Transformers-Natural-Language-Processing-architectures/dp/1800565798)，作者 Denis Rothman。

### 论文

+   [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，2017 年。

### 文章

+   [Transformer 注意力机制](https://machinelearningmastery.com/the-transformer-attention-mechanism/)

+   [Transformer 模型](https://machinelearningmastery.com/the-transformer-model/)

+   [用于语言理解的 Transformer 模型](https://www.tensorflow.org/text/tutorials/transformer)

## 总结

在本教程中，你发现了变压器中的位置编码。

具体来说，你学到了：

+   什么是位置编码，它为何需要。

+   如何使用 NumPy 在 Python 中实现位置编码

+   如何可视化位置编码矩阵

在本文中讨论的位置编码有任何问题吗？请在下面的评论中提出您的问题，我会尽力回答。
