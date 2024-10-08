# Transformer 注意力机制

> 原文：[`machinelearningmastery.com/the-transformer-attention-mechanism/`](https://machinelearningmastery.com/the-transformer-attention-mechanism/)

在引入 Transformer 模型之前，用于神经机器翻译的注意力使用 RNN-based 编码器-解码器架构实现。Transformer 模型通过摒弃循环和卷积，并仅依赖自注意力机制，彻底改变了注意力的实现方式。

在本教程中，我们首先关注 Transformer 注意力机制，随后在另一个教程中回顾 Transformer 模型。

在本教程中，您将了解神经机器翻译的 Transformer 注意力机制。

完成本教程后，您将了解到：

+   Transformer 注意力机制与其前身有何不同

+   Transformer 如何计算缩放点积注意力

+   Transformer 如何计算多头注意力

**启动您的项目**，阅读我的书 [使用注意力构建 Transformer 模型](https://machinelearningmastery.com/transformer-models-with-attention/)。它提供了带有 **工作代码** 的 **自学教程**，引导您构建一个完全可工作的 Transformer 模型，能够

*将句子从一种语言翻译成另一种语言*...

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/10/transformer_cover.jpg)

Transformer 注意力机制

照片由 [Andreas Gücklhorn](https://unsplash.com/photos/mawU2PoJWfU) 提供，某些权利保留。

## **教程概览**

本教程分为两部分；它们是：

+   介绍 Transformer 注意力机制

+   Transformer 注意力机制

    +   缩放点积注意力

    +   多头注意力

## **先决条件**

对于本教程，我们假设您已经熟悉：

+   [注意力的概念](https://machinelearningmastery.com/what-is-attention/)

+   [注意力机制](https://machinelearningmastery.com/the-attention-mechanism-from-scratch/)

+   [Bahdanau 注意力机制](https://machinelearningmastery.com/?p=12940&preview=true)

+   [Luong 注意力机制](https://machinelearningmastery.com/the-luong-attention-mechanism/)

## **介绍 Transformer 注意力机制**

到目前为止，您已经熟悉了在 RNN-based 编码器-解码器架构中使用注意力机制。其中两个最流行的模型是由 [Bahdanau et al. (2014)](https://arxiv.org/abs/1409.0473) 和 [Luong et al. (2015)](https://arxiv.org/abs/1508.04025) 提出的。

Transformer 架构通过摒弃依赖于循环和卷积的方式，彻底改变了注意力的使用。

> *… 变压器是第一个完全依赖自注意力计算输入和输出表示的转导模型，而无需使用序列对齐的 RNN 或卷积。*
> 
> *–* [注意力机制全靠它](https://arxiv.org/abs/1706.03762)，2017。

在他们的论文《注意力机制全靠它》中，[Vaswani 等人 (2017)](https://arxiv.org/abs/1706.03762) 解释了变压器模型如何完全依赖于自注意力机制，其中序列（或句子）的表示是通过关联同一序列中的不同单词来计算的。

> *自注意力，有时称为内注意力，是一种注意力机制，通过关联单个序列的不同位置来计算该序列的表示。*
> 
> *–* [注意力机制全靠它](https://arxiv.org/abs/1706.03762)，2017。

**变压器注意力机制**

变压器注意力机制使用的主要组件如下：

+   $\mathbf{q}$ 和 $\mathbf{k}$ 分别表示维度为 $d_k$ 的查询和键向量

+   $\mathbf{v}$ 表示维度为 $d_v$ 的值向量

+   $\mathbf{Q}$、$\mathbf{K}$ 和 $\mathbf{V}$ 分别表示打包在一起的查询、键和值的矩阵。

+   $\mathbf{W}^Q$、$\mathbf{W}^K$ 和 $\mathbf{W}^V$ 分别表示用于生成查询、键和值矩阵不同子空间表示的投影矩阵

+   $\mathbf{W}^O$ 表示用于多头输出的投影矩阵

实质上，注意力函数可以被视为查询与一组键值对之间的映射，得到一个输出。

> *输出作为值的加权和计算，其中每个值分配的权重由查询与相应键的兼容性函数计算得出。*
> 
> *–* [注意力机制全靠它](https://arxiv.org/abs/1706.03762)，2017。

Vaswani 等人提出了一种 *缩放点积注意力*，并在此基础上提出了 *多头注意力*。在神经机器翻译的背景下，作为这些注意力机制输入的查询、键和值是同一句输入的不同投影。

直观地说，提出的注意力机制通过捕捉同一句子中不同元素（在这种情况下是单词）之间的关系来实现自注意力。

### 想要开始构建带有注意力机制的变压器模型吗？

立即参加我的免费 12 天电子邮件速成课程（附带示例代码）。

点击以注册并获取课程的免费 PDF 电子书版本。

## **缩放点积注意力**

变压器实现了一种缩放点积注意力，这遵循了你之前见过的 [通用注意力机制](https://machinelearningmastery.com/the-attention-mechanism-from-scratch/) 的过程。

正如名称所示，缩放点积注意力首先对每个查询$\mathbf{q}$与所有键$\mathbf{k}$计算一个*点积*。随后，它将每个结果除以$\sqrt{d_k}$，并应用 softmax 函数。这样，它获得了用于*缩放*值$\mathbf{v}$的权重。

![](https://machinelearningmastery.com/wp-content/uploads/2021/09/tour_3.png)

缩放点积注意力

取自“[Attention Is All You Need](https://arxiv.org/abs/1706.03762)“

实际上，缩放点积注意力执行的计算可以高效地同时应用于整个查询集。为此，矩阵—$\mathbf{Q}$、$\mathbf{K}$和$\mathbf{V}$—作为输入提供给注意力函数：

$$\text{attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V$$

Vaswani 等人解释说，他们的缩放点积注意力与[Luong 等人（2015）](https://arxiv.org/abs/1508.04025)的乘法注意力是相同的，唯一的不同是添加了缩放因子$\tfrac{1}{\sqrt{d_k}}$。

引入这个缩放因子的目的是为了抵消当$d_k$的值很大时，点积增长幅度较大的效果，此时应用 softmax 函数会返回极小的梯度，导致著名的梯度消失问题。因此，缩放因子旨在将点积乘法生成的结果拉低，从而防止这个问题。

Vaswani 等人进一步解释说，他们选择乘法注意力而非[Bahdanau 等人（2014）](https://arxiv.org/abs/1409.0473)的加法注意力是基于前者的计算效率。

> *…点积注意力在实践中要快得多且空间效率更高，因为它可以使用高度优化的矩阵乘法代码实现。*
> 
> *–* [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.

因此，计算缩放点积注意力的逐步过程如下：

1.  通过将查询矩阵$\mathbf{Q}$中的查询集合与矩阵$\mathbf{K}$中的键相乘来计算对齐分数。如果矩阵$\mathbf{Q}$的大小为$m \times d_k$，而矩阵$\mathbf{K}$的大小为$n \times d_k$，则结果矩阵的大小将为$m \times n$：

$$

\mathbf{QK}^T =

\begin{bmatrix}

e_{11} & e_{12} & \dots & e_{1n} \\

e_{21} & e_{22} & \dots & e_{2n} \\

\vdots & \vdots & \ddots & \vdots \\

e_{m1} & e_{m2} & \dots & e_{mn} \\

\end{bmatrix}

$$

1.  将每个对齐分数缩放为$\tfrac{1}{\sqrt{d_k}}$：

$$

\frac{\mathbf{QK}^T}{\sqrt{d_k}} =

\begin{bmatrix}

\tfrac{e_{11}}{\sqrt{d_k}} & \tfrac{e_{12}}{\sqrt{d_k}} & \dots & \tfrac{e_{1n}}{\sqrt{d_k}} \\

\tfrac{e_{21}}{\sqrt{d_k}} & \tfrac{e_{22}}{\sqrt{d_k}} & \dots & \tfrac{e_{2n}}{\sqrt{d_k}} \\

\vdots & \vdots & \ddots & \vdots \\

\tfrac{e_{m1}}{\sqrt{d_k}} & \tfrac{e_{m2}}{\sqrt{d_k}} & \dots & \tfrac{e_{mn}}{\sqrt{d_k}} \\

\end{bmatrix}

$$

1.  然后通过应用 softmax 操作来进行缩放过程，以获得一组权重：

$$

\text{softmax} \left( \frac{\mathbf{QK}^T}{\sqrt{d_k}} \right) =

\begin{bmatrix}

\text{softmax} ( \tfrac{e_{11}}{\sqrt{d_k}} & \tfrac{e_{12}}{\sqrt{d_k}} & \dots & \tfrac{e_{1n}}{\sqrt{d_k}} ) \\

\text{softmax} ( \tfrac{e_{21}}{\sqrt{d_k}} & \tfrac{e_{22}}{\sqrt{d_k}} & \dots & \tfrac{e_{2n}}{\sqrt{d_k}} ) \\

\vdots & \vdots & \ddots & \vdots \\

\text{softmax} ( \tfrac{e_{m1}}{\sqrt{d_k}} & \tfrac{e_{m2}}{\sqrt{d_k}} & \dots & \tfrac{e_{mn}}{\sqrt{d_k}} ) \\

\end{bmatrix}

$$

1.  最后，将生成的权重应用于矩阵 $\mathbf{V}$ 中的值，大小为 $n \times d_v$：

$$

\begin{aligned}

& \text{softmax} \left( \frac{\mathbf{QK}^T}{\sqrt{d_k}} \right) \cdot \mathbf{V} \\

=&

\begin{bmatrix}

\text{softmax} ( \tfrac{e_{11}}{\sqrt{d_k}} & \tfrac{e_{12}}{\sqrt{d_k}} & \dots & \tfrac{e_{1n}}{\sqrt{d_k}} ) \\

\text{softmax} ( \tfrac{e_{21}}{\sqrt{d_k}} & \tfrac{e_{22}}{\sqrt{d_k}} & \dots & \tfrac{e_{2n}}{\sqrt{d_k}} ) \\

\vdots & \vdots & \ddots & \vdots \\

\text{softmax} ( \tfrac{e_{m1}}{\sqrt{d_k}} & \tfrac{e_{m2}}{\sqrt{d_k}} & \dots & \tfrac{e_{mn}}{\sqrt{d_k}} ) \\

\end{bmatrix}

\cdot

\begin{bmatrix}

v_{11} & v_{12} & \dots & v_{1d_v} \\

v_{21} & v_{22} & \dots & v_{2d_v} \\

\vdots & \vdots & \ddots & \vdots \\

v_{n1} & v_{n2} & \dots & v_{nd_v} \\

\end{bmatrix}

\end{aligned}

$$

## **多头注意力**

在其单个注意力函数基础上，接下来构建了一个多头注意力机制，该函数以矩阵 $\mathbf{Q}$、$\mathbf{K}$ 和 $\mathbf{V}$ 作为输入，正如您刚刚审查的那样，Vaswani 等人还提出了一个多头注意力机制。

多头注意力机制通过$h$次线性投影来处理查询、键和值，每次使用不同的学习投影。然后，单个注意力机制并行应用于这$h$个投影中的每一个，以产生$h$个输出，然后这些输出被串联并再次投影以产生最终结果。

![](https://machinelearningmastery.com/wp-content/uploads/2021/09/tour_4.png)

多头注意力

取自“[Attention Is All You Need](https://arxiv.org/abs/1706.03762)“

多头注意力的理念是允许注意力函数从不同的表示子空间中提取信息，这在单个注意力头中是不可能的。

多头注意力功能可以表示如下：

$$\text{multihead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}^O$$

在这里，每个 $\text{head}_i$，$i = 1, \dots, h$，实现了一个由自己的学习投影矩阵特征化的单一注意力函数：

$$\text{head}_i = \text{attention}(\mathbf{QW}^Q_i, \mathbf{KW}^K_i, \mathbf{VW}^V_i)$$

计算多头注意力的逐步过程如下：

1.  通过与各自的权重矩阵$\mathbf{W}^Q_i$、$\mathbf{W}^K_i$和$\mathbf{W}^V_i$相乘，计算查询、键和值的线性投影版本，每个$\text{head}_i$一个。

1.  对每个头应用单一的注意力函数，步骤包括（1）乘以查询和键矩阵，（2）应用缩放和 softmax 操作，以及（3）加权值矩阵以生成每个头的输出。

1.  连接头的输出，$\text{head}_i$，$i = 1, \dots, h$。

1.  通过与权重矩阵$\mathbf{W}^O$相乘，将连接的输出进行线性投影，以生成最终结果。

## **进一步阅读**

本节提供了更多关于该主题的资源，供您深入了解。

### **书籍**

+   [《深入学习 Python》](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X)，2019 年。

### **论文**

+   [《Attention Is All You Need》](https://arxiv.org/abs/1706.03762)，2017 年。

+   [《通过联合学习对齐和翻译的神经机器翻译》](https://arxiv.org/abs/1409.0473)，2014 年。

+   [《基于注意力的神经机器翻译的有效方法》](https://arxiv.org/abs/1508.04025)，2015 年。

## **总结**

在本教程中，你发现了用于神经机器翻译的 Transformer 注意力机制。

具体来说，你学到了：

+   Transformer 注意力与其前身的区别。

+   Transformer 如何计算缩放点积注意力。

+   Transformer 如何计算多头注意力。

你有任何问题吗？

在下方评论中提出你的问题，我会尽力回答。
