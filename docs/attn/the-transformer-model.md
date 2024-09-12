# Transformer 模型

> 原文：[`machinelearningmastery.com/the-transformer-model/`](https://machinelearningmastery.com/the-transformer-model/)

我们已经熟悉了由 Transformer 注意力机制实现的自注意力概念，用于神经机器翻译。现在我们将把焦点转移到 Transformer 架构的细节上，以探索如何在不依赖于递归和卷积的情况下实现自注意力。

在本教程中，您将了解 Transformer 模型的网络架构。

完成本教程后，您将了解：

+   Transformer 架构如何实现编码器-解码器结构而不依赖于递归和卷积

+   Transformer 编码器和解码器的工作原理

+   Transformer 自注意力与使用递归和卷积层的比较

**用我的书[Building Transformer Models with Attention](https://machinelearningmastery.com/transformer-models-with-attention/)来启动您的项目**。它提供了具有**工作代码**的**自学教程**，指导您构建一个完全工作的 Transformer 模型，能够

*将一种语言的句子翻译成另一种语言*...

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/10/transformer_cover-1-scaled.jpg)

Transformer 模型

照片由[Samule Sun](https://unsplash.com/photos/vuMTQj6aQQ0)拍摄，部分权利保留。

## **教程概述**

本教程分为三个部分；它们是：

+   Transformer 架构

    +   编码器

    +   解码器

+   总结：Transformer 模型

+   与递归和卷积层的比较

## **先决条件**

对于本教程，我们假设您已经熟悉：

+   [注意力机制的概念](https://machinelearningmastery.com/what-is-attention/)

+   [注意力机制](https://machinelearningmastery.com/the-attention-mechanism-from-scratch/)

+   [Transformer 注意力机制](https://machinelearningmastery.com/the-transformer-attention-mechanism)

## **Transformer 架构**

Transformer 架构遵循编码器-解码器结构，但不依赖于递归和卷积以生成输出。

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png)

Transformer 架构的编码器-解码器结构

摘自“[Attention Is All You Need](https://arxiv.org/abs/1706.03762)”

简而言之，Transformer 架构左半部分的编码器的任务是将输入序列映射到一系列连续的表示，然后输入到解码器中。

解码器位于架构的右半部分，接收来自编码器的输出以及前一个时间步的解码器输出，生成一个输出序列。

> *在每一步中，模型都是自回归的，生成下一个符号时会消耗先前生成的符号作为额外的输入。*
> 
> *–* [注意力机制](https://arxiv.org/abs/1706.03762)，2017 年。

### **编码器**

![](https://machinelearningmastery.com/wp-content/uploads/2021/10/transformer_1.png)

Transformer 架构的编码器块

取自“[注意力机制](https://arxiv.org/abs/1706.03762)“

编码器由 $N$ = 6 个相同的层组成，每个层由两个子层组成：

1.  第一个子层实现了多头自注意力机制。[你已经看到](https://machinelearningmastery.com/the-transformer-attention-mechanism) 多头机制实现了 $h$ 个头，每个头接收查询、键和值的（不同的）线性投影版本，每个头并行生成 $h$ 个输出，然后用于生成最终结果。

1.  第二个子层是一个全连接的前馈网络，由两个线性变换组成，中间有 ReLU 激活：

$$\text{FFN}(x) = \text{ReLU}(\mathbf{W}_1 x + b_1) \mathbf{W}_2 + b_2$$

Transformer 编码器的六层将相同的线性变换应用于输入序列中的所有单词，但*每*层使用不同的权重（$\mathbf{W}_1, \mathbf{W}_2$）和偏置（$b_1, b_2$）参数来实现。

此外，这两个子层都有绕它们的残差连接。

每个子层之后还跟着一个标准化层，$\text{layernorm}(.)$，它对子层输入 $x$ 和子层生成的输出 $\text{sublayer}(x)$ 之间计算的和进行归一化：

$$\text{layernorm}(x + \text{sublayer}(x))$$

需要注意的一点是，Transformer 架构本质上不能捕获序列中单词之间的相对位置信息，因为它不使用递归。此信息必须通过引入*位置编码*到输入嵌入中来注入。

位置编码向量的维度与输入嵌入相同，使用不同频率的正弦和余弦函数生成。然后，它们简单地与输入嵌入求和，以*注入*位置信息。

### **解码器**

![](https://machinelearningmastery.com/wp-content/uploads/2021/10/transformer_2.png)

Transformer 架构的解码器块

取自“[注意力机制](https://arxiv.org/abs/1706.03762)“

解码器与编码器有几个相似之处。

解码器也由 $N$ = 6 个相同的层组成，每层包含三个子层：

1.  第一个子层接收解码器堆栈的先前输出，用位置信息增强，并在其上实现多头自注意力。虽然编码器被设计为无论输入序列中单词的位置如何都能关注，解码器修改为只关注前面的单词。因此，在多头注意力机制中（并行实现多个单注意力函数），通过引入一个掩码来阻止由缩放矩阵$\mathbf{Q}$和$\mathbf{K}$乘法产生的值。这种屏蔽通过抑制矩阵值来实现，否则这些值将对应于非法连接：

$$

\text{mask}(\mathbf{QK}^T) =

\text{mask} \left( \begin{bmatrix}

e_{11} & e_{12} & \dots & e_{1n} \\

e_{21} & e_{22} & \dots & e_{2n} \\

\vdots & \vdots & \ddots & \vdots \\

e_{m1} & e_{m2} & \dots & e_{mn} \\

\end{bmatrix} \right) =

\begin{bmatrix}

e_{11} & -\infty & \dots & -\infty \\

e_{21} & e_{22} & \dots & -\infty \\

\vdots & \vdots & \ddots & \vdots \\

e_{m1} & e_{m2} & \dots & e_{mn} \\

\end{bmatrix}

$$

![](https://machinelearningmastery.com/wp-content/uploads/2021/09/tour_3.png)

在解码器中的多头注意力机制实现了几个掩码的单注意力功能。

取自“[注意力机制是你所需要的](https://arxiv.org/abs/1706.03762)”

> *屏蔽使得解码器单向（不像双向编码器）。*
> 
> *–* [Python 深度学习进阶](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X)，2019 年。

1.  第二层实现了一种类似于编码器第一子层中实现的多头自注意力机制。在解码器侧，这个多头机制接收来自前一个解码器子层的查询，并从编码器的输出中获取键和值。这使得解码器能够关注输入序列中的所有单词。

1.  第三层实现一个全连接的前馈网络，类似于编码器第二子层中实现的网络。

此外，解码器侧的三个子层周围还有残差连接，并且后接一个标准化层。

位置编码也以与编码器相同的方式添加到解码器的输入嵌入中。

### 想要开始构建带有注意力的 Transformer 模型吗？

现在获取我的免费 12 天电子邮件速成课程（附带示例代码）。

点击注册并获得免费的课程 PDF 电子书版本。

## **总结：Transformer 模型**

Transformer 模型的运行如下：

1.  形成输入序列的每个单词都转换为一个$d_{\text{model}}$维嵌入向量。

1.  每个表示输入词的嵌入向量通过与相同 $d_{\text{model}}$ 长度的位置信息向量逐元素相加，从而将位置信息引入输入。

1.  增强的嵌入向量被输入到包含上述两个子层的编码器块中。由于编码器会关注输入序列中的所有词，无论这些词是否在当前考虑的词之前或之后，因此 Transformer 编码器是*双向的*。

1.  解码器在时间步 $t – 1$ 收到其自身预测的输出词作为输入。

1.  解码器的输入也通过与编码器侧相同的方式进行位置编码增强。

1.  增强的解码器输入被输入到包含上述三个子层的解码器块中。掩蔽被应用于第一个子层，以防止解码器关注后续词。在第二个子层，解码器还接收到编码器的输出，这使得解码器能够关注输入序列中的所有词。

1.  解码器的输出最终经过一个全连接层，然后是一个 softmax 层，以生成对输出序列下一个词的预测。

## **与递归层和卷积层的比较**

[Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) 解释了他们放弃使用递归和卷积的动机是基于几个因素：

1.  自注意力层在处理较短序列长度时比递归层更快，并且对于非常长的序列长度，可以限制只考虑输入序列中的一个邻域。

1.  递归层所需的序列操作数是基于序列长度的，而自注意力层的这个数字保持不变。

1.  在卷积神经网络中，卷积核的宽度直接影响输入和输出位置对之间可以建立的长期依赖关系。追踪长期依赖关系需要使用大卷积核或卷积层堆栈，这可能会增加计算成本。

## **进一步阅读**

如果你希望深入了解这个话题，本节提供了更多资源。

### **书籍**

+   [Advanced Deep Learning with Python](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X), 2019。

### **论文**

+   [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017。

## **总结**

在本教程中，你了解了 Transformer 模型的网络架构。

具体来说，你学到了：

+   Transformer 架构如何在没有递归和卷积的情况下实现编码器-解码器结构

+   Transformer 编码器和解码器如何工作

+   Transformer 自注意力与递归层和卷积层的比较

你有任何问题吗？

在下方评论中提出你的问题，我会尽力回答。
