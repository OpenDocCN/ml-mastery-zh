# Bahdanau 注意力机制

> 原文：[`machinelearningmastery.com/the-bahdanau-attention-mechanism/`](https://machinelearningmastery.com/the-bahdanau-attention-mechanism/)

用于机器翻译的传统编码器-解码器架构将每个源句编码成一个固定长度的向量，无论其长度如何，解码器将生成一个翻译。这使得神经网络难以处理长句子，实质上导致了性能瓶颈。

Bahdanau 注意力被提出来解决传统编码器-解码器架构的性能瓶颈，相较于传统方法实现了显著改进。

在本教程中，您将了解神经机器翻译的 Bahdanau 注意力机制。

完成本教程后，您将了解：

+   Bahdanau 注意力机制名称的来源及其解决的挑战

+   形成 Bahdanau 编码器-解码器架构的不同组成部分的角色

+   Bahdanau 注意力算法执行的操作

**启动您的项目**，使用我的书籍[使用注意力构建 Transformer 模型](https://machinelearningmastery.com/transformer-models-with-attention/)。它提供了**自学教程**和**工作代码**，指导您构建一个完全工作的 Transformer 模型，可以

*将一种语言的句子翻译成另一种语言*...

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/09/bahdanau_cover-scaled.jpg)

Bahdanau 注意力机制

图片由[Sean Oulashin](https://unsplash.com/photos/KMn4VEeEPR8)拍摄，部分权利保留。

## **教程概览**

本教程分为两部分；它们是：

+   介绍 Bahdanau 注意力

+   Bahdanau 架构

    +   编码器

    +   解码器

    +   Bahdanau 注意力算法

## **先决条件**

对于本教程，我们假设您已经熟悉：

+   [递归神经网络（RNNs）](https://machinelearningmastery.com/an-introduction-to-recurrent-neural-networks-and-the-math-that-powers-them/)

+   [编码器-解码器 RNN 架构](https://machinelearningmastery.com/encoder-decoder-recurrent-neural-network-models-neural-machine-translation/)

+   [注意力的概念](https://machinelearningmastery.com/what-is-attention/)

+   [注意力机制](https://machinelearningmastery.com/the-attention-mechanism-from-scratch/)

## **介绍 Bahdanau 注意力**

Bahdanau 注意力机制的名称源自其发表论文的第一作者。

它遵循了[Cho et al. (2014)](https://arxiv.org/abs/1406.1078)和[Sutskever et al. (2014)](https://arxiv.org/abs/1409.3215)的研究，他们也采用了 RNN 编码器-解码器框架进行神经机器翻译，具体通过将变长的源句子编码成固定长度的向量，然后再解码为变长的目标句子。

[Bahdanau et al. (2014)](https://arxiv.org/abs/1409.0473)认为，将变长输入编码为固定长度向量会*挤压*源句子的的信息，无论其长度如何，导致基本的编码器-解码器模型随着输入句子长度的增加而性能迅速恶化。他们提出的方法用变长向量替代固定长度向量，以提高基本编码器-解码器模型的翻译性能。

> *这种方法与基本的编码器-解码器方法的最重要的区别在于，它不会试图将整个输入句子编码为单一的固定长度向量。相反，它将输入句子编码为向量序列，并在解码翻译时自适应地选择这些向量的子集。*
> 
> *–* [通过联合学习对齐和翻译的神经机器翻译](https://arxiv.org/abs/1409.0473)，2014 年。

### 想开始构建带有注意力的 Transformer 模型吗？

立即参加我的免费 12 天电子邮件速成课程（附带示例代码）。

点击注册并获取课程的免费 PDF 电子书版本。

## **Bahdanau 架构**

Bahdanau 编码器-解码器架构使用的主要组件如下：

+   $\mathbf{s}_{t-1}$是前一时间步$t-1$的*隐藏解码器状态*。

+   $\mathbf{c}_t$是时间步$t$的*上下文向量*。它在每个解码器步骤中独特生成，以生成目标单词$y_t$。

+   $\mathbf{h}_i$是一个*注释*，它捕捉了构成整个输入句子$\{ x_1, x_2, \dots, x_T \}$的单词中的信息，特别关注第$i$个单词（共$T$个单词）。

+   $\alpha_{t,i}$是分配给当前时间步$t$的每个注释$\mathbf{h}_i$的*权重*值。

+   $e_{t,i}$是由对齐模型$a(.)$生成的*注意力分数*，用来评分$\mathbf{s}_{t-1}$和$\mathbf{h}_i$的匹配程度。

这些组件在 Bahdanau 架构的不同阶段发挥作用，该架构使用双向 RNN 作为编码器，RNN 解码器，并且在两者之间有一个注意力机制：

![](https://machinelearningmastery.com/wp-content/uploads/2021/09/bahdanau_1.png)

Bahdanau 架构

摘自“[通过联合学习对齐和翻译的神经机器翻译](https://arxiv.org/abs/1409.0473)“

### **编码器**

编码器的角色是为输入句子中每个单词$x_i$生成一个注释$\mathbf{h}_i$，输入句子的长度为$T$个单词。

为了实现这一目的，Bahdanau 等人采用了一个双向 RNN，它首先正向读取输入句子以生成前向隐藏状态 $\overrightarrow{\mathbf{h}_i}$，然后反向读取输入句子以生成后向隐藏状态 $\overleftarrow{\mathbf{h}_i}$。对于某个特定词 $x_i$，其注释将这两个状态连接起来：

$$\mathbf{h}_i = \left[ \overrightarrow{\mathbf{h}_i^T} \; ; \; \overleftarrow{\mathbf{h}_i^T} \right]^T$$

以这种方式生成每个注释的思想是捕获前面和后面单词的摘要。

> *通过这种方式，注释 $\mathbf{h}_i$ 包含了前面单词和后续单词的摘要。*
> 
> *–* [神经机器翻译：联合学习对齐和翻译](https://arxiv.org/abs/1409.0473)，2014 年。

生成的注释然后传递给解码器以生成上下文向量。

### **解码器**

解码器的作用是通过关注源句子中包含的最相关信息来生成目标词语。为此，它利用了一个注意力机制。

> *每当提议的模型在翻译中生成一个词时，它（软）搜索源句子中信息最集中的一组位置。然后，基于与这些源位置相关的上下文向量和之前生成的所有目标词语，模型预测目标词语。*
> 
> *–* [神经机器翻译：联合学习对齐和翻译](https://arxiv.org/abs/1409.0473)，2014 年。

解码器将每个注释与对齐模型 $a(.)$ 和前一个隐藏解码器状态 $\mathbf{s}_{t-1}$ 一起提供，这生成一个注意力分数：

$$e_{t,i} = a(\mathbf{s}_{t-1}, \mathbf{h}_i)$$

这里由对齐模型实现的函数将 $\mathbf{s}_{t-1}$ 和 $\mathbf{h}_i$ 使用加法操作组合起来。因此，Bahdanau 等人实现的注意力机制被称为*加性注意力*。

这可以通过两种方式实现，要么 (1) 在连接向量 $\mathbf{s}_{t-1}$ 和 $\mathbf{h}_i$ 上应用权重矩阵 $\mathbf{W}$，要么 (2) 分别对 $\mathbf{s}_{t-1}$ 和 $\mathbf{h}_i$ 应用权重矩阵 $\mathbf{W}_1$ 和 $\mathbf{W}_2$：

1.  $$a(\mathbf{s}_{t-1}, \mathbf{h}_i) = \mathbf{v}^T \tanh(\mathbf{W}[\mathbf{h}_i \; ; \; \mathbf{s}_{t-1}])$$

1.  $$a(\mathbf{s}_{t-1}, \mathbf{h}_i) = \mathbf{v}^T \tanh(\mathbf{W}_1 \mathbf{h}_i + \mathbf{W}_2 \mathbf{s}_{t-1})$$

这里，$\mathbf{v}$ 是一个权重向量。

对齐模型被参数化为一个前馈神经网络，并与其余系统组件一起进行训练。

随后，对每个注意力分数应用 softmax 函数以获得相应的权重值：

$$\alpha_{t,i} = \text{softmax}(e_{t,i})$$

softmax 函数的应用本质上将注释值归一化到 0 到 1 的范围，因此，结果权重可以视为概率值。每个概率（或权重）值反映了 $\mathbf{h}_i$ 和 $\mathbf{s}_{t-1}$ 在生成下一个状态 $\mathbf{s}_t$ 和下一个输出 $y_t$ 时的重要性。

> *直观地说，这在解码器中实现了一个注意力机制。解码器决定要关注源句子的哪些部分。通过让解码器具备注意力机制，我们减轻了编码器必须将源句子中的所有信息编码成固定长度向量的负担。*
> 
> *–* [神经机器翻译：通过联合学习对齐和翻译](https://arxiv.org/abs/1409.0473)，2014 年。

最终计算上下文向量作为注释的加权和：

$$\mathbf{c}_t = \sum^T_{i=1} \alpha_{t,i} \mathbf{h}_i$$

### **Bahdanau 注意力算法**

总结来说，Bahdanau 等人提出的注意力算法执行以下操作：

1.  编码器从输入句子生成一组注释 $\mathbf{h}_i$。

1.  这些注释被输入到对齐模型和之前的隐藏解码器状态中。对齐模型使用这些信息生成注意力分数 $e_{t,i}$。

1.  对注意力分数应用了 softmax 函数，将其有效地归一化为权重值，$\alpha_{t,i}$，范围在 0 到 1 之间。

1.  结合先前计算的注释，这些权重用于通过注释的加权和生成上下文向量 $\mathbf{c}_t$。

1.  上下文向量与之前的隐藏解码器状态和先前输出一起输入解码器，以计算最终输出 $y_t$。

1.  步骤 2-6 会重复直到序列结束。

Bahdanau 等人对其架构进行了英法翻译任务的测试。他们报告称，他们的模型显著优于传统的编码器-解码器模型，无论句子长度如何。

已经有几个对 Bahdanau 注意力的改进，例如 [Luong 等人 (2015)](https://arxiv.org/abs/1508.04025) 提出的改进，我们将在单独的教程中回顾。

## **进一步阅读**

本节提供了更多关于该主题的资源，如果你想深入了解。

### **书籍**

+   [深入学习与 Python](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X)，2019 年。

### **论文**

+   [神经机器翻译：通过联合学习对齐和翻译](https://arxiv.org/abs/1409.0473)，2014 年。

## **总结**

在本教程中，你发现了 Bahdanau 注意力机制在神经机器翻译中的应用。

具体来说，你学到了：

+   Bahdanau 注意力的名字来源于哪里以及它所解决的挑战。

+   组成 Bahdanau 编码器-解码器架构的不同组件的作用

+   Bahdanau 注意力算法执行的操作

你有什么问题吗？

在下面的评论中提出你的问题，我会尽力回答。
