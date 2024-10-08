# [注意力机制架构之旅](https://machinelearningmastery.com/a-tour-of-attention-based-architectures/)

> 原文：[`machinelearningmastery.com/a-tour-of-attention-based-architectures/`](https://machinelearningmastery.com/a-tour-of-attention-based-architectures/)

随着注意力在机器学习中的流行，整合注意力机制的神经架构列表也在增长。

在本教程中，您将了解与注意力结合使用的显著神经架构。

完成本教程后，您将更好地理解注意力机制如何被整合到不同的神经架构中，以及其目的。

**用我的书 [使用注意力构建变形金刚模型](https://machinelearningmastery.com/transformer-models-with-attention/) 开始您的项目**。它提供了**自学教程**和**可工作的代码**，帮助您构建一个完全可工作的变形金刚模型，可以

*将一种语言的句子翻译成另一种语言*...

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/09/tour_cover2-scaled.jpg)

注意力机制架构之旅

照片由[Lucas Clara](https://unsplash.com/photos/hvPB-UCAmmU)拍摄，部分权利保留。

## **教程概述**

本教程分为四个部分；它们是：

+   编码器-解码器架构

+   变形金刚

+   图神经网络

+   增强记忆神经网络

## **编码器-解码器架构**

编码器-解码器架构已被广泛应用于序列到序列（seq2seq）任务，例如语言处理中的机器翻译和图像字幕。

> *注意力最早作为 RNN 基础编码器-解码器框架的一部分用于编码长输入句子[Bahdanau et al. 2015]。因此，注意力在这种架构中被广泛使用。*
> 
> – [注意模型的关注性调查](https://arxiv.org/abs/1904.02874?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%253A+arxiv%252FQSXk+%2528ExcitingAds%2521+cs+updates+on+arXiv.org%2529)，2021 年。

在机器翻译的背景下，这样的 seq2seq 任务将涉及将输入序列$I = \{ A, B, C, <EOS> \}$翻译成长度不同的输出序列$O = \{ W, X, Y, Z, <EOS> \}$。

对于没有注意力的基于 RNN 的编码器-解码器架构，[展开每个 RNN](https://machinelearningmastery.com/rnn-unrolling/)将产生以下图表：

![](https://machinelearningmastery.com/wp-content/uploads/2021/09/tour_1.png)

未展开的基于 RNN 的编码器和解码器

摘自“[神经网络的序列到序列学习](https://arxiv.org/abs/1409.3215)“

在这里，编码器一次读取一个词的输入序列，每次更新其内部状态。遇到<EOS>符号时，表明*序列结束*。编码器生成的隐藏状态本质上包含了输入序列的向量表示，解码器将处理这个表示。

解码器一次生成一个词的输出序列，以前一个时间步（$t$ – 1）的词作为输入生成输出序列中的下一个词。解码端的<EOS>符号表示解码过程已结束。

[正如我们之前提到的](https://machinelearningmastery.com/what-is-attention/)，没有注意力机制的编码器-解码器架构的问题在于，当不同长度和复杂性的序列由固定长度的向量表示时，可能会导致解码器遗漏重要信息。

为了解决这个问题，基于注意力的架构在编码器和解码器之间引入了注意力机制。

![](https://machinelearningmastery.com/wp-content/uploads/2021/09/tour_2.jpg)

带有注意力机制的编码器-解码器架构

摘自“[心理学、神经科学与机器学习中的注意力](https://www.frontiersin.org/articles/10.3389/fncom.2020.00029/full)”

在这里，注意力机制（$\phi$）学习一组注意力权重，这些权重捕捉编码向量（v）与解码器的隐藏状态（h）之间的关系，通过对编码器所有隐藏状态的加权求和生成上下文向量（c）。这样，解码器能够访问整个输入序列，特别关注生成输出时最相关的输入信息。

## **Transformer**

Transformer 的架构还实现了编码器和解码器。然而，与上述回顾的架构不同，它不依赖于递归神经网络。因此，本文将单独回顾这一架构及其变体。

Transformer 架构摒弃了任何递归，而是完全依赖于*自注意力*（或内部注意力）机制。

> *在计算复杂性方面，当序列长度 n 小于表示维度 d 时，自注意力层比递归层更快……*
> 
> – [高级深度学习与 Python](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X)，2019。

自注意力机制依赖于使用*查询*、*键*和*值*，这些是通过将编码器对相同输入序列的表示与不同的权重矩阵相乘生成的。Transformer 使用点积（或*乘法*）注意力，在生成注意力权重的过程中，每个查询通过点积操作与键的数据库进行匹配。这些权重然后与值相乘，以生成最终的注意力向量。

![](https://machinelearningmastery.com/wp-content/uploads/2021/09/tour_3.png)

乘法注意力

取自 “[Attention Is All You Need](https://arxiv.org/abs/1706.03762)“

直观地说，由于所有查询、键和值都源于相同的输入序列，自注意力机制捕捉了同一序列中不同元素之间的关系，突出显示了彼此之间最相关的元素。

由于 transformer 不依赖于 RNN，序列中每个元素的位置信息可以通过增强编码器对每个元素的表示来保留。这意味着 transformer 架构还可以应用于信息可能不一定按顺序相关的任务，例如图像分类、分割或标注的计算机视觉任务。

> *Transformers 可以捕捉输入和输出之间的全局/长期依赖，支持并行处理，要求最少的归纳偏置（先验知识），展示了对大序列和数据集的可扩展性，并允许使用类似的处理块进行多模态（文本、图像、语音）的领域无关处理。*
> 
> – [An Attentive Survey of Attention Models](https://arxiv.org/abs/1904.02874?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%253A+arxiv%252FQSXk+%2528ExcitingAds%2521+cs+updates+on+arXiv.org%2529)，2021 年。

此外，多个注意力层可以并行堆叠，这被称为*多头注意力*。每个头部在相同输入的不同线性变换上并行工作，然后将头部的输出连接起来生成最终的注意力结果。使用多头模型的好处是每个头部可以关注序列的不同元素。

![](https://machinelearningmastery.com/wp-content/uploads/2021/09/tour_4.png)

多头注意力

取自 “[Attention Is All You Need](https://arxiv.org/abs/1706.03762)“

**一些解决原始模型局限性的 transformer 架构变体包括：**

***   Transformer-XL：引入了递归机制，使其能够学习超越训练过程中通常使用的碎片化序列的固定长度的长期依赖。

+   XLNet：一个双向变换器，它通过引入基于排列的机制在 Transformer-XL 的基础上进行构建，其中训练不仅在输入序列的原始顺序上进行，还包括输入序列顺序的不同排列。

### 想开始构建带有注意力的变换器模型吗？

立即获取我的免费 12 天电子邮件速成课程（附示例代码）。

点击注册，并免费获得课程的 PDF Ebook 版本。

## **图神经网络**

图可以定义为一组*节点*（或顶点），它们通过*连接*（或边）链接在一起。

> *图是一种多功能的数据结构，非常适合许多现实世界场景中数据的组织方式。*
> 
> – [深入学习 Python](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X)，2019 年。

例如，考虑一个社交网络，其中用户可以通过图中的节点表示，朋友之间的关系通过边表示。或者是一个分子，其中节点是原子，边表示它们之间的化学键。

> *我们可以将图像视为图形，其中每个像素是一个节点，直接连接到其邻近的像素…*
> 
> – [深入学习 Python](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X)，2019 年。

特别感兴趣的是*图注意力网络*（GAT），它在图卷积网络（GCN）中使用自注意力机制，其中后者通过在图的节点上执行卷积来更新状态向量。卷积操作应用于中心节点和邻近节点，使用加权滤波器来更新中心节点的表示。GCN 中的滤波器权重可以是固定的或可学习的。

![](https://machinelearningmastery.com/wp-content/uploads/2021/09/tour_5.png)

在中心节点（红色）及其邻域节点上的图卷积

摘自“[图神经网络的综合调查](https://arxiv.org/abs/1901.00596)“

相比之下，GAT 使用注意力分数为邻近节点分配权重。

这些注意力分数的计算遵循与上述 seq2seq 任务中的方法类似的程序：（1）首先计算两个邻近节点的特征向量之间的对齐分数，然后（2）通过应用 softmax 操作计算注意力分数，最后（3）通过对所有邻居的特征向量进行加权组合，计算每个节点的输出特征向量（相当于 seq2seq 任务中的上下文向量）。

多头注意力也可以以非常类似于先前在变换器架构中提议的方式应用。在图中的每个节点会分配多个头，最终层中会对它们的输出进行平均。

一旦生成最终输出，这可以作为后续任务特定层的输入。可以通过图解决的任务包括将个体节点分类到不同的群组中（例如，在预测一个人将决定加入哪些俱乐部时）。或者可以对个体边进行分类，以确定两个节点之间是否存在边（例如，在社交网络中预测两个人是否可能是朋友），甚至可以对整个图进行分类（例如，预测分子是否有毒）。

**用我的书 [使用注意力构建 Transformer 模型](https://machinelearningmastery.com/transformer-models-with-attention/) 开始你的项目**。它提供了带有**工作代码**的**自学教程**，指导你构建一个完全工作的 Transformer 模型。

*将一种语言的句子翻译成另一种语言*...

## **记忆增强神经网络**

到目前为止审查的基于编码器-解码器的注意力架构中，编码输入序列的向量集可以被视为外部记忆，编码器写入并解码器读取。然而，由于编码器只能写入此内存，解码器只能读取，因此存在限制。

记忆增强神经网络（MANNs）是最近旨在解决这一限制的算法。

神经图灵机（NTM）是 MANN 的一种类型。它由一个神经网络控制器组成，接受输入并产生输出，并对内存执行读写操作。

![](https://machinelearningmastery.com/wp-content/uploads/2021/09/tour_6.png)

神经图灵机架构

取自“[神经图灵机](https://arxiv.org/abs/1410.5401)”

读头执行的操作类似于用于 seq2seq 任务的注意力机制，其中注意力权重指示考虑的向量在形成输出时的重要性。

> *读头总是读取完整的记忆矩阵，但通过对不同记忆向量进行不同强度的关注来执行此操作。*
> 
> – [Python 深度学习进阶](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X)，2019 年。

读操作的输出由记忆向量的加权和定义。

写头还利用注意力向量，与擦除和添加向量一起工作。根据注意力和擦除向量的值擦除内存位置，并通过添加向量写入信息。

MANNs 的应用示例包括问答和聊天机器人，其中外部记忆存储大量序列（或事实），神经网络利用这些信息。关注机制在选择对当前任务更相关的数据库事实时起着关键作用。

## **进一步阅读**

本节提供了更多关于该主题的资源，如果你想深入了解的话。

### **书籍**

+   [Python 高级深度学习](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X)，2019 年。

+   [深度学习精粹](https://www.amazon.com/Deep-Learning-Essentials-hands-fundamentals/dp/1785880365)，2018 年。

### **论文**

+   [注意力模型的综述](https://arxiv.org/abs/1904.02874?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%253A+arxiv%252FQSXk+%2528ExcitingAds%2521+cs+updates+on+arXiv.org%2529)，2021 年。

+   [心理学、神经科学和机器学习中的注意力](https://www.frontiersin.org/articles/10.3389/fncom.2020.00029/full)，2020 年。

## **摘要**

在本教程中，你发现了与注意力机制结合使用的显著神经网络架构。

具体来说，你更好地理解了注意力机制如何融入不同的神经网络架构以及其目的。

你有任何问题吗？

在下面的评论中提问，我会尽力回答。**
