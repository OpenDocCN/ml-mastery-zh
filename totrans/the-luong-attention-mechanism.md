# Luong 注意力机制

> 原文：[`machinelearningmastery.com/the-luong-attention-mechanism/`](https://machinelearningmastery.com/the-luong-attention-mechanism/)

Luong 注意力旨在对 Bahdanau 模型进行若干改进，特别是通过引入两种新的注意力机制：一种是 *全局* 方法，关注所有源单词，另一种是 *局部* 方法，只关注在预测目标句子时选择的单词子集。

在本教程中，你将发现 Luong 注意力机制在神经机器翻译中的应用。

完成本教程后，你将了解：

+   Luong 注意力算法执行的操作

+   全局和局部注意力模型如何工作。

+   Luong 注意力与 Bahdanau 注意力的比较

**用我的书** [《构建带有注意力的 Transformer 模型》](https://machinelearningmastery.com/transformer-models-with-attention/) **来启动你的项目**。它提供了 **自学教程** 和 **可运行的代码**，帮助你构建一个完全运行的 Transformer 模型。

*将句子从一种语言翻译成另一种语言*...

开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/10/luong_cover-scaled.jpg)

Luong 注意力机制

图片来源 [Mike Nahlii](https://unsplash.com/photos/BskqKfpR4pw)，版权所有。

## **教程概述**

本教程分为五部分；它们是：

+   Luong 注意力简介

+   Luong 注意力算法

+   全局注意力模型

+   局部注意力模型

+   与 Bahdanau 注意力的比较

## **先决条件**

在本教程中，我们假设你已经熟悉：

+   [注意力机制的概念](https://machinelearningmastery.com/what-is-attention/)

+   [注意力机制](https://machinelearningmastery.com/the-attention-mechanism-from-scratch/)

+   [Bahdanau 注意力机制](https://machinelearningmastery.com/?p=12940&preview=true)

## **Luong 注意力简介**

[Luong 等人 (2015)](https://arxiv.org/abs/1508.04025) 从先前的注意力模型中汲取灵感，提出了两种注意力机制：

> *在这项工作中，我们以简洁和有效性为目标，设计了两种新型的基于注意力的模型：一种是全局方法，它总是关注所有源单词，另一种是局部方法，它仅关注一次性选择的源单词子集。*
> 
> *–* [基于注意力的神经机器翻译的有效方法](https://arxiv.org/abs/1508.04025)，2015 年。

*全局* 注意力模型类似于 [Bahdanau 等人 (2014)](https://arxiv.org/abs/1409.0473) 模型，关注 *所有* 源单词，但旨在在结构上简化它。

*局部* 注意力模型受到 [Xu 等人 (2016)](https://arxiv.org/abs/1502.03044) 的硬注意力和软注意力模型的启发，只关注 *少量* 源位置。

两种注意力模型在预测当前词的许多步骤中是相似的，但主要在于它们计算上下文向量的方式不同。

让我们首先看看整体的 Luong 注意力算法，然后再深入探讨全局和局部注意力模型之间的差异。

### 想开始构建具有注意力机制的 Transformer 模型吗？

立即参加我的免费 12 天电子邮件速成课程（包含示例代码）。

点击注册，还可以获得课程的免费 PDF 电子书版本。

## **Luong 注意力算法**

Luong 等人的注意力算法执行以下操作：

1.  编码器从输入句子中生成一组注释，$H = \mathbf{h}_i, i = 1, \dots, T$。

1.  当前的解码器隐藏状态计算公式为：$\mathbf{s}_t = \text{RNN}_\text{decoder}(\mathbf{s}_{t-1}, y_{t-1})$。这里，$\mathbf{s}_{t-1}$ 表示先前的隐藏解码器状态，而 $y_{t-1}$ 是前一个解码器输出。

1.  对齐模型 $a(.)$ 使用注释和当前解码器隐藏状态来计算对齐分数：$e_{t,i} = a(\mathbf{s}_t, \mathbf{h}_i)$。

1.  将 softmax 函数应用于对齐分数，有效地将其归一化为介于 0 和 1 之间的权重值：$\alpha_{t,i} = \text{softmax}(e_{t,i})$。

1.  与之前计算的注释一起，这些权重被用于通过加权求和生成上下文向量：$\mathbf{c}_t = \sum^T_{i=1} \alpha_{t,i} \mathbf{h}_i$。

1.  基于上下文向量和当前解码器隐藏状态的加权连接计算注意力隐藏状态：$\widetilde{\mathbf{s}}_t = \tanh(\mathbf{W_c} [\mathbf{c}_t \; ; \; \mathbf{s}_t])$。

1.  解码器通过输入加权注意力隐藏状态来生成最终输出：$y_t = \text{softmax}(\mathbf{W}_y \widetilde{\mathbf{s}}_t)$。

1.  步骤 2-7 重复直到序列结束。

## **全局注意力模型**

全局注意力模型在生成对齐分数时考虑了输入句子中的所有源词，最终在计算上下文向量时也会考虑这些源词。

> *全局注意力模型的思想是，在推导上下文向量 $\mathbf{c}_t$ 时考虑编码器的所有隐藏状态。*
> 
> *–* [基于注意力的神经机器翻译的有效方法](https://arxiv.org/abs/1508.04025)，2015 年。

为了实现这一点，Luong 等人提出了三种计算对齐分数的替代方法。第一种方法类似于 Bahdanau 的方法。它基于 $\mathbf{s}_t$ 和 $\mathbf{h}_i$ 的连接，而第二种和第三种方法则实现了 *乘法* 注意力（与 Bahdanau 的 *加法* 注意力相对）：

1.  $$a(\mathbf{s}_t, \mathbf{h}_i) = \mathbf{v}_a^T \tanh(\mathbf{W}_a [\mathbf{s}_t \; ; \; \mathbf{h}_i])$$

1.  $$a(\mathbf{s}_t, \mathbf{h}_i) = \mathbf{s}^T_t \mathbf{h}_i$$

1.  $$a(\mathbf{s}_t, \mathbf{h}_i) = \mathbf{s}^T_t \mathbf{W}_a \mathbf{h}_i$$

在这里，$\mathbf{W}_a$是一个可训练的权重矩阵，类似地，$\mathbf{v}_a$是一个权重向量。

从直观上讲，*乘法*注意力中使用点积可以解释为提供了向量$\mathbf{s}_t$和$\mathbf{h}_i$之间的相似性度量。

> *……如果向量相似（即对齐），则乘法结果将是一个大值，注意力将集中在当前的 t,i 关系上。*
> 
> – [用 Python 进行高级深度学习](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X)，2019。

结果对齐向量$\mathbf{e}_t$的长度根据源词的数量而变化。

## **局部注意力模型**

在关注所有源词时，全局注意力模型计算开销大，可能会使其在翻译较长句子时变得不切实际。

局部注意力模型试图通过专注于较小的源词子集来生成每个目标词，从而解决这些局限性。为此，它从[Xu 等人（2016）](https://arxiv.org/abs/1502.03044)的图像描述生成工作中的*硬*和*软*注意力模型中获得灵感：

+   *软*注意力等同于全局注意力方法，其中权重软性地分布在所有源图像区域上。因此，软注意力将整个源图像考虑在内。

+   *硬*注意力一次关注一个图像区域。

Luong 等人的局部注意力模型通过计算在对齐位置$p_t$中心窗口内注释集$\mathbf{h}_i$上的加权平均来生成上下文向量：

$$[p_t – D, p_t + D]$$

虽然$D$的值是通过经验选择的，但 Luong 等人考虑了计算$p_t$值的两种方法：

1.  *单调*对齐：源句子和目标句子假定是单调对齐的，因此$p_t = t$。

1.  *预测*对齐：基于可训练的模型参数$\mathbf{W}_p$和$\mathbf{v}_p$以及源句子长度$S$对对齐位置进行预测：

$$p_t = S \cdot \text{sigmoid}(\mathbf{v}^T_p \tanh(\mathbf{W}_p, \mathbf{s}_t))$$

高斯分布在计算对齐权重时围绕$p_t$中心，以偏好窗口中心附近的源词。

这一次，结果对齐向量$\mathbf{e}_t$具有固定长度$2D + 1$。

**启动你的项目**，请参见我的书[使用注意力构建 Transformer 模型](https://machinelearningmastery.com/transformer-models-with-attention/)。它提供了**自学教程**和**有效代码**，引导你构建一个完整的 Transformer 模型。

*将句子从一种语言翻译成另一种语言*……

## **与 Bahdanau 注意力的比较**

Bahdanau 模型和 Luong 等人的全局注意力方法大致相似，但两者之间存在关键差异：

> *尽管我们的全球注意力方法在精神上类似于 Bahdanau 等人（2015 年）提出的模型，但存在若干关键区别，这些区别反映了我们如何从原始模型中进行简化和概括。*
> 
> *–* [基于注意力的神经机器翻译的有效方法](https://arxiv.org/abs/1508.04025)，2015 年。

1.  最显著的是，Luong 全球注意力模型中对齐得分 $e_t$ 的计算依赖于当前解码器隐藏状态 $\mathbf{s}_t$，而非 Bahdanau 注意力中的前一个隐藏状态 $\mathbf{s}_{t-1}$。

![](https://machinelearningmastery.com/wp-content/uploads/2021/10/luong_1.png)

Bahdanau 架构（左）与 Luong 架构（右）

摘自 “[深入学习 Python](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X)”

1.  Luong 等人舍弃了 Bahdanau 模型中使用的双向编码器，而是利用编码器和解码器顶部 LSTM 层的隐藏状态。

1.  Luong 等人的全球注意力模型研究了使用乘法注意力作为 Bahdanau 加性注意力的替代方案。

## **进一步阅读**

本节提供了更多相关资源，供你深入了解。

### **书籍**

+   [深入学习 Python](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X)，2019 年。

### **论文**

+   [基于注意力的神经机器翻译的有效方法](https://arxiv.org/abs/1508.04025)，2015 年。

## **总结**

在本教程中，你了解了 Luong 注意力机制在神经机器翻译中的应用。

具体来说，你学到了：

+   Luong 注意力算法执行的操作

+   全球和局部注意力模型如何工作

+   Luong 注意力与 Bahdanau 注意力的比较

你有任何问题吗？

在下方评论中提出你的问题，我会尽力回答。
