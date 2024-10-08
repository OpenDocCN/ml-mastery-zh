# 浅谈神经机器翻译

> 原文： [`machinelearningmastery.com/introduction-neural-machine-translation/`](https://machinelearningmastery.com/introduction-neural-machine-translation/)

计算机最早的目标之一是将文本从一种语言自动转换为另一种语言。

考虑到人类语言的流动性，自动或机器翻译可能是最具挑战性的人工智能任务之一。传统上，基于规则的系统被用于这项任务，在 20 世纪 90 年代用统计方法取代了这一系统。最近，深度神经网络模型在一个恰当地命名为神经机器翻译的领域中实现了最先进的结果。

在这篇文章中，您将发现机器翻译的挑战和神经机器翻译模型的有效性。

阅读这篇文章后，你会知道：

*   鉴于人类语言固有的模糊性和灵活性，机器翻译具有挑战性。
*   统计机器翻译将经典的基于规则的系统替换为学习从示例翻译的模型。
*   神经机器翻译模型适合单个模型而不是微调模型的管道，并且目前实现最先进的结果。

让我们开始吧。

![A Gentle Introduction to Neural Machine Translation](img/794b6df79869c917f098df7b3999c43c.jpg)

神经机器翻译的温和介绍
[Fabio Achilli](https://www.flickr.com/photos/travelourplanet/6218704200/) 的照片，保留一些权利。

## 什么是机器翻译？

机器翻译是将一种语言的源文本自动转换为另一种语言的文本的任务。

> 在机器翻译任务中，输入已经由某种语言的符号序列组成，并且计算机程序必须将其转换为另一种语言的符号序列。

- 第 98 页，[深度学习](http://amzn.to/2xBEsBJ)，2016 年。

给定源语言中的一系列文本，该文本没有一个单一的最佳翻译成另一种语言。这是因为人类语言的自然模糊性和灵活性。这使得自动机器翻译的挑战变得困难，也许是人工智能中最难的一个：

> 事实是，准确的翻译需要背景知识，以解决歧义并确定句子的内容。

- 第 21 页，[人工智能，现代方法](http://amzn.to/2wUZesr)，第 3 版，2009 年。

经典机器翻译方法通常涉及将源语言中的文本转换为目标语言的规则。这些规则通常由语言学家开发，可以在词汇，句法或语义层面上运作。这种对规则的关注给出了这个研究领域的名称：基于规则的机器翻译或 RBMT。

> RBMT 的特点是明确使用和手动创建语言知情规则和表示。

- 第 133 页，[自然语言处理和机器翻译手册](http://amzn.to/2jYUFfy)，2011。

经典机器翻译方法的主要局限性是开发规则所需的专业知识，以及所需的大量规则和例外。

## 什么是统计机器翻译？

统计机器翻译（简称 SMT）是使用统计模型来学习将文本从源语言翻译成目标语言，从而提供大量的示例。

使用统计模型的任务可以正式说明如下：

> 给定目标语言中的句子 T，我们寻找翻译者产生 T 的句子 S.我们知道通过选择最可能给出 T 的句子 S 来最小化我们的错误机会。因此，我们希望选择 S 所以为了最大化 Pr（S | T）。

- [机器翻译的统计方法](https://dl.acm.org/citation.cfm?id=92860)，1990。

这种形式化的规范使输出序列的概率最大化，给定文本的输入序列。它还使得存在一套候选翻译的概念明确，并且需要搜索过程或解码器从模型的输出概率分布中选择最可能的翻译。

> 鉴于源语言中的文本，目标语言中最可能的翻译是什么？ [...]如何构建一个统计模型，为“好”翻译分配高概率，为“坏”翻译分配低概率？

- 第 xiii 页，[基于句法的统计机器翻译](http://amzn.to/2xCrl3p)，2017。

该方法是数据驱动的，只需要包含源语言和目标语言文本的示例语料库。这意味着语言学家不再需要指定翻译规则。

> 这种方法不需要复杂的语际概念本体论，也不需要源语言和目标语言的手工语法，也不需要手工标记的树库。它所需要的只是数据样本翻译，从中可以学习翻译模型。

- 第 909 页，[人工智能，现代方法](http://amzn.to/2wUZesr)，第 3 版，2009 年。

很快，机器翻译的统计方法优于传统的基于规则的方法，成为事实上的标准技术集。

> 自 20 世纪 80 年代末该领域开始以来，最流行的统计机器翻译模型基于序列。在这些模型中，翻译的基本单位是单词或单词序列[...]这些模型简单有效，适用于人类语言对

- [基于句法的统计机器翻译](http://amzn.to/2xCrl3p)，2017。

最广泛使用的技术是基于短语的，并且侧重于分段翻译源文本的子序列。

> 几十年来，统计机器翻译（SMT）一直是主流的翻译范式。 SMT 的实际实现通常是基于短语的系统（PBMT），其翻译长度可以不同的单词或短语的序列

- [谷歌的神经机器翻译系统：缩小人机翻译之间的差距](https://arxiv.org/abs/1609.08144)，2016。

虽然有效，但统计机器翻译方法很少关注被翻译的短语，失去了目标文本的更广泛性质。对数据驱动方法的高度关注也意味着方法可能忽略了语言学家已知的重要语法区别。最后，统计方法需要仔细调整转换管道中的每个模块。

## 什么是神经机器翻译？

神经机器翻译（简称 NMT）是利用神经网络模型来学习机器翻译的统计模型。

该方法的主要好处是可以直接在源文本和目标文本上训练单个系统，不再需要统计机器学习中使用的专用系统的管道。

> 与传统的基于短语的翻译系统不同，翻译系统由许多单独调整的小子组件组成，神经机器翻译尝试构建和训练单个大型神经网络，该网络读取句子并输出正确的翻译。

- [通过联合学习对齐和翻译的神经机器翻译](https://arxiv.org/abs/1409.0473)，2014。

因此，神经机器翻译系统被称为端到端系统，因为翻译仅需要一个模型。

> NMT 的优势在于它能够以端到端的方式直接学习从输入文本到相关输出文本的映射。

- [谷歌的神经机器翻译系统：缩小人机翻译之间的差距](https://arxiv.org/abs/1609.08144)，2016。

### 编解码器模型

多层感知机神经网络模型可用于机器转换，尽管模型受固定长度输入序列的限制，其中输出必须具有相同的长度。

最近，通过使用组织成编解码器结构的循环神经网络，这些早期模型得到了极大的改进，该结构允许可变长度的输入和输出序列。

> 编码器神经网络将源句子读取并编码为固定长度的向量。然后，解码器从编码向量输出转换。整个编解码器系统，包括用于语言对的编码器和解码器，被联合训练以最大化给定源句子的正确翻译的概率。

- [通过联合学习对齐和翻译的神经机器翻译](https://arxiv.org/abs/1409.0473)，2014。

编解码器架构的关键是模型将源文本编码为称为上下文向量的内部固定长度表示的能力。有趣的是，一旦编码，原则上可以使用不同的解码系统将上下文翻译成不同的语言。

> ...一个模型首先读取输入序列并发出一个汇总输入序列的数据结构。我们将此摘要称为“上下文”C. [...]第二种模式，通常是 RNN，然后读取上下文 C 并生成目标语言的句子。

- 第 461 页，[深度学习](http://amzn.to/2xBEsBJ)，2016 年。

有关编解码器循环神经网络架构的更多信息，请参阅帖子：

*   [编解码器长短期记忆网络](https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/)

### 带注意的编解码器

虽然有效，但编解码器架构在要翻译的长文本序列方面存在问题。

问题源于必须用于解码输出序列中每个单词的固定长度内部表示。

解决方案是使用注意机制，该机制允许模型在输出序列的每个字被解码时学习将注意力放在输入序列的哪个位置。

> 使用固定大小的表示来捕获很长句子的所有语义细节是非常困难的。 [...]然而，更有效的方法是阅读整个句子或段落[...]，然后一次一个地产生翻译的单词，每次都集中在他输入句子的不同部分以收集所需的语义细节生成下一个输出字。

- 第 462 页，[深度学习](http://amzn.to/2xBEsBJ)，2016 年。

目前关注的编解码器循环神经网络架构是机器翻译的一些基准问题的最新技术。此架构用于谷歌翻译服务中使用的谷歌神经机器翻译系统（GNMT）的核心。
https://translate.google.com

> ......当前最先进的机器翻译系统由引起注意的模型提供动力。

- 第 209 页，[自然语言处理中的神经网络方法](http://amzn.to/2wPrW37)，2017。

有关关注的更多信息，请参阅帖子：

*   [长期短期记忆循环神经网络](https://machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/)的注意事项

虽然有效，但神经机器翻译系统仍然存在一些问题，例如缩放到较大的单词词汇表以及训练模型的速度慢。目前有大型生产神经翻译系统的重点领域，例如 Google 系统。

> 神经机器翻译的三个固有缺点：它的训练速度和推理速度较慢，处理稀有单词的效率低下，有时无法翻译源句中的所有单词。

- [谷歌的神经机器翻译系统：缩小人机翻译之间的差距](https://arxiv.org/abs/1609.08144)，2016。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 图书

*   [自然语言处理中的神经网络方法](http://amzn.to/2wPrW37)，2017。
*   [基于句法的统计机器翻译](http://amzn.to/2xCrl3p)，2017。
*   [深度学习](http://amzn.to/2xBEsBJ)，2016 年。
*   [统计机器翻译](http://amzn.to/2xCe1vP)，2010。
*   [自然语言处理和机器翻译手册](http://amzn.to/2jYUFfy)，2011。
*   [人工智能，现代方法](http://amzn.to/2wUZesr)，第 3 版，2009 年。

### 文件

*   [机器翻译的统计方法](https://dl.acm.org/citation.cfm?id=92860)，1990。
*   [评论文章：基于实例的机器翻译](https://link.springer.com/article/10.1023/A:1008109312730)，1999。
*   [使用 RNN 编解码器进行统计机器翻译的学习短语表示](https://arxiv.org/abs/1406.1078)，2014。
*   [通过联合学习对齐和翻译的神经机器翻译](https://arxiv.org/abs/1409.0473)，2014。
*   [谷歌的神经机器翻译系统：缩小人机翻译之间的差距](https://arxiv.org/abs/1609.08144)，2016。
*   [用神经网络进行序列学习的序列](https://arxiv.org/abs/1409.3215)，2014。
*   [循环连续翻译模型](http://www.aclweb.org/anthology/D13-1176)，2013。
*   [基于短语的统计机器翻译的连续空间翻译模型](https://aclweb.org/anthology/C/C12/C12-2104.pdf)，2013。

### 额外

*   [机器翻译档案](http://www.mt-archive.info/)
*   [维基百科上的神经机器翻译](https://en.wikipedia.org/wiki/Neural_machine_translation)
*   [第十三章，神经机器翻译，统计机器翻译](https://arxiv.org/abs/1709.07809)，2017。

## 摘要

在这篇文章中，您发现了机器翻译的挑战和神经机器翻译模型的有效性。

具体来说，你学到了：

*   鉴于人类语言固有的模糊性和灵活性，机器翻译具有挑战性。
*   统计机器翻译将经典的基于规则的系统替换为学习从示例翻译的模型。
*   神经机器翻译模型适合单个模型而不是精细调整模型的管道，并且目前实现最先进的结果。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。