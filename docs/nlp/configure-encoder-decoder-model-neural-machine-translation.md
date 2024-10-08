# 如何配置神经机器翻译的编解码器模型

> 原文： [`machinelearningmastery.com/configure-encoder-decoder-model-neural-machine-translation/`](https://machinelearningmastery.com/configure-encoder-decoder-model-neural-machine-translation/)

用于循环神经网络的编解码器架构在标准机器翻译基准上实现了最先进的结果，并且正在用于工业翻译服务的核心。

该模型很简单，但考虑到训练它所需的大量数据，调整模型中的无数设计决策以获得最佳表现问题可能实际上难以处理。值得庆幸的是，研究科学家已经使用 Google 规模的硬件为我们完成这项工作，并提供了一套启发式方法，用于如何配置编解码器模型进行神经机器翻译和一般的序列预测。

在这篇文章中，您将了解如何最好地配置编解码器循环神经网络以进行神经机器翻译和其他自然语言处理任务的详细信息。

阅读这篇文章后，你会知道：

*   Google 研究调查了编解码器模型中的每个模型设计决策，以隔离它们的影响。
*   设计决策的结果和建议，如字嵌入，编码器和解码器深度以及注意机制。
*   一组基本模型设计决策，可用作您自己的序列到序列项目的起点。

让我们开始吧。

![How to Configure an Encoder-Decoder Model for Neural Machine Translation](img/f51461220fc281d9dea38cd12d3be911.jpg)

如何配置神经机器翻译的编解码器模型
照片由 [Sporting Park](https://www.flickr.com/photos/sporting_su/34385197705/) ，保留一些权利。

## 神经机器翻译的编解码器模型

用于循环神经网络的编解码器架构正在取代基于经典短语的统计机器翻译系统，以获得最先进的结果。

作为证据，他们的 2016 年论文“[谷歌的神经机器翻译系统：缩小人机翻译之间的差距](https://arxiv.org/abs/1609.08144)”，谷歌现在将这种方法用于他们的谷歌翻译服务的核心。

这种架构的一个问题是模型很大，反过来需要非常大的数据集来训练。这具有模型训练的效果，需要数天或数周并且需要通常非常昂贵的计算资源。因此，关于不同设计选择对模型的影响及其对模型技能的影响的工作很少。

[Denny Britz](http://blog.dennybritz.com/) 等人明确解决了这个问题。在他们的 2017 年论文“[大规模探索神经机器翻译架构](https://arxiv.org/abs/1703.03906)。”在本文中，他们设计了标准英语 - 德语翻译任务的基线模型，并列举了一套不同的模型设计选择和描述它们对模型技能的影响。他们声称完整的实验耗时超过 250,000 GPU 计算时间，这至少可以说令人印象深刻。

> 我们报告了数百次实验运行的经验结果和方差数，相当于标准 WMT 英语到德语翻译任务的超过 250,000 GPU 小时数。我们的实验为构建和扩展 NMT 架构提供了新颖的见解和实用建议。

在这篇文章中，我们将看一下本文的一些发现，我们可以用来调整我们自己的神经机器翻译模型，以及一般的序列到序列模型。

有关编解码器架构和注意机制的更多背景信息，请参阅帖子：

*   [编解码器长短期记忆网络](https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/)
*   [长期短期记忆循环神经网络](https://machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/)的注意事项

## 基线模型

我们可以通过描述用作所有实验起点的基线模型来开始。

选择基线模型配置，使得模型在翻译任务上表现得相当好。

*   嵌入：512 维
*   RNN Cell：门控循环单元或 GRU
*   编码器：双向
*   编码器深度：2 层（每个方向 1 层）
*   解码器深度：2 层
*   注意：巴达瑙式
*   优化者：亚当
*   dropout：投入 20％

每个实验都从基线模型开始，并且不同的一个元素试图隔离设计决策对模型技能的影响，在这种情况下，BLEU 分数。

![Encoder-Decoder Architecture for Neural Machine Translation](img/38abff20be7b017cc50576763f7328b3.jpg)

用于神经机器翻译的编解码器架构
取自“神经机器翻译架构的大规模探索”。

## 嵌入尺寸

[字嵌入](https://machinelearningmastery.com/what-are-word-embeddings/)用于表示输入到编码器的字。

这是一种分布式表示，其中每个单词被映射到固定大小的连续值向量。这种方法的好处是具有相似含义的不同单词将具有类似的表示。

通常在将模型拟合到训练数据上时学习该分布式表示。嵌入大小定义用于表示单词的向量的长度。人们普遍认为，较大的维度将导致更具表现力的表现形式，从而产生更好的技能。

有趣的是，结果表明，测试的最大尺寸确实达到了最佳效果，但增加尺寸的好处总体上是微不足道的。

> [结果显示] 2048 维嵌入产生了总体最佳结果，它们只是略微提高了。即使是小型的 128 维嵌入也表现出色，而收敛速度几乎快了两倍。

**建议**：从小型嵌入开始，例如 128，可能会稍后增加尺寸，以提高技能。

## RNN 细胞类型

通常使用三种类型的循环神经网络细胞：

*   简单的 RNN。
*   长期短期记忆或 LSTM。
*   门控循环单元或 GRU。

LSTM 的开发是为了解决简单 RNN 的消失梯度问题，这限制了深 RNN 的训练。 GRU 的开发旨在简化 LSTM。

结果显示 GRU 和 LSTM 均明显优于 Simple RNN，但 LSTM 总体上更好。

> 在我们的实验中，LSTM 细胞始终优于 GRU 细胞

**建议**：在您的模型中使用 LSTM RNN 单位。

## 编解码器深度

通常，深层网络被认为比浅层网络具有更好的表现。

关键是要在网络深度，模型技能和训练时间之间找到平衡点。这是因为如果对技能的好处很小，我们通常没有无限的资源来训练非常深的网络。

作者探讨了编码器和解码器模型的深度以及对模型技能的影响。

当涉及编码器时，发现深度对技能没有显着影响，更令人惊讶的是，1 层单向模型仅比 4 层单向配置略差。双层双向编码器的表现略优于其他测试配置。

> 我们没有发现明确的证据表明超过两层的编码器深度是必要的。

**建议**：使用 1 层双向编码器并扩展到 2 个双向层，以提高技能。

解码器出现了类似的故事。具有 1,2 和 4 层的解码器之间的技能在 4 层解码器略微更好的情况下有所不同。 8 层解码器在测试条件下没有收敛。

> 在解码器方面，较深的模型以较小的幅度优于较浅的模型。

**建议**：使用 1 层解码器作为起点，并使用 4 层解码器以获得更好的结果。

## 编码器输入方向

源文本序列的顺序可以通过多种方式提供给编码器：

*   前进或正常。
*   逆转。
*   前进和后退都在同一时间。

作者探讨了输入序列顺序对模型技能的影响，比较了各种单向和双向配置。

通常，他们证实了先前的发现，即反向序列优于正向序列，并且双向略好于反向序列。

> ...双向编码器通常优于单向编码器，但不是很大。具有反向源的编码器始终优于其未反转的对应物。

**建议**：使用反向顺序输入序列或移动到双向以获得模型技能的小升力。

## 注意机制

朴素的编解码器模型的问题在于编码器将输入映射到固定长度的内部表示，解码器必须从该表示产生整个输出序列。

注意是对模型的改进，其允许解码器在输出序列中输出每个字时“注意”输入序列中的不同字。

作者研究了简单注意机制的一些变化。结果表明，注意力会比没有注意力的情况下产生明显更好的表现。

> 虽然我们确实期望基于注意力的模型明显优于没有注意机制的模型，但我们对[无注意]模型的表现有多么惊讶。

Bahdanau 等人描述的简单加权平均风格注意。在他们的 2015 年论文中，“[神经机器翻译通过联合学习对齐和翻译](https://arxiv.org/abs/1409.0473)”被发现表现最佳。

**推荐**：使用注意力并更喜欢 Bahdanau 式加权平均风格的关注。

## 推理

在神经机器翻译系统中通常使用集束搜索来对模型输出的序列中的单词的概率进行采样。

光束宽度越宽，搜索越详尽，并且据信，结果越好。

结果表明，3-5 的适度集束宽度表现最佳，通过使用长度惩罚可以非常轻微地改善。作者通常建议在每个特定问题上调整集束宽度。

> 我们发现，良好调谐的光束搜索对于获得良好的结果至关重要，并且它可以导致多个 BLEU 点的一致增益。

**推荐**：从贪婪搜索开始（集束= 1）并根据您的问题进行调整。

## 最终模型

作者将他们的研究结果汇总到一个“最佳模型”中，并将该模型的结果与其他表现良好的模型和最先进的结果进行比较。

该模型的具体配置总结在下表中，摘自论文。在为 NLP 应用程序开发自己的编解码器模型时，这些参数可以作为一个好的或最好的起点。

![Summary of Model Configuration for the Final NMT Model](img/c508993d3836a302efe11a3c00a0861d.jpg)

最终 NMT 模型的模型配置总结
摘自“神经机器翻译架构的大规模探索”。

该系统的结果显示令人印象深刻，并且通过更简单的模型获得了接近最新技术的技能，这不是本文的目标。

> ...我们确实表明，通过仔细的超参数调整和良好的初始化，可以在标准 WMT 基准测试中实现最先进的表现

重要的是，作者将所有代码作为一个名为 [tf-seq2seq](https://github.com/google/seq2seq) 的开源项目提供。由于其中两位作者是 Google Brain 驻留计划的成员，他们的工作在 Google Research 博客上公布，标题为“[介绍 tf-seq2seq：TensorFlow 中的开源序列到序列框架](https://research.googleblog.com/2017/04/introducing-tf-seq2seq-open-source.html)” ，2017 年。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

*   [神经机器翻译架构的大规模探索](https://arxiv.org/abs/1703.03906)，2017。
*   [Denny Britz 主页](http://blog.dennybritz.com/)
*   [WildML 博客](http://www.wildml.com/)
*   [介绍 tf-seq2seq：TensorFlow 中的开源序列到序列框架](https://research.googleblog.com/2017/04/introducing-tf-seq2seq-open-source.html)，2017。
*   [tf-seq2seq：Tensorflow](https://github.com/google/seq2seq) 的通用编解码器框架
*   [tf-seq2seq 项目文件](https://google.github.io/seq2seq/)
*   [tf-seq2seq 教程：神经机器翻译背景](https://google.github.io/seq2seq/nmt/)
*   [神经机器翻译通过联合学习调整和翻译](https://arxiv.org/abs/1409.0473)，2015。

## 摘要

在这篇文章中，您了解了如何最好地配置编解码器循环神经网络，用于神经机器翻译和其他自然语言处理任务。

具体来说，你学到了：

*   Google 研究调查了编解码器模型中的每个模型设计决策，以隔离它们的影响。
*   设计决策的结果和建议，如字嵌入，编码器和解码器深度以及注意机制。
*   一组基本模型设计决策，可用作您自己序列的起点，以对项目进行排序。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。