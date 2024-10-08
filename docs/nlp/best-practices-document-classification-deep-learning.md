# 深度学习文档分类的最佳实践

> 原文： [`machinelearningmastery.com/best-practices-document-classification-deep-learning/`](https://machinelearningmastery.com/best-practices-document-classification-deep-learning/)

文本分类描述了一类问题，例如预测推文和电影评论的情感，以及将电子邮件分类为垃圾邮件。

深度学习方法在文本分类方面证明非常好，在一系列标准学术基准问题上实现了最先进的结果。

在这篇文章中，您将发现在开发用于文本分类的深度学习模型时要考虑的一些最佳实践。

阅读这篇文章后，你会知道：

*   深度学习方法的一般组合，在开始文本分类问题时要考虑。
*   第一个尝试提供有关如何配置超参数的具体建议的架构。
*   在灵活性和能力方面，更深层次的网络可能是该领域的未来。

让我们开始吧。

![Best Practices for Document Classification with Deep Learning](img/a27fd8667f8f49d17583292a3c5e62ab.jpg)

深度学习文档分类的最佳实践
[storebukkebruse](https://www.flickr.com/photos/tusnelda/6140792529/) 的照片，保留一些权利。

## 概观

本教程分为 5 个部分;他们是：

1.  词嵌入+ CNN =文本分类
2.  使用单层 CNN 架构
3.  拨入 CNN 超参数
4.  考虑字符级 CNN
5.  考虑更深入的 CNN 进行分类

## 1\. 词嵌入+ CNN =文本分类

文本分类的操作方法涉及使用单词嵌入来表示单词和卷积神经网络（CNN）来学习如何区分分类问题的文档。

Yoav Goldberg 在自然语言处理深度学习的入门读物中评论说，神经网络通常提供比经典线性分类器更好的表现，特别是与预训练的字嵌入一起使用时。

> 网络的非线性以及容易集成预训练的字嵌入的能力通常会带来更高的分类准确率。

- [自然语言处理神经网络模型入门](https://arxiv.org/abs/1510.00726)，2015。

他还评论说卷积神经网络在文档分类方面是有效的，即因为它们能够以对输入序列中的位置不变的方式选择显着特征（例如令牌或令牌序列）。

> 具有卷积和池化层的网络对于分类任务很有用，在这些分类任务中我们期望找到关于类成员资格的强大的局部线索，但是这些线索可以出现在输入中的不同位置。 [...]我们希望了解某些单词序列是该主题的良好指标，并不一定关心它们在文档中的出现位置。卷积层和汇集层允许模型学习如何找到这样的本地指标，而不管它们的位置如何。

- [自然语言处理神经网络模型入门](https://arxiv.org/abs/1510.00726)，2015。

因此，该架构由三个关键部分组成：

1.  **单词嵌入**：单词的分布式表示，其中具有相似含义的不同单词（基于其用法）也具有相似的表示。
2.  **卷积模型**：一种特征提取模型，用于学习从使用单词嵌入表示的文档中提取显着特征。
3.  **完全连接模型**：根据预测输出解释提取的特征。

Yoav Goldberg 在他的书中强调了 CNN 作为特征提取器模型的作用：

> ... CNN 本质上是一种特征提取架构。它本身并不构成一个独立的，有用的网络，而是意味着要集成到一个更大的网络中，并接受训练以与其协同工作以产生最终结果。 CNNs 层的职责是提取对手头的整体预测任务有用的有意义的子结构。

- 第 152 页，[自然语言处理的神经网络方法](http://amzn.to/2vRopQz)，2017。

将这三个要素结合在一起，可能是下一节中描述的最广泛引用的组合实例之一。

## 2.使用单层 CNN 架构

使用单层 CNN 可以获得用于文档分类的良好结果，可能在过滤器上具有不同大小的内核，以允许以不同比例对单词表示进行分组。

Yoon Kim 在研究使用预训练的单词向量进行卷积神经网络分类任务时发现，使用预先训练的静态单词向量非常有效。他建议在非常大的文本语料库中训练的预训练的单词嵌入，例如从谷歌新闻中训练的 1000 亿个令牌的免费 word2vec 向量，可以提供用于自然语言处理的良好通用特征。

> 尽管对超参数进行了很少的调整，但是具有一层卷积的简单 CNN 表现得非常好。我们的结果增加了公认的证据，即无监督的单词向量训练是 NLP 深度学习的重要组成部分

- [用于句子分类的卷积神经网络](https://arxiv.org/abs/1408.5882)，2014。

他还发现，对单词向量进行进一步的任务特定调整可以提供额外的表现改进。

Kim 描述了使用 CNN 进行自然语言处理的一般方法。句子被映射到嵌入向量，并且可用作模型的矩阵输入。使用不同大小的内核（例如，一次 2 或 3 个字）逐字输入地执行卷积。然后使用最大池层处理所得到的特征图以压缩或汇总所提取的特征。

该架构基于 Ronan Collobert 等人使用的方法。在他们的论文“[自然语言处理（几乎）来自 Scratch](https://arxiv.org/abs/1103.0398) ”，2011 年。在其中，他们开发了一个单一的端到端神经网络模型，具有卷积和汇集层，可用于一系列基本自然语言处理问题。

Kim 提供了一个图表，有助于查看使用不同大小的内核作为不同颜色（红色和黄色）的过滤器的采样。

![Example of a CNN Filter and Polling Architecture for Natural Language Processing](img/d4ad4d52b98d4d58a038dfe86c2651bc.jpg)

用于自然语言处理的 CNN 过滤器和轮询架构的示例。
取自“用于句子分类的卷积神经网络”，2014 年。

有用的是，他报告了他所选择的模型配置，通过网格搜索发现并在一系列 7 个文本分类任务中使用，总结如下：

*   传递函数：整流线性。
*   内核大小：2,4,5。
*   过滤器数量：100
*   dropout 率：0.5
*   权重正则化（L2）：3
*   批量大小：50
*   更新规则：Adadelta

这些配置可用于激发您自己实验的起点。

## 3.拨入 CNN 超参数

在对文档分类问题调整卷积神经网络时，一些超参数比其他参数更重要。

Ye Zhang 和 Byron Wallace 对配置单层卷积神经网络进行文档分类所需的超参数进行了灵敏度分析。该研究的动机是他们声称模型对其配置很敏感。

> 不幸的是，基于 CNN 的模型（即使是简单的模型）的缺点是它们需要从业者指定要使用的确切模型架构并设置伴随的超参数。对于初学者来说，做出这样的决定看起来像黑色艺术，因为模型中有许多自由参数。

- [用于句子分类的卷积神经网络（和从业者指南）的灵敏度分析](https://arxiv.org/abs/1510.03820)，2015 年。

他们的目标是提供可用于在新文本分类任务上配置 CNN 的常规配置。

它们提供了对模型架构的良好描述以及用于配置模型的决策点，如下所示。

![Convolutional Neural Network Architecture for Sentence Classification](img/22ce3c817c06500a10b4356c1da2db68.jpg)

用于句子分类的卷积神经网络结构
取自“_ 用于句子分类的卷积神经网络（和从业者指南）的灵敏度分析 _”，2015。

该研究提供了许多有用的发现，可以作为配置浅 CNN 模型进行文本分类的起点。

一般调查结果如下：

*   预训练的 word2vec 和 GloVe 嵌入的选择因问题而异，并且两者都比使用单热编码的单词向量表现更好。
*   内核的大小很重要，应针对每个问题进行调整。
*   要素图的数量也很重要，应该进行调整。
*   1-max 池通常优于其他类型的池。
*   Dropout 对模型表现影响不大。

他们继续提供更具体的启发式方法，如下所示：

*   使用 word2vec 或 GloVe 字嵌入作为起点，并在拟合模型时进行调整。
*   跨不同内核大小的网格搜索，以找到问题的最佳配置，范围为 1-10。
*   搜索 100-600 的过滤器数量，并在同一搜索中探索 0.0-0.5 的丢失。
*   使用 tanh，relu 和线性激活功能进行探索。

关键的警告是，调查结果是基于使用单个句子作为输入的二元文本分类问题的实证结果。

我建议阅读全文以获取更多详细信息：

*   [用于句子分类的卷积神经网络（和从业者指南）的灵敏度分析](https://arxiv.org/abs/1510.03820)，2015 年。

## 4.考虑字符级 CNN

文本文档可以使用卷积神经网络在角色级别建模，该卷积神经网络能够学习单词，句子，段落等的相关层次结构。

张翔，等。使用基于字符的文本表示作为卷积神经网络的输入。该方法的承诺是，如果 CNN 可以学习抽象显着的细节，那么清理和准备文本所需的所有劳动密集型工作都可以克服。

> ......除了先前研究的结论之外，ConvNets 不需要语言的语法或语义结构，因此深入的 ConvNets 不需要语言知识。对于可以用于不同语言的单个系统而言，这种工程简化可能是至关重要的，因为无论是否可以分割成单词，字符总是构成必要的构造。仅处理字符还具有以下优点：可以自然地学习诸如拼写错误和表情符号之类的异常字符组合。

- 用于文本分类的[字符级卷积网络](https://arxiv.org/abs/1509.01626)，2015 年。

该模型以固定大小的字母表读入单热门的编码字符。编码字符以块或 1,024 个字符的序列读取。随后是具有池化的 6 个卷积层的栈，在网络的输出端具有 3 个完全连接的层以便做出预测。

![Character-based Convolutional Neural Network for Text Classification](img/c4866f7ed8a5fff365f7c34641a4bd14.jpg)

基于字符的文本分类卷积神经网络
取自“_ 用于文本分类的字符级卷积网络 _”，2015。

该模型取得了一些成功，在提供更大文本语料的问题上表现更好。

> ...分析表明，人物级别的 ConvNet 是一种有效的方法。 [...]我们的模型在比较中的表现有多好取决于许多因素，例如数据集大小，文本是否经过策划以及字母表的选择。

- 用于文本分类的[字符级卷积网络](https://arxiv.org/abs/1509.01626)，2015 年。

使用这种方法的扩展版本的结果在下一节中介绍的后续文章中被推到了最新技术水平。

## 5.考虑更深入的 CNN 进行分类

尽管标准和可重用的架构尚未被用于分类任务，但是使用非常深的卷积神经网络可以实现更好的表现。

Alexis Conneau，et al。评论用于自然语言处理的相对较浅的网络以及用于计算机视觉应用的更深层网络的成功。例如，Kim（上文）将模型限制为单个卷积层。

本文中使用的用于自然语言的其他架构限于 5 层和 6 层。这与用于计算机视觉的成功架构形成对比，具有 19 层甚至 152 层。

他们建议并证明使用非常深的卷积神经网络模型（称为 VDCNN）对分层特征学习有益。

> ...我们建议使用多个卷积层的深层架构来实现这一目标，最多使用 29 层。我们的架构设计受到最近计算机视觉进展的启发[...]所提出的深度卷积网络显示出比以前的 ConvNets 方法更好的结果。

他们的方法的关键是嵌入单个字符，而不是单词嵌入。

> 我们提出了一种用于文本处理的新架构（VDCNN），它直接在字符级别操作，并且仅使用小卷积和池化操作。

- [用于文本分类的非常深的卷积网络](https://arxiv.org/abs/1509.01626)，2016。

一套 8 个大型文本分类任务的结果显示出比更浅层网络更好的表现。具体而言，在撰写本文时，除了两个测试数据集之外的所有数据集都具有最先进的结果。

通常，他们通过探索更深层的架构方法得出一些重要发现：

*   非常深的架构在小型和大型数据集上运行良好。
*   更深的网络减少了分类错误。
*   与其他更复杂的池化类型相比，最大池化可获得更好的结果。
*   通常更深入会降低准确率;架构中使用的快捷方式连接很重要。

> ......这是第一次在 NLP 中显示卷积神经网络的“深度效益”。

- [用于文本分类的非常深的卷积网络](https://arxiv.org/abs/1606.01781)，2016。

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

*   [自然语言处理神经网络模型入门](https://arxiv.org/abs/1510.00726)，2015。
*   [用于句子分类的卷积神经网络](https://arxiv.org/abs/1103.0398)，2014。
*   [自然语言处理（几乎）来自 Scratch](https://arxiv.org/abs/1103.0398) ，2011。
*   [用于文本分类的非常深的卷积网络](https://arxiv.org/abs/1606.01781)，2016。
*   [用于文本分类的字符级卷积网络](https://arxiv.org/abs/1509.01626)，2015 年。
*   [用于句子分类的卷积神经网络（和从业者指南）的灵敏度分析](https://arxiv.org/abs/1510.03820)，2015 年。

您是否在文档分类的深度学习方面遇到了一些很好的资源？
请在下面的评论中告诉我。

## 摘要

在这篇文章中，您发现了一些用于开发文档分类深度学习模型的最佳实践。

具体来说，你学到了：

*   一个关键的方法是使用字嵌入和卷积神经网络进行文本分类。
*   单层模型可以很好地解决中等规模的问题，以及如何配置它的想法。
*   直接在文本上运行的更深层模型可能是自然语言处理的未来。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。