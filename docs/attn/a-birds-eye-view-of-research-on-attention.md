# 对注意力研究的总体概述

> 原文：[`machinelearningmastery.com/a-birds-eye-view-of-research-on-attention/`](https://machinelearningmastery.com/a-birds-eye-view-of-research-on-attention/)

注意力是一个在多个学科中科学研究的概念，包括心理学、神经科学，以及最近的机器学习。虽然各个学科可能对注意力有不同的定义，但它们都一致认为，注意力是使生物和人工神经系统更具灵活性的机制。

在本教程中，你将发现关于注意力研究进展的概述。

完成本教程后，你将了解到：

+   对不同科学学科具有重要意义的注意力概念

+   注意力如何在机器学习中引发革命，特别是在自然语言处理和计算机视觉领域

**启动你的项目**，请参阅我的书籍 [《使用注意力构建变换器模型》](https://machinelearningmastery.com/transformer-models-with-attention/)。它提供了**自学教程**和**工作代码**，指导你构建一个完全可用的变换器模型。

*将句子从一种语言翻译成另一种语言*...

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_cover-scaled.jpg)

对注意力研究的总体概述

图片由 [Chris Lawton](https://unsplash.com/photos/6tfO1M8_gas) 提供，部分权利保留。

## **教程概述**

本教程分为两个部分，分别是：

+   注意力的概念

+   机器学习中的注意力

    +   自然语言处理中的注意力

    +   计算机视觉中的注意力

## **注意力的概念**

对注意力的研究源于心理学领域。

> *对注意力的科学研究始于心理学，通过细致的行为实验可以精准展示注意力在不同情况下的倾向和能力。*
> 
> *–* [心理学、神经科学和机器学习中的注意力](https://www.frontiersin.org/articles/10.3389/fncom.2020.00029/full)，2020 年。

从这些研究中得出的观察结果可以帮助研究人员推断出这些行为模式背后的心理过程。

尽管心理学、神经科学以及最近的机器学习领域都对注意力有各自的定义，但有一个核心特质对所有领域都具有重要意义：

> *注意力是对有限计算资源的灵活控制。*
> 
> *–* [心理学、神经科学和机器学习中的注意力](https://www.frontiersin.org/articles/10.3389/fncom.2020.00029/full)，2020 年。

鉴于此，接下来的部分将回顾注意力在引领机器学习领域革命中的角色。

## **机器学习中的注意力**

机器学习中的注意力概念*非常*松散地受到人脑注意力心理机制的启发。

> *注意力机制在人工神经网络中的使用出现了——就像大脑中对注意力的明显需求一样——作为使神经系统更加灵活的一种手段。*
> 
> *–* [心理学、神经科学与机器学习中的注意力](https://www.frontiersin.org/articles/10.3389/fncom.2020.00029/full)，2020。

这个想法是能够处理一种能够在输入可能具有不同长度、大小或结构，甚至处理几个不同任务的人工神经网络。正是在这种精神下，机器学习中的注意力机制被认为是从心理学中获得灵感的，而不是因为它们复制了人脑的生物学。

> *在最初为人工神经网络（ANNs）开发的注意力形式中，注意力机制在编码器-解码器框架和序列模型的背景下工作……*
> 
> *–* [心理学、神经科学与机器学习中的注意力](https://www.frontiersin.org/articles/10.3389/fncom.2020.00029/full)，2020。

[编码器](https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/)的任务是生成输入的向量表示，而[解码器](https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/)的任务是将这个向量表示转换为输出。注意力机制将二者连接起来。

已经有不同的神经网络架构提议实现注意力机制，这些架构也与其应用的特定领域相关。自然语言处理（NLP）和计算机视觉是最受欢迎的应用之一。

### **自然语言处理中的注意力**

在自然语言处理（NLP）中，早期的注意力应用是机器翻译，其目标是将源语言中的输入句子翻译为目标语言中的输出句子。在这个背景下，编码器会生成一组*上下文*向量，每个词一个。解码器则读取这些上下文向量，以逐字生成目标语言中的输出句子。

> *在没有注意力的传统编码器-解码器框架中，编码器生成一个固定长度的向量，该向量与输入的长度或特征无关，并且在解码过程中保持静态。*
> 
> *–* [心理学、神经科学与机器学习中的注意力](https://www.frontiersin.org/articles/10.3389/fncom.2020.00029/full)，2020。

使用固定长度向量表示输入在处理长序列或结构复杂的序列时尤为棘手，因为这些序列的表示维度必须与较短或较简单序列的表示维度相同。

> *例如，在某些语言中，如日语，最后一个词可能对预测第一个词非常重要，而将英语翻译成法语可能更容易，因为句子的顺序（句子的组织方式）更相似。*
> 
> *–* [心理学、神经科学和机器学习中的注意力](https://www.frontiersin.org/articles/10.3389/fncom.2020.00029/full)，2020 年。

这造成了一个瓶颈，即解码器对由输入提供的信息的访问受到限制，即在固定长度编码向量内可用的信息。另一方面，在编码过程中保持输入序列的长度不变可以使解码器能够灵活地利用其最相关的部分。

注意机制是如何运作的。

> *注意帮助确定应该使用这些向量中的哪一个来生成输出。由于输出序列是逐个元素动态生成的，注意力可以在每个时间点动态突出显示不同的编码向量。这使得解码器能够灵活地利用输入序列中最相关的部分。*
> 
> – 第 186 页，[深度学习基础](https://www.amazon.com/Deep-Learning-Essentials-hands-fundamentals/dp/1785880365)，2018 年。

在早期的机器翻译工作中，试图解决由固定长度向量引起的瓶颈问题的工作之一是由[Bahdanau et al. (2014)](https://arxiv.org/abs/1409.0473)完成的。在他们的工作中，Bahdanau 等人使用递归神经网络（RNNs）进行编码和解码任务：编码器采用双向 RNN 生成一系列*注释*，每个注释包含前后单词的摘要，可以通过加权和映射到*上下文*向量；解码器然后基于这些注释和另一个 RNN 的隐藏状态生成输出。由于上下文向量是通过注释的加权和计算得到的，因此 Bahdanau 等人的注意机制是[*软注意*](https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/)的一个例子。

另一项早期工作是由[Sutskever et al. (2014)](https://arxiv.org/abs/1409.3215)完成的。他们选择使用多层长短期记忆（LSTM）来编码表示输入序列的向量，并使用另一个 LSTM 来将该向量解码为目标序列。

[Luong 等人（2015）](https://arxiv.org/abs/1508.04025) 引入了*全局*与*局部*注意力的概念。在他们的工作中，他们将全局注意力模型描述为在推导上下文向量时考虑编码器的所有隐藏状态。因此，全局上下文向量的计算基于*所有*源序列中的词的加权平均。Luong 等人提到，这在计算上是昂贵的，并且可能使全局注意力难以应用于长序列。局部注意力被提出以解决这个问题，通过专注于每个目标词的源序列中的较小子集。Luong 等人解释说，局部注意力在计算上比软注意力更便宜，但比硬注意力更易于训练。

更近期，[Vaswani 等人（2017）](https://arxiv.org/abs/1706.03762)提出了一种完全不同的架构，已经引导了机器翻译领域的一个新方向。这个被称为*Transformer*的架构完全舍弃了递归和卷积，但实现了*自注意力*机制。源序列中的词首先被并行编码以生成键、查询和值表示。键和值被组合以生成注意力权重，从而捕捉每个词与序列中其他词的关系。这些注意力权重随后用于缩放值，以便保持对重要词的关注，并消除无关的词。

> *输出是通过对值的加权求和来计算的，其中分配给每个值的权重是通过查询与相应键的兼容性函数计算的。*
> 
> – [《Attention Is All You Need》](https://arxiv.org/pdf/1706.03762.pdf)，2017 年。

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png)

Transformer 架构

摘自《Attention Is All You Need》

当时，提出的 Transformer 架构为英语到德语和英语到法语的翻译任务建立了新的最先进的过程。报告称，它的训练速度也比基于递归或卷积层的架构更快。随后，由[Devlin 等人（2019）](https://arxiv.org/abs/1810.04805)提出的方法 BERT 基于 Vaswani 等人的工作，提出了一个多层双向架构。

如我们很快将看到的那样，Transformer 架构的接受不仅在 NLP 领域迅速增长，而且在计算机视觉领域也迅速扩展。

### **计算机视觉中的注意力**

在计算机视觉中，注意力机制已经在多个应用领域找到了它的位置，例如在图像分类、图像分割和图像描述领域。

例如，如果我们需要将编码器-解码器模型重新构建用于图像描述任务，那么编码器可以是一个卷积神经网络（CNN），它将图像中的显著视觉线索转化为向量表示。解码器则可以是一个 RNN 或 LSTM，将向量表示转换为输出。

> *此外，正如神经科学文献中所述，这些注意力过程可以分为空间注意力和基于特征的注意力。*
> 
> *–* [心理学、神经科学和机器学习中的注意力](https://www.frontiersin.org/articles/10.3389/fncom.2020.00029/full)，2020。

在*空间*注意力中，不同的空间位置被赋予不同的权重。然而，这些相同的权重在不同的空间位置的所有特征通道中保持不变。

一种基于空间注意力的基本图像描述方法由[Xu et al. (2016)](https://arxiv.org/abs/1502.03044)提出。他们的模型将 CNN 作为编码器，提取一组特征向量（或*注释*向量），每个向量对应于图像的不同部分，以便解码器能够有选择地关注特定的图像部分。解码器是一个 LSTM，根据上下文向量、先前的隐藏状态和先前生成的单词生成描述。Xu et al.研究了将[*硬注意力*](https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/)作为计算其上下文向量的[软注意力](https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/)的替代方法。在这里，软注意力在源图像的所有区域上*柔和地*施加权重，而硬注意力则只关注单个区域，同时忽略其余部分。他们报告说，在他们的工作中，硬注意力表现更好。

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_2.png)

图像描述生成模型

摘自《展示、注意和讲述：带有视觉注意力的神经图像描述生成》

相较之下，*特征*注意力允许各个特征图赋予自身的权重值。一个这样的例子，亦应用于图像描述，是[Chen et al. (2018)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_SCA-CNN_Spatial_and_CVPR_2017_paper.pdf)的编码器-解码器框架，它在同一个 CNN 中结合了空间和通道注意力。

与 Transformer 迅速成为 NLP 任务的标准架构类似，它最近也被计算机视觉领域采纳并加以改编。

最早这样做的工作是由 [Dosovitskiy 等人 (2020)](https://arxiv.org/abs/2010.11929) 提出的，他们将 *Vision Transformer* (ViT) 应用于图像分类任务。他们认为，长期以来对 CNN 的依赖并不是必要的，纯变换器也可以完成同样的任务。Dosovitskiy 等人将输入图像重塑为一系列展平的 2D 图像补丁，然后通过可训练的线性投影将其嵌入，以生成 *补丁嵌入*。这些补丁嵌入与 *位置嵌入* 一起，以保留位置信息，被输入到变换器架构的编码器部分，编码器的输出随后被输入到多层感知机 (MLP) 进行分类。

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_3.png)

Vision Transformer 架构

摘自《An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale》

> *受 ViT 启发，并且基于注意力的架构在建模视频中的长程上下文关系时直观有效，我们开发了几个基于变换器的视频分类模型。*
> 
> – [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691)，2021 年。

[Arnab 等人 (2021)](https://arxiv.org/abs/2103.15691) 随后将 ViT 模型扩展为 ViViT，该模型利用视频中的时空信息进行视频分类任务。他们的方法探索了提取时空数据的不同方法，例如通过独立采样和嵌入每一帧，或提取不重叠的管段（一个跨越多个图像帧的图像补丁，形成一个 *管道*）并逐一嵌入。他们还研究了对输入视频的空间和时间维度进行分解的不同方法，以提高效率和可扩展性。

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_4.png)

视频视觉变换器架构

摘自《ViViT: A Video Vision Transformer》

正确性 · Re

在首次应用于图像分类的情况下，Vision Transformer 已经被应用于多个其他计算机视觉领域，如 [动作定位](https://arxiv.org/abs/2106.08061)、[注视估计](https://arxiv.org/abs/2105.14424) 和 [图像生成](https://arxiv.org/abs/2107.04589)。这种计算机视觉从业者的兴趣激增，预示着一个激动人心的近未来，我们将看到更多对变换器架构的适应和应用。

### 想开始构建带有注意力机制的变换器模型吗？

现在就参加我的免费 12 天电子邮件速成课程（包含示例代码）。

点击注册，并获取课程的免费 PDF 电子书版本。

## **进一步阅读**

本节提供了更多关于该主题的资源，如果你想深入了解。

### **书籍**

+   [深度学习要点](https://www.amazon.com/Deep-Learning-Essentials-hands-fundamentals/dp/1785880365)，2018 年。

### **论文**

+   [心理学、神经科学和机器学习中的注意力](https://www.frontiersin.org/articles/10.3389/fncom.2020.00029/full)，2020 年。

+   [通过联合学习对齐和翻译的神经机器翻译](https://arxiv.org/abs/1409.0473)，2014 年。

+   [序列到序列学习与神经网络](https://arxiv.org/abs/1409.3215)，2014 年。

+   [基于注意力的神经机器翻译的有效方法](https://arxiv.org/abs/1508.04025)，2015 年。

+   [注意力机制是你所需的一切](https://arxiv.org/abs/1706.03762)，2017 年。

+   [BERT：深度双向变换器的预训练用于语言理解](https://arxiv.org/abs/1810.04805)，2019 年。

+   [展示、关注和讲述：使用视觉注意力的神经图像描述生成](https://arxiv.org/abs/1502.03044)，2016 年。

+   [SCA-CNN：用于图像描述的卷积网络中的空间和通道注意力](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_SCA-CNN_Spatial_and_CVPR_2017_paper.pdf)，2018 年。

+   [一张图像值 16×16 个词：用于大规模图像识别的变换器](https://arxiv.org/abs/2010.11929)，2020 年。

+   [ViViT：视频视觉变换器](https://arxiv.org/abs/2103.15691)，2021 年。

**示例应用：**

+   [时空动作定位中的关系建模](https://arxiv.org/abs/2106.08061)，2021 年。

+   [使用变换器的注视估计](https://arxiv.org/abs/2105.14424)，2021 年。

+   [ViTGAN：使用视觉变换器训练 GANs](https://arxiv.org/abs/2107.04589)，2021 年。

## **总结**

在本教程中，你了解了关于注意力的研究进展概述。

具体来说，你学到了：

+   注意力的概念对不同科学学科的重要性

+   注意力如何在机器学习中引发革命，特别是在自然语言处理和计算机视觉领域

你有什么问题吗？

请在下方评论中提出你的问题，我会尽力回答。
