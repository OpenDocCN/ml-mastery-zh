# 视觉 Transformer 模型

> 原文：[`machinelearningmastery.com/the-vision-transformer-model/`](https://machinelearningmastery.com/the-vision-transformer-model/)

随着 Transformer 架构在自然语言处理领域实现了令人鼓舞的结果，计算机视觉领域的应用也只是时间问题。这最终通过视觉 Transformer（ViT）的实现得以实现。

在本教程中，您将发现视觉 Transformer 模型的架构，以及它在图像分类任务中的应用。

完成本教程后，您将了解：

+   ViT 在图像分类中的工作原理。

+   ViT 的训练过程。

+   ViT 与卷积神经网络在归纳偏置方面的比较。

+   ViT 在不同数据集上与 ResNets 的比较表现如何。

+   ViT 如何在内部处理数据以实现其性能。

**启动您的项目**，可以参考我的书籍 [《构建注意力的 Transformer 模型》](https://machinelearningmastery.com/transformer-models-with-attention/)。它提供了**自学教程**和**工作代码**，帮助您构建一个完全可用的 Transformer 模型。

*将句子从一种语言翻译成另一种语言*……

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2022/02/vit_cover-scaled.jpg)

视觉 Transformer 模型

图片由 [Paul Skorupskas](https://unsplash.com/photos/7KLa-xLbSXA) 提供，部分权利保留。

## **教程概述**

本教程分为六个部分，它们是：

+   视觉 Transformer（ViT）简介

+   ViT 架构

+   训练 ViT

+   与卷积神经网络相比的归纳偏置

+   ViT 变体与 ResNets 的比较性能

+   数据的内部表示

## **前提条件**

对于本教程，我们假设您已经熟悉：

+   [注意力的概念](https://machinelearningmastery.com/what-is-attention/)

+   [Transformer 注意力机制](https://machinelearningmastery.com/the-transformer-attention-mechanism)

+   [Transformer 模型](https://machinelearningmastery.com/the-transformer-model/)

## **视觉 Transformer（ViT）简介**

我们已经看到，[Vaswani 等人（2017）](https://arxiv.org/abs/1706.03762)的 Transformer 架构如何革新了注意力的使用，避免了依赖于递归和卷积的早期注意力模型。在他们的工作中，Vaswani 等人将他们的模型应用于自然语言处理（NLP）的具体问题。

> *然而，在计算机视觉中，卷积架构仍然占据主导地位……*
> 
> *–* [图像胜过 16×16 个词：用于大规模图像识别的 Transformers](https://arxiv.org/abs/2010.11929)，2021 年。

受到在自然语言处理中的成功启发，Dosovitskiy 等人（2021 年）试图将标准 Transformer 架构应用于图像，我们很快将看到。他们当时的目标应用是图像分类。

### 想要开始构建带注意力的 Transformer 模型吗？

现在就参加我的免费 12 天电子邮件速成课程（附有示例代码）。

点击注册，还可获得免费 PDF 电子书版本的课程。

## **ViT 架构**

请记住，标准 Transformer 模型接收一维序列的单词嵌入作为输入，因为它最初是为自然语言处理设计的。相反，当应用于计算机视觉中的图像分类任务时，Transformer 模型的输入数据以二维图像的形式提供。

为了以类似自然语言处理（NLP）领域中单词序列的方式结构化输入图像数据（意味着有一个单词序列的序列），输入图像的高度为$H$，宽度为$W$，有$C$个通道，被切割成更小的二维补丁。这导致产生$N = \tfrac{HW}{P²}$个补丁，每个补丁的分辨率为($P, P$)像素。

在将数据馈送到 Transformer 之前，执行以下操作：

+   每个图像补丁被扁平化为长度为$P² \times C$的向量$\mathbf{x}_p^n$，其中$n = 1, \dots N$。

+   通过可训练的线性投影$\mathbf{E}$，将扁平化的补丁映射到$D$维度，生成嵌入的图像补丁序列。

+   一个可学习的类别嵌入$\mathbf{x}_{\text{class}}$被前置到嵌入图像补丁序列中。$\mathbf{x}_{\text{class}}$的值代表分类输出$\mathbf{y}$。

+   最终，补丁嵌入向量最终与一维位置嵌入$\mathbf{E}_{\text{pos}}$相结合，从而将位置信息引入输入中，该信息在训练期间也被学习。

由上述操作产生的嵌入向量序列如下：

$$\mathbf{z}_0 = [ \mathbf{x}_{\text{class}}; \; \mathbf{x}_p¹ \mathbf{E}; \; \dots ; \; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{\text{pos}}$$

Dosovitskiy 等人利用了 Vaswani 等人 Transformer 架构的编码器部分。

为了进行分类，他们在 Transformer 编码器的输入处输入$\mathbf{z}_0$，该编码器由$L$个相同的层堆叠而成。然后，他们继续从编码器输出的第$L^{\text{th}}$层取$\mathbf{x}_{\text{class}}$的值，并将其馈送到分类头部。

> *在预训练阶段，分类头部由具有一个隐藏层的 MLP 实现，在微调阶段则由单一线性层实现。*
> 
> *–* [图像价值 16×16 字：大规模图像识别中的 Transformer](https://arxiv.org/abs/2010.11929), 2021.

形成分类头的多层感知机（MLP）实现了高斯误差线性单元（GELU）非线性。

总结来说，ViT 使用了原始 Transformer 架构的编码器部分。编码器的输入是一个嵌入图像块的序列（包括一个附加到序列前面的可学习类别嵌入），并且还增加了位置位置信息。附加在编码器输出上的分类头接收可学习类别嵌入的值，以生成基于其状态的分类输出。所有这些都在下图中进行了说明：

![](https://machinelearningmastery.com/wp-content/uploads/2022/02/vit_1.png)

视觉变换器（ViT）的架构

摘自 “[一张图片值 16×16 个词：用于大规模图像识别的变换器](https://arxiv.org/abs/2010.11929)”

Dosovitskiy 等人提到的另一个注意事项是，原始图像也可以在传递给 Transformer 编码器之前先输入到卷积神经网络（CNN）中。图像块序列将从 CNN 的特征图中获得，而后续的特征图块嵌入、添加类别标记和增加位置位置信息的过程保持不变。

## **训练 ViT**

ViT 在更大的数据集上进行预训练（如 ImageNet、ImageNet-21k 和 JFT-300M），然后对较少的类别进行微调。

在预训练过程中，附加在编码器输出上的分类头由一个具有一个隐藏层和 GELU 非线性函数的 MLP 实现，如前所述。

在微调过程中，MLP 被替换为一个大小为 $D \times K$ 的单层（零初始化）前馈层，其中 $K$ 表示与当前任务对应的类别数量。

微调是在比预训练时使用的图像分辨率更高的图像上进行的，但输入图像被切割成的块大小在训练的所有阶段保持不变。这导致在微调阶段的输入序列长度比预训练阶段使用的更长。

输入序列长度更长的含义是，微调需要比预训练更多的位置嵌入。为了解决这个问题，Dosovitskiy 等人通过在二维上插值预训练位置嵌入，根据它们在原始图像中的位置，得到一个与微调过程中使用的图像块数量相匹配的更长序列。

## **与卷积神经网络的归纳偏差比较**

归纳偏差指的是模型为泛化训练数据和学习目标函数所做的任何假设。

> *在 CNN 中，本地性、二维邻域结构和平移等变性被嵌入到模型的每一层中。*
> 
> *–* [图像的价值相当于 16×16 个词：用于大规模图像识别的 Transformers](https://arxiv.org/abs/2010.11929)，2021。

在卷积神经网络（CNNs）中，每个神经元仅与其邻域内的其他神经元连接。此外，由于同一层上的神经元共享相同的权重和偏置值，当感兴趣的特征落在其感受野内时，这些神经元中的任何一个都会被激活。这导致了一个对特征平移等变的特征图，这意味着如果输入图像被平移，则特征图也会相应平移。

Dosovitskiy 等人认为在 ViT 中，只有 MLP 层具有局部性和平移等变性。另一方面，自注意力层被描述为全局的，因为在这些层上进行的计算并不局限于局部的二维邻域。

他们解释说，对于图像的二维邻域结构的偏置仅在以下情况下使用：

+   在模型输入端，每个图像被切割成补丁，从而固有地保留了每个补丁内像素之间的空间关系。

+   在微调过程中，预训练的位置嵌入根据它们在原始图像中的位置进行二维插值，以生成一个更长的序列，这个序列的长度与微调过程中使用的图像补丁数量相匹配。

## **ViT 变体与 ResNet 的比较性能**

Dosovitskiy 等人将三个逐渐增大的 ViT 模型与两个不同尺寸的修改版 ResNet 进行对比。实验结果得出了几个有趣的发现：

+   **实验 1** – 在 ImageNet 上进行微调和测试：

    +   当在最小的数据集（ImageNet）上进行预训练时，两个较大的 ViT 模型的表现不如其较小的对应模型。所有 ViT 模型的表现普遍低于 ResNet。

    +   当在较大的数据集（ImageNet-21k）上进行预训练时，三个 ViT 模型的表现彼此相似，也与 ResNet 的表现相当。

    +   当在最大的数据集（JFT-300M）上进行预训练时，较大 ViT 模型的表现超过了较小 ViT 模型和 ResNet 的表现。

+   **实验 2** – 在 JFT-300M 数据集的随机子集上进行训练，并在 ImageNet 上进行测试，以进一步调查数据集大小的影响：

    +   在数据集的较小子集上，ViT 模型的过拟合程度高于 ResNet 模型，并且表现显著较差。

    +   在数据集的较大子集上，较大 ViT 模型的表现超过了 ResNet 模型的表现。

> *这一结果加强了这样的直觉：卷积的归纳偏置对较小的数据集是有用的，但对于较大的数据集，从数据中直接学习相关模式是足够的，甚至是有利的。*
> 
> *–* [图像的价值相当于 16×16 个词：用于大规模图像识别的 Transformers](https://arxiv.org/abs/2010.11929)，2021。

## **数据的内部表示**

在分析 ViT 中图像数据的内部表示时，Dosovitskiy 等人发现以下内容：

+   初始应用于 ViT 第一层图像补丁的学习嵌入滤波器，类似于能够提取每个补丁内低级特征的基础功能：

![](https://machinelearningmastery.com/wp-content/uploads/2022/02/vit_2.png)

学习嵌入滤波器

摘自“[一张图像值 16×16 个词：大规模图像识别的变压器](https://arxiv.org/abs/2010.11929)”

+   原始图像中空间接近的图像补丁，其学习的位置嵌入相似：

![](https://machinelearningmastery.com/wp-content/uploads/2022/02/vit_3.png)

学习位置嵌入

摘自“[一张图像值 16×16 个词：大规模图像识别的变压器](https://arxiv.org/abs/2010.11929)”

+   在模型最低层的几个自注意力头部已经关注了大部分图像信息（基于它们的注意力权重），展示了自注意力机制在整合整个图像信息方面的能力：

![](https://machinelearningmastery.com/wp-content/uploads/2022/02/vit_4.png)

不同自注意力头部关注的图像区域大小

摘自“[一张图像值 16×16 个词：大规模图像识别的变压器](https://arxiv.org/abs/2010.11929)”

## **进一步阅读**

如果您希望深入了解该主题，本节提供了更多资源。

### **论文**

+   [一张图像值 16×16 个词：大规模图像识别的变压器](https://arxiv.org/abs/2010.11929)，2021 年。

+   [注意力机制就是你所需要的](https://arxiv.org/abs/1706.03762)，2017 年。

## **摘要**

在本教程中，您了解了 Vision Transformer 模型的架构及其在图像分类任务中的应用。

具体而言，您学到了：

+   ViT 在图像分类背景下的工作原理。

+   ViT 的训练过程包括哪些内容。

+   ViT 在归纳偏差方面与卷积神经网络的比较。

+   ViT 在不同数据集上与 ResNets 的对比表现如何。

+   ViT 内部如何处理数据以实现其性能。

您是否有任何问题？

在下方评论区提出您的问题，我会尽力回答。
