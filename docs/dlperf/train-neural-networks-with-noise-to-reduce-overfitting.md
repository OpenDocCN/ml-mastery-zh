# 使用噪声训练神经网络来减少过拟合

> 原文：<https://machinelearningmastery.com/train-neural-networks-with-noise-to-reduce-overfitting/>

最后更新于 2019 年 8 月 6 日

用小数据集训练神经网络会导致网络记住所有的训练示例，进而导致[过拟合](https://machinelearningmastery.com/introduction-to-regularization-to-reduce-overfitting-and-improve-generalization-error/)，在保持数据集上表现不佳。

给定高维输入空间中点的不完整或稀疏采样，小数据集也可能代表神经网络更难学习的映射问题。

使输入空间更平滑、更容易学习的一种方法是在训练过程中给输入添加噪声。

在这篇文章中，你会发现在训练过程中给神经网络添加噪声可以提高网络的鲁棒性，从而获得更好的泛化能力和更快的学习速度。

看完这篇文章，你会知道:

*   小数据集会给神经网络的学习带来挑战，并且示例可以记忆。
*   在训练过程中加入噪声可以使训练过程更加稳健，减少泛化误差。
*   传统上，噪声会添加到输入中，但也可以添加到权重、梯度甚至激活函数中。

**用我的新书[更好的深度学习](https://machinelearningmastery.com/better-deep-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![Train Neural Networks With Noise to Reduce Overfitting](img/e4ba563c34e86b06a3ceee4240ccf21d.png)

用噪声训练神经网络以减少过拟合。

## 概观

本教程分为五个部分；它们是:

1.  小训练数据集的挑战
2.  训练期间添加随机噪声
3.  如何以及在哪里添加噪音
4.  训练期间添加噪音的示例
5.  训练时增加噪音的技巧

## 小训练数据集的挑战

训练大型神经网络时，小数据集会带来问题。

第一个问题是网络可能会有效地记忆训练数据集。该模型可以学习特定的输入示例及其相关输出，而不是学习从输入到输出的一般映射。这将导致模型在训练数据集上表现良好，而在新数据(如保持数据集)上表现不佳。

第二个问题是，小数据集提供较少的机会来描述输入空间的结构及其与输出的关系。更多的训练数据提供了更丰富的问题描述，模型可以从中学习。更少的数据点意味着，这些点可能代表不和谐和不连贯的结构，而不是平滑的输入空间，这可能导致一个困难的(如果不是不可理解的)映射函数。

获取更多的数据并不总是可能的。此外，获得更多数据可能无法解决这些问题。

## 训练期间添加随机噪声

一种改进泛化误差和改进映射问题结构的方法是添加随机噪声。

> 许多研究[…]已经注意到，向训练数据中添加少量输入噪声(抖动)通常有助于泛化和容错。

—第 273 页，[神经锻造:前馈人工神经网络中的监督学习](https://amzn.to/2Dxo4XU)，1999。

起初，这听起来像是让学习更具挑战性的方法。这是一个提高表现的反直觉建议，因为在训练过程中，人们会期望噪声会降低模型的表现。

> 试探性地，我们可能预期噪声将“涂抹”每个数据点，并使网络难以精确拟合单个数据点，因此将减少过拟合。在实践中，已经证明用噪声进行训练确实可以提高网络的泛化能力。

—第 347 页，[用于模式识别的神经网络](https://amzn.to/2I9gNMP)，1995。

在神经网络模型的训练期间添加噪声具有正则化效果，并且反过来提高了模型的鲁棒性。已经表明，与权重正则化方法的情况一样，增加惩罚项对损失函数有类似的影响。

> 众所周知，在训练期间向神经网络的输入数据添加噪声在某些情况下会导致泛化表现的显著提高。先前的工作表明，这种带噪声的训练相当于一种正则化形式，其中在误差函数中增加了一个额外的项。

——[带噪训练相当于 Tikhonov 正则化](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1995.7.1.108)，2008。

实际上，添加噪声会扩大训练数据集的大小。每次训练样本暴露在模型中时，随机噪声都会被添加到输入变量中，使它们每次暴露在模型中时都不同。这样，给输入样本添加噪声就是[数据扩充](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/)的一种简单形式。

> 在神经网络的输入中注入噪声也可以被视为一种数据扩充形式。

—第 241 页，[深度学习](https://amzn.to/2NJW3gE)，2016。

添加噪声意味着网络记忆训练样本的能力降低，因为它们一直在变化，导致网络权重更小，网络更健壮，泛化误差更低。

噪声意味着好像从已知样本附近的域中抽取新样本，平滑输入空间的结构。这种平滑可能意味着映射函数对于网络来说更容易学习，从而导致更好更快的学习。

> ……输入噪声和权重噪声分别促使神经网络输出成为输入或其权重的平滑函数。

——[反向传播训练中添加噪声对泛化表现的影响](https://ieeexplore.ieee.org/document/6796981/)，1996。

## 如何以及在哪里添加噪音

训练期间使用的最常见的噪声类型是向输入变量添加高斯噪声。

高斯噪声或白噪声的平均值为零，标准偏差为 1，可以根据需要使用伪随机数发生器生成。在信号处理中使用术语“电路中不相关的随机噪声”之后，将高斯噪声添加到神经网络的输入中传统上被称为“T0”抖动或“T2”随机抖动。

添加的噪声量(例如扩展或标准偏差)是一个可配置的超参数。太少的噪声没有影响，而太多的噪声使得映射函数太具挑战性而难以学习。

> 这通常是通过在每个输入模式呈现给网络之前向其添加一个随机向量来完成的，因此，如果模式正在被回收，则每次添加一个不同的随机向量。

——[带噪训练相当于 Tikhonov 正则化](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1995.7.1.108)，2008。

随机噪声的标准偏差控制传播量，并且可以根据每个输入变量的比例进行调整。如果首先对输入变量的规模进行标准化，配置起来会更容易。

噪音只在训练时增加。在模型评估期间或当模型用于对新数据进行预测时，不会添加噪声。

噪声的添加也是自动特征学习的重要部分，例如在自动编码器的情况下，所谓的去噪自动编码器明确要求模型在存在添加到输入的噪声的情况下学习鲁棒的特征。

> 我们已经看到，重建标准本身不能保证有用特征的提取，因为它可能导致显而易见的解决方案“简单地复制输入”或类似的不感兴趣的解决方案，这些解决方案通常会最大化相互信息。[……]我们改变了重建标准，以实现一个更具挑战性和更有趣的目标:清理部分损坏的输入，或者简而言之，去噪。

——[堆叠去噪自动编码器:利用局部去噪标准在深度网络中学习有用的表示](http://www.jmlr.org/papers/v11/vincent10a.html)，2010。

尽管输入端的额外噪声是最常见且研究最广泛的方法，但在训练过程中，随机噪声可能会添加到网络的其他部分。一些例子包括:

*   **给激活**添加噪声，即每层的输出。
*   **给权重**添加噪声，即输入的替代。
*   **给梯度**添加噪点，即更新权重的方向。
*   **给输出**添加噪声，即标签或目标变量。

将噪声添加到层激活允许在网络中的任何点使用噪声。这对非常深的网络可能是有益的。噪声可以添加到层输出本身，但这更有可能通过使用噪声激活功能来实现。

将噪声添加到权重允许该方法以一致的方式在整个网络中使用，而不是将噪声添加到输入和层激活中。这在递归神经网络中特别有用。

> 另一种将噪声用于模型调整的方法是将其加入权重。这项技术主要用于递归神经网络。[……]应用于权重的噪声也可以被解释为等同于(在某些假设下)更传统形式的正则化，鼓励要学习的函数的稳定性。

—第 242 页，[深度学习](https://amzn.to/2NJW3gE)，2016。

向梯度添加噪声更侧重于提高优化过程本身的鲁棒性，而不是输入域的结构。噪音的量可以在训练开始时很高，并随着时间的推移而减少，很像一个衰减的学习率。这种方法已被证明是非常深的网络和各种不同网络类型的有效方法。

> 当优化各种各样的模型时，包括非常深的全连接网络，以及用于问题回答和算法学习的专用架构，我们不断看到注入的梯度噪声带来的改进。[……]我们的实验表明，通过衰减方差添加退火高斯噪声比使用固定高斯噪声效果更好

——[添加梯度噪声改善超深度网络的学习](https://arxiv.org/abs/1511.06807)，2015。

向激活、权重或梯度添加噪声都提供了一种更通用的方法来添加噪声，该方法对于提供给模型的输入变量类型是不变的。

如果问题域被认为或预期有错误标记的例子，那么向类标签添加噪声可以提高模型对此类错误的鲁棒性。虽然，这可能很容易破坏学习过程。

在回归或时间序列预测的情况下，向连续目标变量添加噪声很像向输入变量添加噪声，可能是更好的使用情形。

## 训练期间添加噪音的示例

本节总结了一些在训练过程中加入噪声的例子。

Lasse Holmstrom 在 1992 年发表的题为“在反向传播训练中使用加性噪声”的论文中，用 MLPs 分析和实验研究了随机噪声的加入他们建议首先标准化输入变量，然后使用交叉验证来选择训练期间使用的噪声量。

> 如果应该建议单一的通用噪声设计方法，我们会选择最大化交叉验证的似然函数。该方法易于实现，完全由数据驱动，并具有理论一致性结果支持的有效性

Klaus Gref 等人在他们 2016 年发表的题为《 [LSTM:搜索空间奥德赛》](https://ieeexplore.ieee.org/abstract/document/7508408/)》的论文中，使用超参数搜索一组序列预测任务的输入变量上高斯噪声的标准偏差，发现它几乎普遍会导致更差的表现。

> 输入上的加性高斯噪声，神经网络的传统正则化器，也被用于 LSTM。然而，我们发现，它不仅几乎总是损害表现，还会略微增加训练次数。

亚历克斯·格雷夫斯(Alex Graves)等人在他们 2013 年的开创性论文《使用深度递归神经网络的语音识别》》中称，语音识别取得了当时最先进的结果，但在训练过程中，噪声增加了 LSTMs 的权重。

> ……使用了权重噪声(在训练期间向网络权重添加高斯噪声)。每个训练序列添加一次权重噪声，而不是在每个时间步长添加。权重噪声倾向于“简化”神经网络，即减少传输参数所需的信息量，从而提高泛化能力。

在 2011 年之前的一篇研究不同类型的静态和自适应权重噪声的论文中，标题为“神经网络的实用变分推理”，“格雷夫斯建议使用提前停止，并结合权重噪声的加入来使用低信噪比模型。

> ……在实践中，当用重量噪音训练时，需要提前停止以防止过度训练。

## 训练时增加噪音的技巧

本节提供了一些在神经网络训练过程中添加噪声的技巧。

### 添加噪音的问题类型

无论正在解决的问题是什么类型，噪声都可能会加入到训练中。

尝试向分类和回归类型问题添加噪声是合适的。

噪声类型可以专用于用作模型输入的数据类型，例如，图像情况下的二维噪声和音频数据情况下的信号噪声。

### 给不同的网络类型增加噪音

在训练过程中添加噪声是一种通用方法，无论使用哪种类型的神经网络都可以使用。

这是一种主要用于多层感知器的方法，给出了它们的优先优势，但是可以并且正在用于卷积和递归神经网络。

### 先重新缩放数据

重要的是，噪声的加入对模型有一致的影响。

这需要重新调整输入数据的比例，以便所有变量都具有相同的比例，这样，当噪声以固定的方差添加到输入时，它具有相同的效果。也适用于向权重和梯度添加噪声，因为它们也受到输入比例的影响。

这可以通过输入变量的标准化或规范化来实现。

如果在数据缩放后添加了随机噪声，那么可能需要重新缩放变量，也许是每一个小批量。

### 测试噪音量

您无法知道在您的训练数据集中，噪声对您的特定模型有多大好处。

尝试不同的噪音量，甚至不同类型的噪音，以便发现什么效果最好。

系统化并使用受控实验，也许是在一系列数值的较小数据集上。

### 仅噪音训练

噪声仅在模型训练期间添加。

请确保在评估模型的过程中，或者在使用模型对新数据进行预测时，没有添加任何噪声源。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   第 7.5 节噪声鲁棒性，[深度学习](https://amzn.to/2NJW3gE)，2016。
*   第十七章带噪声输入的训练，[神经锻造:前馈人工神经网络中的监督学习](https://amzn.to/2Dxo4XU)，1999。
*   第 9.3 节，带噪声的训练，[用于模式识别的神经网络](https://amzn.to/2I9gNMP)，1995。

### 报纸

*   [创建推广](https://www.sciencedirect.com/science/article/pii/0893608091900332)的人工神经网络，1991 年。
*   [用于鲁棒视觉识别的深度网络](https://dl.acm.org/citation.cfm?id=3104456)，2010。
*   [堆叠去噪自动编码器:利用局部去噪标准在深度网络中学习有用的表示](http://www.jmlr.org/papers/v11/vincent10a.html)，2010。
*   [分析自动编码器和深度网络中的噪声](https://arxiv.org/abs/1406.1831)，2014。
*   [反向传播训练中添加噪声对泛化表现的影响](https://ieeexplore.ieee.org/document/6796981/)，1996。
*   [带噪训练相当于 Tikhonov 正则化](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1995.7.1.108)，2008。
*   [添加梯度噪声改善超深度网络的学习](https://arxiv.org/abs/1511.06807)，2016。
*   [噪声激活功能](http://proceedings.mlr.press/v48/gulcehre16.html)，2016 年。

### 文章

*   什么是抖动？(带噪训练)，神经网络常见问题。
*   [抖动，维基百科](https://en.wikipedia.org/wiki/Jitter)。

## 摘要

在这篇文章中，你发现在训练过程中向神经网络添加噪声可以提高网络的鲁棒性，从而获得更好的泛化能力和更快的学习速度。

具体来说，您了解到:

*   小数据集会给神经网络的学习带来挑战，并且示例可以记忆。
*   在训练过程中加入噪声可以使训练过程更加稳健，减少泛化误差。
*   传统上，噪声会添加到输入中，但也可以添加到权重、梯度甚至激活函数中。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。