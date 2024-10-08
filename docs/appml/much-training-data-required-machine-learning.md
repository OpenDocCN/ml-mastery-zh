# 机器学习需要多少训练数据？

> 原文： [`machinelearningmastery.com/much-training-data-required-machine-learning/`](https://machinelearningmastery.com/much-training-data-required-machine-learning/)

您需要的数据量取决于问题的复杂程度和所选算法的复杂程度。

这是一个事实，但如果你处于机器学习项目的尖端，它对你没有帮助。

我被问到的一个常见问题是：

**_ 我需要多少数据？_**

我不能直接为你或任何人回答这个问题。但我可以给你一些思考这个问题的方法。

在这篇文章中，我列出了一套方法，您可以使用这些方法来考虑将机器学习应用到您的问题所需的训练数据量。

我希望这些方法中的一个或多个可以帮助您理解问题的难度以及它如何与您试图解决的归纳问题的核心紧密结合。

让我们深入研究它。

注意：您是否有自己的启发式方法来决定机器学习需要多少数据？请在评论中分享。

![How Much Training Data is Required for Machine Learning?](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/07/How-Much-Training-Data-is-Required-for-Machine-Learning.jpg)

机器学习需要多少训练数据？
[Seabamirum](https://www.flickr.com/photos/seabamirum/2926154653/) 的照片，保留一些权利。

## 你为什么问这个问题？

重要的是要知道为什么要询问训练数据集所需的大小。

答案可能会影响您的下一步。

例如：

*   **你有太多的数据吗？** 考虑开发一些学习曲线，以找出代表性样本有多大（下图）。或者，考虑使用大数据框架以使用所有可用数据。
*   **你的数据太少了吗？** 考虑确认你确实有太少的数据。考虑收集更多数据，或使用数据增加方法人为地增加样本量。
*   **你还没有收集数据吗？** 考虑收集一些数据并评估它是否足够。或者，如果是用于研究或数据收集是昂贵的，请考虑与域专家和统计学家交谈。

更一般地说，您可能会有更多的行人问题，例如：

*   我应该从数据库导出多少条记录？
*   需要多少样品才能达到理想的表现水平？
*   训练集必须有多大才能获得对模型表现的充分估计？
*   需要多少数据来证明一个模型比另一个更好？
*   我应该使用训练/测试拆分或 k 折交叉验证吗？

这篇文章中的建议可能试图解决这些后面的问题。

在实践中，我自己使用学习曲线（见下文）回答这个问题，在小数据集上使用重采样方法（例如 k 折交叉验证和自举），并在最终结果中添加置信区间。

您询问机器学习所需样品数量的原因是什么？
请在评论中告诉我。

_ 那么，您需要多少数据？_

## 它取决于;没有人能告诉你

没有人能告诉您预测性建模问题需要多少数据。

这是不可知的：一个棘手的问题，你必须通过实证调查发现答案。

机器学习所需的数据量取决于许多因素，例如：

*   **问题的复杂性**，名义上是未知的基础函数，它最好地将输入变量与输出变量联系起来。
*   **学习算法**的复杂性，名义上是用于从特定示例中归纳地学习未知底层映射函数的算法。

这是我们的出发点。

并且“_ 取决于 _”是大多数从业者在您第一次提出时会给您的答案。

## 比喻的理由

很多人在你面前研究过很多应用的机器学习问题。

其中一些人已公布了他们的结果。

也许您可以查看与您类似的问题的研究，作为对可能需要的数据量的估计。

同样，通常会对算法表现如何随数据集大小进行扩展进行研究。也许这些研究可以告诉您使用特定算法需要多少数据。

也许你可以平均多次研究。

在 Google， [Google Scholar](https://scholar.google.com) 和 [Arxiv](https://arxiv.org/) 上搜索论文。

## 3.使用域专业知识

您需要一个代表您要解决的问题的问题数据样本。

通常，示例必须是独立且相同的。

请记住，在机器学习中，我们正在学习将输入数据映射到输出数据的功能。学习的映射函数只能与您提供的数据一样好。

这意味着需要有足够的数据来合理地捕获输入要素之间以及输入要素和输出要素之间可能存在的关系。

使用您的领域知识，或找到领域专家和有关域的合理性以及可能需要合理捕获问题中有用复杂性所需的数据规模。

## 4.使用统计启发式

有统计启发式方法可用于计算合适的样本量。

我见过的大部分启发式算法都是分类问题，它是类，输入特征或模型参数的函数。一些启发式方法看起来很严谨，其他启发式方式看起来很特别。

以下是您可以考虑的一些示例：

*   **类别的因子**：每个类必须有 x 个独立的例子，其中 x 可以是数十，数百或数千（例如 5,50,500,5000）。
*   **输入特征数量的因子**：必须有比输入特征多 x％的例子，其中 x 可以是十（例如 10）。
*   **模型参数数量的因子**：模型中每个参数必须有 x 个独立的例子，其中 x 可以是数十（例如 10）。

它们看起来都像是特殊的缩放因子给我。

你使用过这些启发式算法吗？
怎么回事？请在评论中告诉我。

在关于这个主题的理论工作（不是我的专业领域！）中，分类器（例如 k-最近邻居）经常与最优贝叶斯决策规则形成对比，并且难度在维数灾难的背景下表征;也就是说随着输入特征数量的增加，问题的难度呈指数增长。

例如：

*   [统计模式识别中的小样本量效应：对从业者的建议](http://sci2s.ugr.es/keel/pdf/specific/articulo/raudys91.pdf)，1991
*   [模式识别实践中的维度和样本量考虑](http://www.sciencedirect.com/science/article/pii/S0169716182020422)，1982

结果建议避免使用局部方法（如 k-最近邻居）来处理来自高维问题的稀疏样本（例如，少量样本和许多输入特征）。

有关该主题的更友善的讨论，请参阅：

*   第 2.5 节高维局部方法，[统计学习的要素：数据挖掘，推理和预测](http://www.amazon.com/dp/0387848576?tag=inspiredalgor-20)，2008。

## 5.非线性算法需要更多数据

更强大的机器学习算法通常被称为非线性算法。

根据定义，他们能够学习输入和输出特征之间复杂的非线性关系。您可能正在使用这些类型的算法或打算使用它们。

这些算法通常更灵活，甚至是非参数的（除了这些参数的值之外，它们还可以确定模拟问题所需的参数数量）。它们也是高方差，意味着预测因用于训练它们的特定数据而异。这种增加的灵活性和功率需要更多的训练数据，通常需要更多的数据。

实际上，一些非线性算法（如深度学习方法）可以在您为其提供更多数据时继续提高技能。

如果线性算法通过每个类的数百个示例实现了良好的表现，则每个类可能需要数千个示例用于非线性算法，如随机森林或人工神经网络。

## 6.评估数据集大小与模型技能

在开发新的机器学习算法时，通常会演示甚至解释算法的表现以响应数据量或问题的复杂性。

这些研究可能会或可能不会由算法的作者执行和发布，并且可能存在或可能不存在您正在使用的算法或问题类型。

我建议您使用可用数据和一个表现良好的算法（如随机森林）进行自己的研究。

设计一项评估模型技能与训练数据集大小的研究。

将结果绘制为具有 x 轴上的训练数据集大小和 y 轴上的模型技能的线图，将使您了解数据大小如何影响模型对特定问题的技能。

该图称为[学习曲线](https://en.wikipedia.org/wiki/Learning_curve)。

从该图中，您可以预测开发熟练模型所需的数据量，或者在达到收益递减拐点之前实际需要的数据量。

我强烈推荐这种方法，以便在全面理解问题的背景下开发健壮的模型。

## 7.朴素的猜测

在应用机器学习算法时，您需要大量数据。

通常，您需要的数据超出了传统统计中可能需要的数据。

我经常回答轻率响应需要多少数据的问题：

_ 获取并使用尽可能多的数据。_

如果按下问题，并且对问题的具体细节一无所知，我会说一些朴素的事情：

*   你需要成千上万的例子。
*   不下百个。
*   理想情况下，数十或数十万的“平均”建模问题。
*   数百万或数百万的“硬”问题，如深度学习所解决的问题。

同样，这只是更多的临时猜测，但如果你需要它，它就是一个起点。所以开始吧！

## 8.获取更多数据（无关紧要！？）

大数据通常与机器学习一起讨论，但您可能不需要大数据来适应您的预测模型。

有些问题需要大数据，所有数据都有。例如，简单的统计机器翻译：

*   [数据的不合理有效性](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/35179.pdf)（和 [Peter Norvig 的谈话](https://www.youtube.com/watch?v=yvDCzhbjYWs)）

如果您正在执行传统的预测性建模，那么训练集大小的回报可能会有所减少，您应该研究您的问题和您选择的模型以查看该点的位置。

请记住，机器学习是一个归纳过程。该模型只能捕获它所看到的内容。如果您的训练数据不包含边缘情况，则模型很可能不支持它们。

## 不要拖延;入门

现在，停止准备为您的问题建模并对其进行建模。

不要让训练集大小的问题阻止您开始预测性建模问题。

在许多情况下，我认为这个问题是拖延的一个原因。

获取所有可用的数据，使用您拥有的数据，并查看模型对您的问题的有效性。

学习一些东西，然后采取行动，通过进一步分析更好地了解您拥有的内容，通过扩充扩展您拥有的数据，或从您的域中收集更多数据。

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

在 Quora，StackOverflow 和 CrossValidated 等 Q＆amp; A 网站上围绕这个问题进行了大量讨论。以下是一些可能有用的选择示例。

*   [需要多大的训练集？](https://stats.stackexchange.com/questions/51490/how-large-a-training-set-is-needed)
*   [考虑维数诅咒的神经网络训练集大小](https://stats.stackexchange.com/questions/161982/training-set-size-for-neural-networks-considering-curse-of-dimensionality)
*   [如何减少训练集大小？](https://stats.stackexchange.com/questions/22291/how-to-decrease-training-set-size)
*   [训练集大小的增加是否有助于永久提高准确度或是否存在饱和点？](https://stats.stackexchange.com/questions/181573/does-increase-in-training-set-size-help-in-increasing-the-accuracy-perpetually-o)
*   [如何为小样本数据选择训练，交叉验证和测试集大小？](https://stats.stackexchange.com/questions/113994/how-to-choose-the-training-cross-validation-and-test-set-sizes-for-small-sampl)
*   [训练神经网络时，很少有训练样例太少？](https://stats.stackexchange.com/questions/226672/how-few-training-examples-is-too-few-when-training-a-neural-network)
*   [训练深度神经网络的建议最小训练数据集大小是多少？](https://www.quora.com/What-is-the-recommended-minimum-training-dataset-size-to-train-a-deep-neural-network)

我希望对这个问题有一些很好的统计研究;这里有一些我能找到的。

*   [分类模型的样本量规划](https://arxiv.org/abs/1211.1323)，1991
*   [模式识别实践中的维度和样本量考虑](http://www.sciencedirect.com/science/article/pii/S0169716182020422)，1982
*   [统计模式识别中的小样本量效应：对从业者的建议](http://sci2s.ugr.es/keel/pdf/specific/articulo/raudys91.pdf)，1991
*   [预测分类表现所需的样本量](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/1472-6947-12-8)，2012

其他相关文章。

*   [您需要多少训练数据？](https://medium.com/@malay.haldar/how-much-training-data-do-you-need-da8ec091e956)
*   [我们需要更多的训练数据吗？](http://web.mit.edu/vondrick/bigdata.pdf)
*   [数据的不合理效力](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/35179.pdf)，（和 [Peter Norvig 的谈话](https://www.youtube.com/watch?v=yvDCzhbjYWs)）

如果您了解更多信息，请在下面的评论中告诉我们。

## 摘要

在这篇文章中，您发现了一套思考和推理回答常见问题的方法：

**_ 机器学习需要多少训练数据？_**

这些方法有帮助吗？
请在下面的评论中告诉我。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。
当然，除了 _ 你 _ 特别需要多少数据的问题。