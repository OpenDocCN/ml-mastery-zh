# 机器学习最大似然估计的温和介绍

> 原文：<https://machinelearningmastery.com/what-is-maximum-likelihood-estimation-in-machine-learning/>

最后更新于 2019 年 11 月 5 日

密度估计是估计来自问题域的观测样本的概率分布的问题。

有许多技术可以解决密度估计，尽管机器学习领域中使用的一个常见框架是最大似然估计。最大似然估计涉及定义似然函数，用于计算给定概率分布和分布参数时观察数据样本的条件概率。这种方法可以用来搜索可能的分布和参数的空间。

这种灵活的概率框架也为许多机器学习算法提供了基础，包括分别用于预测数值和类标签的重要方法，如线性回归和逻辑回归，但也更普遍地用于深度学习人工神经网络。

在这篇文章中，你会发现最大似然估计的温和介绍。

看完这篇文章，你会知道:

*   最大似然估计是解决密度估计问题的概率框架。
*   它包括最大化似然函数，以便找到最能解释观测数据的概率分布和参数。
*   它为机器学习中的预测建模提供了一个框架，在这个框架中，寻找模型参数可以作为一个优化问题。

**用我的新书[机器学习概率](https://machinelearningmastery.com/probability-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![A Gentle Introduction to Maximum Likelihood Estimation for Machine Learning](img/01d5a7a54e03c1fb41ec836ae841278c.png)

机器学习最大似然估计简介
图片由[吉勒姆·维尔鲁特](https://www.flickr.com/photos/o_0/29028143047/)提供，版权所有。

## 概观

本教程分为三个部分；它们是:

1.  概率密度估计问题
2.  最大似然估计
3.  与机器学习的关系

## 概率密度估计问题

一个常见的建模问题涉及如何估计数据集的联合概率分布。

例如，给定来自域( *x1，x2，x3，…，xn* )的观察样本( *X* ，其中每个观察都是从具有相同概率分布(所谓的独立同分布，即，即，或接近于它)的域中独立得出的。

密度估计包括选择一个概率分布函数和该分布的参数，最好地解释观测数据的联合概率分布( *X* )。

*   概率分布函数怎么选？
*   如何选择概率分布函数的参数？

这个问题变得更具挑战性，因为从总体中抽取的样本( *X* )很小并且有噪声，这意味着对估计的概率密度函数及其参数的任何评估都会有一些误差。

有许多技术可以解决这个问题，尽管有两种常见的方法:

*   最大后验概率，一种贝叶斯方法。
*   最大似然估计，频率法。

主要区别在于，最大似然估计假设所有的解决方案事先都是相同的，而最大似然估计允许利用关于解决方案形式的事先信息。

在这篇文章中，我们将仔细研究最大似然估计方法及其与应用机器学习的关系。

## 最大似然估计

概率密度估计的一种解决方案称为最大似然估计，简称 MLE。

[最大似然估计](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)涉及将问题作为优化或搜索问题来处理，其中我们寻求一组参数，该组参数导致数据样本的联合概率的最佳拟合( *X* )。

首先，它包括定义一个名为*θ*的参数，该参数定义了概率密度函数的选择和该分布的参数。它可能是一个数值向量，其值平滑地变化，并映射到不同的概率分布及其参数。

在最大似然估计中，我们希望在给定特定概率分布及其参数的情况下，从联合概率分布中观测数据的概率最大化，正式表述为:

*   p(X |θ)

这个条件概率经常用分号(；)符号而不是条形符号(|)，因为*θ*不是随机变量，而是未知参数。例如:

*   p(X；θ)

或者

*   P(x1，x2，x3，…，xn；θ)

这个结果条件概率被称为观察给定模型参数的数据的可能性，并使用符号 *L()* 来表示[可能性函数](https://en.wikipedia.org/wiki/Likelihood_function)。例如:

*   l(X；θ)

最大似然估计的目标是找到使似然函数最大化的一组参数(*θ*)，例如产生最大似然值。

*   最大化 L(X；θ)

我们可以解包由似然函数计算的条件概率。

假设样本由 n 个示例组成，我们可以将其框定为给定概率分布参数(*θ*)的 *X* 中的观测数据样本 *x1，x2，x3，…，xn* 的联合概率。

*   L(x1，x2，x3，…，xn；θ)

给定分布参数，联合概率分布可以重申为观察每个例子的条件概率的乘积。

*   产品一至产品二(Xi；θ)

将许多小概率相乘在实际中可能在数值上不稳定，因此，通常将这个问题重述为给定模型参数时观察每个例子的对数条件概率的总和。

*   对数和(Xi；θ))

其中通常使用以-e 为底的对数，称为自然对数。

> 这种在许多概率上的乘积可能是不方便的[……]它容易出现数字下溢。为了得到一个更方便但等价的优化问题，我们观察到取似然的对数不会改变它的 arg max，但可以方便地将乘积转化为和

—第 132 页，[深度学习](https://amzn.to/2lnc3vL)，2016。

鉴于对数在似然函数中的频繁使用，它通常被称为对数似然函数。

在优化问题中，通常倾向于最小化成本函数，而不是最大化成本函数。因此，使用对数似然函数的负值，通常称为负对数似然函数。

*   最小化对数和(Xi；θ))

> 在软件中，我们经常把两者都称为最小化成本函数。最大似然因此成为负对数似然的最小化…

—第 133 页，[深度学习](https://amzn.to/2lnc3vL)，2016。

## 与机器学习的关系

这个密度估计问题与应用机器学习直接相关。

我们可以将拟合机器学习模型的问题框架为概率密度估计问题。具体来说，模型和模型参数的选择被称为建模假设 *h* ，问题涉及到寻找最能解释数据 *X* 的 *h* 。

*   p(X；h)

因此，我们可以找到最大化似然函数的建模假设。

*   最大化 L(X；h)

或者更确切地说:

*   最大化对数和(Xi；h))

这为估计数据集的概率密度提供了基础，通常用于无监督的机器学习算法；例如:

*   聚类算法。

> 在著名的“期望最大化”或 EM 算法的背景下，使用期望对数联合概率作为具有隐藏变量的概率模型中学习的关键量更为人所知。

—第 365 页，[数据挖掘:实用机器学习工具与技术](https://amzn.to/2lnW5S7)，2016 年第 4 版。

最大似然估计框架也是监督机器学习的有用工具。

这适用于我们有输入和输出变量的数据，在回归和分类预测建模的情况下，输出变量可以是数值或类别标签。

我们可以将其表述为给定输入( *X* )给定建模假设( *h* )的输出( *y* )的条件概率。

*   最大化 L(y | X；h)

或者更确切地说:

*   最大化对数和(Xi；h))

> 最大似然估计器可以很容易地推广到我们的目标是估计条件概率 P(y | x；这实际上是最常见的情况，因为它构成了大多数监督学习的基础。

—第 133 页，[深度学习](https://amzn.to/2lnc3vL)，2016。

这意味着通常用于密度估计的相同最大似然估计框架可以用于寻找监督学习模型和参数。

这为基本的线性建模技术提供了基础，例如:

*   线性回归，用于预测数值。
*   逻辑回归，用于二分类。

在线性回归的情况下，模型被约束到一条线上，并且包括为最适合观察数据的线找到一组系数。幸运的是，这个问题可以通过分析来解决(例如，直接使用线性代数)。

在逻辑回归的情况下，该模型定义了一条线，并涉及为最好地分隔类别的线找到一组系数。这不能通过分析来解决，通常通过使用诸如 BFGS 算法或变体的有效优化算法来搜索可能系数值的空间来解决。

使用更通用的优化算法，如随机梯度下降，这两种方法的求解效率也较低。

事实上，大多数机器学习模型都可以在最大似然估计框架下构建，这为将预测建模作为优化问题来处理提供了一种有用且一致的方法。

机器学习中最大似然估计器的一个重要好处是，随着数据集大小的增加，估计器的质量不断提高。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   第五章机器学习基础，[深度学习](https://amzn.to/2lnc3vL)，2016。
*   第二章概率分布，[模式识别和机器学习](https://amzn.to/2JwHE7I)，2006。
*   第八章模型推断和平均，[统计学习的要素](https://amzn.to/2YVqu8s)，2016。
*   第九章概率方法，[数据挖掘:实用机器学习工具和技术](https://amzn.to/2lnW5S7)，第 4 版，2016。
*   第二十二章最大似然和聚类，[信息论，推理和学习算法](https://amzn.to/31q6fBo)，2003。
*   第八章学习分布，[贝叶斯推理和机器学习](https://amzn.to/31D2VTD)，2011。

### 文章

*   [最大似然估计，维基百科](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)。
*   [最大似然，Wolfram MathWorld](http://mathworld.wolfram.com/MaximumLikelihood.html) 。
*   [似然函数，维基百科](https://en.wikipedia.org/wiki/Likelihood_function)。
*   [理解最大似然法中函数定义的一些问题，交叉验证](https://stats.stackexchange.com/questions/49077/some-problems-understanding-the-definition-of-a-function-in-a-maximum-likelihood)。

## 摘要

在这篇文章中，你发现了最大似然估计的温和介绍。

具体来说，您了解到:

*   最大似然估计是解决密度估计问题的概率框架。
*   它包括最大化似然函数，以便找到最能解释观测数据的概率分布和参数。
*   它为机器学习中的预测建模提供了一个框架，在这个框架中，寻找模型参数可以作为一个优化问题。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。