# 机器学习最大后验概率的温和介绍

> 原文：<https://machinelearningmastery.com/maximum-a-posteriori-estimation/>

密度估计是估计来自问题域的观测样本的概率分布的问题。

通常，估计整个分布是很难的，相反，我们很高兴有分布的期望值，如平均值或模式。最大后验概率(简称 MAP)是一种基于贝叶斯的方法，用于估计最能解释观测数据集的分布和模型参数。

这种灵活的概率框架可用于为许多机器学习算法提供贝叶斯基础，包括分别用于预测数值和类别标签的线性回归和逻辑回归等重要方法，并且与最大似然估计不同，明确允许系统地纳入关于候选模型的先验信念。

在这篇文章中，你会发现最大后验估计的温和介绍。

看完这篇文章，你会知道:

*   最大后验估计是解决密度估计问题的概率框架。
*   MAP 包括计算观察给定模型的数据的条件概率，该概率由关于模型的先验概率或信念加权。
*   MAP 为机器学习的最大似然估计提供了另一种概率框架。

**用我的新书[机器学习概率](https://machinelearningmastery.com/probability-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![A Gentle Introduction to Maximum a Posteriori (MAP) for Machine Learning](img/7d273109a3c747379a6ee1831ca0f6d4.png)

机器学习最大后验概率(MAP)的温和介绍[吉勒姆·维尔鲁特](https://www.flickr.com/photos/o_0/44585280464/)摄，版权所有。

## 概观

本教程分为三个部分；它们是:

1.  密度估计
2.  最大后验概率
3.  地图和机器学习

## 密度估计

一个常见的建模问题涉及如何估计数据集的联合概率分布。

例如，给定来自域( *x1，x2，x3，…，xn* )的观察样本( *X* ，其中每个观察都是从具有相同概率分布(所谓的独立同分布，即，即，或接近于它)的域中独立得出的。

密度估计包括选择一个概率分布函数和该分布的参数，最好地解释观测数据的联合概率分布( *X* )。

通常估计密度太具挑战性；相反，我们对目标分布的点估计感到满意，例如平均值。

有许多技术可以解决这个问题，尽管有两种常见的方法:

*   最大后验概率，一种贝叶斯方法。
*   最大似然估计，一种频率方法。

这两种方法都将问题框架为优化，并涉及到搜索最能描述观测数据的分布和分布参数集。

在[最大似然估计](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)中，我们希望在给定特定概率分布及其参数的情况下，从联合概率分布中观测数据的概率最大化，正式表述为:

*   p(X；θ)

或者

*   P(x1，x2，x3，…，xn；θ)

这种产生的条件概率被称为在给定模型参数的情况下观察数据的可能性。

最大似然估计的目标是找到使[似然函数](https://en.wikipedia.org/wiki/Likelihood_function)最大化的一组参数(*θ*，例如，产生最大似然值。

*   最大化 P(X；θ)

另一种与此密切相关的方法是从贝叶斯概率的角度考虑优化问题。

> 最大化可能性的一个流行替代方法是最大化参数的贝叶斯后验概率密度。

—第 306 页，[信息论，推理和学习算法](https://amzn.to/2zn1Eny)，2003。

## 最大后验概率

回想一下，贝叶斯定理提供了一种计算条件概率的原则性方法。

它包括计算一个结果给定另一个结果的条件概率，使用这种关系的倒数，陈述如下:

*   P(A | B) = (P(B | A) * P(A)) / P(B)

我们正在计算的量通常被称为给定 *B* 的 *A* 的后验概率，而 *P(A)* 被称为 *A* 的先验概率。

可以去掉 *P(B)* 的归一化常数，可以证明后验与给定的 *A* 乘以前验的 *B* 的概率成正比。

*   P(A | B)与 P(B | A) * P(A)成正比

或者，简单地说:

*   P(A | B) = P(B | A) * P(A)

这是一个有用的简化，因为我们对估计概率不感兴趣，而是对优化一个量感兴趣。一个成比例的数量就足够了。

现在，我们可以将该计算与我们对估计分布和参数(*θ*)的期望联系起来，该分布和参数最好地解释了我们的数据集( *X* )，正如我们在上一节中所描述的那样。这可以表述为:

*   P(θ| X)= P(X |θ)* P(θ)

在θ范围内最大化这个量解决了用于估计后验概率的中心趋势(例如，分布的模型)的优化问题。因此，这种技术被称为“*最大后验估计*，或简称 MAP 估计，有时简称为“*最大后验估计*”

*   最大化 P(X |θ)* P(θ)

我们通常不计算完全后验概率分布，事实上，对于许多感兴趣的问题，这可能是不容易处理的。

> ……寻找 MAP 假设通常比贝叶斯学习容易得多，因为它需要解决一个优化问题，而不是一个大的求和(或积分)问题。

—第 804 页，[人工智能:现代方法](https://amzn.to/2Y7yCpO)，第 3 版，2009 年。

相反，我们在计算一个点估计，比如分布的一个矩，就像模式，最常见的值，它与正态分布的平均值相同。

> 希望进行点估计的一个常见原因是，对于大多数感兴趣的模型，涉及贝叶斯后验的大多数操作都是难以处理的，而点估计提供了一个易于处理的近似。

—第 139 页，[深度学习](https://amzn.to/2lnc3vL)，2016。

**注**:这与最大似然估计非常相似，增加了分布和参数上的先验概率。

事实上，如果我们假设*θ*的所有值都是同样可能的，因为我们没有任何先验信息(例如，统一的先验)，那么两个计算是等价的。

由于这种等价性，对于许多机器学习算法来说，最大似然估计和最大似然估计往往收敛到同一个优化问题。情况并非总是如此；如果最大似然估计和最大似然估计优化问题的计算不同，则为算法找到的最大似然估计和最大似然估计解也可能不同。

> ……最大似然假设可能不是 MAP 假设，但如果假设先验概率一致，那么它就是。

—第 167 页，[机器学习](https://amzn.to/2jWd51p)，1997。

## 地图和机器学习

在机器学习中，最大后验概率优化提供了一个贝叶斯概率框架，用于将模型参数拟合到训练数据，并提供了一个可能更常见的最大似然估计框架的替代和同级框架。

> 给定数据，最大后验概率(MAP)学习选择一个最有可能的假设。仍然使用假设先验，并且该方法通常比完全贝叶斯学习更容易处理。

—第 825 页，[人工智能:现代方法](https://amzn.to/2Y7yCpO)，第 3 版，2009 年。

一个框架并不比另一个好，正如前面提到的，在许多情况下，两个框架从不同的角度构建了同一个优化问题。

相反，MAP 适用于存在一些先验信息的问题，例如，可以设置有意义的先验来权衡不同分布和参数或模型参数的选择。在没有这种先验的情况下，最大似然估计更为合适。

> 贝叶斯方法可以用来确定给定数据的最可能假设——最大后验概率(MAP)假设。这是最佳假设，因为没有其他假设更有可能。

—第 197 页，[机器学习](https://amzn.to/2jWd51p)，1997。

事实上，在最大似然估计之前添加先验可以被认为是最大似然估计计算的一种正则化。这种洞察力允许在 MAP 贝叶斯推理的框架下解释其他正则化方法(例如，使用输入加权和的模型中的 L2 范数)。例如，L2 是一种偏差或先验，它假设一组系数或权重具有较小的平方和值。

> ……特别是，L2 正则化相当于权重具有高斯先验的 MAP 贝叶斯推断。

—第 236 页，[深度学习](https://amzn.to/2lnc3vL)，2016。

我们可以通过将优化问题重新框架为在候选建模假设(*中的 *h* )上执行，而不是在更抽象的分布和参数(*θ*)上执行，来使 MAP 和机器学习之间的关系更加清晰；例如:*

*   最大化 P(X | h) * P(h)

在这里，我们可以看到，我们想要一个模型或假设( *h* )来最好地解释观察到的训练数据集( *X* )并且先验( *P(h)* )是我们关于假设被期望有多有用的信念，一般来说，不管训练数据如何。优化问题包括估计每个候选假设的后验概率。

> 我们可以通过使用贝叶斯定理计算每个候选假设的后验概率来确定 MAP 假设。

—第 157 页，[机器学习](https://amzn.to/2jWd51p)，1997。

和最大似然法一样，优化问题的求解取决于模型的选择。对于更简单的模型，如线性回归，有解析解。对于更复杂的模型，如逻辑回归，需要使用一阶和二阶导数的数值优化。对于更棘手的问题，可能需要随机优化算法。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   第六章贝叶斯学习，[机器学习](https://amzn.to/2jWd51p)，1997。
*   第十二章最大熵模型，[机器学习基础](https://amzn.to/2Zp9f3A)，2018。
*   第九章概率方法，[数据挖掘:实用机器学习工具和技术](https://amzn.to/2lnW5S7)，第 4 版，2016。
*   第五章机器学习基础，[深度学习](https://amzn.to/2lnc3vL)，2016。
*   第十三章 MAP 推断，[概率图形模型:原理和技术](https://amzn.to/324l0tT)，2009。

### 文章

*   [最大后验估计，维基百科](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation)。
*   [贝叶斯统计，维基百科](https://en.wikipedia.org/wiki/Bayesian_statistics)。

## 摘要

在这篇文章中，你发现了对最大后验估计的温和介绍。

具体来说，您了解到:

*   最大后验估计是解决密度估计问题的概率框架。
*   MAP 包括计算观察给定模型的数据的条件概率，该概率由关于模型的先验概率或信念加权。
*   MAP 为机器学习的最大似然估计提供了另一种概率框架。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。