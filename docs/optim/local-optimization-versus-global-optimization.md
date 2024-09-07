# 局部优化和全局优化的对比

> 原文：<https://machinelearningmastery.com/local-optimization-versus-global-optimization/>

最后更新于 2021 年 10 月 12 日

优化是指找到目标函数的一组输入，从而得到目标函数的最大或最小输出。

用**局部相对于全局优化**来描述优化问题是很常见的。

同样，根据局部搜索和全局搜索来描述优化算法或搜索算法也很常见。

在本教程中，您将发现局部优化和全局优化之间的实际差异。

完成本教程后，您将知道:

*   局部优化包括为搜索空间的特定区域寻找最优解，或者为没有局部最优解的问题寻找全局最优解。
*   全局优化包括寻找包含局部最优解的问题的最优解。
*   如何以及何时使用局部和全局搜索算法，以及如何同时使用这两种方法。

**用我的新书[机器学习优化](https://machinelearningmastery.com/optimization-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

Let’s get started.![Local Optimization Versus Global Optimization](img/4813222ccdc2a6e5f4f0a5e63c7177d6.png)

局部优化对比全局优化
图片由[马尔科·韦奇](https://www.flickr.com/photos/160866001@N07/50136642527/)提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  局部优化
2.  全局优化
3.  局部优化与全局优化

## 局部优化

局部最优值是输入空间给定区域目标函数的极值(最小值或最大值)，例如最小化问题中的一个流域。

> …我们寻找一个仅局部最优的点，这意味着它在它附近的可行点中最小化目标函数…

—第 9 页，[凸优化](https://amzn.to/2FXnvsn)，2004。

一个目标函数可能有多个局部最优解，也可能只有一个局部最优解，在这种情况下，局部最优解也是全局最优解。

*   **局部优化**:从被认为包含最优值的起点(例如，一个盆)定位目标函数的最优值。

[局部优化](https://en.wikipedia.org/wiki/Local_search_(optimization))或局部搜索是指搜索局部最优。

局部优化算法，也称为局部搜索算法，是一种旨在定位局部最优解的算法。它适用于遍历搜索空间的给定区域，并接近(或准确地找到)该区域中函数的极值。

> …局部优化方法被广泛应用于那些即使不是最好的也有价值的应用中。

—第 9 页，[凸优化](https://amzn.to/2FXnvsn)，2004。

局部搜索算法通常对单个候选解进行操作，并且涉及迭代地对候选解进行小的改变，并且评估该改变以查看它是否导致改进并且被作为新的候选解。

局部优化算法将定位全局最优值:

*   如果本地最优值是全局最优值，或者
*   如果正在搜索的区域包含全局最优值。

这些定义了使用本地搜索算法的理想用例。

对于什么是本地搜索算法可能会有争论；然而，使用我们的定义的本地搜索算法的三个例子包括:

*   NelderMead 算法
*   BFGS 算法
*   爬山算法

现在我们已经熟悉了局部优化，让我们来看看全局优化。

## 全局优化

全局最优是整个输入搜索空间的目标函数的极值(最小值或最大值)。

> 全局优化，算法通过使用搜索空间中较大部分的机制来搜索全局最优。

—第 37 页，[计算智能:导论](https://amzn.to/2FUZp1v)，2007。

一个目标函数可能有一个或多个全局最优解，如果有多个全局最优解，则称之为[多模态优化](https://en.wikipedia.org/wiki/Multimodal_distribution)问题，每个最优解将有不同的输入和相同的目标函数评估。

*   **全局优化**:为可能包含局部最优值的目标函数定位最优值。

一个目标函数总是有一个全局最优值(否则我们不会对优化它感兴趣)，尽管它也可能有局部最优值，其目标函数评估不如全局最优值。

全局最优值可能与局部最优值相同，在这种情况下，将优化问题称为局部优化而不是全局优化更合适。

局部最优解的存在是定义全局优化问题难度的主要因素，因为定位局部最优解相对容易，而定位全局最优解相对困难。

[全局优化](https://en.wikipedia.org/wiki/Global_optimization)或全局搜索是指搜索全局最优解。

全局优化算法，也称为全局搜索算法，旨在定位全局最优解。它适合遍历整个输入搜索空间，并接近(或准确地找到)函数的极值。

> 全局优化用于变量少的问题，计算时间并不关键，找到真正全局解的价值非常高。

—第 9 页，[凸优化](https://amzn.to/2FXnvsn)，2004。

全局搜索算法可能涉及管理单个或一群候选解，从这些候选解中迭代地生成和评估新的候选解，以查看它们是否导致改进并作为新的工作状态。

关于什么是全局搜索算法，可能会有争论；然而，使用我们的定义的全局搜索算法的三个例子包括:

*   遗传算法
*   模拟退火
*   粒子群优化算法

现在我们已经熟悉了全局优化和局部优化，让我们对两者进行比较和对比。

## 局部优化与全局优化

局部和全局搜索优化算法解决不同的问题或回答不同的问题。

当您知道自己处于全局最优区域或目标函数包含单个最优值(例如单峰)时，应使用局部优化算法。

当您对目标函数响应面的结构知之甚少时，或者当您知道函数包含局部最优时，应该使用全局优化算法。

> 局部优化，算法可能陷入局部最优而没有找到全局最优。

—第 37 页，[计算智能:导论](https://amzn.to/2FUZp1v)，2007。

将局部搜索算法应用于需要全局搜索算法的问题将会产生较差的结果，因为局部搜索会被局部最优解捕获(欺骗)。

*   **本地搜索**:当你在全局最优区域时。
*   **全局搜索**:当你知道有局部最优时。

只要算法所做的假设成立，局部搜索算法通常会给出与定位全局最优解相关的计算复杂度保证。

全局搜索算法通常很少给出关于定位全局最优解的授权。因此，全局搜索通常用于难度足够大的问题，即“T0”好的“T1”或“T2”足够好的“T3”解决方案比根本没有解决方案更受欢迎。这可能意味着相对较好的局部最优解，而不是真正的全局最优解，如果定位全局最优解很难的话。

多次重新运行或重新启动算法并记录每次运行找到的最优解通常是合适的，这样可以让您确信已经找到了相对较好的解决方案。

*   **局部搜索**:针对需要全局解的狭窄问题。
*   **全局搜索**:寻找全局最优可能难以解决的大问题。

我们通常对目标函数的响应面知之甚少，例如，局部或全局搜索算法是否最合适。因此，可能希望用局部搜索算法建立表现基线，然后探索全局搜索算法，看看它是否能表现得更好。如果不能，它可能表明问题确实是单峰的，或者适合于局部搜索算法。

*   **最佳实践**:通过局部搜索建立基线，然后在未知的目标函数上探索全局搜索。

局部优化比全局优化更容易解决。因此，绝大多数关于数学优化的研究都集中在局部搜索技术上。

> 对一般非线性规划的研究很大一部分集中在局部优化方法上，因此得到了很好的发展。

—第 9 页，[凸优化](https://amzn.to/2FXnvsn)，2004。

全局搜索算法在搜索空间的导航中通常是粗糙的。

> 许多种群方法在全局搜索中表现良好，能够避免局部极小值并找到设计空间的最佳区域。不幸的是，与下降法相比，这些方法在局部搜索中表现不佳。

—第 162 页，[优化算法](https://amzn.to/2G1Wt2K)，2019。

因此，他们可能会为一个好的局部最优值或全局最优值定位流域，但可能无法在流域内找到最佳解决方案。

> 局部和全局优化技术可以结合形成混合训练算法。

—第 37 页，[计算智能:导论](https://amzn.to/2FUZp1v)，2007。

因此，将局部搜索应用于由全局搜索算法找到的最优候选解是一种良好的做法。

*   **最佳实践**:对通过全局搜索找到的解决方案应用本地搜索。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   [凸优化](https://amzn.to/2FXnvsn)，2004。
*   [计算智能:导论](https://amzn.to/2FUZp1v)，2007。
*   [优化算法](https://amzn.to/2G1Wt2K)，2019。

### 文章

*   [本地搜索(优化)，维基百科](https://en.wikipedia.org/wiki/Local_search_(optimization))。
*   [全局优化，维基百科](https://en.wikipedia.org/wiki/Global_optimization)。

## 摘要

在本教程中，您发现了局部优化和全局优化之间的实际差异。

具体来说，您了解到:

*   局部优化包括为搜索空间的特定区域寻找最优解，或者为没有局部最优解的问题寻找全局最优解。
*   全局优化包括寻找包含局部最优解的问题的最优解。
*   如何以及何时使用局部和全局搜索算法，以及如何同时使用这两种方法。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。