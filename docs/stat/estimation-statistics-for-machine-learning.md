# 机器学习中的评估统计的温和介绍

> 原文： [`machinelearningmastery.com/estimation-statistics-for-machine-learning/`](https://machinelearningmastery.com/estimation-statistics-for-machine-learning/)

统计假设检验可用于指示两个样本之间的差异是否是由于随机机会，但不能评论差异的大小。

被称为“_ 新统计 _”的一组方法正在增加使用而不是 p 值或者除了 p 值之外，以便量化效应的大小和估计值的不确定性的量。这组统计方法被称为“_ 估计统计 _”。

在本教程中，您将发现对估计统计量的温和介绍，作为统计假设检验的替代或补充。

完成本教程后，您将了解：

*   效应大小方法涉及量化样本之间的关联或差异。
*   区间估计方法涉及量化点估计周围的不确定性。
*   Meta 分析涉及量化多个类似独立研究中影响的程度。

让我们开始吧。

![A Gentle Introduction to Estimation Statistics for Machine Learning](img/467ac6ed2b02ce9fa484e6c28408d327.jpg)

机器学习估算统计的温和介绍
[NicolásBoullosa](https://www.flickr.com/photos/faircompanies/2184522893/)的照片，保留一些权利。

## 教程概述

本教程分为 5 个部分;他们是：

1.  假设检验的问题
2.  估算统计
3.  规模效应
4.  区间估计
5.  Meta 分析

## 假设检验的问题

统计假设检验和 p 值的计算是呈现和解释结果的流行方式。

像 Student's t 检验这样的测试可用于描述两个样本是否具有相同的分布。它们可以帮助解释两个样本均值之间的差异是否真实或由于随机机会。

虽然它们被广泛使用，但它们[存在一些问题](https://en.wikipedia.org/wiki/Misunderstandings_of_p-values)。例如：

*   计算的 p 值很容易被误用和误解。
*   即使差异很小，样本之间总会有一些显着差异。

有趣的是，在过去的几十年里，人们一直在反对在研究报告中使用 p 值。例如，在 20 世纪 90 年代，[流行病学](https://journals.lww.com/epidem/pages/default.aspx)的期刊禁止使用 p 值。医学和心理学的许多相关领域也纷纷效仿。

尽管仍然可以使用 p 值，但是使用估计统计量推动了结果的呈现。

## 估算统计

估计统计是指尝试量化发现的方法。

这可能包括量化效果的大小或特定结果或结果的不确定性的数量。

> ......'估算统计'，一个术语，描述着重于效果大小估计（点估计）及其置信区间（精确估计）的方法。

- [估算统计量应取代 2016 年的显着性检验](https://www.nature.com/articles/nmeth.3729)。

估算统计是描述三种主要方法类别的术语。三类主要方法包括：

*   **效果大小**。用于量化治疗或干预的效果大小的方法。
*   **区间估计**。量化值的不确定性的方法。
*   **Meta 分析**。在多个类似研究中量化结果的方法。

我们将在以下部分中更详细地介绍这些方法组。

虽然它们并不是新的，但它们被称为“_ 新统计 _”，因为它们在统计假设检验的研究文献中的使用越来越多。

> 新的统计量涉及估计，荟萃分析和其他技术，帮助研究人员将重点从[零假设统计检验]转移。这些技术并不是新的，并且通常用于某些学科，但对于[零假设统计检验]学科，它们的使用将是新的和有益的。

- [了解新统计：影响大小，置信区间和元分析](http://amzn.to/2HxDjgC)，2012。

从统计假设方法向估计系统转变的主要原因是结果更容易在领域或研究问题的背景下进行分析和解释。

效果和不确定性的量化大小允许声明更容易理解和使用。结果更有意义。

> 知道和思考效应的幅度和精确度对于定量科学比考虑观察至少那个极端数据的概率更有用，假设绝对没有效果。

— [Estimation statistics should replace significance testing](https://www.nature.com/articles/nmeth.3729), 2016.

在统计假设检验谈论样本是否来自相同分布的情况下，估计统计可以描述差异的大小和置信度。这允许您评论一种方法与另一种方法的不同之处。

> 估计思维的重点是效果有多大;知道这通常比知道效果是否为零更有价值，这是二元思维的四肢。估计思维促使我们计划一个实验，以解决“多少......？”或“在多大程度上？？”的问题，而不仅仅是二分法无效假设统计检验的问题，“有效果吗？”

— [Understanding The New Statistics: Effect Sizes, Confidence Intervals, and Meta-Analysis](http://amzn.to/2HxDjgC), 2012.

## 规模效应

效应大小描述了治疗的大小或两个样本之间的差异。

假设检验可以评论样本之间的差异是偶然的结果还是真实的结果，而效应大小则表示样本的差异程度。

测量效果的大小是应用机器学习的重要组成部分，事实上，研究也是如此。

> 我有时被问到，研究人员做了什么？简短的回答是我们估计效果的大小。无论我们选择研究什么样的现象，我们基本上都在花费我们的职业生涯来思考新的更好的方法来估计效应量。

- 第 3 页，[影响大小的基本指南：统计功效，Meta 分析和研究结果的解释](http://amzn.to/2p8Ckfs)，2010。

有两种主要的技术用于量化影响的程度;他们是：

*   **协会**。两个样本一起变化的程度。
*   **差异**。两个样本不同的程度。

例如，关联效应大小包括相关性的计算，例如 Pearson 相关系数和 r ^ 2 确定系数。它们可以量化两个样本中的观察结果一起变化的线性或单调方式。

差异效应大小可以包括诸如科恩统计量的方法，其提供关于两个群体的平均值如何不同的标准化度量。他们寻求量化两个样本中观察值之间差异的大小。

> 效果可以是在组（例如，治疗组和未治疗组）之间的比较中显示的治疗的结果，或者它可以描述两个相关变量（例如治疗剂量和健康）之间的关联程度。

- 第 4 页，[影响大小的基本指南：统计力量，Meta 分析和研究结果的解释](http://amzn.to/2p8Ckfs)，2010。

## 区间估计

区间估计是指用于量化观察的不确定性的统计方法。

间隔将点估计转换为一个范围，该范围提供有关估计的更多信息，例如其精度，使其更易于比较和解释。

> 点估计是点，间隔表示这些点估计的不确定性。

- 第 9 页，[了解新统计：影响大小，置信区间和元分析](http://amzn.to/2HxDjgC)，2012。

通常计算有三种主要类型的间隔。他们是：

*   **容差区间**：具有特定置信水平的分布的一定比例的界限或覆盖范围。
*   **置信区间**：总体参数估计的界限。
*   **预测区间**：单次观察的界限。

公差区间可用于设定对群体中观察的期望或帮助识别异常值。置信区间可用于解释数据样本的平均值的范围，随着样本量的增加，该范围可以变得更加精确。预测间隔可用于从模型提供预测或预测的范围。

例如，当呈现模型的平均估计技能时，可以使用置信区间来提供估计精度的界限。如果要比较模型，这也可以与 p 值组合。

> 因此，置信区间为人口价值提供了一系列可能性，而不是仅基于统计显着性的任意二分法。它以牺牲 P 值的精度为代价传达更多有用的信息。然而，除了置信区间之外，实际 P 值是有用的，并且优选地两者都应该被呈现。但是，如果必须排除一个，那么它应该是 P 值。

- [置信区间而不是 P 值：估计而不是假设检验](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1339793/)，1986。

## Meta 分析

荟萃分析指的是使用多个相似研究的权重来量化更广泛的交叉研究效果。

当许多小型和类似的研究已经进行了嘈杂和相互矛盾的研究时，Meta 研究很有用。与其以任何单一研究相比，统计学方法不是将研究结论用于面值，而是将多项研究结果结合起来。

> ...更好地称为荟萃分析，完全忽略了其他人所得出的结论，而忽视了已经观察到的效果。目的是将这些独立观察结果与平均效应大小相结合，并得出关于现实世界效应的方向和幅度的总体结论。

- 第 90 页，[影响大小的基本指南：统计功效，Meta 分析和研究结果的解释](http://amzn.to/2p8Ckfs)，2010。

尽管不常用于应用机器学习，但有必要注意元分析，因为它们构成了新统计方法信任的一部分。

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   描述如何在机器学习项目中使用估算统计量的三个示例。
*   找出并总结三种对使用统计假设检验的批评。
*   搜索并找到三篇利用区间估计的研究论文。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 图书

*   [了解新统计：影响大小，置信区间和元分析](http://amzn.to/2HxDjgC)，2012。
*   [新统计学概论：估计，开放科学及其他](http://amzn.to/2FEudOI)，2016 年。
*   [影响大小的基本指南：统计力量，Meta 分析和研究结果的解释](http://amzn.to/2p8Ckfs)，2010。

### 文件

*   [估算统计量应取代 2016 年的显着性检验](https://www.nature.com/articles/nmeth.3729)。
*   [置信区间而不是 P 值：估计而不是假设检验](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1339793/)，1986。

### 用品

*   [维基百科](https://en.wikipedia.org/wiki/Estimation_statistics)的估算统计
*   [维基百科上的效果大小](https://en.wikipedia.org/wiki/Effect_size)
*   [维基百科的间隔估计](https://en.wikipedia.org/wiki/Interval_estimation)
*   [维基百科上的元分析](https://en.wikipedia.org/wiki/Meta-analysis)

## 摘要

在本教程中，您发现了对估计统计量的温和介绍，作为统计假设检验的替代或补充。

具体来说，你学到了：

*   效应大小方法涉及量化样本之间的关联或差异。
*   区间估计方法涉及量化点估计周围的不确定性。
*   Meta 分析涉及量化多个类似独立研究中影响的程度。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。