# 机器学习降维介绍

> 原文：<https://machinelearningmastery.com/dimensionality-reduction-for-machine-learning/>

最后更新于 2020 年 6 月 30 日

数据集的输入变量或特征的数量称为其维数。

降维是指减少数据集中输入变量数量的技术。

更多的输入特征通常会使预测建模任务更具挑战性，更一般地说，这被称为维数灾难。

高维统计和降维技术常用于数据可视化。然而，这些技术可以用于应用机器学习，以简化分类或回归数据集，从而更好地拟合预测模型。

在这篇文章中，你会发现一个关于机器学习降维的温和介绍

看完这篇文章，你会知道:

*   大量的输入特征会导致机器学习算法表现不佳。
*   降维是与减少输入特征的数量有关的一般研究领域。
*   降维方法包括特征选择、线性代数方法、投影方法和自动编码器。

**用我的新书[机器学习的数据准备](https://machinelearningmastery.com/data-preparation-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2020 年 5 月更新**:更改章节标题更准确。

![A Gentle Introduction to Dimensionality Reduction for Machine Learning](img/ad1b3717ce25569678b62d1d2ed65326.png)

机器学习降维的温和介绍
凯文·贾勒特摄，版权所有。

### 概观

本教程分为三个部分；它们是:

1.  多输入变量问题
2.  降维
3.  降维技术
    1.  特征选择方法
    2.  矩阵分解
    3.  流形学习
    4.  自动编码器方法
    5.  降维技巧

## 多输入变量问题

当输入变量过多时，机器学习算法的表现会下降。

如果您的数据是用行和列来表示的，例如在电子表格中，那么输入变量就是作为输入提供给模型以预测目标变量的列。输入变量也称为特征。

我们可以将 n 维特征空间上表示维度的数据列和数据行视为该空间中的点。这是数据集的一种有用的几何解释。

特征空间中有大量的维度可能意味着该空间的体积非常大，反过来，我们在该空间中的点(数据行)通常代表一个小的且不具有代表性的样本。

这可能会极大地影响机器学习算法在具有许多输入特征的数据上的表现，通常被称为“维度诅咒””

因此，通常希望减少输入特征的数量。

这减少了特征空间的维数，因此得名“*降维*”

## 降维

降维是指减少训练数据中输入变量数量的技术。

> 当处理高维数据时，通过将数据投影到捕捉数据“本质”的低维子空间来降低维数通常是有用的。这叫做降维。

—第 11 页，[机器学习:概率视角](https://amzn.to/2ucStHi)，2012。

高维可能意味着数百、数千甚至数百万个输入变量。

更少的输入维数通常意味着机器学习模型中相应更少的参数或更简单的结构，称为自由度。具有太多自由度的模型可能会过度训练数据集，因此在新数据上可能表现不佳。

希望有简单的模型能够很好地概括，进而输入数据中的输入变量很少。对于输入数量和模型自由度通常密切相关的线性模型尤其如此。

> 维数灾难的根本原因是高维函数可能比低维函数复杂得多，而且这些复杂性更难辨别。打破诅咒的唯一方法是整合关于正确数据的知识。

—第 15 页，[图案分类](https://amzn.to/2RlneT5)，2000 年。

降维是在建模之前对数据执行的数据准备技术。它可以在数据清理和数据缩放之后、训练预测模型之前执行。

> …降维产生了目标概念的更紧凑、更容易解释的表示，将用户的注意力集中在最相关的变量上。

—第 289 页，[数据挖掘:实用机器学习工具与技术](https://amzn.to/2tlRP9V)，2016 年第 4 版。

因此，在使用最终模型进行预测时，对训练数据执行的任何降维也必须对新数据执行，例如测试数据集、验证数据集和数据。

## 降维技术

有许多技术可以用于降维。

在本节中，我们将回顾主要的技术。

### 特征选择方法

也许最常见的是所谓的特征选择技术，它使用评分或统计方法来选择保留哪些特征和删除哪些特征。

> ……执行特征选择，以删除对分类问题帮助不大的“无关”特征。

—第 86 页，[机器学习:概率视角](https://amzn.to/2ucStHi)，2012。

两类主要的特征选择技术包括包装方法和过滤方法。

有关一般特征选择的更多信息，请参见教程:

*   [特征选择介绍](https://machinelearningmastery.com/an-introduction-to-feature-selection/)

顾名思义，包装器方法包装一个机器学习模型，用输入特征的不同子集拟合和评估该模型，并选择导致最佳模型表现的子集。RFE 是包装特征选择方法的一个例子。

过滤方法使用评分方法，如特征和目标变量之间的相关性，来选择最具预测性的输入特征子集。例子包括皮尔逊相关和卡方检验。

有关基于过滤器的特征选择方法的更多信息，请参见教程:

*   [如何选择机器学习的特征选择方法](https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/)

### 矩阵分解

线性代数的技术可以用于降维。

具体来说，矩阵分解方法可用于将数据集矩阵简化为其组成部分。

例子包括特征分解和奇异值分解。

有关矩阵分解的更多信息，请参见教程:

*   [机器学习矩阵分解的简单介绍](https://machinelearningmastery.com/introduction-to-matrix-decompositions-for-machine-learning/)

然后，可以对这些部分进行排序，并且可以选择这些部分的子集，该子集最好地捕捉可用于表示数据集的矩阵的显著结构。

最常用的成分排序方法是主成分分析，简称 PCA。

> 最常见的降维方法叫做主成分分析。

—第 11 页，[机器学习:概率视角](https://amzn.to/2ucStHi)，2012。

有关主成分分析的更多信息，请参见教程:

*   [如何在 Python 中从头计算主成分分析](https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/)

### 流形学习

来自高维统计的技术也可以用于降维。

> 在数学中，投影是一种以某种方式转换数据的函数或映射。

—第 304 页，[数据挖掘:实用机器学习工具与技术](https://amzn.to/2tlRP9V)，2016 年第 4 版。

这些技术有时被称为“*流形学习*”，用于创建高维数据的低维投影，通常用于数据可视化的目的。

投影被设计成既创建数据集的低维表示，同时最好地保留数据中的显著结构或关系。

多种学习技术的例子包括:

*   [科霍宁自组织映射(SOM)](https://en.wikipedia.org/wiki/Self-organizing_map) 。
*   [萨蒙斯地图](https://en.wikipedia.org/wiki/Sammon_mapping)
*   多维标度(MDS)
*   分布式随机邻居嵌入(t-SNE)。

投影中的特征通常与原始列关系不大，例如它们没有列名，这可能会使初学者感到困惑。

### 自动编码器方法

可以构建深度学习神经网络来执行降维。

一种流行的方法叫做自动编码器。这包括构建一个自我监督的学习问题，其中模型必须正确地再现输入。

有关自我监督学习的更多信息，请参见教程:

*   [机器学习中的 14 种不同类型的学习](https://machinelearningmastery.com/types-of-learning-in-machine-learning/)

使用网络模型，试图将数据流压缩到比原始输入数据维度少得多的瓶颈层。模型中位于瓶颈之前并包含瓶颈的部分称为编码器，模型中读取瓶颈输出并重构输入的部分称为解码器。

> 自动编码器是一种用于降维和特征发现的无监督神经网络。更准确地说，自动编码器是一个被训练来预测输入本身的前馈神经网络。

—第 1000 页，[机器学习:概率视角](https://amzn.to/2ucStHi)，2012。

训练后，解码器被丢弃，瓶颈的输出被直接用作输入的降维。然后，由该编码器转换的输入可以被馈送到另一个模型，不一定是神经网络模型。

> 深度自动编码器是非线性降维的有效框架。一旦这样的网络已经建立，编码器的最顶层，代码层 hc，可以被输入到监督分类过程。

—第 448 页，[数据挖掘:实用机器学习工具与技术](https://amzn.to/2tlRP9V)，2016 年第 4 版。

编码器的输出是一种投影类型，与其他投影方法一样，瓶颈输出与原始输入变量之间没有直接关系，这使得它们难以解释。

有关自动编码器的示例，请参见教程:

*   [LSTM 自动编码器简介](https://machinelearningmastery.com/lstm-autoencoders/)

### 降维技巧

没有最好的降维技术，也没有技术到问题的映射。

相反，最好的方法是使用系统的受控实验来发现什么样的降维技术，当与您选择的模型配对时，会在您的数据集上产生最佳表现。

典型地，线性代数和流形学习方法假设所有输入特征具有相同的规模或分布。这表明，如果输入变量具有不同的比例或单位，那么在使用这些方法之前对数据进行规范化或标准化是一种良好的做法。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [特征选择介绍](https://machinelearningmastery.com/an-introduction-to-feature-selection/)
*   [如何选择机器学习的特征选择方法](https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/)
*   [机器学习矩阵分解的简单介绍](https://machinelearningmastery.com/introduction-to-matrix-decompositions-for-machine-learning/)
*   [如何在 Python 中从头计算主成分分析](https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/)
*   [机器学习中的 14 种不同类型的学习](https://machinelearningmastery.com/types-of-learning-in-machine-learning/)
*   [LSTM 自动编码器简介](https://machinelearningmastery.com/lstm-autoencoders/)

### 书

*   [机器学习:概率视角](https://amzn.to/2ucStHi)，2012。
*   [数据挖掘:实用机器学习工具与技术](https://amzn.to/2tlRP9V)，第 4 版，2016。
*   [图案分类](https://amzn.to/2RlneT5)，2000。

### 应用程序接口

*   [流形学习，Sklearn](https://Sklearn.org/stable/modules/manifold.html) 。
*   [分解分量中的信号(矩阵分解问题)，Sklearn](https://Sklearn.org/stable/modules/decomposition.html) 。

### 文章

*   [降维，维基百科](https://en.wikipedia.org/wiki/Dimensionality_reduction)。
*   [维度的诅咒，维基百科](https://en.wikipedia.org/wiki/Curse_of_dimensionality)。

## 摘要

在这篇文章中，你发现了一个关于机器学习降维的温和介绍。

具体来说，您了解到:

*   大量的输入特征会导致机器学习算法表现不佳。
*   降维是与减少输入特征的数量有关的一般研究领域。
*   降维方法包括特征选择、线性代数方法、投影方法和自动编码器。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。