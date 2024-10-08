# 统计采样和重采样的简要介绍

> 原文： [`machinelearningmastery.com/statistical-sampling-and-resampling/`](https://machinelearningmastery.com/statistical-sampling-and-resampling/)

数据是应用机器学习的货币。因此，有效地收集和使用它是很重要的。

数据采样是指用于从域中选择观测值的统计方法，其目的是估计总体参数。而数据重采样指的是经济地使用收集的数据集来改进总体参数的估计并且有助于量化估计的不确定性的方法。

数据采样和数据重采样都是预测性建模问题所需的方法。

在本教程中，您将发现用于收集和充分利用数据的统计采样和统计重采样方法。

完成本教程后，您将了解：

*   采样是一个积极的过程，用于收集观察结果，旨在估算人口变量。
*   重采样是一种经济地使用数据样本来提高准确率并量化群体参数的不确定性的方法。
*   实际上，重采样方法使用嵌套的重采样方法。

让我们开始吧。

![A Gentle Introduction to Statistical Sampling and Resampling](img/2947bc249dab5f0cf32f497588309bf1.jpg)

统计采样和重采样
的照片由 [Ed Dunens](https://www.flickr.com/photos/blachswan/33929764716/) 拍摄，保留一些权利。

## 教程概述

本教程分为两部分;他们是：

1.  统计采样
2.  统计重采样

## 统计采样

每行数据代表对世界某事物的观察。

处理数据时，我们通常无法访问所有可能的观察结果。这可能有很多原因;例如：

*   进行更多观察可能很困难或者成本很高。
*   将所有观察结果收集在一起可能具有挑战性。
*   预计将来会有更多的观察结果。

在域中进行的观察表示可以在域中进行的所有可能观察的一些更广泛的理想化和未知群体的样本。这是一个有用的概念化，因为我们可以看到观察与理想化人口之间的分离和关系。

我们还可以看到，即使我们打算在所有可用数据上使用大数据基础设施，数据仍然代表了理想化人口的观测样本。

然而，我们可能希望估计人口的属性。我们通过使用观察样本来做到这一点。

> 采样包括选择一部分人口进行观察，以便人们可以对整个人口进行估算。

- 第 1 页，[采样](http://amzn.to/2HNgJAQ)，第三版，2012 年。

### 如何取样

统计采样是从人口中选择实例子集的过程，目的是估计人口的属性。

采样是一个积极的过程。目标是估算人口属性并控制采样的发生方式。该控制不能影响产生每个观察的过程，例如进行实验。因此，作为场的采样整齐地位于纯不受控制的观察和对照试验之间。

> 采样通常与实验设计密切相关的领域区别开来，因为在实验中，人们故意扰乱一部分人口，以便了解该行动的影响。 [...]采样通常也与观察性研究区别开来，在这些研究中，人们很少或根本无法控制对人群的观察结果。

- 第 1-2 页，[采样](http://amzn.to/2HNgJAQ)，第三版，2012。

与使用更全面或完整的数据集相比，采样有许多好处，包括降低成本和提高速度。

为了执行采样，需要您仔细定义人口以及选择（并可能拒绝）观察结果作为数据样本一部分的方法。这可能很好地通过您希望使用样本估计的总体参数来定义。

在收集数据样本之前要考虑的一些方面包括：

*   **样本目标**。您希望使用样本估算的人口属性。
*   **人口**。理论上可以进行观察的范围或领域。
*   **选择标准**。用于接受或拒绝样本中观察结果的方法。
*   **样本量**。构成样本的观察数量。

> 一些明显的问题是如何最好地获取样本并进行观察，并且一旦样本数据掌握，如何最好地使用它们来估计整个人口的特征。获得观察结果涉及样本大小，如何选择样本，使用何种观察方法以及记录哪些测量值等问题。

— Page 1, [Sampling](http://amzn.to/2HNgJAQ), Third Edition, 2012.

统计采样是一个很大的研究领域，但在应用机器学习中，您可能会使用三种类型的采样：简单随机采样，系统采样和分层采样。

*   **简单随机采样**：从域中以均匀概率抽取样本。
*   **系统采样**：使用预先指定的模式（例如每隔一段时间）绘制样本。
*   **分层采样**：在预先指定的类别（即分层）内抽取样本。

虽然这些是您可能遇到的更常见的采样类型，但还有其他技术。

### 采样错误

采样要求我们从一小组观察中对人口进行统计推断。

我们可以将样本中的属性推广到总体。这种估计和推广过程比使用所有可能的观察要快得多，但会包含错误。在许多情况下，我们可以量化估算的不确定性并添加误差条，例如置信区间。

有很多方法可以将错误引入数据样本。

两种主要类型的错误包括选择偏差和采样误差。

*   **选择偏差**。当绘制观察的方法以某种方式使样本偏斜时引起。
*   **采样错误**。由于绘图观察的随机性质导致以某种方式偏斜样本。

可能存在其他类型的错误，例如观察或测量的方式中的系统误差。

在这些情况下以及更多情况下，样本的统计特性可能与理想化人口中的预期不同，这反过来可能影响正在估计的人口的特性。

简单的方法，例如检查原始观察，摘要统计和可视化，可以帮助揭示简单的错误，例如测量损坏和一类观察的过度或不足。

然而，在采样和在采样时得出有关人口的结论时，必须小心谨慎。

## 统计重采样

一旦我们有了数据样本，它就可以用来估计总体参数。

问题是我们只对人口参数进行了单一估计，对估计的可变性或不确定性知之甚少。

解决此问题的一种方法是从我们的数据样本中多次估算人口参数。这称为重采样。

统计重采样方法是描述如何经济地使用可用数据来估计总体参数的过程。结果可以是更准确的参数估计（例如取估计的平均值）和估计的不确定性的量化（例如添加置信区间）。

重采样方法非常易于使用，几乎不需要数学知识。与专业统计方法相比，它们是易于理解和实现的方法，这些方法可能需要深入的技术技能才能选择和解释。

> 重采样方法易于学习且易于应用。除了介绍性的高中代数之外，它们不需要数学，并且适用于范围极广的学科领域。

- 第 xiii 页，[重采样方法：数据分析实用指南](http://amzn.to/2G6gMKP)，2005 年。

这些方法的缺点是它们在计算上可能非常昂贵，需要数十，数百甚至数千个重采样，以便开发人口参数的稳健估计。

> 关键的想法是重采样形成原始数据 - 直接或通过拟合模型 - 来创建复制数据集，从中可以评估感兴趣的分位数的可变性，而无需冗长且容易出错的分析计算。因为这种方法涉及使用许多复制数据集重复原始数据分析过程，所以这些有时被称为计算机密集型方法。

- 第 3 页， [自举法及其应用](http://amzn.to/2FVsmVY)，1997。

来自原始数据样本的每个新子样本用于估计总体参数。然后可以使用统计工具考虑估计的人口参数样本，以量化预期值和方差，提供估计不确定性的度量。

统计采样方法可用于从原始样本中选择子样本。

关键的区别是过程必须重复多次。这样的问题在于样本之间将存在一些关系，作为将在多个子样本之间共享的观察。这意味着子样本和估计的总体参数不是严格相同且独立分布的。这对于对下游估计的种群参数的样本进行的统计测试具有影响，即可能需要成对的统计测试。

您可能遇到的两种常用的重采样方法是 k-fold 交叉验证和引导程序。

*   **Bootstrap** 。从替换的数据集中抽取样本（允许相同的样本在样本中出现多次），其中未被抽入数据样本的那些实例可用于测试集。
*   **k 折交叉验证**。数据集被划分为 k 个组，其中每个组被赋予被用作保持测试集的机会，其余组作为训练集。

k 折交叉验证方法特别适用于评估预测模型，该预测模型在数据的一个子集上重复训练并在第二个保持的数据子集上进行评估。

> 通常，用于估计模型表现的重采样技术类似地操作：样本子集用于拟合模型，剩余样本用于估计模型的功效。重复此过程多次，并汇总和汇总结果。技术上的差异通常围绕选择子样本的方法。

- 第 69 页， [Applied Predictive Modeling](http://amzn.to/2Fmrbib) ，2013。

引导方法可以用于相同的目的，但是用于估计总体参数的更通用和更简单的方法。

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   列举两个机器学习项目中需要统计采样的例子。
*   列出在机器学习项目中需要统计重采样时的两个示例。
*   查找使用重采样方法的论文，该方法又使用嵌套统计采样方法（提示：k-fold 交叉验证和分层采样）。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 图书

*   [采样](http://amzn.to/2HNgJAQ)，第三版，2012 年。
*   [采样技术](http://amzn.to/2FMh8XF)，第 3 版，1977 年。
*   [重采样方法：数据分析实用指南](http://amzn.to/2G6gMKP)，2005。
*   [引导程序简介](http://amzn.to/2praHye)，1994。
*   [自举法及其应用](http://amzn.to/2FVsmVY)，1997。
*   [Applied Predictive Modeling](http://amzn.to/2Fmrbib) ，2013。

### 用品

*   [维基百科上的样本（统计量）](https://en.wikipedia.org/wiki/Sample_(statistics))
*   [维基百科上的简单随机样本](https://en.wikipedia.org/wiki/Simple_random_sample)
*   [维基百科上的系统采样](https://en.wikipedia.org/wiki/Systematic_sampling)
*   [维基百科上的分层采样](https://en.wikipedia.org/wiki/Stratified_sampling)
*   [维基百科上的重新取样（统计）](https://en.wikipedia.org/wiki/Resampling_(statistics))
*   [维基百科上的引导（统计）](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))
*   [维基百科](https://en.wikipedia.org/wiki/Cross-validation_(statistics))的交叉验证（统计）

## 摘要

在本教程中，您发现了用于收集和充分利用数据的统计采样和统计重采样方法。

具体来说，你学到了：

*   采样是收集观察意图估计人口变量的积极过程。
*   重采样是一种经济地使用数据样本来提高准确率并量化群体参数的不确定性的方法。
*   实际上，重采样方法使用嵌套的重采样方法。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。