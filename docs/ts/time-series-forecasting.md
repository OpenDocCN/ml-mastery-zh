# 什么是时间序列预测？

> 原文： [`machinelearningmastery.com/time-series-forecasting/`](https://machinelearningmastery.com/time-series-forecasting/)

时间序列预测是机器学习的一个重要领域，往往被忽视。

这很重要，因为有很多涉及时间成分的预测问题。这些问题被忽略了，因为正是这个时间组件使得时间序列问题更难以处理。

在这篇文章中，您将发现时间序列预测。

阅读这篇文章后，你会知道：

*   时间序列，时间序列分析和时间序列预测的标准定义。
*   时间序列数据中要考虑的重要组成部分。
*   时间序列的例子，使您的理解具体化。

让我们开始吧。

![What is Time Series Forecasting?](img/f012e3d8f614f2a6bcf26e571a8ee1b8.jpg)

什么是时间序列预测？
照片由 [Dennis Kleine](https://www.flickr.com/photos/theartificer/6473624063/) 拍摄，保留一些权利。

## 时间序列

普通机器学习数据集是观察的集合。

例如：

```py
observation #1
observation #2
observation #3
```

时间确实在普通机器学习数据集中发挥作用。

当在未来某个日期之前可能不知道实际结果时，对新数据做出预测。未来正在被预测，但所有先前的观察几乎总是被平等对待。也许用一些非常小的时间动力来克服“_ 概念漂移 _”的想法，例如仅使用去年的观测而不是所有可用的数据。

时间序列数据集是不同的。

时间序列在观察之间增加了明确的顺序依赖：时间维度。

此附加维度既是约束，也是提供附加信息源的结构。

> 时间序列是按时间顺序进行的一系列观察。

- 第 1 页，[时间序列分析：预测和控制](http://www.amazon.com/dp/1118675029?tag=inspiredalgor-20)。

例如：

```py
Time #1, observation
Time #2, observation
Time #3, observation
```

## 描述与预测

根据我们是否有兴趣了解数据集或做出预测，我们有不同的目标。

了解一个名为 _ 时间序列分析 _ 的数据集可以帮助做出更好的预测，但不是必需的，并且可能导致大量的技术投资在时间和专业知识上与预期结果没有直接对应，这预测了未来。

> 在描述性建模或时间序列分析中，时间序列被建模以根据季节性模式，趋势，与外部因素的关系等来确定其组成部分。 ...相比之下，时间序列预测使用时间序列中的信息（可能包含其他信息）来预测该系列的未来值

- 第 18-19 页， [R 实用时间序列预测：动手指南](http://www.amazon.com/dp/0997847913?tag=inspiredalgor-20)。

### 时间序列分析

使用经典统计时，主要关注的是时间序列的分析。

时间序列分析涉及开发最能捕获或描述观察到的时间序列的模型，以便了解根本原因。该研究领域寻求时间序列数据集背后的“_ 为什么 _”。

这通常涉及对数据形式进行假设并将时间序列分解为构成组件。

描述性模型的质量取决于它描述所有可用数据的程度以及它为更好地通知问题域提供的解释。

> 时间序列分析的主要目标是开发数学模型，从样本数据中提供合理的描述

- 第 11 页，[时间序列分析及其应用：R 实例](http://www.amazon.com/dp/144197864X?tag=inspiredalgor-20)

### 时间序列预测

对时间做出预测称为时间序列数据的经典统计处理中的外推。

更多现代领域专注于该主题，并将其称为时间序列预测。

预测涉及使模型适合历史数据并使用它们来预测未来的观测。

描述性模型可以借用未来（即平滑或消除噪声），它们只是寻求最好地描述数据。

预测中的一个重要区别是，未来完全不可用，只能根据已经发生的事情进行估算。

> 时间序列分析的目的通常是双重的：理解或模拟产生观测序列的随机机制，并根据该序列的历史预测或预测序列的未来值

- 第 1 页，[时间序列分析：应用于 R](http://www.amazon.com/dp/0387759581?tag=inspiredalgor-20) 。

时间序列预测模型的技能取决于其在预测未来方面的表现。这通常是以能够解释为什么进行特定预测，置信区间甚至更好地理解问题背后的根本原因为代价的。

## 时间序列的组成部分

时间序列分析提供了一系列技术来更好地理解数据集。

也许最有用的是将时间序列分解为 4 个组成部分：

1.  **等级**。系列的基线值，如果它是一条直线。
2.  **趋势**。随着时间的推移，系列的可选且通常线性增加或减少的行为。
3.  **季节性**。随时间变化的可选重复模式或行为周期。
4.  **噪音**。观测中的可选可变性，无法通过模型解释。

所有时间序列都有一个级别，大多数都有噪音，趋势和季节性是可选的。

> 许多时间序列的主要特征是趋势和季节变化......大多数时间序列的另一个重要特征是在时间上紧密相连的观察往往是相关的（依赖于序列）

- 第 2 页， [R](http://www.amazon.com/dp/0387886974?tag=inspiredalgor-20) 入门时间序列

可以认为这些组成成分以某种方式组合以提供观察到的时间序列。例如，它们可以一起添加以形成如下模型：

```py
y = level + trend + seasonality + noise
```

可以在行为和组合方式上对这些组件做出假设，这允许使用传统的统计方法对它们进行建模。

这些组件也可能是预测未来价值的最有效方式，但并非总是如此。

在这些经典方法不能产生有效表现的情况下，这些组件可能仍然是有用的概念，甚至是替代方法的输入。

## 关注预测

在预测时，了解您的目标非常重要。

使用苏格拉底方法并提出许多问题来帮助您放大预测性建模问题的具体细节。例如：

1.  **你有多少数据，你可以一起收集这些数据吗？** 更多数据通常更有帮助，为探索性数据分析，模型测试和调整以及模型保真度提供了更多机会。
2.  **预测所需的时间范围是多少？短期，中期还是长期？** 更短的时间范围通常更容易预测，信心更高。
3.  **预测可以随时间更新，还是必须进行一次并保持静态？** 在新信息可用时更新预测通常可以获得更准确的预测。
4.  **需要预测的时间频率是多少？** 通常可以在较低或较高频率下做出预测，从而可以利用下采样和数据上采样，从而在建模时提供优势。

时间序列数据通常需要清理，缩放甚至转换。

例如：

*   **频率**。也许数据的提供频率太高而无法建模，或者在某些模型中需要重采样的时间间隔不均匀。
*   **异常值**。也许存在需要识别和处理的腐败或极端异常值。
*   **缺少**。也许存在需要插补或估算的间隙或缺失数据。

通常，时间序列问题是实时的，不断为预测提供新的机会。这为时间序列预测增添了诚实，可以快速排除错误假设，建模错误以及我们可能欺骗自己的所有其他方式。

## 时间序列预测的例子

时间序列预测问题几乎无穷无尽。

以下是一系列行业的 10 个例子，使时间序列分析和预测的概念更加具体。

*   每年按州预测玉米产量。
*   预测几秒钟内的脑电图迹线是否表明患者是否癫痫发作。
*   每天预测股票的收盘价。
*   每年预测一个城市所有医院的出生率。
*   预测商店每天销售的产品销售额。
*   每天预测通过训练站的乘客数量。
*   每个季度预测一个州的失业率。
*   每小时预测服务器上的利用率需求。
*   预测每个繁殖季节的兔子种群数量。
*   每天预测一个城市的汽油平均价格。

我希望您能够将这些示例中的一个或多个与您自己的时间序列相关联，预测您想要解决的问题。

## 摘要

在这篇文章中，您发现了时间序列预测。

具体来说，你学到了：

*   关于时间序列数据以及时间序列分析和时间序列预测之间的差异。
*   在执行分析时可以将时间序列分解成的组成部分。
*   时间序列预测问题的例子，使这些想法具体化。

您对时间序列预测或此帖有任何疑问吗？
在下面的评论中提出您的问题。