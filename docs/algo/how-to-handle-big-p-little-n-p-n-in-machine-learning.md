# 如何在机器学习中处理大`p`小`n`（`p >> n`）

> 原文：<https://machinelearningmastery.com/how-to-handle-big-p-little-n-p-n-in-machine-learning/>

最后更新于 2020 年 8 月 19 日

#### 如果我的数据集中的列比行多怎么办？

机器学习数据集通常是由行和列组成的结构化或表格数据。

作为模型输入的列称为预测器或“ *p* ”，行是样本“ *n* ”。大多数机器学习算法假设样本比预测值多得多，表示为 p < < n。

有时情况并非如此，数据集中的预测因子比样本多得多，称为“**大 p，小 n** ，记为**p>T5【n】T3。这些问题通常需要专门的数据准备和建模算法来正确解决。**

在本教程中，您将发现大 p、小 n 或 p >> n 机器学习问题的挑战。

完成本教程后，您将知道:

*   大多数机器学习问题比预测器有更多的样本，并且大多数机器学习算法在训练过程中做出这种假设。
*   有些建模问题的预测因子比样本多得多，称为 p >> n。
*   建模预测因子多于样本的机器学习数据集时要探索的算法。

**用我的新书[掌握机器学习算法](https://machinelearningmastery.com/master-machine-learning-algorithms/)启动你的项目**，包括*分步教程*和所有示例的 *Excel 电子表格*文件。

我们开始吧。

![How to Handle Big-p, Little-n (p >> n) in Machine Learning](img/382735c7d23e5fe06ecad92453012df3.png)

如何处理机器学习中的大 p，小 n(p > > n)Phil Dolby 摄，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

*   预测值(p)和样本(n)
*   机器学习假设 p << n
*   如何处理 p >> n

## 预测值(p)和样本(n)

考虑一个预测建模问题，例如分类或回归。

数据集是结构化数据或表格数据，就像您可能在 Excel 电子表格中看到的那样。

有列和行。大多数列将用作模型的输入，一列将代表要预测的输出或变量。

> 输入有不同的名称，如预测因子、独立变量、特征，或者有时只是变量。输出变量——在本例中为销售额——通常被称为响应或因变量，通常用符号 y 表示

—第 15 页，[R](https://amzn.to/2NNJfaT)中应用的统计学习介绍，2017。

每列代表一个变量或样本的一个方面。表示模型输入的列称为预测器。

每一行代表一个样本，其值跨越每一列或特征。

*   **预测器**:数据集的输入列，也称为输入变量或特征。
*   **样本**:数据集的行，也称为观察、示例或实例。

在机器学习中，通常用预测器和样本来描述训练数据集。

使用术语“ *p* 描述数据集中预测因子的数量，使用术语“ *n* 或有时使用“ *N* 描述数据集中样本的数量”。

*   **p** :数据集中预测因子的个数。
*   **n** :数据集中的样本数。

具体来说，我们来看看[鸢尾花分类](https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv)的问题。

下面是该数据集前五行的示例。

```py
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5.0,3.6,1.4,0.2,Iris-setosa
...
```

该数据集有五列 150 行。

前四列是输入，第五列是输出，这意味着有四个预测器。

我们将鸢尾花数据集描述为:

*   p=4，n=150。

## 机器学习假设 p << n

几乎总是预测器的数量( *p* )会小于样本的数量( *n* )。

通常要小得多。

我们可以把这个期望概括为 p << n, where “<数学不等式，意思是“*比*少很多。”

*   **p < < n** :通常我们的预测因子比样本少。

为了演示这一点，让我们看几个更标准的机器学习数据集:

*   [皮马印第安人糖尿病](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv) : p=8，n=768
*   [玻璃标识](https://raw.githubusercontent.com/jbrownlee/Datasets/master/glass.csv) : p=9，n=214
*   [波士顿住宅](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv) : p=13，n=506

大多数机器学习算法基于样本比预测值多得多的假设进行操作。

考虑预测因子和样本的一种方法是采取几何透视法。

考虑一个超立方体，其中预测数( *p* )定义了超立方体的维数。这个超立方体的体积是可以从域中提取的可能样本的范围。样本数量( *n* )是从域中抽取的实际样本，您必须使用这些样本来建模您的预测建模问题。

这是应用机器学习中公理“[获取尽可能多的数据](https://machinelearningmastery.com/much-training-data-required-machine-learning/)”的基本原理。这是一个收集足够有代表性的 T2 问题领域样本的愿望。

随着维数( *p* )的增加，域的体积呈指数级增加。这又需要来自域的更多样本( *n* )来为学习算法提供域的有效覆盖。我们不需要领域的全面覆盖，只需要可能观察到的东西。

这种对高维空间进行有效采样的挑战通常被称为维度诅咒。

机器学习算法通过假设从输入到输出的映射函数的数据和结构来克服维数灾难。他们增加了偏见。

> 维数灾难的根本原因是高维函数可能比低维函数复杂得多，而且这些复杂性更难辨别。打破诅咒的唯一方法是整合关于正确数据的知识。

—第 15 页，[图案分类](https://amzn.to/2RlneT5)，2000 年。

随着预测器数量的增加，许多使用距离度量和其他局部模型(在特征空间中)的机器学习算法的表现通常会下降。

> 当特征 p 的数量较大时，KNN 和其他仅使用必须进行预测的测试观测值附近的观测值进行预测的局部方法的表现趋于恶化。这种现象被称为维数灾难，它与非参数方法在 p 较大时通常表现不佳的事实有关。

—第 168 页，[R](https://amzn.to/2NNJfaT)中应用的统计学习介绍，2017。

预测器的数量并不总是少于样本的数量。

## 如何处理 p >> n

根据定义，一些预测建模问题具有比样本更多的预测因子。

预测因子往往比样本多得多。

这经常被描述为“*大-p，小-n* ”、“*大-p，小-n* ”，或者更紧凑地描述为“p > > n”，其中“> >”是一个[数学不等式](https://en.wikipedia.org/wiki/Inequality_(mathematics))运算符，意思是“*比*多得多。”

> …特征数 p 远大于观测数 N 的预测问题，通常写成 p >> N

—第 649 页，[统计学习的要素:数据挖掘、推理和预测](https://amzn.to/3aD4dDd)，2016。

从几何角度考虑这个问题。

现在，领域不再有几十个维度(或更少)，而是有几千个维度，只有几十个样本来自这个空间。我们不能期望有任何领域的代表性样本。

p >> n 问题的很多例子来自医学领域，那里的患者人群少，描述性特征多。

> 与此同时，出现了一些应用，其中实验单元的数量相对较少，但底层维度却很庞大；说明性的例子可能包括图像分析、微阵列分析、文件分类、天文学和大气科学。

——[高维数据的统计挑战](https://royalsocietypublishing.org/doi/full/10.1098/rsta.2009.0159)，2009。

p >> n 问题的一个常见例子是[基因表达阵列](https://en.wikipedia.org/wiki/Gene_expression)，其中可能有数千个基因(预测因子)，只有几十个样本。

> 基因表达阵列通常有 50 到 100 个样本和 5000 到 20000 个变量(基因)。

——[表达式数组与 p > > n 问题](http://ww.web.stanford.edu/~hastie/Papers/pgtn.pdf)，2003。

鉴于大多数机器学习算法假设的样本比预测值多得多，这在建模时带来了挑战。

具体来说，标准机器学习模型所做的假设可能会导致模型出现意外行为、提供误导性结果或完全失败。

> ……模型不能“开箱即用”使用，因为标准拟合算法都要求 p < n；事实上，通常的经验法则是样本数量是变量的五到十倍。

——[表达式数组与 p > > n 问题](http://ww.web.stanford.edu/~hastie/Papers/pgtn.pdf)，2003。

当使用机器学习模型时，p >> n 问题的一个主要问题是过度拟合训练数据集。

由于缺少样本，大多数模型无法进行归纳，而是学习训练数据中的统计噪声。这使得模型在训练数据集上表现良好，但在问题域的新示例上表现不佳。

这也是一个很难诊断的问题，因为缺少样本不允许使用测试或验证数据集来评估模型过拟合。因此，在 p >> n 问题上评估模型时，通常使用省略式交叉验证(LOOCV)。

有许多方法可以处理 p >> n 型分类或回归问题。

一些例子包括:

### 忽略 p 和 n

一种方法是忽略 p 和 n 的关系，评估标准的机器学习模型。

这可以被认为是比较任何其他更专门的干预措施的基线方法。

### 特征选择

[特征选择](https://machinelearningmastery.com/an-introduction-to-feature-selection/)包括选择预测器的子集，用作预测模型的输入。

常见的技术包括基于特征与目标变量的统计关系(例如相关性)选择特征的过滤方法，以及在预测目标变量时基于特征对模型的贡献选择特征的包装方法(例如 RFE)。

可以对一套特征选择方法进行评估和比较，也许可以采用积极的方式将输入特征的数量大幅减少到那些被确定为最关键的特征。

> 当 p 较大时，特征选择是分类器的重要科学要求。

—第 658 页，[统计学习的要素:数据挖掘、推理和预测](https://amzn.to/3aD4dDd)，2016。

有关特征选择的更多信息，请参见教程:

*   [如何选择机器学习的特征选择方法](https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/)

### 投影方法

投影方法创建了样本的低维表示，保留了数据中观察到的关系。

它们通常用于可视化，尽管这些技术的降维特性也可能使它们作为减少预测器数量的数据转换非常有用。

这可能包括线性代数的技术，例如[奇异值分解](https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/)和[主成分分析](https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/)。

> 当 p > N 时，通过奇异值分解，计算可以在 N 维空间而不是 p 维空间进行

—第 659 页，[统计学习的要素:数据挖掘、推理和预测](https://amzn.to/3aD4dDd)，2016。

它还可能包括经常用于可视化的多种学习算法，如 t-SNE。

### 正则化算法

标准机器学习算法可以适于在训练过程中使用正则化。

这将根据所用特征的数量或特征的权重来惩罚模型，鼓励模型表现良好，并最小化模型中使用的预测器的数量。

这可以在训练期间充当一种自动特征选择，并且可以涉及扩充现有模型(例如，正则化线性回归和正则化逻辑回归)或者使用诸如 LARS 和 LASSO 的专门方法。

没有最好的方法，建议使用对照实验来测试一套不同的方法。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 报纸

*   [表达式数组与 p > > n 问题](http://ww.web.stanford.edu/~hastie/Papers/pgtn.pdf)，2003。
*   [高维数据的统计挑战](https://royalsocietypublishing.org/doi/full/10.1098/rsta.2009.0159)，2009。

### 书

*   [图案分类](https://amzn.to/2RlneT5)，2000。
*   第十八章，高维问题:p >> N，[统计学习的要素:数据挖掘、推理和预测](https://amzn.to/3aD4dDd)，2016。
*   [R](https://amzn.to/2NNJfaT)中应用的统计学习导论，2017。

### 文章

*   [不等式(数学)，维基百科](https://en.wikipedia.org/wiki/Inequality_(mathematics))。
*   [维度的诅咒，维基百科](https://en.wikipedia.org/wiki/Curse_of_dimensionality)。
*   [大 p，小 n](https://www.johndcook.com/blog/2016/01/07/big-p-little-n/) ，2016 年。

## 摘要

在本教程中，您发现了大 p、小 n 或 p >> n 机器学习问题的挑战。

具体来说，您了解到:

*   机器学习数据集可以用预测数(p)和样本数(n)来描述。
*   大多数机器学习问题比预测器有更多的样本，并且大多数机器学习算法在训练过程中做出这种假设。
*   有些建模问题的预测因子比样本多得多，比如医学上的问题，称为 p >> n，可能需要使用专门的算法。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。