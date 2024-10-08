# 机器学习中的线性回归

> 原文： [`machinelearningmastery.com/linear-regression-for-machine-learning/`](https://machinelearningmastery.com/linear-regression-for-machine-learning/)

线性回归可能是统计学和机器学习中最知名且易于理解的算法之一。

在这篇文章中，您将发现线性回归算法，它如何工作以及如何在机器学习项目中最好地使用它。在这篇文章中，您将学习：

*   为什么线性回归属于统计学和机器学习。
*   知道线性回归的众多名称。
*   用于创建线性回归模型的表示和学习算法。
*   如何使用线性回归进行建模时最好地准备数据。

您不需要知道任何统计量或线性代数来理解线性回归。这是对该技术的温和高级介绍，为您提供足够的背景，以便能够有效地使用它来解决您自己的问题。

让我们开始吧。

![Linear Regression for Machine Learning](img/eb5e9c48b845758d310012c31ed92a57.jpg)

机器学习的线性回归
照片由 [Nicolas Raymond](https://www.flickr.com/photos/82955120@N05/7645683004/) 拍摄，保留一些权利。

## 统计学不是线性回归吗？

在我们深入了解线性回归的细节之前，您可能会问自己为什么要查看此算法。

这不是一项统计技术吗？

机器学习，更具体地说是预测性建模领域，主要关注的是最小化模型的误差或使可能性最准确的预测，但代价是可解释性。在应用机器学习中，我们将借用，重用和窃取来自许多不同领域的算法，包括统计量并将其用于这些目的。

因此，在统计学领域开发了线性回归，并且作为理解输入和输出数值变量之间关系的模型进行了研究，但是已经被机器学习所借鉴。它既是统计算法又是机器学习算法。

接下来，让我们回顾一下用于指代线性回归模型的一些常用名称。

## 获取免费算法思维导图

![Machine Learning Algorithms Mind Map](img/2ce1275c2a1cac30a9f4eea6edd42d61.jpg)

方便的机器学习算法思维导图的样本。

我已经创建了一个由类型组织的 60 多种算法的方便思维导图。

下载，打印并使用它。

## 许多线性回归的名称

当你开始研究线性回归时，事情会变得非常混乱。

原因是因为线性回归已存在很长时间（超过 200 年）。它已经从每个可能的角度进行了研究，并且通常每个角度都有一个新的和不同的名称。

线性回归是**线性模型**，例如，假设输入变量（x）和单个输出变量（y）之间存在线性关系的模型。更具体地，y 可以从输入变量（x）的线性组合计算。

当存在单个输入变量（x）时，该方法称为**简单线性回归**。当存在**多个输入变量**时，来自统计学的文献通常将该方法称为多元线性回归。

可以使用不同的技术从数据中准备或训练线性回归方程，其中最常见的是**普通最小二乘**。因此，通常将以这种方式制备的模型称为普通最小二乘线性回归或仅最小二乘回归。

现在我们知道了一些用于描述线性回归的名称，让我们仔细看看所使用的表示。

## 线性回归模型表示

[线性回归](https://en.wikipedia.org/wiki/Linear_regression)是一个有吸引力的模型，因为表示非常简单。

该表示是一个线性方程，它组合了一组特定的输入值（x），该解决方案是该组输入值（y）的预测输出。因此，输入值（x）和输出值都是数字。

线性方程为每个输入值或列分配一个比例因子，称为系数，由大写希腊字母 Beta（B）表示。还增加了一个附加系数，使线具有额外的自由度（例如，在二维图上上下移动），并且通常称为截距或偏置系数。

例如，在一个简单的回归问题（单个 x 和单个 y）中，模型的形式为：

y = B0 + B1 * x

在具有多个输入（x）的较高维度中，该线被称为平面或超平面。因此，该表示是等式的形式和用于系数的特定值（例如，在以上示例中为 B0 和 B1）。

谈论像线性回归这样的回归模型的复杂性是很常见的。这是指模型中使用的系数数量。

当系数变为零时，它有效地消除了输入变量对模型的影响，因此也消除了模型的预测（0 * x = 0）。如果你看一下改变学习算法的正则化方法，通过对系数的绝对大小施加压力，将某些系数调到零来降低回归模型的复杂性，这就变得相关了。

现在我们已经理解了用于线性回归模型的表示，让我们回顾一下我们可以从数据中学习这种表示的一些方法。

![What is Linear Regression?](img/ede1561b2c8b5228d0d5c4f7ccc45168.jpg)

什么是线性回归？
[Estitxu Carton](https://www.flickr.com/photos/bichuas/3961559679/) 的照片，保留一些权利。

## 线性回归学习模型

学习线性回归模型意味着使用我们可用的数据估计表示中使用的系数的值。

在本节中，我们将简要介绍准备线性回归模型的四种技术。这不足以从零开始实现它们，但足以让人了解所涉及的计算和权衡。

还有更多技术，因为模型研究得很好。注意普通的最小二乘法，因为它是一般使用的最常用的方法。还要注意 Gradient Descent，因为它是机器学习课程中最常用的技术。

### 1.简单线性回归

通过简单的线性回归，当我们有一个输入时，我们可以使用统计来估计系数。

这要求您根据数据计算统计特性，例如均值，标准差，相关性和协方差。所有数据必须可用于遍历和计算统计量。

这在 excel 中很有趣，但在实践中并没有真正有用。

### 2.普通的最小二乘法

当我们有多个输入时，我们可以使用普通最小二乘来估计系数的值。

[普通最小二乘](https://en.wikipedia.org/wiki/Ordinary_least_squares)程序试图最小化残差平方和。这意味着给定数据的回归线，我们计算从每个数据点到回归线的距离，将其平方，并将所有平方误差加在一起。这是普通最小二乘法寻求最小化的数量。

该方法将数据视为矩阵，并使用线性代数运算来估计系数的最佳值。这意味着所有数据都必须可用，并且您必须有足够的内存来适应数据并执行矩阵运算。

除非作为线性代数中的练习，否则自己实现普通最小二乘法是不常见的。您更有可能在线性代数库中调用过程。此过程计算速度非常快。

### 3.梯度下降

当有一个或多个输入时，您可以通过迭代最小化训练数据模型的误差来使用优化系数值的过程。

此操作称为[梯度下降](https://en.wikipedia.org/wiki/Gradient_descent)，其工作原理是从每个系数的随机值开始。计算每对输入和输出值的平方误差之和。学习率用作比例因子，并且系数在朝向最小化误差的方向上更新。重复该过程直到达到最小和平方误差或者不可能进一步改进。

使用此方法时，必须选择学习率（alpha）参数，该参数确定要在过程的每次迭代中采用的改进步骤的大小。

通常使用线性回归模型来教授梯度下降，因为它相对简单易懂。实际上，当您在行数或可能不适合内存的列数中拥有非常大的数据集时，它非常有用。

### 4.正规化

线性模型的训练有扩展，称为正则化方法。这些都试图最小化模型在训练数据上的平方误差之和（使用普通最小二乘），但也降低模型的复杂性（如模型中所有系数之和的数量或绝对大小） 。

线性回归正则化程序的两个常见例子是：

*   [套索回归](https://en.wikipedia.org/wiki/Lasso_(statistics))：修改普通最小二乘法以最小化系数的绝对和（称为 L1 正则化）。
*   [岭回归](https://en.wikipedia.org/wiki/Tikhonov_regularization)：修改普通最小二乘法以最小化系数的平方绝对和（称为 L2 正则化）。

当输入值存在共线性且普通最小二乘法会过拟合训练数据时，这些方法可以有效使用。

既然您已经了解了一些在线性回归模型中学习系数的技术，那么让我们看一下如何使用模型对新数据做出预测。

## 用线性回归做出预测

如果表示是一个线性方程，那么做出预测就像解决一组特定输入的方程一样简单。

让我们以一个例子来具体化。想象一下，我们从高度（x）预测重量（y）。我们对这个问题的线性回归模型表示如下：

y = B0 + B1 * x1

要么

重量= B0 + B1 *高度

其中 B0 是偏差系数，B1 是高度列的系数。我们使用学习技术来找到一组好的系数值。一旦找到，我们可以插入不同的高度值来预测重量。

例如，让我们使用 B0 = 0.1 和 B1 = 0.5。让我们将它们插入并计算出身高 182 厘米的人的体重（千克）。

重量= 0.1 + 0.5 * 182

重量= 91.1

您可以看到上面的等式可以绘制为二维线。无论我们有多高，B0 都是我们的起点。我们可以在 100 到 250 厘米的高度上运行并将它们插入等式并获得重量值，从而创建我们的生产线。

![Sample Height vs Weight Linear Regression](img/6c34d39327369150d1a1e8c38e1b8efe.jpg)

样本高度和权重线性回归

既然我们已经知道如何在学习线性回归模型的情况下做出预测，那么让我们看看准备数据的一些经验法则，以充分利用这种类型的模型。

## 准备线性回归数据

对线性回归进行了长时间的研究，并且有很多关于如何构建数据以充分利用模型的文献。

因此，在谈论可能令人生畏的这些要求和期望时，有很多复杂性。在实践中，当使用普通最小二乘回归时，您可以更多地使用这些规则，这是最常见的线性回归实现。

使用这些启发式方法尝试不同的数据准备工作，看看哪种方法最适合您的问题。

*   **线性假设**。线性回归假设输入和输出之间的关系是线性的。它不支持任何其他内容。这可能是显而易见的，但是当你有很多属性时，记住它是件好事。您可能需要转换数据以使关系成为线性关系（例如，指数关系的对数转换）。
*   **去除噪音**。线性回归假设您的输入和输出变量没有噪声。请考虑使用数据清理操作，以便更好地公开和阐明数据中的信号。这对输出变量最重要，如果可能，您希望删除输出变量（y）中的异常值。
*   **删除共线性**。当您具有高度相关的输入变量时，线性回归将过拟合您的数据。考虑计算输入数据的成对相关性并删除最相关的数据。
*   **高斯分布**。如果输入和输出变量具有高斯分布，则线性回归将进行更可靠的预测。您可以使用变换（例如 log 或 BoxCox）在变量上获得一些好处，使其分布更加高斯。
*   **重新缩放输入**：如果使用标准化或标准化重新缩放输入变量，线性回归通常会做出更可靠的预测。

有关模型所做假设的优秀列表，请参阅 [Wikipedia 关于线性回归](https://en.wikipedia.org/wiki/Linear_regression#Assumptions)的文章。 [普通最小二乘维基百科文章](https://en.wikipedia.org/wiki/Ordinary_least_squares#Assumptions)也有很多假设。

## 进一步阅读

还有更多关于线性回归的内容。在您进行更多阅读之前开始使用它，但是当您想要深入了解时，下面是您可以使用的一些参考。

### 提及线性回归的机器学习书籍

这些是您可能拥有或可访问的一些机器学习书籍，它们描述了机器学习环境中的线性回归。

*   [机器学习的第一门课程](http://www.amazon.com/dp/1439824142?tag=inspiredalgor-20)，第一章。
*   [统计学习简介：应用于 R](http://www.amazon.com/dp/1461471370?tag=inspiredalgor-20) ，第三章。
*   [Applied Predictive Modeling](http://www.amazon.com/dp/1461468485?tag=inspiredalgor-20) ，第六章。
*   [机器学习在行动](http://www.amazon.com/dp/1617290181?tag=inspiredalgor-20)，第八章。
*   [统计学习要素：数据挖掘，推理和预测，第二版](http://www.amazon.com/dp/0387848576?tag=inspiredalgor-20)，第三章。

### 线性回归的帖子

以下是我遇到的一些有趣的关于线性回归的文章和博客文章。

*   [普通最小二乘回归：视觉解释](http://setosa.io/ev/ordinary-least-squares-regression/)
*   [普通最小二乘线性回归：缺陷，问题和陷阱](http://www.clockbackward.com/2009/06/18/ordinary-least-squares-linear-regression-flaws-problems-and-pitfalls/)
*   [线性回归分析介绍](http://people.duke.edu/~rnau/regintro.htm)
*   [研究人员应该总是测试的多重回归的四个假设](http://pareonline.net/getvn.asp?n=2&v=8)

对线性回归有更多好的参考，并且倾向于机器学习和预测性建模？发表评论并告诉我。

## 摘要

在这篇文章中，您发现了用于机器学习的线性回归算法。

你涵盖了很多方面，包括：

*   描述线性回归模型时使用的通用名称。
*   模型使用的表示。
*   学习算法用于估计模型中的系数。
*   准备用于线性回归的数据时要考虑的经验法则。

尝试线性回归并熟悉它。

您对线性回归或此帖有任何疑问吗？
发表评论并问，我会尽力回答。