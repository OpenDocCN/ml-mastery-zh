# 《偏导数和梯度向量的温和介绍》

> 原文：[`machinelearningmastery.com/a-gentle-introduction-to-partial-derivatives-and-gradient-vectors/`](https://machinelearningmastery.com/a-gentle-introduction-to-partial-derivatives-and-gradient-vectors/)

偏导数和梯度向量在机器学习算法中非常常用，用于寻找函数的最小值或最大值。梯度向量用于神经网络的训练、逻辑回归以及许多其他分类和回归问题中。

在本教程中，你将发现偏导数和梯度向量。

完成本教程后，你将了解：

+   多变量函数

+   等值线、等高线和双变量函数的图形

+   多变量函数的偏导数

+   梯度向量及其含义

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/atifgulzar.jpg)

《偏导数和梯度向量的温和介绍》。照片由 Atif Gulzar 提供，保留所有权利。

## 教程概述

本教程分为三个部分，它们是：

1.  多变量函数

    1.  等值线

    1.  等高线

    1.  图形

1.  偏导数的定义

1.  梯度向量

    1.  梯度向量表示什么

## 多变量函数

你可以在这个 [教程](https://machinelearningmastery.com/a-gentle-introduction-to-multivariate-calculus) 中复习函数和多变量函数的概念。我们将在这里提供更多关于多变量函数的详细信息。

多变量函数具有以下属性：

+   它的定义域是由 (x_1, x_2, x_3, …, x_n) 给出的 n 元组集

+   它的范围是实数集

例如，以下是一个双变量函数（n=2）：

f_1(x,y) = x + y

在上述函数中，x 和 y 是独立变量。它们的和决定了函数的值。该函数的定义域是 XY 直角坐标平面上的所有点集合。该函数的图形需要在 3D 空间中绘制，其中两个轴表示输入点 (x,y)，第三个轴表示 f 的值。

这是另一个双变量函数的示例。 f_2(x,y) = x**x + y**y

为了简化问题，我们将做双变量函数的示例。当然，在机器学习中，你会遇到数百个变量的函数。与双变量函数相关的概念可以扩展到这些情况。

### 双变量函数的等值线和图形

平面上的点集合，其中函数 f(x,y) 具有恒定值，即 f(x,y)=c，称为 f 的等值集或等值曲线。

例如，对于函数 f_1，所有满足以下方程的 (x,y) 点定义了 f_1 的一个等值集：

x + y = 1

我们可以看到，这个等值集具有无限的点集，例如 (0,2)、(1,1)、(2,0) 等。这个等值集定义了 XY 平面上的一条直线。

一般来说，函数 f_1 的所有水平集定义了形式为直线的直线（c 为任何实数常数）：

x + y = c

同样，对于函数 f_2，水平集的一个示例如下：

x**x + y**y = 1

我们可以看到，任何位于半径为 1、中心在 (0,0) 的圆上的点都满足上述表达式。因此，这个水平集由所有位于这个圆上的点组成。类似地，f_2 的任何水平集满足以下表达式（c 为任何实数常数 >= 0）：

x**x + y**y = c

因此，f_2 的所有水平集都是中心在 (0,0) 的圆，每个水平集都有自己独特的半径。

函数 f(x,y) 的图形是所有点 (x,y,f(x,y)) 的集合。它也被称为表面 z=f(x,y)。f_1 和 f_2 的图形如下（左侧）。

![函数 f_1 和 f_2 及其相应的轮廓](https://machinelearningmastery.com/wp-content/uploads/2021/07/grad1.png)

函数 f_1 和 f_2 及其相应的轮廓

### 两变量函数的轮廓

假设我们有一个两变量的函数 f(x,y)。如果我们用平面 z=c 切割表面 z=f(x,y)，则得到满足 f(x,y) = c 的所有点的集合。轮廓曲线是满足 f(x,y)=c 的点在平面 z=c 中的集合。这与水平集略有不同，水平曲线直接在 XY 平面中定义。然而，许多书籍将轮廓和水平曲线视为相同的概念。

上述图形（右侧）显示了 f_1 和 f_2 的轮廓。

### 想开始学习机器学习中的微积分吗？

立即参加我的免费 7 天电子邮件速成课程（包括示例代码）。

点击注册并获得免费 PDF 电子书版的课程。

## 偏导数和梯度

函数 f 关于变量 x 的偏导数表示为 ∂f/∂x。其表达式可以通过对 f 关于 x 求导来确定。例如，对于函数 f_1 和 f_2，我们有：

∂f_1/∂x = 1

∂f_2/∂x = 2x

∂f_1/∂x 表示 f_1 关于 x 的变化率。对于任何函数 f(x,y)，∂f/∂x 表示 f 关于变量 x 的变化率。

对于 ∂f/∂y 也是类似的情况。它表示 f 关于 y 的变化率。你可以在这个 [教程](https://machinelearningmastery.com/a-gentle-introduction-to-multivariate-calculus) 中查看偏导数的正式定义。

当我们求出对所有独立变量的偏导数时，我们得到一个向量。这个向量称为 f 的梯度向量，表示为 ∇f(x,y)。f_1 和 f_2 的梯度的一般表达式如下（其中 i,j 是与坐标轴平行的单位向量）：

∇f_1(x,y) = ∂f_1/∂xi + ∂f_1/∂yj = i+j

∇f_2(x,y) = ∂f_2/∂xi + ∂f_2/∂yj = 2xi + 2yj

从梯度的一般表达式中，我们可以在空间中的不同点上评估梯度。对于 f_1，梯度向量是常数，即：

i+j

无论我们处于三维空间的何处，梯度向量的方向和大小保持不变。

对于函数 f_2，∇f_2(x,y) 随 (x,y) 的值变化。例如，在 (1,1) 和 (2,1) 处，f_2 的梯度由以下向量给出：

∇f_2(1,1) = 2i + 2j

∇f_2(2,1) = 4i + 2j

## 梯度向量在某一点上表示什么？

多变量函数在任何点的梯度向量表示最大变化速率的方向。

我们可以将梯度向量与[切线](https://machinelearningmastery.com/a-gentle-introduction-to-slopes-and-tangents) 联系起来。如果我们站在空间中的一个点，并制定了一个规则，告诉我们沿着该点的轮廓的切线行走。这意味着无论我们在哪里，我们都找到该点的轮廓切线，并沿着它行走。如果我们遵循这个规则，我们将沿着 f 的轮廓行走。函数值将保持不变，因为函数值在 f 的轮廓上是恒定的。

另一方面，梯度向量垂直于切线，指向最大增速的方向。如果我们沿着梯度的方向前进，我们会遇到下一个函数值大于前一个值的点。

梯度的正方向表示最大增速的方向，而负方向则表示最大降速的方向。下图显示了函数 f_2 的不同轮廓点处的梯度向量的正方向。正梯度的方向由红色箭头指示。轮廓的切线以绿色显示。

![轮廓和梯度向量的方向](https://machinelearningmastery.com/wp-content/uploads/2021/07/grad2.png)

轮廓和梯度向量的方向

## 为什么梯度向量在机器学习中很重要？

梯度向量在机器学习算法中非常重要且频繁使用。在分类和回归问题中，我们通常定义均方误差函数。沿着该函数梯度的负方向将使我们找到该函数具有最小值的点。

对于函数也是如此，最大化它们会实现最高的准确性。在这种情况下，我们将沿着该函数的最大增速方向或梯度向量的正方向前进。

## 扩展

本节列出了一些扩展教程的想法，您可能希望进行探索。

+   梯度下降/梯度上升

+   Hessian 矩阵

+   雅可比矩阵

如果您探索了这些扩展内容，我很想知道。请在下面的评论中分享您的发现。

## 进一步阅读

本节提供了有关该主题的更多资源，如果您希望深入了解。

### 教程

+   [导数](https://machinelearningmastery.com/a-gentle-introduction-to-function-derivatives)

+   [斜率与切线](https://machinelearningmastery.com/a-gentle-introduction-to-slopes-and-tangents)

+   [多变量微积分](https://machinelearningmastery.com/a-gentle-introduction-to-multivariate-calculus)

+   [机器学习中的梯度下降](https://machinelearningmastery.com/gradient-descent-for-machine-learning/)

+   [机器学习中的梯度是什么](https://machinelearningmastery.com/gradient-in-machine-learning/)

### 资源

+   关于[机器学习的微积分书籍](https://machinelearningmastery.com/calculus-books-for-machine-learning)的额外资源

### 书籍

+   [托马斯微积分](https://amzn.to/35Yeolv)，第 14 版，2017 年。（基于**乔治·B·托马斯**的原著，由**乔尔·哈斯**、**克里斯托弗·海尔**和**莫里斯·韦尔**修订）

+   [微积分](https://www.amazon.com/Calculus-3rd-Gilbert-Strang/dp/0980232759/ref=as_li_ss_tl?dchild=1&keywords=Gilbert+Strang+calculus&qid=1606171602&s=books&sr=1-1&linkCode=sl1&tag=inspiredalgor-20&linkId=423b93db012f7cc6bb92cb7494a3095f&language=en_US)，第 3 版，2017 年。（**吉尔伯特·斯特朗**）

+   [微积分](https://amzn.to/3kS9I52)，第 8 版，2015 年。（**詹姆斯·斯图尔特**）

## 总结

在本教程中，你了解了什么是多变量函数、偏导数和梯度向量。具体来说，你学到了：

+   多变量函数

    +   多变量函数的轮廓

    +   多变量函数的水平集

+   多变量函数的偏导数

+   梯度向量及其意义

## 你有任何问题吗？

在下面的评论中提问，我会尽力回答。
