# 温和地介绍如何评估极限

> 原文：[`machinelearningmastery.com/a-gentle-introduction-to-evaluating-limits/`](https://machinelearningmastery.com/a-gentle-introduction-to-evaluating-limits/)

函数极限的概念可以追溯到古希腊学者如厄休德斯和阿基米德。虽然他们没有正式定义极限，但他们的许多计算都基于这一概念。艾萨克·牛顿正式定义了极限的概念，柯西对这一概念进行了细化。极限构成了微积分的基础，而微积分又定义了许多机器学习算法的基础。因此，理解如何评估不同类型函数的极限非常重要。

在本教程中，你将发现如何评估不同类型函数的极限。

完成本教程后，你将了解：

+   评估极限的不同规则

+   如何评估多项式和有理函数的极限

+   如何评估带有间断的函数的极限

+   三明治定理

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/IMG_1951-scaled.jpg)

温和地介绍极限和连续性 由 Mehreen Saeed 拍摄，保留部分权利。

## 教程概述

本教程分为 3 部分；它们是：

1.  极限规则

    1.  使用极限规则评估极限的示例

    1.  多项式的极限

    1.  有理表达式的极限

1.  带有间断的函数的极限

1.  三明治定理

## 极限的规则

如果我们知道一些简单的原则，极限是容易评估的，以下列出了这些原则。所有这些规则都基于两个函数 f(x)和 g(x)的已知极限，当 x 接近点 k 时：

![评估极限的简单规则](https://machinelearningmastery.com/wp-content/uploads/2021/06/eval1.png)

评估极限的规则

### 使用规则评估极限的示例

![使用简单规则评估极限的示例](https://machinelearningmastery.com/wp-content/uploads/2021/06/evalex.png)

使用简单规则评估极限的示例

这里有几个使用基本规则评估极限的示例。注意，这些规则适用于在 x 接近某一点时定义的函数。

## 多项式的极限

示例 1 和 2 是多项式的情况。根据极限规则，我们可以看到对于任何多项式，当 x 接近点 k 时，多项式的极限等于多项式在 k 处的值。可以写成：![](https://machinelearningmastery.com/wp-content/uploads/2021/06/poly.png)

因此，我们可以通过直接替代来评估多项式的极限，例如。

lim(x→1) x⁴+3x³+2 = 1⁴+3(1)³+2 = 6

## 有理函数的极限

对于涉及分数的有理函数，有两种情况。一种情况是在 x 接近某一点且函数在该点定义的情况下评估极限。另一种情况涉及在 x 接近某一点且函数在该点处未定义时计算极限。

### 情况 1：函数已定义

类似于多项式的情况，每当我们有一个函数，即形式为 f(x)/g(x)的有理表达式且分母在某一点不为零时，则：

lim(x→k) f(x)/g(x) = f(k)/g(k) if g(k)≠0

因此，我们可以通过直接代入来评估这个极限。例如：

lim(x→0)(x²+1)/(x-1) = -1

在这里，我们可以应用商规则，或者更简单地，替换 x=0 来评估极限。然而，当 x 接近 1 时，这个函数没有极限。请看下图中的第一个图表。

### 情况 2：函数未定义

让我们看另一个例子：

lim(x→2)(x²-4)/(x-2)

在 x=2 时，我们面临一个问题。分母为零，因此在 x=2 处函数未定义。我们可以从图表中看出，这个函数和(x+2)是相同的，除了在点 x=2 处有一个孔。在这种情况下，我们可以取消公共因子，仍然评估(x→2)的极限如下：

lim(x→2)(x²-4)/(x-2) = lim(x→2)(x-2)(x+2)/(x-2) = lim(x→2)(x+2) = 4

以下图像展示了上述两个例子以及另一个类似的 g_3(x)的第三个例子：

![有理函数的极限](https://machinelearningmastery.com/wp-content/uploads/2021/06/evalrat.png)

有理函数计算极限的例子

## 函数有不连续点的情况

假设我们有一个函数 h(x)，它在所有实数上都有定义：

h(x) = (x²+x)/x，如果 x≠0

h(x) = 0，如果 x=0

函数 g(x)在 x=0 处有不连续点，如下图所示。当评估 lim(x→0)h(x)时，我们必须看到当 x 接近 0 时 h(x)的变化（而不是 x=0 时）。当我们从两侧接近 x=0 时，h(x)接近 1，因此 lim(x→0)h(x)=1。

在下图中显示的函数 m(x)是另一个有趣的案例。该函数在所有实数上都有定义，但当 x→0 时，极限不存在。

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/evalundef.png)

在存在不连续性时评估极限

## 夹逼定理

这个定理也称为夹逼定理或者夹紧定理。它陈述了以下情况为真时：

1.  x 接近 k

1.  f(x) <= g(x) <= h(x)

1.  lim(x→k)f(x) = lim(x→k)h(x) = L

那么，在 x→k 时，函数 g(x)的极限如下：

lim(x→k)g(x) = L

![夹逼定理的示意图](https://machinelearningmastery.com/wp-content/uploads/2021/06/sand.png)

夹逼定理

这个定理在下图中有所说明：

利用这个定理，我们可以评估许多复杂函数的极限。一个众所周知的例子涉及正弦函数：

lim(x→0)x²sin(1/x)

![计算极限的夹逼定理](https://machinelearningmastery.com/wp-content/uploads/2021/06/sandEx.png)

使用夹逼定理计算极限

我们知道 sin(x)总是交替在-1 和+1 之间。利用这一事实，我们可以如下解决这个极限：

## 扩展内容

本节列出了一些你可能希望探讨的扩展教程的想法。

+   洛必达法则与不确定型（需要函数导数）

+   函数导数在函数极限中的定义

+   函数积分

如果你探讨了这些扩展内容，我很乐意知道。请在下面的评论中发布你的发现。

## 进一步阅读

学习和理解数学的最佳方式是通过练习和解决更多的问题。本节提供了更多资源，如果你希望深入学习这个主题。

### 教程

+   关于[极限与连续性](https://machinelearningmastery.com/a-gentle-introduction-to-limits-and-continuity)的教程

+   关于[机器学习的微积分书籍](https://machinelearningmastery.com/calculus-books-for-machine-learning/)的额外资源

### 书籍

+   [托马斯微积分](https://amzn.to/35Yeolv)，第 14 版，2017 年。（基于 George B. Thomas 的原著，由 Joel Hass, Christopher Heil, Maurice Weir 修订）

+   [微积分](https://amzn.to/3fqNSEB)，第 3 版，2017 年。（Gilbert Strang）

+   [微积分](https://amzn.to/3kS9I52)，第 8 版，2015 年。（James Stewart）

## 总结

在本教程中，你了解到如何评估不同类型函数的极限。

具体来说，你学到了：

+   不同函数极限的评估规则。

+   多项式和有理函数的极限评估

+   函数中存在不连续性时的极限评估

你有任何问题吗？请在下面的评论中提问，我会尽力回答。享受微积分吧！
