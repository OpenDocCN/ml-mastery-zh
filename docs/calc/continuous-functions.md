# 轻松介绍连续函数

> 原文：[`machinelearningmastery.com/continuous-functions/`](https://machinelearningmastery.com/continuous-functions/)

微积分的许多领域需要了解连续函数。连续函数的特性以及对不连续点的研究对数学界非常感兴趣。由于其重要的属性，连续函数在机器学习算法和优化方法中有实际应用。

在本教程中，你将发现什么是连续函数，它们的性质，以及在优化算法研究中两个重要的定理，即中间值定理和极值定理。

完成本教程后，你将了解到：

+   连续函数的定义

+   中间值定理

+   极值定理

让我们开始吧。

![红玫瑰图片](https://machinelearningmastery.com/wp-content/uploads/2021/06/jeeni.jpg)

轻松介绍连续函数 由 Jeeni Khala 拍摄，版权所有。

## 教程概述

本教程分为 2 部分，它们是：

1.  连续函数的定义

    1.  非正式定义

    1.  正式定义

1.  定理

    1.  中间值定理

    1.  极值定理

## 先决条件

本教程需要了解极限的概念。为了刷新你的记忆，你可以查看 [极限与连续性](https://machinelearningmastery.com/a-gentle-introduction-to-limits-and-continuity)，其中也简要定义了连续函数。在本教程中，我们将深入探讨。

我们还将使用区间。因此，方括号表示闭区间（包括边界点），而圆括号表示开区间（不包括边界点），例如，

+   [a,b] 意味着 a<=x<=b

+   (a,b) 意味着 a<x<b

+   [a,b) 意味着 a<=x<b

从上面你可以注意到，一个区间可以在一侧是开区间，在另一侧是闭区间。

最后，我们将仅讨论定义在实数上的实函数。我们不会讨论复数或定义在复平面上的函数。

## 连续函数的非正式定义

假设我们有一个函数 f(x)。如果我们可以在不抬起手的情况下绘制 f(x) 的图形，那么我们可以很容易地检查它在两个点 a 和 b 之间是否是连续的。例如，考虑一个定义为：

f(x)=2x+1

我们可以在 [0,1] 之间绘制这条直线而不抬起手。事实上，我们可以在任何两个 x 值之间绘制这条直线而不需要抬起手（见下图）。因此，这个函数在整个实数域上是连续的。现在让我们看看当我们绘制 ceiling 函数时会发生什么：

![连续函数（左）和非连续函数（右）](https://machinelearningmastery.com/wp-content/uploads/2021/06/cont1-1.png)

连续函数（左），以及非连续函数（右）

例如，ceil 函数在区间 (0,1] 上的值为 1，比如 ceil(0.5)=1，ceil(0.7)=1，等等。因此，该函数在区间 (0,1] 上是连续的。如果我们将区间调整为 (0,2]，那么当 x>1 时，ceil(x) 跳跃到 2。为了在区间 (0,2] 上绘制 ceil(x)，我们现在必须抬起手，并从 x=2 开始重新绘制。因此，ceil 函数不是一个连续函数。

如果函数在整个实数域上是连续的，那么它就是一个整体连续函数；否则，它不是整体连续的。对于后者类型的函数，我们可以检查它们在什么区间上是连续的。

## 一个正式的定义

函数 f(x) 在点 a 处是连续的，如果当 x 接近 a 时，函数的值接近 f(a)。因此，为了测试函数在点 x=a 处的连续性，请检查以下内容：

1.  f(a) 应该存在

1.  f(x) 在 x 接近 a 时有一个极限

1.  f(x) 在 x->a 时的极限等于 f(a)

如果上述所有条件都成立，那么该函数在点 a 处是连续的。

## 示例

一些示例列在下面，并在图中显示：

+   f(x) = 1/x 在 x=0 处未定义，因此不连续。然而，函数在 x>0 的区间内是连续的。

+   所有多项式函数都是连续函数。

+   三角函数 sin(x) 和 cos(x) 是连续的，并在 -1 和 1 之间振荡。

+   三角函数 tan(x) 不是连续的，因为它在 x=????/2, x=-????/2 等处未定义。

+   sqrt(x) 在 x<0 时未定义，因此不连续。

+   |x| 在任何地方都是连续的。

![连续函数和具有不连续性的函数的示例](https://machinelearningmastery.com/wp-content/uploads/2021/06/cont1.png)

连续函数和具有不连续性的函数的示例

## 连续性与函数导数的关系

从极限的连续性定义出发，我们有一个替代定义。f(x) 在 x 处是连续的，如果：

当 (h→0) 时，f(x+h)-f(x)→ 0

让我们看一下导数的定义：

f'(x) = lim(h→0) (f(x+h)-f(x))/h

因此，如果 f'(x) 在点 a 处存在，则函数在 a 处是连续的。反之则不总是成立。一个函数可能在点 a 处是连续的，但 f'(a) 可能不存在。例如，在上面的图中 |x| 在任何地方都是连续的。我们可以在不抬起手的情况下绘制它，但在 x=0 处，由于曲线的急剧转折，其导数并不存在。

## 中间值定理

中间值定理表明：

如果：

+   函数 f(x) 在 [a,b] 上是连续的

+   并且 f(a) <= K <= f(b)

那么：

+   在 a 和 b 之间存在一个点 c，即 a<=c<=b，使得 f(c) = K

用非常简单的话来说，这个定理表明，如果一个函数在 [a,b] 上是连续的，那么该函数在 f(a) 和 f(b) 之间的所有值都会存在于这个区间内，如下图所示。

![中间值定理的插图（左）和极值定理（右）](https://machinelearningmastery.com/wp-content/uploads/2021/06/cont3.png)

中间值定理的插图（左）和极值定理（右）

## 极值定理

该定理表明：

如果：

+   函数 f(x)在[a,b]上是连续的

那么：

+   在区间[a,b]内有点 x_min 和 x_max，即：

    +   a<=x_min<=b

    +   a<=x_max<=b

+   并且函数 f(x)具有最小值 f(x_min)和最大值 f(x_max)，即：

    +   当 a<=x<=b 时，f(x_min)<=f(x)<=f(x_max)

简单来说，连续函数在一个区间内总是有最小值和最大值，如上图所示。

## 连续函数与优化

连续函数在优化问题的研究中非常重要。我们可以看到，极值定理保证了在一个区间内，总会有一个点使得函数具有最大值。最小值也是如此。许多优化算法源自这一基本属性，并能执行惊人的任务。

## 扩展

本节列出了一些你可能希望探索的扩展教程的想法。

+   收敛和发散序列

+   基于无穷小常数的 Weierstrass 和 Jordan 连续函数定义

如果你探索了这些扩展内容，我很想知道。请在下面的评论中分享你的发现。

## 进一步阅读

本节提供了更多关于该主题的资源，如果你想深入了解。

### 教程

+   [极限与连续性](https://machinelearningmastery.com/a-gentle-introduction-to-limits-and-continuity)

+   [评估极限](https://machinelearningmastery.com/a-gentle-introduction-to-evaluating-limits)

+   [导数](https://machinelearningmastery.com/a-gentle-introduction-to-function-derivatives)

### 资源

+   关于[机器学习的微积分书籍](https://machinelearningmastery.com/calculus-books-for-machine-learning/)的额外资源

### 书籍

+   [托马斯微积分](https://amzn.to/35Yeolv)，第 14 版，2017 年（基于 George B. Thomas 的原著，由 Joel Hass、Christopher Heil、Maurice Weir 修订）

+   [微积分](https://www.amazon.com/Calculus-3rd-Gilbert-Strang/dp/0980232759/ref=as_li_ss_tl?dchild=1&keywords=Gilbert+Strang+calculus&qid=1606171602&s=books&sr=1-1&linkCode=sl1&tag=inspiredalgor-20&linkId=423b93db012f7cc6bb92cb7494a3095f&language=en_US)，第 3 版，2017 年（Gilbert Strang）

+   [微积分](https://amzn.to/3kS9I52)，第 8 版，2015 年（James Stewart）

## 总结

在本教程中，你发现了连续函数的概念。

具体来说，你学到了：

+   什么是连续函数

+   连续函数的正式和非正式定义

+   不连续点

+   中间值定理

+   极值定理

+   为什么连续函数很重要

## 你有任何问题吗？

在下面的评论中提出你的问题，我会尽力回答。
