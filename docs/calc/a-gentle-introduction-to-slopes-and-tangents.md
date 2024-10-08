# 斜率和切线的温和介绍

> 原文：[`machinelearningmastery.com/a-gentle-introduction-to-slopes-and-tangents/`](https://machinelearningmastery.com/a-gentle-introduction-to-slopes-and-tangents/)

直线的斜率及其与曲线切线的关系是微积分中的一个基本概念。它对于函数导数的一般理解非常重要。

在本教程中，你将了解什么是直线的斜率以及什么是曲线的切线。

完成本教程后，你将了解：

+   直线的斜率

+   关于 x 的 f(x)在区间上的平均变化率

+   曲线的斜率

+   曲线在某一点的切线

让我们开始吧。

![在 CMU 走向天空](https://machinelearningmastery.com/wp-content/uploads/2021/06/IMG_4270-scaled.jpg)

斜率和切线的温和介绍 艺术家：Jonathan Borofsky，摄影：Mehreen Saeed，部分版权保留。

## 教程概述

本教程分为两部分；它们是：

1.  直线和曲线的斜率

1.  曲线的切线

## 直线的斜率

让我们从复习直线的斜率开始。在微积分中，直线的斜率定义了其陡峭度，该数字是通过将垂直方向的变化除以在水平方向上的变化来计算的。图示显示了如何从直线上的两个不同点 A 和 B 计算斜率。

![直线的斜率](https://machinelearningmastery.com/wp-content/uploads/2021/06/slopeLine.png)

从直线上的两个点计算的直线斜率

一条直线可以由该直线上的两个点唯一确定。直线的斜率在直线上的每一点都是相同的；因此，任何直线也可以由斜率和直线上的一个点唯一确定。从已知点我们可以根据直线斜率定义的比例移动到直线上的任何其他点。

## 曲线的平均变化率

我们可以将直线的斜率的概念扩展到曲线的斜率。考虑下图左侧的图。如果我们想测量这条曲线的‘陡峭度’，它将在曲线上的不同点变化。从点 A 到点 B 的平均变化率是负的，因为当 x 增加时函数值在减少。从点 B 到点 A 的情况也是如此。因此，我们可以在区间[x0,x1]上定义它为：

(y1-y0)/(x1-x0)

我们可以看到，上图也是包含点 A 和 B 的割线斜率的表达式。为了刷新你的记忆，割线在曲线上交于两点。

同样，点 C 和点 D 之间的平均变化率是正的，它由包含这两个点的割线的斜率给出。

![曲线在区间内的变化率（左）曲线在某一点的变化率（右）](https://machinelearningmastery.com/wp-content/uploads/2021/06/rate.png)

曲线在区间内的变化率与在某一点的变化率

## 定义曲线的斜率

现在让我们看看上述图形的右侧图。当我们将点 B 移向点 A 时会发生什么？我们称新的点为 B'。当点 B' 无限接近 A 时，割线将变成只接触曲线一次的直线。这里 B' 的 x 坐标是 (x0+h)，其中 h 是一个无限小的值。点 B' 的 y 坐标的对应值是该函数在 (x0+h) 处的值，即 f(x0+h)。

区间 [x0,x0+h] 上的平均变化率表示在长度为 h 的非常小的区间上的变化率，其中 h 接近零。这被称为曲线在点 x0 处的斜率。因此，在任何点 A(x0,f(x0))，曲线的斜率定义为：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/slopeofcurve.png)

点 A 处的曲线斜率的表达式等同于 f(x) 在点 x0 处的导数。因此，我们可以使用导数来找到曲线的斜率。你可以在这个 [教程](https://machinelearningmastery.com/a-gentle-introduction-to-function-derivatives) 中回顾导数的概念。

### 曲线斜率的例子

这里有几个曲线斜率的例子。

+   f(x) = 1/x 在任何点 k (k≠0) 处的斜率由 (-1/k²) 给出。作为例子：

    +   f(x) = 1/x 在 (x=2) 处的斜率是 -1/4

    +   f(x) = 1/x 在 (x=-1) 处的斜率是 -1

+   f(x) = x² 在任何点 k 处的斜率由 (2k) 给出。例如：

    +   f(x) = x² 在 (x=0) 处的斜率是 0

    +   f(x) = x² 在 (x=1) 处的斜率是 2

+   f(x) = 2x+1 的斜率是一个等于 2 的常数值。我们可以看到 f(x) 定义了一条直线。

+   f(x) = k（其中 k 是常数）的斜率为零，因为该函数在任何地方都不发生变化。因此，它在任何点的平均变化率为零。

## 切线

之前提到过，任何直线可以通过其斜率和一个经过它的点唯一确定。我们也刚刚定义了曲线在点 A 处的斜率。利用这两个事实，我们将曲线 f(x) 在点 A(x0,f(x0)) 处的切线定义为满足以下两个条件之一的直线：

1.  该直线通过 A 点

1.  直线的斜率等于曲线在点 A 处的斜率

利用上述两个事实，我们可以轻松确定切线在点 (x0,f(x0)) 处的方程。接下来展示了几个例子。

## 切线的例子

### 1. f(x) = 1/x

图中显示了 f(x) 及其在 x=1 和 x=-1 处的切线。下面是确定 x=1 处切线的步骤。

![f(x) = 1/x](https://machinelearningmastery.com/wp-content/uploads/2021/06/oneoverx.png)

f(x) = 1/x

+   具有斜率 m 和 y 截距 c 的直线方程为： y=mx+c

+   任意点的直线斜率由函数 f'(x) = -1/x² 给出

+   曲线在 x=1 处的切线斜率为-1，我们得到 y=-x+c

+   切线经过点 (1,1)，因此代入上述方程我们得到：

    +   1 = -(1)+c ⟹ c = 2

+   切线的最终方程是 y = -x+2

### 2\. f(x) = x²

下面显示了曲线以及在点 x=2、x=-2 和 x=0 处的切线。在 x=0 处，切线与 x 轴平行，因为 f(x) 在 x=0 处的斜率为零。

这就是我们计算 x=2 处切线方程的方法：

![f(x) = x²](https://machinelearningmastery.com/wp-content/uploads/2021/06/xsq.png)

f(x) = x²

+   具有斜率 m 和 y 截距 c 的直线方程为： y=mx+c

+   任意点的切线斜率由函数 f'(x) = 2x 给出

+   曲线在 x=2 处的切线斜率为 4，我们得到 y=4x+c

+   切线经过点 (2,4)，因此代入上述方程我们得到：

    +   4 = 4(2)+c ⟹ c = -4

+   切线的最终方程是 y = 4x-4

### 3\. f(x) = x³+2x+1

下面展示了这个函数以及其在 x=0、x=2 和 x=-2 处的切线。以下是推导 x=0 处切线方程的步骤。

![f(x) = x³+2x+1](https://machinelearningmastery.com/wp-content/uploads/2021/06/cubic.png)

f(x) = x³+2x+1

+   具有斜率 m 和 y 截距 c 的直线方程为： y=mx+c

+   任意点的直线斜率由函数 f'(x) = 3x²+2 给出

+   曲线在 x=0 处的切线斜率为 2，我们得到 y=2x+c

+   切线经过点 (0,1)，因此代入上述方程我们得到：

    +   1 = 2(0)+c ⟹ c = 1

+   切线的最终方程是 y = 2x+1

注意，曲线在 x=2 和 x=-2 处的斜率相同，因此这两条切线是平行的。对于任意 x=k 和 x=-k，这种情况也是成立的，因为 f'(x) = f'(-x) = 3x²+2

## 扩展

本节列出了一些可能扩展教程的想法，你可以考虑探索。

+   速度与加速度

+   函数的积分

如果你探索这些扩展内容，我很想知道。请在下面的评论中分享你的发现。

## 进一步阅读

本节提供了更多资源，如果你希望深入了解这个话题。

### 教程

+   [极限与连续性](https://machinelearningmastery.com/a-gentle-introduction-to-limits-and-continuity)

+   [评估极限](https://machinelearningmastery.com/a-gentle-introduction-to-evaluating-limits)

+   [导数](https://machinelearningmastery.com/a-gentle-introduction-to-function-derivatives)

### 资源

+   关于[机器学习的微积分书籍](https://machinelearningmastery.com/calculus-books-for-machine-learning/)的额外资源

### 书籍

+   [托马斯微积分](https://amzn.to/35Yeolv)，第 14 版，2017 年。（基于**乔治·B·托马斯**的原著，由**乔尔·哈斯**、**克里斯托弗·海尔**、**莫里斯·韦尔**修订）

+   [微积分](https://www.amazon.com/Calculus-3rd-Gilbert-Strang/dp/0980232759/ref=as_li_ss_tl?dchild=1&keywords=Gilbert+Strang+calculus&qid=1606171602&s=books&sr=1-1&linkCode=sl1&tag=inspiredalgor-20&linkId=423b93db012f7cc6bb92cb7494a3095f&language=en_US)，第 3 版，2017 年。（**吉尔伯特·斯特朗**）

+   [微积分](https://amzn.to/3kS9I52)，第 8 版，2015 年。（**詹姆斯·斯图尔特**）

## 总结

在本教程中，你了解了曲线在某一点的斜率和曲线在某一点的切线的概念。

具体而言，你学到了：

+   直线的斜率是什么？

+   曲线在某一间隔内相对于 x 的平均变化率是什么？

+   曲线在某一点的斜率

+   在某一点上的曲线的切线

## 你有什么问题吗？

在下方评论中提出你的问题，我将尽力回答。
