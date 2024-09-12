# 对幂次和多项式导数的温和介绍

> 原文：[`machinelearningmastery.com/a-gentle-introduction-to-derivatives-of-powers-and-polynomials/`](https://machinelearningmastery.com/a-gentle-introduction-to-derivatives-of-powers-and-polynomials/)

在机器学习和数据科学算法中，最常用的函数之一是多项式或涉及 x 次幂的函数。因此，理解这些函数的导数如何计算是很重要的。

在本教程中，你将学习如何计算 x 的幂次和多项式的导数。

完成本教程后，你将了解：

+   计算多项式导数的通用规则

+   计算涉及任何非零实数次幂的 x 的函数导数的通用规则

让我们开始吧。

![KhairaGali 的山](https://machinelearningmastery.com/wp-content/uploads/2021/06/murree-e1623865846111.jpg)

幂次和多项式的导数 图片由 Misbah Ghufran 提供，部分权利保留

## 教程概述

本教程分为两个部分，它们是：

1.  涉及 x 的整数次幂的函数的导数

1.  具有任何实数非零次幂的 x 的函数的微分

## 两个函数和的导数

让我们从寻找一个简单规则开始，该规则描述了两个函数之和的导数。假设我们有两个函数 f(x) 和 g(x)，那么它们之和的导数可以如下计算。如果需要复习，可以参考 [导数的定义](https://machinelearningmastery.com/a-gentle-introduction-to-function-derivatives)。

![两个函数和的导数](https://machinelearningmastery.com/wp-content/uploads/2021/06/sumtwofunctions.png)

两个函数和的导数

这里有一个通用规则，即两个函数之和的导数等于这两个函数各自导数的和。

## x 的整数次幂的导数

在讨论 x 的整数次幂的导数之前，让我们复习一下二项式定理，该定理告诉我们如何展开以下表达式（其中 C(n,k) 是组合函数）：

(a+b)^n = a^n + C(n,1)a^(n-1)b + C(n,2)a^(n-2)b² + … + C(n,n-1)ab^(n-1) + b^n

我们将推导出一个简单规则，用于寻找涉及 x^n 的函数的导数，其中 n 是一个整数且 n>0。让我们回到这个 [教程](https://machinelearningmastery.com/a-gentle-introduction-to-function-derivatives) 中讨论的导数定义，并将其应用于 kx^n，其中 k 是常数。

![kx^n 的导数](https://machinelearningmastery.com/wp-content/uploads/2021/06/powerderiv.png)

kx^n 的导数

以下是应用此规则的一些示例：

+   x² 的导数是 2x

+   3x⁵ 的导数是 15x⁴

+   4x⁹ 的导数是 36x⁸

## 如何对多项式进行微分？

这两个规则，即两个函数和的导数规则和整数次幂的导数规则，使我们能够对多项式进行求导。如果我们有一个度数为 n 的多项式，我们可以将其视为涉及不同次幂的 x 的各个函数的和。假设我们有一个度数为 n 的多项式 P(x)，那么它的导数是 P'(x) 如下：

![多项式的导数](https://machinelearningmastery.com/wp-content/uploads/2021/06/poly-1.png)

多项式的导数

这表明，多项式的导数是一个度数为 (n-1) 的多项式。

## 示例

下方展示了一些示例，其中多项式函数及其导数都一起绘制。蓝色曲线表示函数本身，而红色曲线表示该函数的导数。

![多项式函数及其导数的示例](https://machinelearningmastery.com/wp-content/uploads/2021/06/polyDeriv.png)

多项式函数及其导数的示例

## 那么 x 的非整数次幂呢？

上述导数规则扩展到 x 的非整数实数次幂，这些次幂可以是分数、负数或无理数。通用规则如下，其中 a 和 k 可以是任何非零实数。

f(x) = kx^a

f'(x) = kax^(a-1)

一些示例如下：

+   x^(0.2) 的导数是 (0.2)x^(-0.8)

+   x^(????) 的导数是 ????x^(????-1)

+   x^(-3/4) 的导数是 (-3/4)x^(-7/4)

这里有一些示例，这些示例与其导数一起绘制。蓝色曲线表示函数本身，红色曲线表示相应的导数：

![包含 x 的实数次幂的表达式的导数示例](https://machinelearningmastery.com/wp-content/uploads/2021/06/nonlinDeriv.png)

包含 x 的实数次幂的表达式的导数示例

## 扩展

本节列出了一些扩展教程的想法，你可能希望探索这些内容。

+   两个函数乘积导数的规则

+   有理函数导数的规则

+   积分

如果你探索了这些扩展内容，我很想知道。请在下面的评论中发布你的发现。

## 进一步阅读[¶](http://localhost:8888/notebooks/work/upwork/MLM/derivative/Untitled.ipynb?kernel_name=python3#Further-Reading)

本节提供了更多相关资源，如果你希望深入了解该主题。

### 教程

+   [极限与连续性](https://machinelearningmastery.com/a-gentle-introduction-to-limits-and-continuity)

+   [评估极限](https://machinelearningmastery.com/a-gentle-introduction-to-evaluating-limits)

+   [导数](https://machinelearningmastery.com/a-gentle-introduction-to-function-derivatives)

### 资源[¶](http://localhost:8888/notebooks/work/upwork/MLM/derivative/Untitled.ipynb?kernel_name=python3#Resources)

+   机器学习的 [微积分书籍](https://machinelearningmastery.com/calculus-books-for-machine-learning/) 额外资源

### 书籍

+   [托马斯微积分](https://amzn.to/35Yeolv)，第 14 版，2017 年。（基于乔治·B·托马斯的原著，由乔尔·哈斯、克里斯托弗·海尔、莫里斯·威尔修订）

+   [微积分](https://www.amazon.com/Calculus-3rd-Gilbert-Strang/dp/0980232759/ref=as_li_ss_tl?dchild=1&keywords=Gilbert+Strang+calculus&qid=1606171602&s=books&sr=1-1&linkCode=sl1&tag=inspiredalgor-20&linkId=423b93db012f7cc6bb92cb7494a3095f&language=en_US)，第 3 版，2017 年。（吉尔伯特·斯特朗）

+   [微积分](https://amzn.to/3kS9I52)，第 8 版，2015 年。（詹姆斯·斯图尔特）

## 摘要

在本教程中，你发现了如何对多项式函数和涉及非整数次幂和的函数进行微分。

具体来说，你学到了：

+   两个函数和的导数

+   常数乘以 x 的整数次幂的导数

+   多项式函数的导数

+   包含 x 的非整数次幂的表达式之和的导数

## 你有任何问题吗？

在下面的评论中提问，我将尽力回答。
