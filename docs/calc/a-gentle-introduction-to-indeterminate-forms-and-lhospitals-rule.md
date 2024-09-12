# 温和介绍未定型和洛必达法则

> 原文：[`machinelearningmastery.com/a-gentle-introduction-to-indeterminate-forms-and-lhospitals-rule/`](https://machinelearningmastery.com/a-gentle-introduction-to-indeterminate-forms-and-lhospitals-rule/)

在评估函数的极限时，经常会遇到未定型，极限在数学和微积分中扮演着重要角色。它们对学习导数、梯度、黑塞矩阵以及更多内容至关重要。

在本教程中，你将学习如何评估未定型的极限以及解决它们的洛必达法则。

完成本教程后，你将了解：

+   如何评估具有 0/0 和 ∞/∞ 未定型形式的函数的极限

+   洛必达法则用于评估未定型

+   如何转换更复杂的未定型并将洛必达法则应用于它们

开始吧。

![温和介绍未定型和洛必达法则，照片由 Mehreen Saeed 提供，保留所有权利。](https://machinelearningmastery.com/wp-content/uploads/2021/06/IMG_9247-scaled.jpg)

温和介绍未定型和洛必达法则，照片由 Mehreen Saeed 提供，保留所有权利。

## 教程概述

本教程分为 2 部分；它们是：

1.  0/0 和 ∞/∞ 的未定型

    1.  如何将洛必达法则应用于这些类型

    1.  这两种未定型的解题示例

1.  更复杂的未定型

    1.  如何将更复杂的未定型转换为 0/0 和 ∞/∞ 形式

    1.  这类类型的解题示例

## 前提条件

本教程需要对以下两个主题有基本了解：

+   [极限与连续性](https://machinelearningmastery.com/a-gentle-introduction-to-limits-and-continuity)

+   [评估极限](https://machinelearningmastery.com/a-gentle-introduction-to-evaluating-limits)

如果你不熟悉这些主题，可以通过点击上述链接进行复习。

## 什么是未定型？

在评估极限时，我们会遇到[评估极限的基本规则](https://machinelearningmastery.com/a-gentle-introduction-to-evaluating-limits)可能会失效的情况。例如，对于有理函数，我们可以应用商法则：

lim(x→a) f(x)/g(x) = (lim(x→a)f(x))/(lim(x→a)g(x))          如果 lim(x→a)g(x)≠0

只有在分母中的表达式在 x 接近 a 时不趋近于零时，才能应用上述规则。如果当 x 接近 a 时，分子和分母都趋近于零，则会出现更复杂的情况。这被称为 0/0 的未定型。同样，还有 ∞/∞ 的未定型，表示为：

lim(x→a) f(x)/g(x) = (lim(x→a)f(x))/(lim(x→a)g(x)) 当 lim(x→a)f(x)=∞ 和 lim(x→a)g(x)=∞

## 什么是洛必达法则？

洛必达法则如下：

![洛必达法则](https://machinelearningmastery.com/wp-content/uploads/2021/06/rule.png)

洛必达法则

### 何时应用洛必达法则

需要注意的重要一点是，洛必达法则仅在满足 f(x) 和 g(x) 的条件时适用。例如：

+   lim(????→0) sin(x)/(x+1) 不能应用洛必达法则，因为这不是 0/0 形式

+   lim(????→0) sin(x)/x 可以应用该法则，因为这是 0/0 形式

+   lim(????→∞) (e^x)/(1/x+1) 不能应用洛必达法则，因为这不是 ∞/∞ 形式

+   lim(????→∞) (e^x)/x 可以应用洛必达法则，因为这是 ∞/∞ 形式

## 0/0 和 ∞/∞ 的示例

下面展示了一些这两种类型的示例以及如何解决它们。你还可以参考下图中的函数。

### 示例 1.1: 0/0

计算 lim(????→2) ln(x-1)/(x-2) （见图中的左侧图）

![lim(????→2) ln(x-1)/(x-2)=1](https://machinelearningmastery.com/wp-content/uploads/2021/06/ex11.png)

lim(????→2) ln(x-1)/(x-2)=1

### 示例 1.2: ∞/∞

计算 lim(????→∞) ln(x)/x （见图中的右侧图）

![lim(????→∞) ln(x)/x=0](https://machinelearningmastery.com/wp-content/uploads/2021/06/ex12.png)

lim(????→∞) ln(x)/x=0

![示例 1.1 和 1.2 的图像](https://machinelearningmastery.com/wp-content/uploads/2021/06/hosp1.png)

示例 1.1 和 1.2 的图像

## 更多不确定形式

洛必达法则仅告诉我们如何处理 0/0 或 ∞/∞ 形式。然而，还有更多涉及乘积、差值和幂的不确定形式。那么我们如何处理剩下的呢？我们可以使用一些数学巧妙的方法将乘积、差值和幂转换为商。这可以使我们几乎在所有不确定形式中轻松应用洛必达法则。下表展示了各种不确定形式以及如何处理它们。

![如何解决更复杂的不确定形式](https://machinelearningmastery.com/wp-content/uploads/2021/06/hosp2.png)

如何解决更复杂的不确定形式

## 示例

以下示例展示了如何将一种不确定形式转换为 0/0 或 ∞/∞ 形式，并应用洛必达法则来求解极限。在解决示例之后，你还可以查看所有计算了极限的函数的图像。

### 示例 2.1: 0.∞

计算 lim(????→∞) x.sin(1/x) （见图中的第一张图）

![lim(????→∞) x.sin(1/x)=1](https://machinelearningmastery.com/wp-content/uploads/2021/06/ex21.png)

lim(????→∞) x.sin(1/x)=1

### 示例 2.2: ∞-∞

计算 lim(????→0) 1/(1-cos(x)) – 1/x （见下图的第二张图）

![lim(????→0) 1/(1-cos(x)) - 1/x = ∞](https://machinelearningmastery.com/wp-content/uploads/2021/06/ex22.png)

lim(????→0) 1/(1-cos(x)) – 1/x = ∞

### 示例 2.3：幂型

评估 lim(????→∞) (1+x)^(1/x)（参见下面图形中的第三张图）

![lim(????→∞) (1+x)^(1/x)=1](https://machinelearningmastery.com/wp-content/uploads/2021/06/ex23.png)

lim(????→∞) (1+x)^(1/x)=1

![示例 2.1、2.2 和 2.3 的图形](https://machinelearningmastery.com/wp-content/uploads/2021/06/hosp3.png)

示例 2.1、2.2 和 2.3 的图形

## 扩展

本节列出了一些扩展教程的想法，你可能希望探索。

+   柯西均值定理

+   罗尔定理

如果你探索了这些扩展功能，我很想知道。请在下面的评论中分享你的发现。

## 进一步阅读

本节提供了更多关于该主题的资源，如果你想深入了解。

### 教程

+   [极限与连续性](https://machinelearningmastery.com/a-gentle-introduction-to-limits-and-continuity)

+   [评估极限](https://machinelearningmastery.com/a-gentle-introduction-to-evaluating-limits)

+   [导数](https://machinelearningmastery.com/a-gentle-introduction-to-function-derivatives)

### 资源

+   关于[机器学习的微积分书籍](https://machinelearningmastery.com/calculus-books-for-machine-learning/)的额外资源

### 书籍

+   [托马斯微积分](https://amzn.to/35Yeolv)，第 14 版，2017 年。（基于乔治·B·托马斯的原著，由乔尔·哈斯、克里斯托弗·海尔、莫里斯·韦尔修订）

+   [微积分](https://www.amazon.com/Calculus-3rd-Gilbert-Strang/dp/0980232759/ref=as_li_ss_tl?dchild=1&keywords=Gilbert+Strang+calculus&qid=1606171602&s=books&sr=1-1&linkCode=sl1&tag=inspiredalgor-20&linkId=423b93db012f7cc6bb92cb7494a3095f&language=en_US)，第 3 版，2017 年。（吉尔伯特·斯特朗）

+   [微积分](https://amzn.to/3kS9I52)，第 8 版，2015 年。（詹姆斯·斯图尔特）

## 总结

在本教程中，你了解了不确定型的概念及其评估方法。

具体来说，你学习了：

+   类型 0/0 和∞/∞的不确定形式

+   评估类型 0/0 和∞/∞的洛必达法则

+   类型 0.∞、∞-∞和幂型的不确定形式，以及如何评估它们。

## 你有任何问题吗？

在下面的评论中提问，我会尽力回答。
