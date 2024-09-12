# **拉格朗日乘数法的温和介绍**

> 原文：[`machinelearningmastery.com/a-gentle-introduction-to-method-of-lagrange-multipliers/`](https://machinelearningmastery.com/a-gentle-introduction-to-method-of-lagrange-multipliers/)

拉格朗日乘数法是一种简单而优雅的方法，用于在存在等式或不等式约束时找到函数的局部最小值或局部最大值。拉格朗日乘数也称为未确定乘数。在本教程中，我们将讨论当给定等式约束时的这种方法。

在本教程中，你将探索拉格朗日乘数法以及如何在存在等式约束时找到函数的局部最小值或最大值。

完成本教程后，你将了解：

+   如何在等式约束下找到函数的局部最大值或最小值

+   带等式约束的拉格朗日乘数法

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/IMG_4464-scaled.jpg)

**拉格朗日乘数法的温和介绍**。照片由 Mehreen Saeed 提供，保留部分权利。

## 教程概述

本教程分为 2 部分；它们是：

1.  带等式约束的拉格朗日乘数法

1.  两个解决示例

## **前提条件**

对于本教程，我们假设你已经了解以下内容：

+   [函数的导数](https://machinelearningmastery.com/a-gentle-introduction-to-function-derivatives/)

+   [多变量函数、偏导数和梯度向量](https://machinelearningmastery.com/a-gentle-introduction-to-partial-derivatives-and-gradient-vectors)

+   [优化的温和介绍](https://machinelearningmastery.com/a-gentle-introduction-to-optimization-mathematical-programming)

+   [梯度下降](https://machinelearningmastery.com/a-gentle-introduction-to-gradient-descent-procedure)

你可以通过点击上面给出的链接来复习这些概念。

## **拉格朗日乘数法的等式约束是什么？**

假设我们有以下优化问题：

最小化 f(x)

主题：

g_1(x) = 0

g_2(x) = 0

…

g_n(x) = 0

拉格朗日乘数法首先构造一个称为拉格朗日函数的函数，如下式所示。

L(x, **????**) = f(x) + ????_1 g_1(x) + ????_2 g_2(x) + … + ????_n g_n(x)

这里**????**表示拉格朗日乘数的一个向量，即：

**????**= [ ????_1, ????_2, …, ????_n]^T

要在等式约束下找到 f(x) 的局部最小值，我们求解拉格朗日函数 L(x, **????**）的驻点，即我们解以下方程：

∇xL = 0

∂L/∂????_i = 0（对于 i = 1..n）

因此，我们得到总共 m+n 个方程需要求解，其中

m = f 的变量数量

n = 等式约束的数量。

简而言之，局部最小值点将是以下方程的解：

∂L/∂x_j = 0（对于 j = 1..m）

g_i(x) = 0（对于 i = 1..n）

### 想要开始学习机器学习的微积分吗？

现在获取我的免费 7 天电子邮件速成课程（附带示例代码）。

点击注册并免费获得课程的 PDF 电子书版本。

## 已解决的示例

本节包含两个已解决的示例。如果你解决了这两个问题，你将对如何将拉格朗日乘子法应用于两个以上变量的函数以及更多的等式约束有一个很好的了解。

### **示例 1：一个等式约束**

让我们解决以下最小化问题：

最小化： f(x) = x² + y²

约束条件： x + 2y – 1 = 0

第一步是构建拉格朗日函数：

L(x, y, ????) = x² + y² + ????(x + 2y – 1)

我们有以下三个方程需要解：

∂L/∂x = 0

2x + ???? = 0      (1)

∂L/∂y = 0

2y + 2???? = 0     (2)

∂L/∂???? = 0

x + 2y -1 = 0    (3)

使用 (1) 和 (2)，我们得到：

???? = -2x = -y

将其代入 (3) 得到：

x = 1/5

y = 2/5

因此，局部最小点位于 (1/5, 2/5)，如右图所示。左图显示了函数的图像。

![函数图（左），轮廓线、约束和局部最小值（右）](https://machinelearningmastery.com/wp-content/uploads/2021/07/lagrange2.png)

函数图（左）。轮廓线、约束和局部最小值（右）

### **示例 2：两个等式约束**

假设我们想要在给定约束下找到以下函数的最小值：

最小化 g(x, y) = x² + 4y²

约束条件：

x + y = 0

x² + y² – 1 = 0

这个问题的解可以通过首先构建拉格朗日函数来找到：

L(x, y, ????_1, ????_2) = x² + 4y² + ????_1(x + y) + ????_2(x² + y² – 1)

我们需要解 4 个方程：

∂L/∂x = 0

2x + ????_1 + 2x ????_2 = 0    (1)

∂L/∂y = 0

8y + ????_1 + 2y ????_2 = 0    (2)

∂L/∂????_1 = 0

x + y = 0         (3)

∂L/∂????_2 = 0

x² + y² – 1 = 0    (4)

解上述方程组可得到 (x,y) 的两个解，即我们得到两个点：

(1/sqrt(2), -1/sqrt(2))

(-1/sqrt(2), 1/sqrt(2))

函数及其约束条件和局部最小值如下所示。

![函数图（左），轮廓线、约束和局部最小值（右）](https://machinelearningmastery.com/wp-content/uploads/2021/07/lagrange1.png)

函数图（左）。轮廓线、约束和局部最小值（右）

## 与最大化问题的关系

如果你有一个需要最大化的函数，可以以类似的方式解决，记住最大化和最小化是等价的问题，即，

最大化 f(x)                 等同于                   最小化 -f(x)

## 拉格朗日乘子法在机器学习中的重要性

许多知名的机器学习算法都使用了拉格朗日乘数法。例如，主成分分析（PCA）的理论基础是使用具有等式约束的拉格朗日乘数法构建的。类似地，支持向量机（SVM）的优化问题也使用这种方法解决。然而，在 SVM 中，还涉及到不等式约束。

## **扩展**

本节列出了一些可能希望探索的扩展教程的想法。

+   带有不等式约束的优化

+   KKT 条件

+   支持向量机

如果你探索了这些扩展中的任何一个，我很想知道。请在下方评论中发布你的发现。

## **进一步阅读**

本节提供了更多资源，供你深入了解该主题。

### **教程**

+   [导数](https://machinelearningmastery.com/a-gentle-introduction-to-function-derivatives)

+   [机器学习中的梯度下降](https://machinelearningmastery.com/gradient-descent-for-machine-learning/)

+   [机器学习中的梯度是什么](https://machinelearningmastery.com/gradient-in-machine-learning/)

+   [偏导数和梯度向量](https://machinelearningmastery.com/a-gentle-introduction-to-partial-derivatives-and-gradient-vectors)

+   [如何选择优化算法](https://machinelearningmastery.com/tour-of-optimization-algorithms/)

### **资源**

+   关于[机器学习中的微积分书籍](https://machinelearningmastery.com/calculus-books-for-machine-learning/)的额外资源

### **书籍**

+   [托马斯微积分](https://amzn.to/35Yeolv)，第 14 版，2017 年（基于乔治·B·托马斯的原著，由乔尔·哈斯、克里斯托弗·海尔、莫里斯·威尔修订）

+   [微积分](https://www.amazon.com/Calculus-3rd-Gilbert-Strang/dp/0980232759/ref=as_li_ss_tl?dchild=1&keywords=Gilbert+Strang+calculus&qid=1606171602&s=books&sr=1-1&linkCode=sl1&tag=inspiredalgor-20&linkId=423b93db012f7cc6bb92cb7494a3095f&language=en_US)，第 3 版，2017 年（吉尔伯特·斯特朗）

+   [微积分](https://amzn.to/3kS9I52)，第 8 版，2015 年（詹姆斯·斯图尔特）

## **总结**

在本教程中，你了解了拉格朗日乘数法。具体来说，你学到了：

+   拉格朗日乘数与拉格朗日函数

+   如何在给定等式约束的情况下解决优化问题

## **你有任何问题吗？**

在下方评论中提出你的问题，我会尽力回答。
