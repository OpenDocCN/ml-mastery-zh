# 温和介绍 Hessian 矩阵

> 原文：[`machinelearningmastery.com/a-gentle-introduction-to-hessian-matrices/`](https://machinelearningmastery.com/a-gentle-introduction-to-hessian-matrices/)

Hessian 矩阵属于包含二阶导数的数学结构类别。它们通常用于机器学习和数据科学算法中，以优化感兴趣的函数。

在本教程中，你将发现 Hessian 矩阵及其对应的判别式，并了解其重要性。所有概念通过示例进行说明。

完成本教程后，你将知道：

+   Hessian 矩阵

+   通过 Hessian 矩阵计算的判别式

+   判别式中包含的信息

让我们开始吧。

![靠近穆里瀑布的照片。照片由 Beenish Fatima 提供，版权所有](https://machinelearningmastery.com/wp-content/uploads/2021/07/beenish-2.jpg)

温和介绍 Hessian 矩阵。照片由 Beenish Fatima 提供，版权所有。

## 教程概述

本教程分为三部分；它们是：

1.  函数的 Hessian 矩阵及其对应的判别式的定义

1.  计算 Hessian 矩阵和判别式的示例

1.  Hessian 和判别式告诉我们关于感兴趣的函数的信息

## **前提条件**

在本教程中，我们假设你已经知道：

+   [函数的导数](https://machinelearningmastery.com/a-gentle-introduction-to-function-derivatives/)

+   [多个变量的函数、偏导数和梯度向量](https://machinelearningmastery.com/a-gentle-introduction-to-partial-derivatives-and-gradient-vectors)

+   [高阶导数](https://machinelearningmastery.com/higher-order-derivatives/)

你可以通过点击上面给出的链接来回顾这些概念。

## **什么是 Hessian 矩阵？**

Hessian 矩阵是一个二阶偏导数的矩阵。假设我们有一个 n 变量的函数 f，即，

$$f: R^n \rightarrow R$$

f 的 Hessian 由左侧的矩阵给出。一个二变量函数的 Hessian 也显示在右侧。

![Hessian n 变量的函数（左）。f(x,y)的 Hessian（右）](https://machinelearningmastery.com/wp-content/uploads/2021/07/hessian1-1.png)

Hessian n 变量的函数（左）。f(x,y)的 Hessian（右）

从我们关于梯度向量的教程中，我们已经知道梯度是一个一阶偏导数的向量。类似地，Hessian 矩阵是一个二阶偏导数的矩阵，由函数 f 定义域内所有变量对组成。

### 想要开始学习机器学习中的微积分吗？

立即参加我的免费 7 天电子邮件速成课程（包括示例代码）。

点击注册并获得免费的 PDF 电子书版本。

## **什么是判别式？**

Hessian 的 **行列式** 也称为 f 的判别式。对于一个二维函数 f(x, y)，它由以下公式给出：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/hessian2.png)

f(x, y) 的判别式

## **Hessian 矩阵和判别式的示例**

假设我们有以下函数：

g(x, y) = x³ + 2y² + 3xy²

然后 Hessian H_g 和判别式 D_g 为：

![Hessian 和 g(x, y) = x³ + 2y² + 3xy² 的判别式](https://machinelearningmastery.com/wp-content/uploads/2021/07/hessian3.png)

Hessian 和 g(x, y) = x³ + 2y² + 3xy² 的判别式

让我们在不同的点上评估判别式：

D_g(0, 0) = 0

D_g(1, 0) = 36 + 24 = 60

D_g(0, 1) = -36

D_g(-1, 0) = 12

## **Hessian 和判别式有什么意义？**

Hessian 和相应的判别式用于确定函数的局部极值点。评估它们有助于理解多个变量的函数。以下是一些重要规则，适用于判别式为 D(a, b) 的点 (a,b)：

1.  如果 f_xx(a, b) > 0 且判别式 D(a,b) > 0，则函数 f 有一个 **局部最小值**

1.  如果 f_xx(a, b) < 0 且判别式 D(a,b) > 0，则函数 f 有一个 **局部最大值**

1.  如果 D(a, b) < 0，函数 f 有一个鞍点

1.  如果 D(a, b) = 0，我们不能得出任何结论，需要更多的测试。

### 示例：g(x, y)

对于函数 g(x,y)：

1.  对于点 (0, 0) 我们不能得出任何结论

1.  f_xx(1, 0) = 6 > 0 和 D_g(1, 0) = 60 > 0，因此 (1, 0) 是局部最小值

1.  点 (0,1) 是一个鞍点，因为 D_g(0, 1) < 0

1.  f_xx(-1,0) = -6 < 0 和 D_g(-1, 0) = 12 > 0，因此 (-1, 0) 是局部最大值

下图展示了函数 g(x, y) 的图形及其相应的等高线。

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/hessian4.png)

g(x,y) 的图形和 g(x,y) 的等高线

## **为什么 Hessian 矩阵在机器学习中很重要？**

Hessian 矩阵在许多机器学习算法中扮演着重要角色，这些算法涉及优化给定函数。尽管计算可能很昂贵，但它包含了一些关于被优化函数的关键信息。它可以帮助确定鞍点和函数的局部极值。在训练神经网络和深度学习架构中被广泛使用。

## **扩展**

本节列出了一些扩展教程的想法，你可能会希望探索。

+   优化

+   Hessian 矩阵的特征值

+   Hessian 矩阵的逆矩阵和神经网络训练

如果你探索了这些扩展内容，我很想知道。请在下面的评论中发布你的发现。

### **进一步阅读**

本节提供了更多关于该主题的资源，如果你希望深入了解。

### **教程**

+   [导数](https://machinelearningmastery.com/a-gentle-introduction-to-function-derivatives)

+   [机器学习中的梯度下降](https://machinelearningmastery.com/gradient-descent-for-machine-learning/)

+   [机器学习中的梯度是什么](https://machinelearningmastery.com/gradient-in-machine-learning/)

+   [偏导数和梯度向量](https://machinelearningmastery.com/a-gentle-introduction-to-partial-derivatives-and-gradient-vectors)

+   [高阶导数](https://machinelearningmastery.com/higher-order-derivatives/)

+   [如何选择优化算法](https://machinelearningmastery.com/tour-of-optimization-algorithms/)

### **资源**

+   额外资源：[机器学习的微积分书籍](https://machinelearningmastery.com/calculus-books-for-machine-learning/)

### **书籍**

+   [托马斯微积分](https://amzn.to/35Yeolv)，第 14 版，2017 年（基于乔治·B·托马斯的原著，由乔尔·哈斯、克里斯托弗·海尔、莫里斯·韦尔修订）

+   [微积分](https://www.amazon.com/Calculus-3rd-Gilbert-Strang/dp/0980232759/ref=as_li_ss_tl?dchild=1&keywords=Gilbert+Strang+calculus&qid=1606171602&s=books&sr=1-1&linkCode=sl1&tag=inspiredalgor-20&linkId=423b93db012f7cc6bb92cb7494a3095f&language=en_US)，第 3 版，2017 年（吉尔伯特·斯特朗）

+   [微积分](https://amzn.to/3kS9I52)，第 8 版，2015 年（詹姆斯·斯图尔特）

## **总结**

在本教程中，你了解到什么是 Hessian 矩阵。具体来说，你学习了：

+   Hessian 矩阵

+   函数的判别式

## **你有任何问题吗？**

在下方评论中提问，我会尽力回答。
