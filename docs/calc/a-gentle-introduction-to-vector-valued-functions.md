# 向量值函数的温和介绍

> 原文：[`machinelearningmastery.com/a-gentle-introduction-to-vector-valued-functions/`](https://machinelearningmastery.com/a-gentle-introduction-to-vector-valued-functions/)

向量值函数经常在机器学习、计算机图形学和计算机视觉算法中遇到。它们特别适用于定义空间曲线的参数方程。理解向量值函数的基本概念对于掌握更复杂的概念非常重要。

在本教程中，您将了解什么是向量值函数，如何定义它们以及一些示例。

完成本教程后，您将了解：

+   向量值函数的定义

+   向量值函数的导数

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/mano.jpg)

对向量值函数的温和介绍。照片由 Noreen Saeed 拍摄，部分权利保留。

## 教程概述

本教程分为两个部分；它们是：

1.  向量值函数的定义和示例

1.  向量值函数的微分

## 向量值函数的定义

向量值函数也称为向量函数。它是具有以下两个属性的函数：

1.  定域是一组实数

1.  范围是一组向量

因此，向量函数简单地是标量函数的扩展，其中定义域和值域都是实数集。

在本教程中，我们将考虑其值域是二维或三维向量集的向量函数。因此，这些函数可以用来定义空间中的一组点。

给定与 x 轴、y 轴、z 轴平行的单位向量 i,j,k，我们可以将三维向量值函数写成：

r(t) = x(t)i + y(t)j + z(t)k

它也可以写成：

r(t) = <x(t), y(t), z(t)>

上述两种符号是等价的，并且在各种教科书中经常使用。

### 空间曲线和参数方程

在前面的部分我们定义了一个向量函数 r(t)。对于不同的 t 值，我们得到相应的 (x,y,z) 坐标，由函数 x(t), y(t) 和 z(t) 定义。因此生成的点集 (x,y,z) 定义了一个称为空间曲线 C 的曲线。因此，x(t), y(t) 和 z(t) 的方程也称为曲线 C 的参数方程。

## 向量函数的例子

本节展示了一些定义空间曲线的向量值函数的例子。所有的例子也都在例子后面的图中绘制出来。

### 1.1 一个圆

让我们从一个简单的二维空间中的向量函数的例子开始：

r_1(t) = cos(t)i + sin(t)j

这里的参数方程是：

x(t) = cos(t)

y(t) = sin(t)

参数方程定义的空间曲线是二维空间中的圆，如图所示。如果我们将 t 从 -???? 变化到 ????，我们将生成所有落在圆上的点。

### 1.2 螺旋线

我们可以扩展示例 1.1 中的 r_1(t) 函数，以便在三维空间中轻松生成螺旋线。我们只需要添加沿 z 轴随 t 变化的值。因此，我们有以下函数：

r_2(t) = cos(t)i + sin(t)j + tk

### 1.3 扭曲的立方体

我们还可以定义一种具有有趣形状的曲线，称为扭曲的立方体，如下所示：

r_3(t) = ti + t²j + t³k

![参数曲线](https://machinelearningmastery.com/wp-content/uploads/2021/07/vecfun1.png)

参数曲线

### 想要开始学习机器学习中的微积分吗？

立即领取我的免费 7 天邮件速成课程（附样例代码）。

点击注册并获得免费的 PDF 电子书版本课程。

## 向量函数的导数

我们可以很容易地将标量函数的导数的概念扩展到向量函数的导数。由于向量函数的值范围是一组向量，因此其导数也是一个向量。

如果

r(t) = x(t)i + y(t)j + z(t)k

那么 r(t) 的导数为 r'(t)，计算公式如下：

r'(t) = x'(t)i + y'(t)i + z'(t)k

## 向量函数的导数示例

我们可以找到前一个示例中定义的函数的导数，如下所示：

### 2.1 圆

2D 中圆的参数方程为：

r_1(t) = cos(t)i + sin(t)j

因此，其导数是通过计算 x(t) 和 y(t) 的相应导数得到的，如下所示：

x'(t) = -sin(t)

y'(t) = cos(t)

这给我们：

r_1′(t) = x'(t)i + y'(t)j

r_1′(t) = -sin(t)i + cos(t)j

由参数方程定义的空间曲线在 2D 空间中是一个圆，如图所示。如果我们将 t 从 -???? 变为 π，我们将生成所有位于圆上的点。

### 2.2 螺旋线

类似于之前的例子，我们可以计算 r_2(t) 的导数，如下所示：

r_2(t) = cos(t)i + sin(t)j + tk

r_2′(t) = -sin(t)i + cos(t)j + k

### 2.3 扭曲的立方体

r_3(t) 的导数为：

r_3(t) = ti + t²j + t³k

r_3′(t) = i + 2tj + 3t²k

所有上述示例都显示在图中，其中导数以红色绘制。注意，圆的导数也在空间中定义了一个圆。

![参数函数及其导数](https://machinelearningmastery.com/wp-content/uploads/2021/07/vecfun2.png)

参数函数及其导数

## 更复杂的示例

一旦你对这些函数有了基本了解，你可以通过定义各种形状和曲线在空间中获得很多乐趣。数学界使用的其他流行示例如下所定义，并在图中进行了说明。

**环形螺旋**：

r_4(t) = (4 + sin(20t))cos(t)i + (4 + sin(20t))sin(t)j + cos(20t)k

**三叶结**：

r_5(t) = (2 + cos(1.5t))cos (t)i + (2 + cos(1.5t))sin(t)j + sin(1.5t)k

**心形曲线：**

r_6(t) = cos(t)(1-cos(t))i + sin(t)(1-cos(t))j

![更复杂曲线的图像](https://machinelearningmastery.com/wp-content/uploads/2021/07/vecfunc3.png)

更复杂的曲线

## 向量值函数在机器学习中的重要性

向量值函数在机器学习算法中扮演着重要角色。作为标量值函数的扩展，您会在多类分类和多标签问题等任务中遇到它们。[核方法](https://en.wikipedia.org/wiki/Kernel_methods_for_vector_output)，作为机器学习中的一个重要领域，可能涉及计算向量值函数，这些函数可以在多任务学习或迁移学习中使用。

## 扩展

本节列出了一些扩展教程的想法，您可能希望探索这些内容。

+   向量函数的积分

+   抛体运动

+   空间中的弧长

+   [向量输出的核方法](https://en.wikipedia.org/wiki/Kernel_methods_for_vector_output)

如果您探索了这些扩展，我很想知道。请在下方评论中发布您的发现。

## 进一步阅读

本节提供了更多资源，如果您想深入了解这个主题。

### 教程

+   [多变量微积分的温和介绍](https://machinelearningmastery.com/a-gentle-introduction-to-multivariate-calculus)

+   [导数](https://machinelearningmastery.com/a-gentle-introduction-to-function-derivatives)

### 资源

+   关于[机器学习的微积分书籍](https://machinelearningmastery.com/calculus-books-for-machine-learning/)的额外资源

### 书籍

+   [托马斯微积分](https://amzn.to/35Yeolv)，第 14 版，2017 年。（基于乔治·B·托马斯的原著，由乔尔·哈斯、克里斯托弗·海尔、莫里斯·韦尔修订）

+   [微积分](https://www.amazon.com/Calculus-3rd-Gilbert-Strang/dp/0980232759/ref=as_li_ss_tl?dchild=1&keywords=Gilbert+Strang+calculus&qid=1606171602&s=books&sr=1-1&linkCode=sl1&tag=inspiredalgor-20&linkId=423b93db012f7cc6bb92cb7494a3095f&language=en_US)，第 3 版，2017 年。（吉尔伯特·斯特朗）

+   [微积分](https://amzn.to/3kS9I52)，第 8 版，2015 年。（詹姆斯·斯图尔特）

## 总结

在本教程中，您了解了什么是向量函数以及如何对其进行微分。

具体来说，您学到了：

+   向量函数的定义

+   参数曲线

+   向量函数的微分

## 您有任何问题吗？

在下方评论中提出您的问题，我会尽力回答。
