# 函数导数的温和介绍

> 原文：[`machinelearningmastery.com/a-gentle-introduction-to-function-derivatives/`](https://machinelearningmastery.com/a-gentle-introduction-to-function-derivatives/)

导数的概念是微积分许多主题的基础。它对理解积分、梯度、Hessian 矩阵等非常重要。

在本教程中，您将发现导数的定义、符号以及如何根据此定义计算导数。您还将发现为什么函数的导数本身也是一个函数。

完成本教程后，您将了解：

+   函数的导数定义

+   如何根据定义计算函数的导数

+   为什么有些函数在某点没有导数

让我们开始吧。

![Ayubia 国家公园。函数导数的温和介绍。](https://machinelearningmastery.com/wp-content/uploads/2021/06/IMG_5405-2-scaled.jpg)

函数导数的温和介绍 由 Mehreen Saeed 拍摄，保留部分权利

## 教程概览

本教程分为三个部分；它们是：

1.  函数导数的定义和符号

1.  如何根据定义计算函数的导数

1.  为什么有些函数在某点没有导数

## 函数的导数是什么

用非常简单的语言来说，函数 f(x) 的导数表示其变化率，通常用 f'(x) 或 df/dx 表示。我们首先来看它的定义和导数的图示。

![函数导数定义的图示](https://machinelearningmastery.com/wp-content/uploads/2021/06/derivDef.png)

函数导数定义的图示

在图中，Δx 表示 x 值的变化。我们不断缩小 x 和 (x+Δx) 之间的间隔，直到它变得微小。因此，我们有极限 (Δ????→0)。分子 f(x+Δx)-f(x) 表示函数 f 在 Δx 间隔内的变化量。这使得函数 f 在某点 x 的导数，即 f 在该点的变化率。

需要注意的一个重要点是 Δx，即 x 的变化量，可以是负值或正值。因此：

0<|Δx|< ????，

其中 ???? 是一个极小的值。

### 关于符号

函数的导数可以用 f'(x) 和 df/dx 表示。数学巨匠牛顿使用 f'(x) 来表示函数的导数。另一位数学英雄莱布尼茨使用 df/dx。因此 df/dx 是一个单一的术语，不要与分数混淆。它被读作函数 f 对 x 的导数，也表示 x 是自变量。

### 与速度的联系

最常引用的导数示例之一是速度的导数。速度是距离关于时间的变化率。因此，如果 f(t) 表示时间 t 处的行驶距离，则 f'(t) 是时间 t 处的速度。接下来的章节展示了计算导数的各种示例。

## 微分示例

找到函数的导数的方法称为微分。在本节中，我们将看到如何使用导数的定义来找到不同函数的导数。稍后，一旦你对定义更加熟悉，你可以使用定义的规则来对函数进行微分。

### 示例 1: m(x) = 2x+5

让我们从一个简单的线性函数 m(x) = 2x+5 开始。我们可以看到 m(x) 以恒定的速率变化。我们可以如下求导这个函数。

![m(x) = 2x+5 的导数](https://machinelearningmastery.com/wp-content/uploads/2021/06/mx.png)

m(x) = 2x+5 的导数

上图显示了函数 m(x) 的变化方式，并且无论选择哪个 x 值，m(x) 的变化率始终为 2。

### 示例 2: g(x) = x²

假设我们有函数 g(x) = x²。下图显示了关于 x 的 g(x) 导数是如何计算的。图中还显示了函数及其导数的绘图。

![g(x) = x² 的导数](https://machinelearningmastery.com/wp-content/uploads/2021/06/gx.png)

g(x) = x² 的导数

由于 g'(x) = 2x，因此 g'(0) = 0，g'(1) = 2，g'(2) = 4，g'(-1) = -2，g'(-2) = -4。

从图中可以看出，对于较大的负 x 值，g(x) 的值非常大。当 x < 0 时，增加 x 会减少 g(x)，因此对于 x < 0，g'(x) < 0。当 x=0 时，图形变平，此时 g(x) 的导数或变化率为零。当 x > 0 时，g(x) 随着 x 的增加呈二次增长，因此导数也是正的。

### 示例 3: h(x) = 1/x

假设我们有函数 h(x) = 1/x。下面展示了 h(x) 关于 x 的微分（对于 x ≠ 0），以及说明导数的图。蓝色曲线表示 h(x)，红色曲线表示其对应的导数。

![h(x) = 1/x 的导数](https://machinelearningmastery.com/wp-content/uploads/2021/06/hx.png)

h(x) = 1/x 的导数

## 可微性和连续性

例如示例 3，函数 h(x) = 1/x 在点 x=0 处未定义。因此，其导数 (-1/x²) 在 x=0 处也不被定义。如果函数在某点不连续，则该点没有导数。以下是几种函数不可微的情形：

1.  如果在某一点函数没有定义

1.  函数在该点没有极限

1.  如果函数在某点不连续

1.  函数在某点有突然跃升

以下是几个示例：

![没有导数的点示例](https://machinelearningmastery.com/wp-content/uploads/2021/06/noDeriv.png)

没有导数的点示例

## 扩展

本节列出了一些可能想要探索的扩展教程的想法。

+   速度和瞬时变化率

+   导数规则

+   积分

如果你探索了这些扩展内容，我很想知道。请在下面的评论中分享你的发现。

## 进一步阅读

本节提供了更多关于该主题的资源，如果你想深入了解。

## 教程

+   [极限与连续性](https://machinelearningmastery.com/a-gentle-introduction-to-limits-and-continuity)

+   [极限评估](https://machinelearningmastery.com/a-gentle-introduction-to-evaluating-limits)

### 资源

+   关于[机器学习的微积分书籍](https://machinelearningmastery.com/calculus-books-for-machine-learning/)的附加资源

### 书籍

+   [托马斯微积分](https://amzn.to/35Yeolv)，第 14 版，2017 年。（基于乔治·B·托马斯的原作，由乔尔·哈斯、克里斯托弗·海尔、莫里斯·威尔修订）

+   [微积分](https://www.amazon.com/Calculus-3rd-Gilbert-Strang/dp/0980232759/ref=as_li_ss_tl?dchild=1&keywords=Gilbert+Strang+calculus&qid=1606171602&s=books&sr=1-1&linkCode=sl1&tag=inspiredalgor-20&linkId=423b93db012f7cc6bb92cb7494a3095f&language=en_US)，第 3 版，2017 年。（吉尔伯特·斯特朗）

+   [微积分](https://amzn.to/3kS9I52)，第 8 版，2015 年。 （詹姆斯·斯图尔特）

## 总结

在本教程中，你了解了函数导数和函数微分的基础知识。

具体来说，你学习了：

+   函数导数的定义和符号

+   如何使用定义来微分一个函数

+   当一个函数不可微分时

你有任何问题吗？在下面的评论中提出你的问题，我会尽力回答。
