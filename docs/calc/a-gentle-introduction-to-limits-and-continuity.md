# 极限和连续性的温和介绍

> 原文：[`machinelearningmastery.com/a-gentle-introduction-to-limits-and-continuity/`](https://machinelearningmastery.com/a-gentle-introduction-to-limits-and-continuity/)

毫无疑问，微积分是一个难度较大的学科。然而，如果你掌握了基础知识，你不仅能够理解更复杂的概念，还会觉得它们非常有趣。要理解机器学习算法，你需要了解诸如函数的梯度、矩阵的赫西矩阵以及优化等概念。极限和连续性的概念为所有这些主题提供了基础。

在这篇文章中，你将学习如何评估一个函数的极限，以及如何确定一个函数是否连续。

阅读完这篇文章后，你将能够：

+   确定一个函数 f(x) 当 x 趋近于某个值时是否存在极限

+   计算一个函数 f(x) 当 x 趋近于 a 时的极限

+   确定一个函数在某点或区间内是否连续

我们开始吧。

## 教程概述

本教程分为两个部分

1.  极限

    1.  确定函数在某点的极限是否存在

    1.  计算函数在某点的极限

    1.  极限的正式定义

    1.  极限的例子

    1.  左极限和右极限

1.  连续性

    1.  连续性的定义

    1.  确定一个函数在某点或区间内是否连续

    1.  连续函数的例子

![极限和连续性的温和介绍](https://machinelearningmastery.com/wp-content/uploads/2021/06/mainpic.png)

极限和连续性的温和介绍

摄影：Mehreen Saeed，部分版权保留。

## 一个简单的例子

让我们先来看一个简单的函数 f(x)，由以下给出：

f(x) = 1+x

f(x) 在 -1 附近会发生什么？

我们可以看到，f(x) 在 x 越来越接近 -1 时，f(x) 越来越接近 0，无论是从 x=-1 的哪一侧。 在 x=-1 时，函数正好为零。我们说当 x 趋近于 -1 时，f(x) 的极限等于 0。

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/limit1.png)

### 扩展例子

扩展问题。让我们定义 g(x)：

g(x) = (1-x²)/(1+x)

我们可以将 g(x) 的表达式简化为：

g(x) = (1-x)(1+x)/(1+x)

如果分母不为零，则 g(x) 可以简化为：

g(x) = 1-x, 如果 x ≠ -1

然而，在 (x = -1) 处，分母为零，我们无法除以零。所以在 x=-1 处似乎存在一个孔。尽管存在这个孔，g(x) 在 x 趋近于 -1 时仍然越来越接近 2，如图所示：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/limit2.png)

这是极限的基本概念。如果 g(x) 在不包含 -1 的开区间内定义，并且当 x 趋近于 -1 时，g(x) 越来越接近 2，我们写作：

lim(x→-1) g(x) = 2

一般来说，对于任何函数 f(x)，如果 f(x) 在 x 趋近于 k 时越来越接近一个值 L，我们定义 f(x) 在 x 接近 k 时的极限为 L。这个定义写作：![limit f(x) = L as x approaches k](https://machinelearningmastery.com/wp-content/uploads/2021/06/l1.png)

### 左极限和右极限

对于函数 g(x)，无论我们是增加 x 以接近 -1（从左接近 -1），还是减少 x 以接近 -1（从右接近 -1），g(x) 都会越来越接近 2。这在下面的图中展示了：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/limit3.png)

这引出了单侧极限的概念。左极限在 -1 左侧的区间上定义，不包括 -1，例如 (-1.003, -1)。当我们从左侧接近 -1 时，g(x) 趋近于 2。

类似地，右极限在 -1 右侧的开区间上定义，不包括 -1，例如 (-1, 0.997)。当我们从右侧接近 -1 时，g(x) 的右极限为 2。左极限和右极限都写作：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/l2.png)

我们说 f(x) 在 x 接近 k 时有极限 L，如果它的左极限和右极限都相等。因此，这是一种测试函数在特定点是否有极限的另一种方法，即，![](https://machinelearningmastery.com/wp-content/uploads/2021/06/l3.png)

## 极限的正式定义

在数学中，我们需要对一切有一个精确的定义。为了正式定义极限，我们将使用希腊字母????的概念。数学界一致同意使用????表示任意小的正数，这意味着我们可以将????变得任意小，并且它可以接近于零，只要????>0（因此????不能为零）。

如果对于每一个 ?????0，都存在一个正数 ?????0，使得：

如果 0<|????−????|<????，则 |????(????)−????|<????

这个定义非常简单。x-k 是 x 与 k 的差异，|x-k| 是 x 与 k 之间的距离，不考虑差异的符号。类似地，|f(x)-L| 是 f(x) 与 L 之间的距离。因此，定义表示当 x 与 k 之间的距离接近任意小的值时，f(x) 与 L 之间的距离也接近一个非常小的值。下面的图很好地说明了这个定义：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/limit4.png)

## 极限的例子

下面的图展示了一些例子，并在下面进行了说明：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/limitExample.png)

### 1.1 绝对值示例

f_1(x) = |x|

f_1(x) 的极限在所有 x 值上都存在，例如，lim(x→0) f_1(x) = 0。

### 1.2 多项式示例

f_2(x) = x² + 3x + 1

f_2(x) 的极限在所有 x 值上都存在，例如，lim(x→1) f_2(x) = 1+3+1 = 5。

### 1.3 无限的例子

f_3(x) = 1/x,  如果 x>0

f_3(x) = 0,   如果 x<=0

对于上述情况，当 x 越来越大时，f_3(x) 的值越来越小，趋近于零。因此，lim(x→∞) f_3(x) = 0。

## 不具有极限的函数示例

从极限的定义可以看出，以下函数没有极限：

### 2.1 单位阶跃函数

单位阶跃函数 H(x) 定义为：

H(x) = 0,  如果 x<0

H(x) = 1,  否则

当我们从左侧越来越接近 0 时，函数保持为零。然而，一旦我们到达 x=0，H(x) 跳到 1，因此 H(x) 在 x 趋近于零时没有极限。这个函数在左侧的极限等于零，而右侧的极限等于 1。

左侧和右侧的极限不一致，因为 x→0 时，H(x) 没有极限。因此，这里我们使用左右侧极限的相等作为检验函数在某点是否有极限的测试。

### 2.2 倒数函数

考虑 h_1(x)：

h_1(x) = 1/(x-1)

当我们从左侧接近 x=1 时，函数趋向于大的负值。当我们从右侧接近 x=1 时，h_1(x) 增加到大的正值。因此，当 x 接近 1 时，h_1(x) 的值不会接近一个固定的实数值。因此，x→1 的极限不存在。

### 2.3 向上取整函数

考虑向上取整函数，它将带有非零小数部分的实数四舍五入到下一个整数值。因此，lim(x→1) ceil(x) 不存在。实际上，ceil(x) 在任何整数值上都没有极限。

所有上述示例都在下图中展示：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/nolimit-1.png)

## 连续性

如果你理解了极限的概念，那么理解连续性就容易了。函数 f(x) 在点 a 连续，如果满足以下三个条件：

1.  f(a) 应该存在

1.  f(x) 在 x 趋近于 a 时有极限

1.  当 x 趋近于 a 时，f(x) 的极限等于 f(a)

如果以上条件都成立，那么函数在点 a 处是连续的。以下是一些示例：

### 连续性的示例

连续性的概念与极限密切相关。如果函数在某点定义良好、在该点没有跳跃，并且在该点有极限，那么它在该点是连续的。下图展示了一些示例，并在下文中解释：

### ![](https://machinelearningmastery.com/wp-content/uploads/2021/06/cont.png)

### 3.1 平方函数

以下函数 f_4(x) 对所有 x 值都是连续的：

f_4(x) = x²

### 3.2 有理函数

我们之前使用的函数 g(x):

g(x) = (1-x²)/(1+x)

g(x) 在除了 x=-1 之外的所有地方都是连续的。

我们可以将 g(x) 修改为 g**(x):*

*g**(x) = (1-x²)/(1+x)，如果 x ≠ -1

g*(x) = 2, 否则

现在我们有一个对所有 x 值都连续的函数。

### 3.3 倒数函数

回到我们之前的例子 f_3(x):

f_3(x) = 1/x, 如果 x>0

f_3(x) = 0, 如果 x<=0

f_3(x) 在除 x=0 外的所有地方都是连续的，因为在 x=0 时 f_3(x) 的值有一个很大的跳跃。因此，在 x=0 处存在不连续性。

## 进一步阅读

本节提供了更多相关资源，如果你想深入了解。数学全在于练习，下面是一些提供更多练习和例子的资源列表。

### 资源页面

+   杰森·布朗利关于[机器学习的微积分书籍](https://machinelearningmastery.com/calculus-books-for-machine-learning/)的优秀资源。

### 书籍

+   [托马斯的微积分](https://www.amazon.com/Thomas-Calculus-14th-Joel-Hass/dp/0134438981/ref=as_li_ss_tl?ie=UTF8&linkCode=sl1&tag=inspiredalgor-20&linkId=1fcceb2171bd06294b60a6aa4cd51550&language=en_US)，第 14 版，2017 年（基于乔治·B·托马斯的原著，由乔尔·哈斯、克里斯托弗·海尔、莫里斯·维尔修订）。

+   [微积分](https://www.amazon.com/Calculus-3rd-Gilbert-Strang/dp/0980232759/ref=as_li_ss_tl?dchild=1&keywords=Gilbert+Strang+calculus&qid=1606171602&s=books&sr=1-1&linkCode=sl1&tag=inspiredalgor-20&linkId=423b93db012f7cc6bb92cb7494a3095f&language=en_US)，第 3 版，2017 年（吉尔伯特·斯特朗）。

+   [微积分](https://www.amazon.com/Calculus-James-Stewart/dp/1285740629/ref=as_li_ss_tl?dchild=1&keywords=calculus&qid=1606170839&sr=8-6&linkCode=sl1&tag=inspiredalgor-20&linkId=3bbba00479751e9e9bf3990c98f629c7&language=en_US)，第 8 版，2015 年（詹姆斯·斯图尔特）。

## 总结

在这篇文章中，你了解了关于极限和连续性的微积分概念。

具体来说，你学到了：

+   函数在接近某一点时是否存在极限

+   函数是否在某一点或区间内连续

你有任何问题吗？请在下方评论中提问，我会尽力回答。
