# 梯度下降过程的温和介绍

> 原文：[`machinelearningmastery.com/a-gentle-introduction-to-gradient-descent-procedure/`](https://machinelearningmastery.com/a-gentle-introduction-to-gradient-descent-procedure/)

梯度下降过程在机器学习中具有至关重要的意义。它常用于最小化分类和回归问题中的误差函数，也用于训练神经网络和深度学习架构。

在本教程中，你将发现梯度下降过程。

完成本教程后，你将了解到：

+   梯度下降方法

+   梯度下降在机器学习中的重要性

让我们开始吧。

![对梯度下降的温和介绍。照片由 Mehreen Saeed 提供，部分权利保留。](https://machinelearningmastery.com/wp-content/uploads/2021/07/IMG_9313-scaled.jpg)

对梯度下降的温和介绍。照片由 Mehreen Saeed 提供，部分权利保留。

## 教程概述

本教程分为两个部分，它们是：

1.  梯度下降过程

1.  梯度下降过程的示例

## 前提条件

对于本教程，假定你已具备以下主题的知识：

+   多变量函数

+   [偏导数和梯度向量](https://machinelearningmastery.com/a-gentle-introduction-to-partial-derivatives-and-gradient-vectors)

你可以通过点击上面的链接来复习这些概念。

## 梯度下降过程

梯度下降过程是一个用于寻找函数最小值的算法。

假设我们有一个函数 f(x)，其中 x 是多个变量的元组，即 x = (x_1, x_2, …x_n)。还假设 f(x) 的梯度为 ∇f(x)。我们希望找到使函数最小的变量值 (x_1, x_2, …x_n)。在任何迭代 t 中，我们用 x[t] 表示元组 x 的值。所以 x[t][1] 是迭代 t 中 x_1 的值，x[t][2] 是迭代 t 中 x_2 的值，依此类推。

### 符号说明

我们有以下变量：

+   t = 迭代次数

+   T = 总迭代次数

+   n = f 的定义域中的总变量（也称为 x 的维度）

+   j = 变量编号的迭代器，例如，x_j 表示第 j 个变量

+   ???? = 学习率

+   ∇f(x[t]) = 迭代 t 时 f 的梯度向量值

### 训练方法

梯度下降算法的步骤如下。这也被称为训练方法。

1.  选择一个随机的初始点 x_initial 并设置 x[0] = x_initial

1.  对于迭代 t=1..T

    +   更新 x[t] = x[t-1] – ????∇f(x[t-1])

就是这么简单！

学习率 ???? 是梯度下降过程中的用户定义变量，其值在 [0,1] 范围内。

上述方法说明在每次迭代时，我们需要通过朝梯度向量负方向小步移动来更新 x 的值。如果 ????=0，则 x 不会变化。如果 ????=1，则相当于在梯度向量的负方向上迈出大步。通常， ???? 设置为小值如 0.05 或 0.1。它也可以在训练过程中变化。因此你的算法可以从较大的值（例如 0.8）开始，然后逐渐减小到较小的值。

### 想开始学习机器学习的微积分吗？

现在就报名我的免费 7 天电子邮件速成课程（包括示例代码）。

点击注册并获得免费 PDF 电子书版课程。

## 梯度下降的示例

让我们找出下列两个变量的函数的最小值，其图形和轮廓如下图所示：

f(x,y) = x**x + 2y**y

![f(x,y) = x*x + 2y*y 的图形和轮廓](https://machinelearningmastery.com/wp-content/uploads/2021/07/gradientDescent1.png)

f(x,y) = x*x + 2y*y 的图形和轮廓

梯度向量的一般形式为：

∇f(x,y) = 2xi + 4yj

显示了算法的两个迭代 T=2 和 ????=0.1

1.  初始 t=0

    +   x[0] = (4,3)     # 这只是一个随机选择的点

1.  在 t = 1 时

    +   x[1] = x[0] – ????∇f(x[0])

    +   x[1] = (4,3) – 0.1*(8,12)

    +   x[1] = (3.2,1.8)

1.  在 t=2 时

    +   x[2] = x[1] – ????∇f(x[1])

    +   x[2] = (3.2,1.8) – 0.1*(6.4,7.2)

    +   x[2] = (2.56,1.08)

如果你持续运行上述迭代过程，最终程序会到达函数最小值的点，即 (0,0)。

在迭代 t=1 时，算法如图所示：

![梯度下降过程的示意图](https://machinelearningmastery.com/wp-content/uploads/2021/07/gradientDescent2.png)

梯度下降过程的示意图

## 需要运行多少次迭代？

通常，梯度下降会运行到 x 的值不再改变或 x 的变化低于某个阈值。停止准则也可以是用户定义的最大迭代次数（我们之前定义为 T）。

## 添加动量

梯度下降可能会遇到如下问题：

1.  在两个或多个点之间震荡

1.  陷入局部最小值

1.  超越并错过最小点

为了处理上述问题，可以在梯度下降算法的更新方程中添加动量项，如下所示：

x[t] = x[t-1] – ????∇f(x[t-1]) + ????*Δx[t-1]

其中 Δx[t-1] 代表 x 的变化，即：

Δx[t] = x[t] – x[t-1]

在 t=0 时的初始变化是零向量。对于这个问题 Δx[0] = (0,0)。

## 关于梯度上升

还有一个相关的梯度上升过程，用于寻找函数的最大值。在梯度下降中，我们沿着函数的最大减少率的方向前进，这就是负梯度向量的方向。而在梯度上升中，我们沿着函数最大增加率的方向前进，这就是正梯度向量的方向。我们也可以通过对 f(x)加上负号来将最大化问题转化为最小化问题，即，

```py
maximize f(x) w.r.t x        is equivalent to          minimize -f(x) w.r.t x
```

## 为什么梯度下降在机器学习中重要？

梯度下降算法常用于机器学习问题。在许多分类和回归任务中，均方误差函数用于将模型拟合到数据。梯度下降过程用于识别导致最低均方误差的最佳模型参数。

梯度上升用于类似的情境，解决涉及最大化函数的问题。

## 扩展

本节列出了一些你可能希望探索的教程扩展想法。

+   黑塞矩阵

+   雅可比矩阵

如果你探索了这些扩展内容，我很乐意知道。请在下方评论中发布你的发现。

## 进一步阅读

本节提供了更多关于该主题的资源，如果你希望深入了解。

### 教程

+   [导数](https://machinelearningmastery.com/a-gentle-introduction-to-function-derivatives)

+   [斜率和切线](https://machinelearningmastery.com/a-gentle-introduction-to-slopes-and-tangents)

+   [机器学习中的梯度下降](https://machinelearningmastery.com/gradient-descent-for-machine-learning/)

+   [什么是机器学习中的梯度](https://machinelearningmastery.com/gradient-in-machine-learning/)

+   [偏导数和梯度向量](https://machinelearningmastery.com/a-gentle-introduction-to-partial-derivatives-and-gradient-vectors)

### 资源

+   关于 [机器学习中的微积分书籍](https://machinelearningmastery.com/calculus-books-for-machine-learning/) 的额外资源

### 书籍

+   [托马斯微积分](https://amzn.to/35Yeolv)，第 14 版，2017 年。（基于 George B. Thomas 的原著，由 Joel Hass、Christopher Heil、Maurice Weir 修订）

+   [微积分](https://www.amazon.com/Calculus-3rd-Gilbert-Strang/dp/0980232759/ref=as_li_ss_tl?dchild=1&keywords=Gilbert+Strang+calculus&qid=1606171602&s=books&sr=1-1&linkCode=sl1&tag=inspiredalgor-20&linkId=423b93db012f7cc6bb92cb7494a3095f&language=en_US)，第 3 版，2017 年。（Gilbert Strang）

+   [微积分](https://amzn.to/3kS9I52)，第 8 版，2015 年。（James Stewart）

## 总结

在本教程中，你发现了梯度下降的算法。具体来说，你学到了：

+   梯度下降过程

+   如何应用梯度下降过程来找到函数的最小值

+   如何将一个最大化问题转化为最小化问题

## 你有任何问题吗？

在下方评论中提出你的问题，我会尽力回答。
