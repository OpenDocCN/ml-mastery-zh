# 拉格朗日乘子法：支持向量机背后的理论（第一部分：可分离情况）

> 原文：[`machinelearningmastery.com/method-of-lagrange-multipliers-the-theory-behind-support-vector-machines-part-1-the-separable-case/`](https://machinelearningmastery.com/method-of-lagrange-multipliers-the-theory-behind-support-vector-machines-part-1-the-separable-case/)

本教程旨在为任何希望深入理解拉格朗日乘子如何在支持向量机（SVM）模型构建中使用的人提供指导。SVM 最初设计用于解决二分类问题，后来扩展并应用于回归和无监督学习。它们在解决许多复杂的机器学习分类问题上取得了成功。

在本教程中，我们将讨论最简单的 SVM，假设正例和负例可以通过线性超平面完全分开。

完成本教程后，你将了解：

+   超平面如何作为决策边界

+   对正例和负例的数学约束

+   什么是边际以及如何最大化边际

+   拉格朗日乘子在最大化边际中的作用

+   如何确定可分离情况下的分离超平面

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/11/IMG_9900-scaled.jpg)

拉格朗日乘子法：支持向量机背后的理论（第一部分：可分离情况）

照片由 Mehreen Saeed 拍摄，部分权利保留。

本教程分为三个部分，它们是：

1.  SVM 的数学模型的公式化

1.  通过拉格朗日乘子法寻找最大边际超平面的解

1.  解决的示例以演示所有概念

## 本教程中使用的符号

+   $m$：总训练点数。

+   $n$：所有数据点的特征总数或维度

+   $x$：数据点，是一个 n 维向量。

+   $x^+$：标记为+1 的数据点。

+   $x^-$：标记为-1 的数据点。

+   $i$：用于索引训练点的下标。 $0 \leq i < m$

+   $j$：用于索引数据点的单独维度的下标。 $1 \leq j \leq n$

+   $t$：数据点的标签。

+   T：转置算子。

+   $w$：表示超平面系数的权重向量。它也是一个 n 维向量。

+   $\alpha$：拉格朗日乘子，每个训练点一个。这是一个 m 维向量。

+   $d$：数据点到决策边界的垂直距离。

## 超平面作为决策边界

![](https://machinelearningmastery.com/wp-content/uploads/2021/11/intro1.png)

支持向量机旨在区分属于两个不同类别的数据点。一组点标记为+1，也称为正类。另一组点标记为-1，也称为负类。现在，我们将做一个简化假设，假设两个类别的点可以通过线性超平面进行区分。

SVM 假设两个类别之间有一个线性的决策边界，目标是找到一个超平面，使两个类别之间的分离最大。因此，有时使用术语`最大边界分类器`来指代 SVM。最近的数据点与决策边界之间的垂直距离被称为`边界`。由于边界完全分隔了正负示例，并且不容忍任何错误，因此也被称为`硬边界`。

超平面的数学表达式如下，其中\(w_j\)是系数，\(w_0\)是决定超平面距离原点的任意常数：

$$

w^T x_i + w_0 = 0

$$

对于第$i$个二维点$(x_{i1}, x_{i2})$，上述表达式简化为：

$$

w_1x_{i1} + w_2 x_{i2} + w_0 = 0

$$

### 对正负数据点的数学约束

由于我们希望最大化正负数据点之间的边界，因此我们希望正数据点满足以下约束：

$$

w^T x_i^+ + w_0 \geq +1

$$

同样，负数据点应满足：

$$

w^T x_i^- + w_0 \leq -1

$$

我们可以通过使用$t_i \in \{-1,+1\}$来表示数据点$x_i$的类别标签，使用一个整齐的技巧来写出两个点集的统一方程：

$$

t_i(w^T x_i + w_0) \geq +1

$$

## 最大边界超平面

数据点$x_i$到边界的垂直距离$d_i$由以下公式给出：

$$

d_i = \frac{|w^T x_i + w_0|}{||w||}

$$

为了最大化这个距离，我们可以最小化分母的平方，从而得到一个二次规划问题：

$$

\min \frac{1}{2}||w||² \;\text{ subject to } t_i(w^Tx_i+w_0) \geq +1, \forall i

$$

## 通过拉格朗日乘子法的解

为了解决上述带有不等式约束的二次规划问题，我们可以使用拉格朗日乘子法。因此，拉格朗日函数为：

$$

L(w, w_0, \alpha) = \frac{1}{2}||w||² + \sum_i \alpha_i\big(t_i(w^Tx_i+w_0) – 1\big)

$$

为了解决上述问题，我们设置如下：

\begin{equation}

\frac{\partial L}{ \partial w} = 0, \\

\frac{\partial L}{ \partial \alpha} = 0, \\

\frac{\partial L}{ \partial w_0} = 0 \\

\end{equation}

将上述内容代入拉格朗日函数得到如下优化问题，也称为对偶问题：

$$

L_d = -\frac{1}{2} \sum_i \sum_k \alpha_i \alpha_k t_i t_k (x_i)^T (x_k) + \sum_i \alpha_i

$$

我们必须在以下条件下最大化上述目标：

$$

w = \sum_i \alpha_i t_i x_i

$$

和

$$

0=\sum_i \alpha_i t_i

$$

上述方法的好处在于，我们有一个关于 \(w\) 的表达式，涉及到拉格朗日乘子。目标函数中没有 $w$ 项。每个数据点都有一个相关的拉格朗日乘子。$w_0$ 的计算也在后面解释。

## 决定测试点的分类

任何测试点 $x$ 的分类可以使用这个表达式来确定：

$$

y(x) = \sum_i \alpha_i t_i x^T x_i + w_0

$$

$y(x)$ 的正值意味着 $x\in+1$，负值则意味着 $x\in-1$

### 想要开始学习机器学习中的微积分？

立即获取我的免费 7 天邮件速成课程（包含示例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## Karush-Kuhn-Tucker 条件

此外，Karush-Kuhn-Tucker (KKT) 条件满足上述约束优化问题，如下所示：

\begin{eqnarray}

\alpha_i &\geq& 0 \\

t_i y(x_i) -1 &\geq& 0 \\

\alpha_i(t_i y(x_i) -1) &=& 0

\end{eqnarray}

### KKT 条件的解释

KKT 条件规定，对于每个数据点，以下之一是正确的：

+   拉格朗日乘子为零，即 \(\alpha_i=0\)。因此，这一点在分类中没有作用。

或

+   $ t_i y(x_i) = 1$ 和 $\alpha_i > 0$：在这种情况下，数据点在决定 $w$ 的值时起作用。这样的点被称为支持向量。

### 计算 $w_0$

对于 $w_0$，我们可以选择任何支持向量 $x_s$ 并求解

$$

t_s y(x_s) = 1

$$

给出：

$$

t_s(\sum_i \alpha_i t_i x_s^T x_i + w_0) = 1

$$

## 一个已解决的示例

为了帮助你理解上述概念，这里有一个简单的任意求解示例。当然，对于大量点，你会使用优化软件来解决这个问题。此外，这只是满足所有约束条件的一个可能解决方案。目标函数可以进一步最大化，但超平面的斜率对于最优解将保持不变。此外，对于这个例子，$w_0$ 是通过取所有三个支持向量的 $w_0$ 的平均值来计算的。

这个示例将向你展示模型并不像它看起来那么复杂。![](https://machinelearningmastery.com/wp-content/uploads/2021/11/intro2-1.png)

对于上述点集，我们可以看到 (1,2)、(2,1) 和 (0,0) 是离分隔超平面最近的点，因此，作为支持向量。远离边界的点（例如 (-3,1)）在确定点的分类时没有任何作用。

## 进一步阅读

本节提供更多有关该主题的资源，如果你想深入了解。

### 书籍

+   [模式识别与机器学习](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738) 由 Christopher M. Bishop 编著

+   [托马斯微积分](https://amzn.to/35Yeolv)，第 14 版，2017 年（基于 George B. Thomas 的原著，由 Joel Hass、Christopher Heil 和 Maurice Weir 修订）

### 文章

+   [支持向量机在机器学习中的应用](https://machinelearningmastery.com/support-vector-machines-for-machine-learning/)

+   [支持向量机模式识别教程](https://www.di.ens.fr/~mallat/papiers/svmtutorial.pdf) 作者：克里斯托弗·J.C.·伯吉斯

## 总结

在本教程中，你了解了如何使用拉格朗日乘子法解决通过具有不等式约束的二次规划问题来最大化间隔的问题。

具体来说，你学到了：

+   分离线性超平面的数学表达式

+   最大间隔作为具有不等式约束的二次规划问题的解

+   如何使用拉格朗日乘子法找到正负样本之间的线性超平面

对于这篇文章中讨论的支持向量机，你有任何问题吗？请在下面的评论中提出你的问题，我会尽力回答。
