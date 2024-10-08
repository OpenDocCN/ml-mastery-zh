# 多变量微积分的温和介绍

> 原文：[`machinelearningmastery.com/a-gentle-introduction-to-multivariate-calculus/`](https://machinelearningmastery.com/a-gentle-introduction-to-multivariate-calculus/)

研究依赖于多个变量的函数通常是令人向往的。

多变量微积分通过将我们在微积分中发现的概念（如变化率的计算）扩展到多个变量，为我们提供了工具。它在训练神经网络的过程中起着至关重要的作用，其中梯度被广泛用于更新模型参数。

在本教程中，你将发现多变量微积分的温和介绍。

完成本教程后，你将了解到：

+   多变量函数依赖于多个输入变量来产生输出。

+   多变量函数的梯度是通过在不同方向上找到函数的导数来计算的。

+   多变量微积分在神经网络中被广泛使用，以更新模型参数。

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/multivariate_cover-scaled.jpg)

多变量微积分的温和介绍

图片来源 [Luca Bravo](https://unsplash.com/photos/O453M2Liufs)，版权所有。

## **教程概述**

本教程分为三个部分；它们是：

+   重新审视函数的概念

+   多变量函数的导数

+   多变量微积分在机器学习中的应用

## **重新审视函数的概念**

我们已经对[函数的概念](https://machinelearningmastery.com/what-you-need-to-know-before-you-get-started-a-brief-tour-of-calculus-pre-requisites/)有所了解，即一个定义因变量和自变量之间关系的规则。我们看到，函数通常表示为*y* = *f*(*x*)，其中输入（或自变量）*x*和输出（或因变量）*y*都是单个实数。

这样的函数接受一个单一的独立变量，并在输入和输出之间定义一对一的映射，称为*单变量*函数。

例如，假设我们尝试仅基于温度来预报天气。在这种情况下，天气是我们试图预测的因变量，它是温度（输入变量）的函数。因此，这样的问题可以很容易地框定为一个单变量函数。

但是，假设我们现在不仅基于温度，还想根据湿度水平和风速来进行天气预报。我们不能通过单变量函数来做到这一点，因为单变量函数的输出仅依赖于单一输入。

因此，我们将注意力转向*多变量*函数，因为这些函数可以接受多个变量作为输入。

从形式上讲，我们可以将多变量函数表示为多个实数输入变量 *n* 到一个实数输出的映射：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/multivariate_3.png)

例如，考虑以下抛物面：

*f*(*x*, *y*) = *x*² *+* 2*y*²

这是一个多变量函数，接受两个变量，*x* 和 *y*，作为输入，因此 *n* = 2，生成一个输出。我们可以通过绘制 *x* 和 *y* 在 -1 到 1 之间的值来进行可视化。

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/multivariate_1.png)

抛物面的三维图

同样，我们可以有接受更多变量作为输入的多变量函数。然而，由于涉及的维度数量，进行可视化可能会很困难。

我们甚至可以进一步推广函数的概念，考虑那些将多个输入 *n* 映射到多个输出 *m* 的函数：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/multivariate_4.png)

这些函数通常被称为*向量值*函数。

## **多变量函数的导数**

[回顾](https://machinelearningmastery.com/key-concepts-in-calculus-rate-of-change/) 微积分涉及变化率的研究。对于某些单变量函数，*g*(*x*)，这可以通过计算其导数来实现：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/multivariate_5.png)

> *导数在多个变量的函数中的推广是梯度。*
> 
> *– 第 146 页, [机器学习的数学](https://www.amazon.com/Mathematics-Machine-Learning-Peter-Deisenroth/dp/110845514X/ref=as_li_ss_tl?dchild=1&keywords=calculus+machine+learning&qid=1606171788&s=books&sr=1-3&linkCode=sl1&tag=inspiredalgor-20&linkId=209ba69202a6cc0a9f2b07439b4376ca&language=en_US), 2020.*

查找多个变量函数的梯度的技术涉及每次改变其中一个变量，同时保持其他变量不变。这样，我们每次都会对多变量函数关于每个变量进行*偏导数*计算。

> *梯度则是这些偏导数的集合。*
> 
> *– 第 146 页, [机器学习的数学](https://www.amazon.com/Mathematics-Machine-Learning-Peter-Deisenroth/dp/110845514X/ref=as_li_ss_tl?dchild=1&keywords=calculus+machine+learning&qid=1606171788&s=books&sr=1-3&linkCode=sl1&tag=inspiredalgor-20&linkId=209ba69202a6cc0a9f2b07439b4376ca&language=en_US), 2020.*

为了更好地可视化这种技术，让我们首先考虑一个形式简单的单变量二次函数：

*g*(*x*) = *x*²

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/multivariate_2.png)

单变量二次函数的线图

在某个点*x*上找到这个函数的导数，需要应用我们之前定义的*g*’(*x*)的方程。我们可以选择使用幂法则来简化计算：

*g’(x*) = 2*x*

此外，如果我们要想象切开之前考虑的抛物面，并且用一条通过*y* = 0 的平面来切割，我们会发现*f*(*x*, *y*)的横截面是二次曲线，*g*(*x*) = *x*²。因此，我们可以通过对*f*(*x*, *y*)进行导数计算（或称陡度，或*slope*），在*x*方向上得到抛物面的导数，同时保持*y*不变。我们称之为*f*(*x*, *y*)关于*x*的*偏导*，用*∂*表示，以说明除了*x*之外还有更多变量，但这些变量暂时不考虑。因此，*f*(*x*, *y*)关于*x*的偏导数为：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/multivariate_6.png)

我们可以类似地保持*x*不变（换句话说，通过用一条通过*x*的平面来切割抛物面，以找到抛物面在*y*方向上的横截面），以找到*f*(*x*, *y*)关于*y*的偏导数，如下所示：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/multivariate_7.png)

我们基本上做的是找到*f*(*x*, *y*)在*x*和*y*方向上的单变量导数。将两个单变量导数结合起来作为最终步骤，给我们提供了多变量导数（或梯度）：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/multivariate_8.png)

相同的技术适用于更高维度的函数。

## **多变量微积分在机器学习中的应用**

偏导数在神经网络中被广泛用于更新模型参数（或权重）。

[我们曾看到过](https://machinelearningmastery.com/calculus-in-machine-learning-why-it-works/) 在最小化某些误差函数时，优化算法会试图沿着其梯度下坡。如果这个误差函数是单变量的，因此是一个单一独立权重的函数，那么优化它就只是计算其单变量导数。

然而，神经网络包含许多权重（每个权重对应一个不同的神经元），误差是这些权重的函数。因此，更新权重值需要计算误差曲线对所有这些权重的梯度。

这就是多变量微积分应用的地方。

错误曲线的梯度通过计算误差对每个权重的偏导数来得到；换句话说，就是通过保持除当前考虑的权重以外的所有权重不变来求误差函数的导数。这使得每个权重可以独立更新，从而达到找到最佳权重集的目标。

## **进一步阅读**

本节提供了更多关于该主题的资源，如果你想深入了解，可以参考。

### **书籍**

+   [单变量与多变量微积分](https://www.whitman.edu/mathematics/multivariable/multivariable.pdf)，2020。

+   [机器学习数学](https://www.amazon.com/Mathematics-Machine-Learning-Peter-Deisenroth/dp/110845514X/ref=as_li_ss_tl?dchild=1&keywords=calculus+machine+learning&qid=1606171788&s=books&sr=1-3&linkCode=sl1&tag=inspiredalgor-20&linkId=209ba69202a6cc0a9f2b07439b4376ca&language=en_US)，2020。

+   [优化算法](https://www.amazon.com/Algorithms-Optimization-Press-Mykel-Kochenderfer/dp/0262039427/ref=sr_1_1?dchild=1&keywords=algorithms+for+optimization&qid=1624019308&sr=8-1)，2019。

+   [深度学习](https://www.amazon.com/Deep-Learning-Press-Essential-Knowledge/dp/0262537559/ref=sr_1_4?dchild=1&keywords=deep+learning&qid=1622968138&sr=8-4)，2019。

## **总结**

在本教程中，你发现了对多变量微积分的温和介绍。

具体来说，你学到了：

+   多变量函数依赖于多个输入变量以产生输出。

+   多变量函数的梯度是通过在不同方向上求函数的导数来计算的。

+   多变量微积分在神经网络中被广泛使用，用于更新模型参数。

你有什么问题吗？

在下方评论中提问，我会尽力回答。
