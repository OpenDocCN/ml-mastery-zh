# 导数的应用

> 原文：[`machinelearningmastery.com/applications-of-derivatives/`](https://machinelearningmastery.com/applications-of-derivatives/)

导数定义了一个变量相对于另一个变量的变化率。

这是一个非常重要的概念，在许多应用中极为有用：在日常生活中，导数可以告诉你你的行驶速度，或帮助你预测股市的波动；在机器学习中，导数对于函数优化至关重要。

本教程将探索导数的不同应用，从较为熟悉的开始，然后再到机器学习。我们将深入研究导数告诉我们关于我们所研究的不同函数的内容。

在本教程中，你将发现导数的不同应用。

完成本教程后，你将知道：

+   导数的使用可以应用于我们周围的实际问题。

+   导数在机器学习中对于函数优化至关重要。

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/applications_cover-scaled.jpg)

导数的应用

图片由[Devon Janse van Rensburg](https://unsplash.com/photos/QT0q-nPWIII)提供，版权所有。

## **教程概述**

本教程分为两个部分；它们是：

+   导数在实际生活中的应用

+   导数在优化算法中的应用

## **导数在实际生活中的应用**

我们已经看到，导数模型用于描述变化率。

> *导数回答诸如“多快？”“多陡？”和“多敏感？”这样的问题。这些都是关于变化率的各种形式的问题。*
> 
> *– 第 141 页，[无限的力量](https://www.amazon.com/Infinite-Powers-Calculus-Reveals-Universe/dp/0358299284/ref=as_li_ss_tl?dchild=1&keywords=joy+of+x&qid=1606170381&sr=8-2&linkCode=sl1&tag=inspiredalgor-20&linkId=17ed7cfdd9b7dd013730d0699a8652a1&language=en_US)，2019 年。*

这种变化率表示为，????*y* / ????*x*，从而定义了因变量????*y*相对于自变量????*x*的变化。

让我们从我们周围最熟悉的导数应用之一开始。

> *每次你上车时，你都在目睹微分。*
> 
> *– 第 178 页，[傻瓜微积分](https://www.amazon.com/Calculus-Dummies-Math-Science/dp/1119293499/ref=as_li_ss_tl?dchild=1&keywords=calculus&qid=1606170839&sr=8-2&linkCode=sl1&tag=inspiredalgor-20&linkId=539ed0b89e326b6eb27b1a9a028e9cee&language=en_US)，2016 年。*

当我们说一辆车以每小时 100 公里的速度行驶时，我们刚刚陈述了它的变化率。我们常用的术语是*速度*或*速率*，虽然最好先区分这两者。

在日常生活中，我们通常将*速度*和*速率*互换使用，以描述移动物体的变化速率。然而，这在数学上是不正确的，因为速度始终是正值，而速度引入了方向的概念，因此可以展现正值和负值。因此，在接下来的解释中，我们将考虑速度作为更技术性的概念，其定义为：

速度 = ????*y* / ????*t*

这意味着速度在时间间隔????*t*内给出了汽车位置的变化????*y*。换句话说，速度是位置对时间的*一阶导数*。

汽车的速度可以保持不变，例如如果汽车持续以每小时 100 公里行驶，或者它也可以随时间变化。后者意味着速度函数本身随时间变化，或者更简单地说，汽车可以被认为是在*加速*。加速度定义为速度的第一导数 *v* 和位置 *y* 对时间的第二导数：

加速度 = ????*v* / ????*t =* ????*²**y* / ????*t**²*

我们可以绘制位置、速度和加速度曲线以更好地可视化它们。假设汽车的位置随时间的函数是*y*(*t*) = *t*³ – 8*t*² + 40*t*：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/applications_1.png)

汽车位置随时间变化的折线图

图表显示汽车在旅程开始时位置变化缓慢，直到大约 t = 2.7s 时略微减速，此后其变化速率加快并继续增加，直到旅程结束。这由汽车速度的图表描绘：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/applications_2.jpg)

汽车速度随时间变化的折线图

注意到汽车在整个旅程中保持正速度，这是因为它从未改变方向。因此，如果我们设想自己坐在这辆移动的汽车里，车速表会显示我们刚刚在速度图上绘制的值（由于速度始终为正，否则我们需要找出速度的绝对值来计算速度）。如果我们对*y*(*t*)应用幂法则以找到其导数，我们会发现速度由以下函数定义：

*v*(*t*) = *y*’(*t*) = 3*t*² – 16*t* + 40

我们还可以绘制加速度图：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/applications_3.png)

汽车加速度随时间变化的折线图

我们发现在时间间隔 *t* = [0, 2.7) 秒内，图表现出负加速度特征。这是因为加速度是速度的导数，在这个时间段内，汽车的速度在减小。如果我们必须再次对 *v*(*t*) 应用幂规则来找到其导数，我们会发现加速度由以下函数定义：

*a*(*t*) = *v*’(*t*) = 6*t* – 16

把所有函数放在一起，我们得到以下结果：

*y*(*t*) = *t*³ – 8*t*² + 40*t*

*v*(*t*) = *y*’(*t*) = 3*t*² – 16*t* + 40

*a*(*t*) = *v*’(*t*) = 6*t* – 16

如果我们代入 *t* = 10s，我们可以使用这三个函数来找出，在旅程结束时，汽车行驶了 600 米，其速度为 180 m/s，并且加速度为 44 m/s²。我们可以验证所有这些数值与我们刚刚绘制的图表相符。

我们在找出汽车速度和加速度的背景下讨论了这个特定示例。但是，有许多现实生活现象随时间（或其他变量）变化，可以通过应用导数的概念来研究，就像我们刚刚为这个特定示例所做的那样。例如：

+   人口（无论是人类集合还是细菌群落）随时间的增长率，可用于预测近期人口规模的变化。

+   温度随位置变化的变化，可用于天气预报。

+   随着时间的推移股市的波动，可以用来预测未来的股市行为。

导数还提供了解决优化问题的重要信息，接下来我们将看到。

## **导数在优化算法中的应用**

我们已经[看到](https://machinelearningmastery.com/calculus-in-machine-learning-why-it-works/)，优化算法（如梯度下降）通过应用导数来寻找误差（或成本）函数的全局最小值。

让我们更详细地看一看导数对误差函数的影响，通过进行与汽车示例相同的练习。

为此，让我们考虑以下用于[函数优化的一维测试函数](https://machinelearningmastery.com/1d-test-functions-for-function-optimization/)：

*f*(*x*) = –*x* sin(*x*)

我们可以应用乘积法则来求 *f*(*x*) 的一阶导数，记为 *f*’(*x*)，然后再次应用乘积法则来求 *f*’(*x*) 的二阶导数，记为 *f*’’(*x*)：

*f*’(*x*) = -sin(*x*) – *x* cos(*x*)

*f*’’(*x*) = *x* sin(*x*) – 2 cos(*x*)

我们可以对不同的 *x* 值绘制这三个函数的图像：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/applications_4.png)

函数 *f*(*x*)、它的一阶导数 *f*‘(*x*) 和二阶导数 *f*”(*x*) 的线图

与我们之前在汽车示例中观察到的类似，一阶导数的图示表示了*f*(*x*)的变化情况及其变化量。例如，正的导数表示*f*(*x*)是一个递增函数，而负的导数则表示*f*(*x*)现在在递减。因此，如果优化算法在寻找函数最小值时，根据其学习率ε对输入进行小幅度的变化：

*x_new = x* – ε *f*’(*x*)

然后，算法可以通过移动到导数的相反方向（即改变符号）来减少*f*(*x*)。

我们可能还对寻找函数的二阶导数感兴趣。

> *我们可以将二阶导数视为测量曲率。*
> 
> *– 第 86 页，[深度学习](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618/ref=sr_1_1?dchild=1&keywords=deep+learning&qid=1622968138&sr=8-1)，2017 年。*

例如，如果算法到达一个临界点，此时一阶导数为零，仅凭*f*’(*x*)无法区分该点是局部最大值、局部最小值、鞍点还是平坦区域。然而，当二阶导数介入时，算法可以判断，如果二阶导数大于零，则该临界点是局部最小值。如果是局部最大值，二阶导数则小于零。因此，二阶导数可以告知优化算法应向哪个方向移动。不幸的是，这个测试对于鞍点和平坦区域仍然不确定，因为这两种情况下的二阶导数均为零。

基于梯度下降的优化算法不使用二阶导数，因此被称为*一阶优化算法*。利用二阶导数的优化算法，如牛顿法，通常被称为*二阶优化算法*。

**进一步阅读**

本节提供了更多的资源，如果你想深入了解该主题。

**书籍**

+   [傻瓜微积分](https://www.amazon.com/Calculus-Dummies-Math-Science/dp/1119293499/ref=as_li_ss_tl?dchild=1&keywords=calculus&qid=1606170839&sr=8-2&linkCode=sl1&tag=inspiredalgor-20&linkId=539ed0b89e326b6eb27b1a9a028e9cee&language=en_US)，2016 年。

+   [无限的力量](https://www.amazon.com/Infinite-Powers-Calculus-Reveals-Universe/dp/0358299284/ref=as_li_ss_tl?dchild=1&keywords=joy+of+x&qid=1606170381&sr=8-2&linkCode=sl1&tag=inspiredalgor-20&linkId=17ed7cfdd9b7dd013730d0699a8652a1&language=en_US)，2020 年。

+   [深度学习](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618/ref=sr_1_1?dchild=1&keywords=deep+learning&qid=1622968138&sr=8-1)，2017 年。

+   [优化算法](https://www.amazon.com/Algorithms-Optimization-Press-Mykel-Kochenderfer/dp/0262039427/ref=sr_1_1?dchild=1&keywords=algorithms+for+optimization&qid=1624019308&sr=8-1)，2019 年。

**总结**

在本教程中，你了解了导数的不同应用。

具体来说，你学到了：

+   导数的应用可以解决我们周围实际问题。

+   导数的使用在机器学习中至关重要，特别是在函数优化方面。

你有任何问题吗？

在下方评论区提出你的问题，我会尽力回答。
