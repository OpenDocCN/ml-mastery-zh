# 微分与积分学 – 针对任何事物进行微分

> 原文：[`machinelearningmastery.com/differential-and-integral-calculus-differentiate-with-respect-to-anything/`](https://machinelearningmastery.com/differential-and-integral-calculus-differentiate-with-respect-to-anything/)

积分学是牛顿和莱布尼茨的伟大发现之一。他们的工作独立地证明了微积分基本定理的重要性，并认可了它将积分与导数联系起来。通过积分的发现，之后可以研究面积和体积。

积分学是我们将要探索的微积分旅程的第二部分。

在本教程中，你将发现微分与积分学之间的关系。

完成本教程后，你将了解：

+   微分和积分学的概念通过微积分基本定理相互关联。

+   通过应用微积分基本定理，我们可以计算积分来找到曲线下的面积。

+   在机器学习中，积分学的应用可以为我们提供一个评估分类器性能的指标。

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/integral_cover-scaled.jpg)

微分与积分学 – 针对任何事物进行微分

照片由[Maxime Lebrun](https://unsplash.com/@flub)提供，版权所有。

## **教程概述**

本教程分为三部分，它们是：

+   微分与积分学 – 什么是联系？

+   微积分基本定理

    +   扫描面积类比

    +   微积分基本定理 – 第一部分

    +   微积分基本定理 – 第二部分

+   积分示例

+   积分在机器学习中的应用

## **微分与积分学 – 什么是联系？**

在我们到目前为止的微积分旅程中，我们了解到微分学涉及[变化率的测量](https://machinelearningmastery.com/key-concepts-in-calculus-rate-of-change/)。我们还发现了[微分](https://machinelearningmastery.com/a-gentle-introduction-to-function-derivatives/)，并将其应用于[从基本原理出发的不同函数](https://machinelearningmastery.com/derivative-of-the-sine-and-cosine/)。我们甚至理解了如何[运用规则更快地得到导数](https://machinelearningmastery.com/the-power-product-and-quotient-rules/)。

但我们才刚刚走到一半。

> *从 21 世纪的视角看，微积分通常被视为变化的数学。它使用两个主要概念来量化变化：导数和积分。导数建模变化率…… 积分建模变化的积累……*
> 
> – 第 141 页，[《无限的力量》](https://www.amazon.com/Infinite-Powers-Calculus-Reveals-Universe/dp/0358299284/ref=as_li_ss_tl?dchild=1&keywords=joy+of+x&qid=1606170381&sr=8-2&linkCode=sl1&tag=inspiredalgor-20&linkId=17ed7cfdd9b7dd013730d0699a8652a1&language=en_US)，2020 年。

[回顾](https://machinelearningmastery.com/what-is-calculus/)计算包括两个阶段：切割和重建。

切割阶段将曲线形状分解为极其小且直的片段，这些片段可以单独研究，例如，通过应用导数来建模其变化率或*斜率*。

微积分之旅的这一部分被称为*微分*微积分，我们已经详细研究过。

重建阶段将极其小且直的片段汇总，试图重新组合以研究原始整体。通过这种方式，我们可以在将其切割成无限薄的片段后，确定规则和不规则形状的面积或体积。我们接下来将探讨微积分之旅的第二部分。它被称为*积分*微积分。

将这两个概念联系在一起的重要定理被称为*微积分基本定理*。

## **微积分基本定理**

为了更好地理解微积分基本定理，让我们回顾一下[汽车位置和速度示例](https://machinelearningmastery.com/?p=12598&preview=true)：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/applications_1.png)

汽车位置随时间变化的线图

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/applications_2.jpg)

汽车速度随时间变化的线图

在计算导数时，我们解决了*前向*问题，即通过位置图的斜率找到了速度。*t*。但如果我们想解决*后向*问题，即给定速度图*v*(*t*)，希望找到行驶的距离呢？解决这个问题的方法是计算曲线下的*面积*（阴影区域），直到时间*t*：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/integral_9.png)

阴影区域是曲线下的面积

我们没有具体的公式来直接定义阴影区域的面积。但我们可以应用微积分的数学，将曲线下的阴影区域切割成许多无限薄的矩形，对于这些矩形我们有一个公式：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/integral_10-1.png)

将阴影区域划分为许多宽度为Δt 的矩形

如果我们考虑第*i*个矩形，它被任意选择来覆盖时间间隔Δ*t*，我们可以将其面积定义为其长度乘以宽度：

area_of_rectangle = *v*(*t**[i]*) Δ*t**[i]*

我们可以拥有足够多的矩形来覆盖感兴趣的区间，在这种情况下，就是曲线下的阴影区域。为简便起见，我们将这个闭区间表示为[*a*, *b*]。找出这个阴影区域的面积（因此也是行进的距离），就等于求*数量为 n*的矩形的和：

total_area = *v*(*t*[0]) Δ*t*[0] *+ v*(*t*[1]) Δ*t*[1] *+ … + v*(*t*[n]) Δ*t*[n]

我们可以通过应用带有 sigma 符号的黎曼和来更紧凑地表达这个总和：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/integral_1.png)

如果我们通过有限数量的矩形来切分（或划分）曲线下方的区域，我们会发现黎曼和给出了这个区域的*近似值*，因为这些矩形无法完全贴合曲线下的区域。如果我们需要将矩形的左上角或右上角放置在曲线上，黎曼和将分别给出真实面积的低估或高估。如果每个矩形的中点必须接触曲线，那么矩形超出曲线的部分*大致*补偿了曲线与相邻矩形之间的间隙：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/integral_11.png)

用左和法来近似曲线下的面积

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/integral_12.png)

用右和法来近似曲线下的面积

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/integral_13.png)

用中点和法来近似曲线下的面积

找到曲线下*确切*面积的解决方法是将矩形的宽度减少到无限薄（回想一下微积分中的无穷大原则）。这样，矩形将覆盖整个区域，求和它们的面积就可以得到*定积分*。

> *定积分（“简单”定义）：曲线在 t = a 和 t = b 之间的确切面积由定积分给出，该积分被定义为黎曼和的极限……*
> 
> – 第 227 页, [《傻瓜微积分》](https://www.amazon.com/Calculus-Dummies-Math-Science/dp/1119293499/ref=as_li_ss_tl?dchild=1&keywords=calculus&qid=1606170839&sr=8-2&linkCode=sl1&tag=inspiredalgor-20&linkId=539ed0b89e326b6eb27b1a9a028e9cee&language=en_US), 2016.

因此，定积分可以通过黎曼和在矩形的数量*n*趋近于无穷大时定义。我们也将曲线下的区域表示为*A*(*t*)。然后：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/integral_2.png)

注意符号现在变成了积分符号∫，取代了西格玛Σ。这一变化的原因只是为了表示我们在对大量薄切矩形进行求和。左侧的表达式读作，从*a*到*b*的*v*(*t*)的积分，找到积分的过程称为*积分*。

### **扫动区域类比**

也许一个更简单的类比来帮助我们将积分与微分联系起来，是想象拿着一片薄切片，并在曲线下以极小的步幅向右拖动。当它向右移动时，薄切片将在曲线下扫过一个更大的区域，同时其高度会根据曲线的形状变化。我们想要回答的问题是，当薄片向右扫过时，区域积累的*速率*是多少？

让*dt*表示扫动切片所经过的每个无穷小步长，而*v*(*t*)表示任何时间*t*的高度。那么，这个薄片的无穷小区域*dA*(*t*)可以通过将其高度*v*(*t*)乘以其无穷小宽度*dt*来找到：

*dA*(*t*) = *v*(*t*) *dt*

将方程除以*dt*，我们得到*A*(*t*)的导数，并告诉我们区域积累的速率等于曲线的高度，*v*(*t*)，在时间*t*：

*dA*(*t*) / *dt* = *v*(*t*)

我们终于可以定义微积分基本定理。

### **微积分基本定理 – 第一部分**

我们发现一个区域，*A*(*t*)，在一个函数*v*(*t*)下扫过，可以定义为：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/integral_3.png)

我们还发现区域被扫过的速率等于原始函数*v*(*t*)：

*dA*(*t*) / *dt* = *v*(*t*)

这引出了微积分基本定理的第一部分，它告诉我们如果*v*(*t*)在区间[*a*，*b*]上是连续的，并且它也是*A*(*t*)的导数，那么*A*(*t*)是*v*(*t*)的*原函数*：

*A’*(*t*) = *v*(*t*)

或者更简单地说，积分是微分的逆操作。因此，如果我们首先对*v*(*t*)进行积分，然后对结果进行微分，我们将得到原始函数*v*(*t*)：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/integral_4.png)

### **微积分基本定理 – 第二部分**

定理的第二部分给出了计算积分的捷径，而不必通过计算黎曼和的极限来走更长的路。

它指出，如果函数*v*(*t*)在区间[*a*，*b*]上是连续的，那么：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/integral_5.png)

这里，*F*(*t*) 是*v*(*t*)的任意一个不定积分，积分被定义为在*a*和*b*处的不定积分值的差。

因此，定理的第二部分通过从某个起始点*C*和下限*a*处的曲线下面积减去相同起始点*C*和上限*b*处的面积来计算积分。这有效地计算了*a*到*b*之间感兴趣的区域的面积。

由于常数*C*定义了扫描开始时*x*轴上的点，最简单的不定积分考虑的是*C* = 0 的情况。然而，任何带有*C*任意值的不定积分都可以使用，这只是将起始点设置在*x*轴上不同位置而已。

## **积分示例**

考虑函数*v*(*t*) = *x*³。通过应用幂函数法则，我们可以轻松找到其导数，*v’*(*t*) = 3*x*²。3*x*²的不定积分再次是*x*³ — 我们执行反向操作以获得原始函数。

现在假设我们有一个不同的函数，*g*(*t*) = *x*³ + 2。它的导数也是 3*x*²，同样另一个函数*h*(*t*) = *x*³ – 5 也是如此。这些函数（以及其他类似的函数）都有*x*³作为它们的不定积分。因此，我们通过*不定*积分指定了所有 3*x*²的不定积分的家族：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/integral_6.png)

不定积分不定义计算曲线下面积的界限。常数*C*被包括在内是为了补偿对界限或扫描起始点缺乏信息的情况。

如果我们知道界限，那么我们可以简单地应用第二基本定理来计算*定*积分：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/integral_7.png)

在这种情况下，我们可以将*C*简单地设置为零，因为这不会改变结果。

## **在机器学习中应用积分**

我们考虑了汽车的速度曲线*v*(*t*)作为一个熟悉的例子，以理解积分和导数之间的关系。

> *但是，你可以使用这种矩形面积累加方案来累加任何微小的部分 —— 距离、体积或能量，例如。换句话说，曲线下的面积不一定代表实际的面积。*
> 
> – 第 214 页，[《Dummies 的微积分》](https://www.amazon.com/Calculus-Dummies-Math-Science/dp/1119293499/ref=as_li_ss_tl?dchild=1&keywords=calculus&qid=1606170839&sr=8-2&linkCode=sl1&tag=inspiredalgor-20&linkId=539ed0b89e326b6eb27b1a9a028e9cee&language=en_US)，2016 年。

成功应用机器学习技术的一个重要步骤是选择适当的性能度量。例如，在深度学习中，常见的做法是衡量*精度*和*召回率*。

> *精确度是模型报告的检测中正确的分数，而召回率是检测到的真实事件的分数。*
> 
> – 第 423 页，[《深度学习》](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618/ref=sr_1_1?dchild=1&keywords=deep+learning&qid=1622968138&sr=8-1)，2017 年。

这也是一种常见做法，然后在精确-召回（PR）曲线上绘制精确度和召回率，将召回率放在*x*轴上，精确度放在*y*轴上。希望分类器既有高召回率又有高精确度，这意味着分类器能够正确检测许多真实事件。这样的良好分类性能将通过 PR 曲线下的较大面积来描述。

你可能已经能够猜到接下来会发生什么。

PR 曲线下的面积确实可以通过应用积分微积分来计算，从而允许我们描述分类器的性能。

## **深入阅读**

如果你希望更深入地了解这个主题，本节提供了更多资源。

### **书籍**

+   [《单变量与多变量微积分》](https://www.whitman.edu/mathematics/multivariable/multivariable.pdf)，2020 年。

+   [《微积分入门》](https://www.amazon.com/Calculus-Dummies-Math-Science/dp/1119293499/ref=as_li_ss_tl?dchild=1&keywords=calculus&qid=1606170839&sr=8-2&linkCode=sl1&tag=inspiredalgor-20&linkId=539ed0b89e326b6eb27b1a9a028e9cee&language=en_US)，2016 年。

+   [《无限的力量》](https://www.amazon.com/Infinite-Powers-Calculus-Reveals-Universe/dp/0358299284/ref=as_li_ss_tl?dchild=1&keywords=joy+of+x&qid=1606170381&sr=8-2&linkCode=sl1&tag=inspiredalgor-20&linkId=17ed7cfdd9b7dd013730d0699a8652a1&language=en_US)，2020 年。

+   [《求学者的微积分指南》](https://www.amazon.com/Hitchhikers-Calculus-Classroom-Resource-Materials/dp/1470449625/ref=as_li_ss_tl?dchild=1&keywords=The+Hitchhiker%27s+Guide+to+Calculus&qid=1606170787&sr=8-1&linkCode=sl1&tag=inspiredalgor-20&linkId=f8875fa9736746bf29d78fc0c55553d8&language=en_US)，2019 年。

+   [《深度学习》](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618/ref=sr_1_1?dchild=1&keywords=deep+learning&qid=1622968138&sr=8-1)，2017 年。

## **总结**

在本教程中，你发现了微积分的微分和积分之间的关系。

具体来说，你学到了：

+   微积分的微分和积分概念由微积分基本定理联系在一起。

+   通过应用微积分基本定理，我们可以计算积分以找到曲线下的面积。

+   在机器学习中，应用积分微积分可以为我们提供一个度量标准，用来评估分类器的性能。

你有什么问题吗？

在评论区留言您的问题，我将尽力回答。
