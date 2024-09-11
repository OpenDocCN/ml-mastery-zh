# 幂法则、乘积法则和商法则

> 原文：[`machinelearningmastery.com/the-power-product-and-quotient-rules/`](https://machinelearningmastery.com/the-power-product-and-quotient-rules/)

优化作为许多机器学习算法中的核心过程之一，依赖于导数的使用，以决定如何更新模型的参数值，以最大化或最小化目标函数。

本教程将继续探索我们可以用来找到函数导数的不同技术。特别是，我们将探索幂法则、乘积法则和商法则，这些规则可以帮助我们比从头开始逐一求导更快地得到函数的导数。因此，对于那些特别具有挑战性的函数，掌握这些规则以便找到它们的导数将变得越来越重要。

在本教程中，您将学习如何使用幂法则、乘积法则和商法则来求解函数的导数。

完成本教程后，您将了解：

+   在寻找一个变量基数（提升到固定幂）的导数时应遵循的幂法则。

+   乘积法则如何帮助我们找到一个函数的导数，该函数被定义为另两个（或更多）函数的乘积。

+   商法则如何帮助我们找到一个函数的导数，该函数是两个可微分函数的比值。

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/rules_cover-scaled.jpg)

幂法则、乘积法则和商法则

图片由[Andreas M](https://unsplash.com/photos/vSN9eB6ghio)提供，部分权利保留。

## **教程概述**

本教程分为三个部分，它们是：

+   幂法则

+   乘积法则

+   商法则

## **幂法则**

如果我们有一个变量基数提升到固定幂，则寻找其导数的规则是将幂放在变量基数的前面，然后将幂减去 1。

例如，如果我们有函数 *f*(*x*) = *x**²*，我们想要找到它的导数，我们首先将 2 放在 *x* 前面，然后将幂减少 1：

*f*(*x*) = *x**²*

*f*’(*x*) = 2*x*

为了更好地理解这个规则的来源，让我们走一条较长的路，从导数的定义出发来找到 *f*(*x*) 的导数：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/rules_1.png)

在这里，我们用 *f*(*x*) = *x**²* 进行替代，然后简化表达式：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/rules_2.png)

当 *h* 接近 0 时，该极限接近 2*x*，这与我们之前使用幂法则获得的结果相符。

如果应用于 *f*(*x*) = *x*，幂法则给出的值是 1。这是因为，当我们将 1 带到 *x* 前面，然后将幂减去 1，我们剩下的是指数为 0 的值。由于 *x*⁰ = 1，所以 *f*’(*x*) = (1) (*x*⁰) = 1。

> *理解这个导数的最佳方法是意识到 f(x) = x 是一个符合 y = mx + b 形式的直线，因为 f(x) = x 与 f(x) = 1x + 0（或 y = 1x + 0）相同。这条直线的斜率（m）是 1，因此导数等于 1。或者你也可以记住 x 的导数是 1。如果你忘记了这两个观点，你总是可以使用幂法则。*
> 
> 第 131 页，[《傻瓜微积分》](https://www.amazon.com/Calculus-Dummies-Math-Science/dp/1119293499/ref=as_li_ss_tl?dchild=1&keywords=calculus&qid=1606170839&sr=8-2&linkCode=sl1&tag=inspiredalgor-20&linkId=539ed0b89e326b6eb27b1a9a028e9cee&language=en_US)，2016。

幂法则适用于任何幂，无论是正数、负数还是分数。我们还可以通过首先将其指数（或幂）表示为分数来应用它到根函数上：

*f*(*x*) = √*x* = *x*^(1/2)

*f’*(*x*) = (1 / 2) *x*^(-1/2)

## **乘积法则**

假设我们现在有一个函数 *f*(*x*)，我们希望找到其导数，该函数是另外两个函数的乘积，*u*(*x*) = 2*x*² 和 *v*(*x*) = *x*³：

*f*(*x*) = *u*(*x*) *v*(*x*) = (2*x*²) (*x*³)

为了调查如何找到 *f*(*x*) 的导数，让我们首先直接计算 *u*(*x*) 和 *v*(*x*) 的乘积的导数：

(*u*(*x*) *v*(*x*))’ = ((2*x*²) (*x*³))’ = (2*x*⁵)’ = 10*x*⁴

现在让我们调查一下如果我们分别计算函数的导数然后再将它们相乘会发生什么：

*u’*(*x*) *v’*(*x*) = (2*x*²)’ (*x*³)’ = (4*x*) (3*x*²) = 12*x*³

很明显，第二个结果与第一个结果不一致，这是因为我们没有应用 *乘积法则*。

乘积法则告诉我们，两个函数乘积的导数可以按如下方式计算：

*f’*(*x*) = *u’*(*x*) *v*(*x*) + *u*(*x*) *v’*(*x*)

如果我们通过应用极限的性质来推导乘积法则，从导数的定义开始，我们可以得到乘积法则：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/rules_3.png)

我们知道 *f*(*x*) = *u*(*x*) *v*(*x*)，因此，我们可以代入 *f*(*x*) 和 *f*(*x* + *h*)：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/rules_4.png)

在这个阶段，我们的目标是将分子因式分解成几个可以分别计算的极限。为此，分母的减法项 *u*(*x*) *v*(*x + h*) – *u*(*x*) *v*(*x + h*) 将被引入。它的引入并不改变我们刚刚得到的 *f*’(*x*) 的定义，但它将帮助我们因式分解分子：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/rules_5.png)

所得表达式看起来复杂，但是仔细观察后我们意识到可以因式分解出共同项：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/rules_6-e1624272833744.png)

通过应用极限法则，我们可以进一步简化表达式，将和与乘积分离开来：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/rules_7.png)

现在我们的问题解决方案变得更加清晰。我们可以看到简化表达式中的第一项和最后一项对应于 *u*(*x*) 和 *v*(*x*) 的导数的定义，我们可以分别用 *u*(*x*)’ 和 *v*(*x*)’ 表示。第二项在 *h* 接近 0 时逼近连续可微分函数 *v*(*x*)，而第三项是 *u*(*x*)。

因此，我们再次得出乘积法则：

*f’*(*x*) = *u’*(*x*) *v*(*x*) + *u*(*x*) *v’*(*x*)

有了这个新工具，让我们重新考虑当 *u*(*x*) = 2*x*² 和 *v*(*x*) = *x*³ 时如何找到 *f*’(*x*)：

*f’*(*x*) = *u’*(*x*) *v*(*x*) + *u*(*x*) *v’*(*x*)

*f’*(*x*) = (4*x*) (*x*³) + (2*x*²) (3*x*²) = 4*x*⁴ + 6*x*⁴ = 10*x*⁴

所得到的导数现在正确地匹配了我们之前得到的乘积 (*u*(*x*) *v*(*x*))’ 的导数。

这是一个相当简单的例子，我们本可以直接计算出来。然而，我们可能会遇到更复杂的涉及无法直接相乘的函数的问题，这时我们可以轻松地应用乘积法则。例如：

*f*(*x*) = *x*² sin *x*

*f’*(*x*) = (*x*²)’ (sin *x*) + (*x*²) (sin *x*)’ *=* 2*x* sin *x* + *x*² cos *x*

我们甚至可以将乘积法则扩展到超过两个函数的情况。例如，假设 *f*(*x*) 现在定义为三个函数 *u*(*x*)、*v*(*x*) 和 *w*(*x*) 的乘积：

*f*(*x*) = *u*(*x*) *v*(*x*) *w*(*x*)

我们可以如下应用乘积法则：

*f*’(*x*) = *u*’(*x*) *v*(*x*) *w*(*x*) + *u*(*x*) *v’*(*x*) *w*(*x*) + *u*(*x*) *v*(*x*) *w’*(*x*)

## **商规则**

同样，商规则告诉我们如何找到一个函数 *f*(*x*) 的导数，这个函数是两个可微分函数 *u*(*x*) 和 *v*(*x*) 的比值：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/rules_8.png)

我们可以像求乘法法则那样从基本原理推导商法则，即从导数的定义开始并应用极限的性质。或者我们可以走捷径，使用乘法法则本身来推导商法则。这一次我们采用这种方法：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/rules_9.png)

我们可以在 *u*(*x*) 上应用乘法法则以获得：

*u*’(*x*) = *f*’(*x*) *v*(*x*) + *f*(*x*) *v*’(*x*)

通过求解 *f*’(*x*) 得到：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/rules_10.png)

最后一步代入 *f*(*x*) 来得到商法则：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/rules_11.png)

我们已经看到如何找到[正弦和余弦函数的导数](https://machinelearningmastery.com/?p=12518&preview=true)。使用商法则，我们现在也可以找到正切函数的导数：

*f*(*x*) = tan *x* = sin *x* / cos *x*

应用商法则并简化结果表达式：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/rules_12.png)

从三角函数中的勾股恒等式我们知道 cos²*x* + sin²*x* = 1，因此：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/rules_13.png)

因此，使用商法则，我们轻松找到了正切的导数是平方的正割函数。

## **进一步阅读**

本节提供了更多关于该主题的资源，如果你想深入了解。

### **书籍**

+   [《傻瓜微积分》](https://www.amazon.com/Calculus-Dummies-Math-Science/dp/1119293499/ref=as_li_ss_tl?dchild=1&keywords=calculus&qid=1606170839&sr=8-2&linkCode=sl1&tag=inspiredalgor-20&linkId=539ed0b89e326b6eb27b1a9a028e9cee&language=en_US)，2016 年。

### **文章**

+   [幂法则，维基百科](https://en.wikipedia.org/wiki/Power_rule)。

+   [乘法法则，维基百科](https://en.wikipedia.org/wiki/Product_rule)。

+   [商法则，维基百科](https://en.wikipedia.org/wiki/Quotient_rule)。

## **总结**

在本教程中，你学会了如何应用幂法则、乘法法则和商法则来找出函数的导数。

具体来说，你学到了：

+   计算变量基数（提升到固定幂次）的导数时遵循的幂法则。

+   乘法法则如何使我们能够找到定义为两个（或更多）函数乘积的函数的导数。

+   商法则如何使我们能够找到一个是两个可微函数比率的函数的导数。

你有任何问题吗？

在下面的评论中提问，我会尽力回答。
