# 微积分中的链式法则——更多函数

> 原文：[`machinelearningmastery.com/the-chain-rule-of-calculus-even-more-functions/`](https://machinelearningmastery.com/the-chain-rule-of-calculus-even-more-functions/)

链式法则是一个重要的导数规则，使我们能够处理复合函数。它在理解反向传播算法的工作原理中至关重要，反向传播算法广泛应用链式法则来计算损失函数相对于神经网络每个权重的误差梯度。我们将在之前对链式法则的介绍基础上，处理更具挑战性的函数。

在本教程中，你将发现如何将微积分的链式法则应用于挑战性函数。

完成本教程后，你将了解：

+   将链式法则应用于单变量函数的过程可以扩展到多变量函数。

+   链式法则的应用遵循类似的过程，无论函数多么复杂：首先求外部函数的导数，然后向内移动。在此过程中，可能需要应用其他导数规则。

+   将链式法则应用于多变量函数需要使用偏导数。

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/more_chain_rule_cover-scaled.jpg)

微积分中的链式法则——更多函数

图片由 [Nan Ingraham](https://unsplash.com/photos/mNuLRRjLwjA) 拍摄，保留了一些权利。

## **教程概述**

本教程分为两个部分；它们是：

+   单变量函数的链式法则

+   多变量函数的链式法则

## **先决条件**

对于本教程，我们假设你已经知道以下内容：

+   [多变量函数](https://machinelearningmastery.com/a-gentle-introduction-to-multivariate-calculus/)

+   [幂法则和积法则](https://machinelearningmastery.com/the-power-product-and-quotient-rules/)

+   [偏导数](https://machinelearningmastery.com/a-gentle-introduction-to-partial-derivatives-and-gradient-vectors)

+   [链式法则](https://machinelearningmastery.com/?p=12720&preview=true)

你可以通过点击上面给出的链接来复习这些概念。

## **单变量函数的链式法则**

我们已经发现了单变量和多变量函数的链式法则，但到目前为止我们只看到了一些简单的例子。这里我们将查看一些更具挑战性的例子。我们将首先从单变量函数开始，然后将所学应用于多变量函数。

**示例 1**：让我们通过考虑以下复合函数来提高难度：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/more_chain_rule_1.png)

我们可以将复合函数分解为内部函数，*f*(*x*) = *x*² – 10，以及外部函数，*g*(*x*) = √*x* = (*x*)^(1/2)。内部函数的输出由中间变量*u*表示，并且其值将输入到外部函数中。

第一步是找到复合函数外部部分的导数，同时忽略内部的内容。为此，我们可以应用幂规则：

*dh / du* = (1/2) (*x*² – 10)^(-1/2)

下一步是找到复合函数内部部分的导数，这次忽略外部的内容。我们可以在这里也应用幂规则：

*du / dx* = 2*x*

将两部分结合并简化，我们得到复合函数的导数：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/more_chain_rule_2.png)

**例子 2**：让我们重复这个过程，这次使用另一个复合函数：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/more_chain_rule_3.png)

我们再次使用内部函数的输出*u*作为我们的中间变量。

在这种情况下，外部函数是 cos *x*。找到它的导数，再次忽略内部的部分，给我们：

*dh* / *du* = (cos(*x*³ – 1))’ = -sin(*x*³ – 1)

内部函数是，*x*³ – 1。因此，它的导数变为：

*du* / *dx* = (*x*³ – 1)’ = 3*x*²

将两部分结合，我们得到复合函数的导数：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/more_chain_rule_4.png)

**例子 3**：现在让我们进一步提高难度，考虑一个更具挑战性的复合函数：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/more_chain_rule_5.png)

如果我们仔细观察，我们会发现不仅我们有嵌套函数需要多次应用链式法则，而且我们还有一个乘积需要应用乘积法则。

我们发现最外层的函数是余弦。通过链式法则找到它的导数时，我们将使用中间变量*u*：

*dh* / *du* = (cos(*x* √(*x*² – 10) ))’ = -sin(*x* √(*x*² – 10) )

在余弦内部，我们有乘积，*x* √(x² – 10)，我们将应用乘积法则来找到其导数（注意，我们总是从外部向内部移动，以便发现需要处理的操作）：

*du* / *dx* = (*x* √(x² – 10) )’ = √(x² – 10) + *x* ( √(x² – 10) )’

结果中的一个组成部分是，( √(x² – 10) )’，我们将再次应用链式法则。事实上，我们在上面已经这样做过了，因此我们可以简单地重新使用结果：

(√(x² – 10) )’ = *x* (*x*² – 10)^(-1/2)

将所有部分结合起来，我们得到复合函数的导数：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/more_chain_rule_6.png)

这可以进一步简化为：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/more_chain_rule_7.png)

### 想要开始机器学习的微积分？

立即免费参加我的 7 天电子邮件速成课程（附带示例代码）。

点击注册并获取课程的免费 PDF 电子书版本。

## **多变量函数的链式法则**

**示例 4**：假设我们现在面对一个关于两个独立变量 *s* 和 *t* 的多变量函数，其中每个变量依赖于另外两个独立变量 *x* 和 *y*：

*h* = *g*(*s*, *t*) = *s*² + *t*³

其中函数为 *s* = *xy*，*t* = 2*x* – *y*。

在这里实施链式法则需要计算偏导数，因为我们处理多个独立变量。此外，*s* 和 *t* 也将作为我们的中间变量。我们将使用以下关于每个输入定义的公式：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/more_chain_rule_8.png)

从这些公式中，我们可以看到我们需要找到六个不同的偏导数：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/more_chain_rule_9.png)

现在我们可以继续用这些术语替换∂*h* / ∂*x*和∂*h* / ∂*y*的公式：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/more_chain_rule_10.png)

然后，用 *s* 和 *t* 替代以找到导数：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/more_chain_rule_11.png)

**示例 5**：让我们再次重复，这次是一个关于三个独立变量 $r$, $s$ 和 $t$ 的多变量函数，其中每个变量依赖于另外两个独立变量 $x$ 和 $y$：

$$h=g(r,s,t)=r²-rs+t³$$

其中函数为 $r = x \cos y$，$s=xe^y$，$t=x+y$。

这一次，$r$, $s$ 和 $t$ 将作为我们的中间变量。我们将使用以下关于每个输入定义的公式：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/more_chain_rule_12.png)

从这些公式中，我们现在需要找到九个不同的偏导数：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/more_chain_rule_13.png)

然后，我们继续用这些术语替换∂*h* / ∂*x*和∂*h* / ∂*y*的公式：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/more_chain_rule_14.png)

随后对 $r$、$s$ 和 $t$ 进行代入，以找到导数：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/more_chain_rule_15.png)

可以进一步简化（提示：对 $\partial h/\partial y$ 应用三角恒等式 $2\sin y\cos y=\sin 2y$）：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/more_chain_rule_16.png)

无论表达式多么复杂，遵循的程序仍然相似：

> *你最后的计算告诉你要做的第一件事。*
> 
> – 第 143 页，[傻瓜微积分](https://www.amazon.com/Calculus-Dummies-Math-Science/dp/1119293499/ref=as_li_ss_tl?dchild=1&keywords=calculus&qid=1606170839&sr=8-2&linkCode=sl1&tag=inspiredalgor-20&linkId=539ed0b89e326b6eb27b1a9a028e9cee&language=en_US)，2016 年。

因此，从处理外层函数开始，然后逐步向内处理下一个函数。你可能需要应用其他规则，就像我们在示例 3 中看到的那样。如果你在处理多变量函数时，不要忘记计算偏导数。

## **进一步阅读**

本节提供了更多相关资源，如果你想深入了解这个话题。

### **书籍**

+   [傻瓜微积分](https://www.amazon.com/Calculus-Dummies-Math-Science/dp/1119293499/ref=as_li_ss_tl?dchild=1&keywords=calculus&qid=1606170839&sr=8-2&linkCode=sl1&tag=inspiredalgor-20&linkId=539ed0b89e326b6eb27b1a9a028e9cee&language=en_US)，2016 年。

+   [单变量与多变量微积分](https://www.whitman.edu/mathematics/multivariable/multivariable.pdf)，2020 年。

+   [机器学习数学](https://www.amazon.com/Mathematics-Machine-Learning-Peter-Deisenroth/dp/110845514X/ref=as_li_ss_tl?dchild=1&keywords=calculus+machine+learning&qid=1606171788&s=books&sr=1-3&linkCode=sl1&tag=inspiredalgor-20&linkId=209ba69202a6cc0a9f2b07439b4376ca&language=en_US)，2020 年。

## **总结**

在本教程中，你发现了如何将微积分链式法则应用于复杂函数。

具体来说，你学习了：

+   将链式法则应用于一变量函数的过程可以扩展到多变量函数。

+   无论函数多么复杂，链式法则的应用过程都是类似的：首先取外层函数的导数，然后向内处理。在过程中，可能需要应用其他导数规则。

+   将链式法则应用于多变量函数需要使用偏导数。

你有任何问题吗？

在下面的评论中提问，我会尽力回答。
