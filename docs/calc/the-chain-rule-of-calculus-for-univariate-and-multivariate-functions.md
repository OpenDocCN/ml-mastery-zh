# 一元和多元函数的链式法则

> 原文：[`machinelearningmastery.com/the-chain-rule-of-calculus-for-univariate-and-multivariate-functions/`](https://machinelearningmastery.com/the-chain-rule-of-calculus-for-univariate-and-multivariate-functions/)

链式法则使我们能够找到复合函数的导数。

反向传播算法广泛计算它，以训练前馈神经网络。通过以高效的方式应用链式法则，同时遵循特定的操作顺序，反向传播算法计算损失函数相对于网络每个权重的误差梯度。

在本教程中，你将发现一元和多元函数的链式法则。

完成本教程后，你将了解：

+   复合函数是两个（或更多）函数的组合。

+   链式法则使我们能够找到复合函数的导数。

+   链式法则可以推广到多元函数，并通过树形图表示。

+   链式法则在反向传播算法中被广泛应用，以计算损失函数相对于每个权重的误差梯度。

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/chain_rule_cover-scaled.jpg)

一元和多元函数的链式法则

图片来源：[Pascal Debrunner](https://unsplash.com/photos/WuwKphhRQSM)，部分权利保留。

## **教程概述**

本教程分为四部分，分别是：

+   复合函数

+   链式法则

+   广义链式法则

+   在机器学习中的应用

## **前提条件**

在本教程中，我们假设你已经知道以下内容：

+   [多元函数](https://machinelearningmastery.com/a-gentle-introduction-to-multivariate-calculus/)

+   [幂法则](https://machinelearningmastery.com/the-power-product-and-quotient-rules/)

+   [函数的梯度](https://machinelearningmastery.com/a-gentle-introduction-to-partial-derivatives-and-gradient-vectors)

你可以通过点击上面给出的链接来复习这些概念。

## **复合函数**

到目前为止，我们已经遇到了单变量和多变量函数（即*一元*和*多元*函数）。现在，我们将这两者扩展到它们的*复合*形式。我们最终将看到如何应用链式法则来求导，但稍后会详细讲解。

> *复合函数是两个函数的组合。*
> 
> – 第 49 页，[傻瓜微积分](https://www.amazon.com/Calculus-Dummies-Math-Science/dp/1119293499/ref=as_li_ss_tl?dchild=1&keywords=calculus&qid=1606170839&sr=8-2&linkCode=sl1&tag=inspiredalgor-20&linkId=539ed0b89e326b6eb27b1a9a028e9cee&language=en_US)，2016 年。

考虑两个单一自变量的函数，*f*(*x*) = 2*x* – 1 和 *g*(*x*) = *x*³。它们的复合函数可以定义如下：

*h* = *g*(*f*(*x*))

在此操作中，*g* 是 *f* 的一个函数。这意味着 *g* 应用于将函数 *f* 应用到 *x* 上的结果，生成 *h*。

让我们用上述指定的函数来考虑一个具体的例子，以便更好地理解。

假设 *f*(*x*) 和 *g*(*x*) 是两个级联系统，接收输入 *x* = 5：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/chain_rule_1.png)

级联中的两个系统表示一个复合函数

由于 *f*(*x*) 是级联中的第一个系统（因为它是复合函数中的内层函数），所以它的输出首先被计算：

*f*(5) = (2 × 5) – 1 = 9

然后将此结果作为输入传递给 *g*(*x*)，即级联中的第二个系统（因为它是复合函数中的外层函数），以产生复合函数的最终结果：

*g*(9) = 9³ = 729

如果我们执行以下计算，可以一次性得到最终结果：

*h* = *g*(*f*(*x*)) = (2*x* – 1)³ = 729

函数的复合也可以被认为是一个*链式*过程，使用一个更熟悉的术语，即一个函数的输出传递给链中的下一个函数。

> *复合函数中，顺序是重要的。*
> 
> – 第 49 页， [《傻瓜微积分》](https://www.amazon.com/Calculus-Dummies-Math-Science/dp/1119293499/ref=as_li_ss_tl?dchild=1&keywords=calculus&qid=1606170839&sr=8-2&linkCode=sl1&tag=inspiredalgor-20&linkId=539ed0b89e326b6eb27b1a9a028e9cee&language=en_US)，2016 年。

请记住，函数的复合是一个*非交换*的过程，这意味着在级联（或链）中交换 *f*(*x*) 和 *g*(*x*) 的顺序不会产生相同的结果。因此：

*g*(*f*(*x*)) ≠ *f*(*g*(*x*))

函数的复合也可以扩展到多变量情况：

*h* = *g*(*r, s, t*) = *g*(*r*(*x, y*), *s*(*x, y*), *t*(*x, y*)) = *g*(***f***(*x, y*))

在这里，***f***(*x, y*) 是一个两自变量（或输入）的向量值函数（在这个特定例子中），由三个组件（*r*(*x, y*), *s*(*x, y*) 和 *t*(*x, y*)）组成，也被称为 ***f*** 的*组件*函数。

这意味着 ***f***(*x*, *y*) 将两个输入映射到三个输出，然后将这三个输出传递给链中的连续系统 *g*(*r*, *s*, *t*)，以生成 *h*。

### 想开始学习用于机器学习的微积分吗？

立即报名参加我的免费 7 天电子邮件速成课程（附样例代码）。

点击注册并获取课程的免费 PDF Ebook 版本。

## **链式法则**

链式法则允许我们找到复合函数的导数。

让我们首先定义链式法则如何区分复合函数，然后将其拆分成单独的组件以便更好地理解。如果我们重新考虑复合函数 *h* = *g*(*f*(*x*))，那么其导数由链式法则给出为：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/chain_rule_3.png)

在这里，*u* 是内函数 *f* 的输出（因此，*u* = *f*(*x*)），然后作为输入提供给下一个函数 *g* 以生成 *h*（因此，*h* = *g*(*u*)）。因此，注意链式法则如何通过一个 *中间变量*，*u*，将最终输出 *h* 与输入 *x* 相关联。

回顾一下复合函数的定义如下：

*h*(*x*) = *g*(*f*(*x*)) = (2*x* – 1)³

链式法则的第一个部分，*dh* / *du*，告诉我们首先找到复合函数外部部分的导数，同时忽略内部部分。为此，我们将应用幂法则：

((2*x* – 1)³)’ = 3(2*x* – 1)²

结果然后乘以链式法则的第二个部分 *du* / *dx*，这是复合函数内部部分的导数，这次忽略外部部分：

( (2*x* – 1)’ )³ = 2

由链式法则定义的复合函数的导数如下：

*h*’ = 3(2*x* – 1)² × 2 = 6(2*x* – 1)²

我们在这里考虑了一个简单的例子，但将链式法则应用于更复杂函数的概念保持不变。我们将在另一个教程中考虑更具挑战性的函数。

## **广义链式法则**

我们可以将链式法则推广到单变量情况之外。

考虑 ***x*** ∈ ℝ^m 和 ***u*** ∈ ℝ^n 的情况，这意味着内函数 *f* 将 *m* 个输入映射到 *n* 个输出，而外函数 *g* 接收 *n* 个输入以产生一个输出 *h*。对于 *i* = 1, …, *m*，广义链式法则表述为：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/chain_rule_4.png)

或以其更简洁的形式，对于 *j* = 1, …, *n*：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/chain_rule_5.png)

回顾一下，当我们寻找多变量函数的梯度时，我们使用偏导数。

我们还可以通过树状图来可视化链式法则的工作过程。

假设我们有一个由两个独立变量 *x*[1] 和 *x*[2] 组成的复合函数，定义如下：

*h* = *g*(***f***(*x*[1], *x*[2])) = *g*(*u*1, *u*2)

在这里，*u*[1] 和 *u*[2] 充当中间变量。它的树状图表示如下：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/chain_rule_2.png)

通过树状图表示链式法则

为了推导每个输入 *x*[1] 和 *x*[2] 的公式，我们可以从树状图的左侧开始，沿着其分支向右移动。以这种方式，我们发现形成了以下两个公式（为了简单起见，分支的和已被着色）：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/chain_rule_6.png)

注意链式法则如何通过中间变量 *u**[j]* 将网络输出 *h* 与每个输入 *x**[i]* 关联起来。这是反向传播算法广泛应用的概念，用于优化神经网络的权重。

## **在机器学习中的应用**

观察树状图与典型神经网络表示的相似性（尽管我们通常通过将输入放在左侧，输出放在右侧来表示神经网络）。我们可以通过反向传播算法将链式法则应用于神经网络，与上面应用于树状图的方式非常相似。

> *链式法则极端使用的一个领域是深度学习，其中函数值***y***被计算为多个层级的函数组合。*
> 
> – 第 159 页，[《机器学习数学》](https://www.amazon.com/Mathematics-Machine-Learning-Peter-Deisenroth/dp/110845514X/ref=as_li_ss_tl?dchild=1&keywords=calculus+machine+learning&qid=1606171788&s=books&sr=1-3&linkCode=sl1&tag=inspiredalgor-20&linkId=209ba69202a6cc0a9f2b07439b4376ca&language=en_US)，2020 年。

神经网络确实可以表示为一个巨大的嵌套复合函数。例如：

***y*** = *f*[K] ( *f*[K – 1] ( … ( *f*1) … ))

在这里，***x*** 是神经网络的输入（例如，图像），而 ***y*** 是输出（例如，类别标签）。每个函数 *f*[i]，对于 *i* = 1，…，*K*，都有自己的权重。

将链式法则应用于这样的复合函数使我们能够反向遍历构成神经网络的所有隐藏层，并有效计算损失函数相对于每个权重 *w*[i] 的误差梯度，直到到达输入。

## **进一步阅读**

本节提供了更多资源，供您深入研究。

### **书籍**

+   [《傻瓜微积分》](https://www.amazon.com/Calculus-Dummies-Math-Science/dp/1119293499/ref=as_li_ss_tl?dchild=1&keywords=calculus&qid=1606170839&sr=8-2&linkCode=sl1&tag=inspiredalgor-20&linkId=539ed0b89e326b6eb27b1a9a028e9cee&language=en_US)，2016 年。

+   [《单变量与多变量微积分》](https://www.whitman.edu/mathematics/multivariable/multivariable.pdf)，2020 年。

+   [《深度学习》](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618/ref=sr_1_1?dchild=1&keywords=deep+learning&qid=1622968138&sr=8-1)，2017 年。

+   [机器学习的数学](https://www.amazon.com/Mathematics-Machine-Learning-Peter-Deisenroth/dp/110845514X/ref=as_li_ss_tl?dchild=1&keywords=calculus+machine+learning&qid=1606171788&s=books&sr=1-3&linkCode=sl1&tag=inspiredalgor-20&linkId=209ba69202a6cc0a9f2b07439b4376ca&language=en_US)，2020 年。

## **总结**

在本教程中，你发现了用于单变量和多变量函数的链式法则。

具体来说，你学到了：

+   复合函数是两个（或更多）函数的组合。

+   链式法则允许我们找到复合函数的导数。

+   链式法则可以推广到多变量函数，并通过树形图表示。

+   链式法则被反向传播算法广泛应用，用于计算损失函数关于每个权重的误差梯度。

你有任何问题吗？

在下方评论中提问，我会尽力回答。
