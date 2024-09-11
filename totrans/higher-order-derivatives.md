# 高阶导数

> 原文：[`machinelearningmastery.com/higher-order-derivatives/`](https://machinelearningmastery.com/higher-order-derivatives/)

高阶导数能够捕捉一阶导数无法捕捉到的信息。

一阶导数可以捕捉重要信息，例如变化率，但单独使用时无法区分局部最小值或最大值，在这些点的变化率为零。若干优化算法通过利用高阶导数来解决这一限制，例如在牛顿法中，使用二阶导数来达到优化函数的局部最小值。

在本教程中，你将学习如何计算高阶的单变量和多变量导数。

完成本教程后，你将了解：

+   如何计算单变量函数的高阶导数。

+   如何计算多变量函数的高阶导数。

+   二阶导数如何通过二阶优化算法在机器学习中得到利用。

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/higher_order_cover-scaled.jpg)

高阶导数

图片由[Jairph](https://unsplash.com/photos/aT2jMKShKIs)提供，保留部分权利。

## **教程概述**

本教程分为三个部分，它们是：

+   单变量函数的高阶导数

+   多变量函数的高阶导数

+   在机器学习中的应用

## **单变量函数的高阶导数**

除了[一阶导数](https://machinelearningmastery.com/a-gentle-introduction-to-function-derivatives/)，我们已经看到它能提供有关函数的重要信息，例如其瞬时[变化率](https://machinelearningmastery.com/key-concepts-in-calculus-rate-of-change/)，更高阶的导数也同样有用。例如，二阶导数可以测量一个移动物体的[加速度](https://machinelearningmastery.com/applications-of-derivatives/)，或者它可以帮助优化算法区分局部最大值和局部最小值。

计算单变量函数的高阶（第二阶、第三阶或更高阶）导数并不难。

> *函数的二阶导数只是其一阶导数的导数。三阶导数是二阶导数的导数，四阶导数是三阶导数的导数，以此类推。*
> 
> – 第 147 页，[傻瓜微积分](https://www.amazon.com/Calculus-Dummies-Math-Science/dp/1119293499/ref=as_li_ss_tl?dchild=1&keywords=calculus&qid=1606170839&sr=8-2&linkCode=sl1&tag=inspiredalgor-20&linkId=539ed0b89e326b6eb27b1a9a028e9cee&language=en_US)，2016 年。

因此，计算高阶导数只是涉及到对函数的重复微分。为了做到这一点，我们可以简单地应用我们对[幂规则](https://machinelearningmastery.com/the-power-product-and-quotient-rules/) 的知识。以函数 *f*(*x*) = x³ + 2x² – 4x + 1 为例：

一阶导数：*f*’(*x*) = 3*x*² + 4*x* – 4

二阶导数：*f*’’(*x*) = 6*x* + 4

三阶导数：*f*’’’(*x*) = 6

第四阶导数：*f*^((4))(*x*) = 0

第五阶导数：*f*^((5))(*x*) = 0 *等*

我们所做的是首先对 *f*(*x*) 应用幂规则以获得其一阶导数 *f*’(*x*)，然后对一阶导数应用幂规则以获得二阶导数，如此继续。导数最终会因为重复微分而趋于零。

[乘积和商的规则](https://machinelearningmastery.com/the-power-product-and-quotient-rules/) 的应用在求取高阶导数时仍然有效，但随着阶数的增加，其计算会变得越来越复杂。一般的莱布尼兹规则在这方面简化了任务，将乘积规则推广为：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/higher_order_1.png)

这里，术语 *n*! / *k*!(*n* – *k*)! 是来自二项式定理的二项式系数，而 *f*^(*^k*^) 和 *g*^(*^k*^) 表示函数 *f* 和 *g* 的 *k*^(th) 导数。

因此，按照一般的莱布尼兹规则，找到一阶和二阶导数（因此，分别替代 *n* = 1 和 *n* = 2），我们得到：

(*fg*)^((1)) = (*fg*)’ = *f*^((1)) *g* + *f* *g*^((1))

(*fg*)^((2)) = (*fg*)’’ = *f*^((2)) *g* + 2*f*^((1)) *g*^((1)) + *f* *g*^((2))

注意到乘积规则定义的一阶导数。莱布尼兹规则也可以用来寻找有理函数的高阶导数，因为商可以有效地表达为形式为 *f* *g*^(-1) 的乘积。

## **多变量函数的高阶导数**

高阶[偏导数](https://machinelearningmastery.com/a-gentle-introduction-to-partial-derivatives-and-gradient-vectors) 的定义对于[多变量函数](https://machinelearningmastery.com/?p=12606&preview=true) 类似于一变量情况：*n*^(th) 阶偏导数对于 *n* > 1，是计算 (*n* – 1)^(th) 阶偏导数的偏导数。例如，对具有两个变量的函数进行二阶偏导数运算，会得到四个二阶偏导数：两个 *自身* 偏导数 *f**[xx]* 和 *f**[yy]*，以及两个交叉偏导数 *f**[xy]* 和 *f**[yx]*。

> *为了进行“导数”，我们必须对 x 或 y 进行偏导数，并且有四种方式：先对 x，然后对 x，先对 x，然后对 y，先对 y，然后对 x，先对 y，然后对 y。*
> 
> – 第 371 页，[单变量和多变量微积分](https://www.whitman.edu/mathematics/multivariable/multivariable.pdf)，2020 年。

让我们考虑多元函数，*f*(*x*, *y*) = *x*² + 3*xy* + 4*y*²，我们希望找到其二阶偏导数。该过程始于找到其一阶偏导数，首先：![](https://machinelearningmastery.com/wp-content/uploads/2021/07/higher_order_2.png)

然后通过重复找到偏导数的过程，找到四个二阶偏导数。*自己的*偏导数是最简单找到的，因为我们只需再次针对*x*或*y*进行偏导数过程的重复：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/higher_order_3.png)

先前找到的*f*[x]（即相对于*x*的偏导数）的交叉偏导数通过取其结果相对于*y*的偏导数得到，给出*f*[xy]。类似地，相对于*x*的偏导数取*f*[y]的结果，给出*f*[yx]：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/higher_order_4.png)

交叉偏导数给出相同结果并非偶然。这由克莱罗定理定义，其表明只要交叉偏导数连续，则它们相等。

### 想要开始学习机器学习的微积分吗？

现在就参加我的免费 7 天电子邮件快速课程（附带示例代码）。

点击注册并获得该课程的免费 PDF 电子书版本。

## **在机器学习中的应用**

在机器学习中，主要使用二阶导数。我们此前提到过，二阶导数可以提供第一阶导数无法捕捉的信息。具体来说，它可以告诉我们临界点是局部最小值还是最大值（基于二阶导数大于或小于零的情况），而在这两种情况下第一阶导数都将为零。

有几种利用此信息的*二阶*优化算法，其中之一是牛顿法。

> *另一方面，二阶信息使我们能够对目标函数进行二次近似，并近似计算出达到局部最小值的正确步长……*
> 
> – 第 87 页，[优化算法](https://www.amazon.com/Algorithms-Optimization-Press-Mykel-Kochenderfer/dp/0262039427/ref=sr_1_1?dchild=1&keywords=algorithms+for+optimization&qid=1624019308&sr=8-1)，2019 年。

在单变量情况下，牛顿法使用二阶泰勒级数展开式在目标函数的某一点进行二次近似。牛顿法的更新规则是通过将导数设为零并解出根得到的，这涉及到对二阶导数进行除法运算。如果牛顿法扩展到多变量优化，导数将被梯度取代，而二阶导数的倒数将被 Hessian 矩阵的逆矩阵取代。

我们将在不同的教程中涵盖 Hessian 矩阵和泰勒级数近似方法，这些方法利用了高阶导数。

## **进一步阅读**

如果你想深入了解该主题，本节提供了更多资源。

### **书籍**

+   [单变量与多变量微积分](https://www.whitman.edu/mathematics/multivariable/multivariable.pdf)，2020 年。

+   [傻瓜微积分](https://www.amazon.com/Calculus-Dummies-Math-Science/dp/1119293499/ref=as_li_ss_tl?dchild=1&keywords=calculus&qid=1606170839&sr=8-2&linkCode=sl1&tag=inspiredalgor-20&linkId=539ed0b89e326b6eb27b1a9a028e9cee&language=en_US)，2016 年。

+   [深度学习](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618/ref=sr_1_1?dchild=1&keywords=deep+learning&qid=1622968138&sr=8-1)，2017 年。

+   [优化算法](https://www.amazon.com/Algorithms-Optimization-Press-Mykel-Kochenderfer/dp/0262039427/ref=sr_1_1?dchild=1&keywords=algorithms+for+optimization&qid=1624019308&sr=8-1)，2019 年。

## **总结**

在本教程中，你学会了如何计算单变量和多变量函数的高阶导数。

具体来说，你学到了：

+   如何计算单变量函数的高阶导数。

+   如何计算多变量函数的高阶导数。

+   如何通过二阶优化算法在机器学习中利用二阶导数。

你有什么问题吗？

在下面的评论中提出你的问题，我会尽力回答。
