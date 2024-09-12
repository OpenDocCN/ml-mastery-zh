# 泰勒级数的温和介绍

> 原文：[`machinelearningmastery.com/a-gentle-introduction-to-taylor-series/`](https://machinelearningmastery.com/a-gentle-introduction-to-taylor-series/)

**泰勒级数的温和介绍**

泰勒级数展开是一个了不起的概念，不仅在数学领域，而且在优化理论、函数逼近和机器学习中都非常重要。当需要在不同点估计函数值时，它在数值计算中得到了广泛应用。

在本教程中，你将发现泰勒级数以及如何使用其泰勒级数展开在不同点附近逼近函数值。

完成本教程后，你将知道：

+   函数的泰勒级数展开

+   如何使用泰勒级数展开逼近函数

让我们开始吧。

![泰勒级数的温和介绍。图片由穆罕默德·库拜布·萨尔法兹提供，部分权利保留。](https://machinelearningmastery.com/wp-content/uploads/2021/07/Muhammad-Khubaib-Sarfraz.jpg)

泰勒级数的温和介绍。图片由穆罕默德·库拜布·萨尔法兹提供，部分权利保留。

## **教程概述**

本教程分为 3 部分；它们是：

1.  幂级数和泰勒级数

1.  泰勒多项式

1.  使用泰勒多项式进行函数逼近

## **什么是幂级数？**

以下是关于中心 x=a 和常数系数 c_0, c_1 等的幂级数。

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/tayloreq1.png)

## **什么是泰勒级数？**

令人惊讶的是，具有无限次可微性的函数可以生成一种称为泰勒级数的幂级数。假设我们有一个函数 f(x)，并且 f(x)在给定区间上具有所有阶的导数，那么在 x=a 处由 f(x)生成的泰勒级数为：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/tayloreq2.png)

上述表达式的第二行给出了第 k 个系数的值。

如果我们设定 a=0，那么我们得到一个称为 f(x)的麦克劳林级数展开。

### 想要开始学习机器学习中的微积分？

立即获取我的免费 7 天邮件速成课程（包含示例代码）。

点击注册并免费获得课程的 PDF 电子书版本。

## **泰勒级数展开的示例**

通过对 f(x) = 1/x 进行微分，可以找到泰勒级数，首先需要对函数进行微分，并找到第 k 阶导数的一般表达式。

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/tayloreq3.png)

现在可以找到关于各个点的泰勒级数。例如：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/tayloreq4.png)

## **泰勒多项式**

由 f(x) 在 x=a 生成的阶数为 k 的泰勒多项式表示为：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/tayloreq5.png)

对于 f(x)=1/x 的例子，阶数为 2 的泰勒多项式表示为：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/tayloreq6.png)

## **通过泰勒多项式进行近似**

我们可以使用泰勒多项式来近似 x=a 处的函数值。多项式的阶数越高，多项式中的项数越多，近似值就越接近该点的实际函数值。

在下图中，函数 1/x 在点 x=1（左侧）和 x=3（右侧）附近绘制。绿色线是实际函数 f(x)= 1/x。粉色线表示通过阶数为 2 的多项式进行的近似。

![实际函数（绿色）及其近似值（粉色）](https://machinelearningmastery.com/wp-content/uploads/2021/07/taylor1-1.png)

实际函数（绿色）及其近似值（粉色）

## 泰勒级数的更多示例

我们来看函数 g(x) = e^x。注意到 g(x) 的 kth 阶导数也是 g(x)，g(x) 关于 x=a 的展开表示为：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/tayloreq7.png)

因此，在 x=0 附近，g(x) 的级数展开表示为（通过设置 a=0 得到）：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/tayloreq8.png)

对于函数 e^x 在点 x=0 附近生成的阶数为 k 的多项式表示为：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/tayloreq9.png)

下图显示了不同阶数的多项式在 x=0 附近对 e^x 值的估计。我们可以看到，随着距离零点的远离，我们需要更多的项来更准确地近似 e^x。绿色线代表实际函数，隐藏在阶数为 7 的蓝色近似多项式后面。

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/taylor2.png)

近似 e^x 的不同阶数的多项式

## **泰勒级数在机器学习中的应用**

在机器学习中，一个常用的方法是牛顿法。牛顿法使用二阶多项式来近似函数在某一点的值。这些使用二阶导数的方法称为二阶优化算法。

## **扩展**

本节列出了一些扩展教程的想法，您可能希望探索。

+   牛顿法

+   二阶优化算法

如果你探索了这些扩展内容，我很想知道。请在下面的评论中分享你的发现。

## **进一步阅读**

本节提供了更多关于该主题的资源，适合希望深入了解的读者。

### **教程**

+   [导数的温和介绍](https://machinelearningmastery.com/a-gentle-introduction-to-function-derivatives/)

+   [关于泰勒级数的维基百科文章](https://en.wikipedia.org/wiki/Taylor_series)

### **资源**

+   Jason Brownlee 关于 [机器学习中的微积分书籍](https://machinelearningmastery.com/calculus-books-for-machine-learning/) 的优秀资源

### **书籍**

+   [模式识别与机器学习](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738) 由 Christopher M. Bishop 编著。

+   [深度学习](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618/ref=as_li_ss_tl?dchild=1&keywords=deep+learning&qid=1606171954&s=books&sr=1-1&linkCode=sl1&tag=inspiredalgor-20&linkId=0a0c58945768a65548b639df6d1a98ed&language=en_US) 由 Ian Goodfellow, Joshua Begio, Aaron Courville 编著。

+   [托马斯微积分](https://amzn.to/35Yeolv)，第 14 版，2017 年。（基于 George B. Thomas 的原著，由 Joel Hass、Christopher Heil 和 Maurice Weir 修订）

+   [微积分](https://www.amazon.com/Calculus-3rd-Gilbert-Strang/dp/0980232759/ref=as_li_ss_tl?dchild=1&keywords=Gilbert+Strang+calculus&qid=1606171602&s=books&sr=1-1&linkCode=sl1&tag=inspiredalgor-20&linkId=423b93db012f7cc6bb92cb7494a3095f&language=en_US)，第 3 版，2017 年。（Gilbert Strang）

+   [微积分](https://amzn.to/3kS9I52)，第 8 版，2015 年。（James Stewart）

## **总结**

在本教程中，你了解了函数在某一点的泰勒级数展开。具体来说，你学到了：

+   幂级数和泰勒级数

+   泰勒多项式

+   如何使用泰勒多项式近似某个值附近的函数

## **你有任何问题吗？**

在下方评论中提问，我会尽力回答
