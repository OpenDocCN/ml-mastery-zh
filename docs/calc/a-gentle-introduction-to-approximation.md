# 近似的温和介绍

> 原文：[`machinelearningmastery.com/a-gentle-introduction-to-approximation/`](https://machinelearningmastery.com/a-gentle-introduction-to-approximation/)

当涉及到分类或回归等机器学习任务时，近似技术在从数据中学习中发挥了关键作用。许多机器学习方法通过学习算法近似输入和输出之间的函数或映射。

在本教程中，你将发现什么是近似及其在机器学习和模式识别中的重要性。

完成本教程后，你将了解：

+   什么是近似

+   近似在机器学习中的重要性

让我们开始吧。

![近似的温和介绍。照片由 M Mani 提供，部分权利保留。](https://machinelearningmastery.com/wp-content/uploads/2021/07/MMani.jpg)

近似的温和介绍。照片由 M Mani 提供，部分权利保留。

## **教程概述**

本教程分为三个部分；它们是：

1.  什么是近似？

1.  当函数的形式未知时的近似

1.  当函数的形式已知时的近似

## **什么是近似？**

我们经常会遇到近似。例如，无理数π可以用 3.14 来近似。更准确的值是 3.141593，这仍然是一个近似值。你也可以类似地近似所有无理数的值，如 sqrt(3)、sqrt(7)等。

近似用于当数值、模型、结构或函数要么未知，要么计算困难时。在本文中，我们将重点讨论函数近似，并描述其在机器学习问题中的应用。主要有两种情况：

1.  函数是已知的，但计算其精确值既困难又成本高。在这种情况下，使用近似方法来寻找接近函数实际值的值。

1.  函数本身是未知的，因此使用模型或学习算法来寻找一个可以产生接近未知函数输出的函数。

## **函数形式已知时的近似**

如果知道函数的形式，那么微积分和数学中一个广泛使用的方法是通过泰勒级数进行近似。函数的泰勒级数是无限项的和，这些项是利用函数的导数计算得出的。函数的泰勒级数展开在这个[教程](https://machinelearningmastery.com/a-gentle-introduction-to-taylor-series)中进行了讨论。

另一个广泛使用的微积分和数学中的近似方法是[牛顿法](https://en.wikipedia.org/wiki/Newton%27s_method)。它可以用来近似多项式的根，因此成为近似不同值的平方根或不同数字的倒数等量的有用技术。

### 想开始学习机器学习中的微积分吗？

立即参加我的免费 7 天邮件速成课程（附示例代码）。

点击注册并免费获得课程的 PDF 电子书版本。

## **当函数形式未知时的近似**

在数据科学和机器学习中，假设存在一个基础函数，它揭示了输入和输出之间的关系。这个函数的形式未知。这里，我们讨论了几种使用近似的机器学习问题。

### **回归中的近似**

回归涉及在给定一组输入时预测输出变量。在回归中，真正将输入变量映射到输出的函数是未知的。假设某种线性或非线性回归模型可以近似输入到输出的映射。

例如，我们可能有关于每日消耗卡路里和相应血糖的数据。为了描述卡路里输入和血糖输出之间的关系，我们可以假设一个直线关系/映射函数。因此，直线就是输入到输出映射的近似。像最小二乘法这样的学习方法被用来找到这条直线。

![卡路里计数和血糖之间关系的直线近似](https://machinelearningmastery.com/wp-content/uploads/2021/07/approx1.png)

卡路里计数和血糖之间关系的直线近似

### **分类中的近似**

在分类问题中，近似函数的经典模型之一是神经网络。假设神经网络整体上可以近似一个将输入映射到类别标签的真实函数。然后使用梯度下降或其他学习算法，通过调整神经网络的权重来学习这个函数的近似。

![神经网络近似一个将输入映射到输出的基础函数](https://machinelearningmastery.com/wp-content/uploads/2021/07/approx3.png)

神经网络近似一个将输入映射到输出的基础函数

### **无监督学习中的近似**

以下是一个典型的无监督学习示例。这里我们有 2D 空间中的点，且这些点的标签均未给出。聚类算法通常假设一个模型，根据该模型一个点可以被分配到一个类或标签。例如，k-means 通过假设数据簇是圆形的，从而学习数据的标签，因此将相同标签或类别分配给位于同一个圆圈或在多维数据中属于 n-球的点。在下图中，我们通过圆形函数近似点与其标签之间的关系。

![一种聚类算法可以逼近一个模型，该模型确定输入点的聚类或未知标签](https://machinelearningmastery.com/wp-content/uploads/2021/07/approx2.png)

一种聚类算法可以逼近一个模型，该模型确定输入点的聚类或未知标签

## **扩展**

本节列出了一些扩展教程的想法，供你探索。

+   麦克劳林级数

+   泰勒级数

如果你探索了这些扩展内容，我很想知道。请在下面的评论中分享你的发现。

## **进一步阅读**

本节提供了更多资源，如果你希望深入了解这个话题。

### **教程**

+   [神经网络是函数逼近算法](https://machinelearningmastery.com/neural-networks-are-function-approximators/)

+   [关于逼近的维基百科文章](https://en.wikipedia.org/wiki/Approximation)

### **资源**

+   Jason Brownlee 提供的有关 [机器学习的微积分书籍](https://machinelearningmastery.com/calculus-books-for-machine-learning/) 的优秀资源。

### **书籍**

+   [模式识别与机器学习](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738) 由 Christopher M. Bishop 编著。

+   [深度学习](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618/ref=as_li_ss_tl?dchild=1&keywords=deep+learning&qid=1606171954&s=books&sr=1-1&linkCode=sl1&tag=inspiredalgor-20&linkId=0a0c58945768a65548b639df6d1a98ed&language=en_US) 由 Ian Goodfellow, Joshua Begio, Aaron Courville 编著。

+   [托马斯微积分](https://amzn.to/35Yeolv)，第 14 版，2017 年。（基于 George B. Thomas 的原著，由 Joel Hass、Christopher Heil、Maurice Weir 修订）

## **总结**

在本教程中，你了解了什么是逼近。具体来说，你学到了：

+   逼近

+   当函数的形式已知时的逼近

+   当函数的形式未知时的逼近

## **你有任何问题吗？**

在下面的评论中提问，我会尽力回答
