# 雅可比的温和介绍

> 原文：[`machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/`](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/)

在文献中，*Jacobian*一词通常交替用于指代雅可比矩阵或其行列式。

矩阵和行列式都有有用且重要的应用：在机器学习中，雅可比矩阵汇集了反向传播所需的偏导数；行列式在变量转换过程中很有用。

在本教程中，你将回顾雅可比的温和介绍。

完成本教程后，你将了解：

+   雅可比矩阵收集了多变量函数的所有一阶偏导数，可用于反向传播。

+   雅可比行列式在变量转换中很有用，它作为一个坐标空间与另一个坐标空间之间的缩放因子。

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_cover-scaled.jpg)

雅可比的温和介绍

照片由[Simon Berger](https://unsplash.com/@8moments)拍摄，版权所有。

## **教程概述**

本教程分为三个部分；它们是：

+   机器学习中的偏导数

+   雅可比矩阵

+   雅可比的其他用途

## **机器学习中的偏导数**

迄今为止，我们提到了[梯度和偏导数](https://machinelearningmastery.com/a-gentle-introduction-to-partial-derivatives-and-gradient-vectors)对优化算法的重要性，例如，更新神经网络的模型权重以达到最优权重集。使用偏导数可以让每个权重独立更新，通过计算误差曲线相对于每个权重的梯度。

我们在机器学习中通常使用的许多函数是[多变量](https://machinelearningmastery.com/?p=12606&preview=true)的，[向量值函数](https://machinelearningmastery.com/a-gentle-introduction-to-vector-valued-functions)，这意味着它们将多个实数输入*n*映射到多个实数输出*m*：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_11.png)

例如，考虑一个将灰度图像分类到多个类别的神经网络。这样的分类器所实现的函数会将每个单通道输入图像的*n*像素值映射到*m*输出概率，这些概率表示图像属于不同类别的可能性。

在训练神经网络时，反向传播算法负责将输出层计算出的误差回传到神经网络中各个隐藏层的神经元，直到达到输入层。

> *反向传播算法调整网络中权重的基本原则是，网络中的每个权重应根据网络整体误差对该权重变化的敏感性进行更新。*
> 
> – 第 222 页，[《深度学习》](https://www.amazon.com/Deep-Learning-Press-Essential-Knowledge/dp/0262537559/ref=sr_1_4?dchild=1&keywords=deep+learning&qid=1622968138&sr=8-4)，2019 年。

网络整体误差对某个特定权重变化的敏感性以变化率来衡量，这个变化率是通过对误差相对于相同权重的偏导数计算得到的。

为了简单起见，假设某个特定网络的一个隐藏层仅由一个神经元*k*组成。我们可以用一个简单的计算图来表示这个情况：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_1.png)

一个具有单一输入和单一输出的神经元

为了简单起见，假设一个权重，*w**[k]*，被施加到这个神经元的一个输入上，以根据该神经元实现的函数（包括非线性）生成一个输出，*z**[k]*。然后，这个神经元的权重可以通过以下方式与网络输出的误差相连接（以下公式在形式上被称为*微积分的链式法则*，但更多内容将在后续的单独教程中讲解）：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_2.png)

在这里，导数，*dz**[k]* / *dw**[k]*，首先将权重，*w**[k]*，与输出，*z**[k]*，连接起来，而导数，*d*error / *dz**[k]*，随后将输出，*z**[k]*，与网络误差连接起来。

通常情况下，我们会有许多相互连接的神经元组成网络，每个神经元都被赋予不同的权重。由于我们对这种情况更感兴趣，因此我们可以将讨论从标量情况推广到多个输入和多个输出：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_3.png)

这些项的和可以更紧凑地表示如下：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_4-1.png)

或者，用[向量表示法](https://machinelearningmastery.com/a-gentle-introduction-to-partial-derivatives-and-gradient-vectors)等效地使用增量算子∇来表示误差对权重，**w***[k]*，或输出，**z***[k]*的梯度：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_5.png)

> *反向传播算法包括对图中的每个操作执行这种雅可比-梯度乘积。*
> 
> – 第 207 页， [深度学习](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618/ref=sr_1_1?dchild=1&keywords=deep+learning&qid=1622968138&sr=8-1)，2017。

这意味着反向传播算法可以通过与*雅可比矩阵*的乘法，将网络误差的敏感度与权重的变化联系起来，公式为 (∂**z***[k]* / ∂**w***[k]*)^T。

因此，这个雅可比矩阵包含了什么？

## **雅可比矩阵**

雅可比矩阵收集了多变量函数的所有一阶偏导数。

具体来说，首先考虑一个将 *u* 个实数输入映射到一个实数输出的函数：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_6.png)

对于长度为 *u* 的输入向量 **x**，大小为 1 × *u* 的雅可比向量可以定义如下：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_7.png)

现在，考虑另一个将 *u* 个实数输入映射到 *v* 个实数输出的函数：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_8.png)

对于同一个输入向量，**x**，长度为 *u*，雅可比矩阵现在是一个 *v* × *u* 的矩阵，**J** ∈ ℝ*^(v×)**^u*，定义如下：

![](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_9.png)

将雅可比矩阵重新框架到之前考虑的机器学习问题中，同时保持 *u* 个实数输入和 *v* 个实数输出，我们发现这个矩阵包含以下偏导数：

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/jacobian_10.png)

### 想要开始学习机器学习中的微积分？

立即领取我的 7 天免费电子邮件速成课程（附带示例代码）。

点击注册，并获取课程的免费 PDF 电子书版本。

## **雅可比矩阵的其他用途**

在处理[积分](https://machinelearningmastery.com/?p=12637&preview=true)时，一个重要的技巧是 *变量变换*（也称为 *积分替换* 或 *u-替换*），即将一个积分简化为另一个更易计算的积分。

在单变量情况下，将某个变量 *x* 替换为另一个变量 *u*，可以将原始函数转化为一个更简单的函数，从而更容易找到其不定积分。在双变量情况下，另一个原因可能是我们希望将积分区域的形状转换为不同的形状。

> *在单变量情况下，通常只有一个改变变量的原因：使函数“更好”，以便我们可以找到其不定积分。在双变量情况下，还有第二个潜在原因：我们需要积分的二维区域在某种程度上不太方便，我们希望用 *u* 和 *v* 表示的区域更好——例如，成为一个矩形。*
> 
> – 第 412 页，[单变量与多变量微积分](https://www.whitman.edu/mathematics/multivariable/multivariable.pdf)，2020 年。

当在两个（或可能更多）变量之间进行替换时，过程开始于定义要进行替换的变量。例如，*x* = *f*(*u*, *v*) 和 *y* = *g*(*u*, *v*)。接着，根据函数 *f* 和 *g* 如何将 *u*–*v* 平面转换为 *x*–*y* 平面，转换积分限。最后，计算并包含 *雅可比行列式* 的绝对值，以作为一个坐标空间与另一个坐标空间之间的缩放因子。

## **进一步阅读**

本节提供了更多相关资源，如果你想深入了解的话。

### **书籍**

+   [深度学习](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618/ref=sr_1_1?dchild=1&keywords=deep+learning&qid=1622968138&sr=8-1)，2017 年。

+   [机器学习的数学](https://www.amazon.com/Mathematics-Machine-Learning-Peter-Deisenroth/dp/110845514X/ref=as_li_ss_tl?dchild=1&keywords=calculus+machine+learning&qid=1606171788&s=books&sr=1-3&linkCode=sl1&tag=inspiredalgor-20&linkId=209ba69202a6cc0a9f2b07439b4376ca&language=en_US)，2020 年。

+   [单变量与多变量微积分](https://www.whitman.edu/mathematics/multivariable/multivariable.pdf)，2020 年。

+   [深度学习](https://www.amazon.com/Deep-Learning-Press-Essential-Knowledge/dp/0262537559/ref=sr_1_4?dchild=1&keywords=deep+learning&qid=1622968138&sr=8-4)，2019 年。

### **文章**

+   [雅可比矩阵与行列式，维基百科](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)。

+   [通过替换积分，维基百科](https://en.wikipedia.org/wiki/Integration_by_substitution)。

## **总结**

在本教程中，你了解了关于雅可比矩阵的温和介绍。

具体来说，你学到了：

+   雅可比矩阵收集了多变量函数的所有一阶偏导数，可用于反向传播。

+   雅可比行列式在变量变换中很有用，它作为一个缩放因子在一个坐标空间与另一个坐标空间之间起作用。

你有任何问题吗？

在下面的评论中提出你的问题，我会尽力回答。
