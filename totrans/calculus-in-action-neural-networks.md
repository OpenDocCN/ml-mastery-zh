# 微积分的应用：神经网络

> 原文：[`machinelearningmastery.com/calculus-in-action-neural-networks/`](https://machinelearningmastery.com/calculus-in-action-neural-networks/)

人工神经网络是一个计算模型，用于逼近输入和输出之间的映射。

它的灵感来自于人脑的结构，因为它类似地由一个互联的神经元网络组成，这些神经元在接收到来自邻近神经元的一组刺激后传播信息。

训练神经网络涉及一个过程，该过程同时使用反向传播和梯度下降算法。正如我们将看到的，这两个算法都广泛使用微积分。

在本教程中，你将发现微积分的各个方面如何应用于神经网络。

完成本教程后，你将了解：

+   人工神经网络被组织成神经元和连接的层次结构，其中后者赋予每个权重值。

+   每个神经元实现一个非线性函数，将一组输入映射到一个输出激活。

+   在训练神经网络时，反向传播和梯度下降算法广泛使用微积分。

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/neural_networks_cover-scaled.jpg)

微积分的应用：神经网络

图片由[Tomoe Steineck](https://unsplash.com/photos/T1Wru10gKhg)提供，保留部分版权。

## **教程概述**

本教程分为三个部分，它们是：

+   神经网络简介

+   神经元的数学

+   训练网络

## **先决条件**

对于本教程，我们假设你已经知道以下内容：

+   [函数逼近](https://machinelearningmastery.com/neural-networks-are-function-approximators/)

+   [变化率](https://machinelearningmastery.com/key-concepts-in-calculus-rate-of-change/)

+   [偏导数](https://machinelearningmastery.com/a-gentle-introduction-to-partial-derivatives-and-gradient-vectors)

+   [链式法则](https://machinelearningmastery.com/?p=12720&preview=true)

+   [链式法则在更多函数中的应用](https://machinelearningmastery.com/?p=12732&preview=true)

+   [梯度下降](https://machinelearningmastery.com/a-gentle-introduction-to-gradient-descent-procedure/)

你可以通过点击上面给出的链接来复习这些概念。

## **神经网络简介**

人工神经网络可以被视为函数逼近算法。

在监督学习环境中，当提供多个输入观察值表示关注的问题，以及相应的目标输出时，人工神经网络将尝试逼近这两者之间存在的映射。

> *神经网络是一个计算模型，灵感来自于人脑的结构。*
> 
> – 第 65 页，[深度学习](https://www.amazon.com/Deep-Learning-Press-Essential-Knowledge/dp/0262537559/ref=sr_1_11?dchild=1&keywords=deep+learning&qid=1627991691&sr=8-11)，2019 年。

人脑由一个庞大的互联神经元网络组成（约有一百亿个神经元），每个神经元包括一个细胞体、一组称为树突的纤维和一个轴突：

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/neural_networks_1.png)

人脑中的神经元

树突作为神经元的输入通道，而轴突则作为输出通道。因此，神经元通过其树突接收输入信号，这些树突又连接到其他邻近神经元的（输出）轴突。通过这种方式，一个足够强的电信号（也称为动作电位）可以沿着一个神经元的轴突传递到所有连接到它的其他神经元。这允许信号在大脑结构中传播。

> *因此，神经元充当全或无的开关，接受一组输入并输出一个动作电位或没有输出。*
> 
> – 第 66 页，[深度学习](https://www.amazon.com/Deep-Learning-Press-Essential-Knowledge/dp/0262537559/ref=sr_1_11?dchild=1&keywords=deep+learning&qid=1627991691&sr=8-11)，2019 年。

人工神经网络类似于人脑的结构，因为（1）它由大量互联的神经元组成，（2）这些神经元通过（3）接收来自邻近神经元的一组刺激并将其映射到输出，从而在网络中传播信息，以便传递到下一层神经元。

人工神经网络的结构通常组织成神经元的层级（[回顾](https://machinelearningmastery.com/?p=12720&preview=true)树状图的描述）。例如，以下图示例展示了一个完全连接的神经网络，其中一层中的所有神经元都连接到下一层的所有神经元：

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/neural_networks_21.png)

完全连接的前馈神经网络

输入位于网络的左侧，信息向右传播（或流动）到对侧的输出端。由于信息在网络中以*前馈*方向传播，因此我们也将这种网络称为*前馈神经网络*。

输入层和输出层之间的神经元层称为*隐藏*层，因为它们无法直接访问。

每两个神经元之间的连接（在图中由箭头表示）被赋予一个权重，该权重作用于通过网络的数据，正如我们稍后将看到的。

### 想要开始学习机器学习中的微积分吗？

现在就获取我的免费 7 天邮件速成课程（包括示例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## **神经元的数学**

更具体地说，假设一个特定的人工神经元（或弗兰克·罗森布拉特最初称之为*感知器*）接收 *n* 个输入，[*x*[1], …, *x*[n]]，每个连接都有一个对应的权重[*w*[1], …, *w*[n]]。

执行的第一个操作是将输入值乘以其相应的权重，并将偏置项 *b* 加到它们的总和中，生成输出 *z*：

*z* = ((*x*[1] × *w*[1]) + (*x*[2] × *w*[2]) + … + (*x*[n] × *w*[n])) + *b*

我们可以将这个操作以更紧凑的形式表示如下：

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/neural_networks_5.png)

到目前为止，我们执行的加权和计算是一个线性操作。如果每个神经元必须单独实现这种特定的计算，那么神经网络将被限制于仅学习线性输入输出映射。

> *然而，我们可能希望建模的许多世界中的关系是非线性的，如果我们尝试使用线性模型来建模这些关系，那么模型将非常不准确。*
> 
> – 第 77 页，[深度学习](https://www.amazon.com/Deep-Learning-Press-Essential-Knowledge/dp/0262537559/ref=sr_1_11?dchild=1&keywords=deep+learning&qid=1627991691&sr=8-11)，2019 年。

因此，每个神经元执行第二个操作，通过应用非线性激活函数 *a*(.) 转换加权和：

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/neural_networks_6.png)

如果我们将偏置项作为另一个权重* w *[0]（注意总和现在从 0 开始）集成到和中，我们可以更紧凑地表示每个神经元执行的操作：

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/neural_networks_7.png)

每个神经元执行的操作可以如下所示：

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/neural_networks_3.png)

神经元实现的非线性函数

因此，每个神经元可以被视为实现一个将输入集映射到输出激活的非线性函数。

## **训练网络**

训练人工神经网络涉及寻找最佳建模数据模式的权重集的过程。这是一个同时使用反向传播和梯度下降算法的过程。这两种算法都大量使用微积分。

每当网络向前（或向右）方向遍历时，可以通过损失函数（如平方误差的总和（SSE））计算网络的误差，即网络输出与预期目标之间的差异。然后，反向传播算法计算此误差对权重变化的梯度（或变化率）。为了做到这一点，它需要使用链式法则和偏导数。

为了简单起见，考虑一个由单条激活路径连接的两个神经元组成的网络。如果我们需要打开它们，我们会发现神经元按照以下级联操作进行：

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/neural_networks_4.png)

两个神经元级联执行的操作

链式法则的第一个应用将网络的整体误差连接到激活函数*a*[2]的输入*z*[2]，随后连接到权重*w*[2]，如下所示：

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/neural_networks_8.png)

你可能注意到链式法则的应用涉及到神经元激活函数关于其输入*z*[2]的偏导数乘积，还有其他项。有不同的激活函数可供选择，例如 sigmoid 或 logistic 函数。如果我们以 logistic 函数为例，那么其偏导数将如下计算：

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/neural_networks_9.png)

因此，我们可以如下计算*t*[2]：

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/neural_networks_10.png)

这里，*t*[2]是期望的激活，通过计算*t*[2]和*a*[2]之间的差异，我们因此计算了网络生成的激活与预期目标之间的误差。

由于我们正在计算激活函数的导数，因此它应在整个实数空间上是连续且可微的。在深度神经网络的情况下，误差梯度向后传播经过大量隐藏层。这可能导致误差信号迅速减少到零，尤其是如果导数函数的最大值已经很小（例如，logistic 函数的倒数最大值为 0.25）。这被称为*梯度消失问题*。ReLU 函数在深度学习中非常流行，以减轻这个问题，因为其在其正部分的导数等于 1。

接下来的权重反向传播到网络的深层，因此链式法则的应用也可以类似地扩展，以将整体误差与权重 *w*[1] 连接起来，如下所示：

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/neural_networks_11.png)

如果我们再次以逻辑函数作为激活函数，那么我们将如下计算 ????[1]：

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/neural_networks_12.png)

一旦我们计算了网络误差相对于每个权重的梯度，就可以应用梯度下降算法来更新每个权重，以进行下一个时间点 *t*+1 的 *前向传播*。对于权重 *w*[1]，使用梯度下降的权重更新规则如下：

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/neural_networks_13.png)

即使我们这里考虑的是一个简单的网络，我们经历的过程也可以扩展到评估更复杂和更深的网络，例如卷积神经网络（CNNs）。

如果考虑的网络具有来自多个输入的多个分支（并可能流向多个输出），则其评估将涉及对每条路径的不同导数链的求和，类似于我们之前推导的广义链式法则。

## **进一步阅读**

本节提供了更多关于这个主题的资源，如果你想深入了解。

### **书籍**

+   [深度学习](https://www.amazon.com/Deep-Learning-Press-Essential-Knowledge/dp/0262537559/ref=sr_1_11?dchild=1&keywords=deep+learning&qid=1627991691&sr=8-11)，2019。

+   [模式识别与机器学习](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/1493938436/ref=sr_1_2?dchild=1&keywords=Pattern+Recognition+and+Machine+Learning&qid=1627991645&sr=8-2)，2016。

## **总结**

在本教程中，你发现了微积分在神经网络中的应用。

具体来说，你学到了：

+   人工神经网络被组织成由神经元和连接层组成，后者每个都分配一个权重值。

+   每个神经元实现一个非线性函数，将一组输入映射到输出激活值。

+   在训练神经网络时，反向传播和梯度下降算法广泛使用微积分。

你有什么问题吗？

在下面的评论中提出你的问题，我会尽力回答。
