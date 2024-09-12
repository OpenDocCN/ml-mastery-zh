# 机器学习中的微积分：为什么它有效

> 原文：[`machinelearningmastery.com/calculus-in-machine-learning-why-it-works/`](https://machinelearningmastery.com/calculus-in-machine-learning-why-it-works/)

微积分是机器学习中的核心数学概念之一，它使我们能够理解不同机器学习算法的内部工作原理。

微积分在机器学习中的一个重要应用是梯度下降算法，它与反向传播一起使我们能够训练神经网络模型。

在本教程中，你将发现微积分在机器学习中的关键作用。

完成本教程后，你将知道：

+   微积分在理解机器学习算法的内部工作原理中发挥着重要作用，例如用于最小化误差函数的梯度下降算法。

+   微积分为我们提供了优化复杂目标函数以及具有多维输入的函数所需的工具，这些函数代表了不同的机器学习应用。

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/calculus_in_machine_learning_cover-scaled.jpg)

机器学习中的微积分：为什么它有效

图片由 [Hasmik Ghazaryan Olson](https://unsplash.com/photos/N9OQ2ZHNwCs) 提供，保留部分权利。

## **教程概述**

本教程分为两部分，它们是：

+   机器学习中的微积分

+   为什么微积分在机器学习中有效

## **机器学习中的微积分**

神经网络模型，无论是浅层还是深层，都会实现一个将一组输入映射到期望输出的函数。

神经网络实现的函数通过训练过程学习，该过程迭代地搜索一组权重，以使神经网络能够最好地模拟训练数据的变化。

> *一种非常简单的函数类型是从单一输入到单一输出的线性映射。*
> 
> *《深度学习》第 187 页，2019 年。*

*这样的线性函数可以用一个具有斜率 *m* 和 y 截距 *c* 的直线方程表示：*

*y* = *mx* + *c*

变化每个参数 *m* 和 *c* 会产生定义不同输入输出映射的不同线性模型。

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/calculus_in_machine_learning_1.png)

通过变化斜率和截距产生的不同线性模型的线图

取自《深度学习》

因此，学习映射函数的过程涉及到这些模型参数或 *权重* 的近似，这些参数会导致预测输出与目标输出之间的最小误差。这个误差通过损失函数、成本函数或误差函数（通常可以互换使用）来计算，而最小化损失的过程称为 *函数优化*。

我们可以将微分计算应用于函数优化过程。

为了更好地理解如何将微分计算应用于函数优化，我们回到具有线性映射函数的具体示例。

假设我们有一些单输入特征的数据显示，*x*，以及它们对应的目标输出，*y*。为了衡量数据集上的误差，我们将计算预测输出和目标输出之间的平方误差和（SSE），作为我们的损失函数。

对模型权重的不同值进行参数扫描，*w[0]* = *m* 和 *w[1]* = *c*，生成了形状为凸形的个别误差轮廓。

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/calculus_in_machine_learning_2.png)

错误（SSE）轮廓的线图，当在斜率和截距的范围内进行扫描时生成

摘自《深度学习》

结合个别误差轮廓生成了一个三维误差面，该面也呈凸形。这个误差面位于一个权重空间内，该空间由模型权重的扫掠范围定义，*w[0]* 和 *w[1]*。

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/calculus_in_machine_learning_3.png)

当斜率和截距都变化时生成的误差（SSE）面三维图

摘自《深度学习》

在这个权重空间中移动相当于在不同线性模型之间移动。我们的目标是识别在所有可能的备选模型中最适合数据的模型。最佳模型的特征是数据集上的最低误差，这与误差面的最低点相对应。

> *一个凸形或碗状的误差面对于学习线性函数以建模数据集是非常有用的，因为这意味着学习过程可以被框定为在误差面上寻找最低点。用来寻找这个最低点的标准算法被称为梯度下降。*
> 
> *第 194 页，《深度学习》，2019 年。*

*梯度下降算法作为优化算法，将通过沿着误差面的梯度下坡来寻求达到误差面的最低点。这种下降是基于对误差面梯度或斜率的计算。*

这就是微分计算发挥作用的地方。

> *微积分，特别是微分，是处理变化率的数学领域。*
> 
> *第 198 页，《深度学习》，2019 年。*

*更正式地说，我们可以将我们希望优化的函数表示为：*

*error =* f(*weights*)

通过计算误差相对于权重的变化率或斜率，梯度下降算法可以决定如何改变权重以继续减少误差。

## **为何微积分在机器学习中有效**

我们考虑优化的误差函数相对简单，因为它是凸的，且具有单一的全局最小值。

尽管如此，在机器学习的背景下，我们通常需要优化更复杂的函数，这使得优化任务变得非常具有挑战性。如果函数的输入也是多维的，优化可能会变得更加困难。

微积分为我们提供了解决这两种挑战所需的工具。

假设我们有一个更通用的函数，希望将其最小化，该函数接受一个实数输入*x*，并产生一个实数输出*y*：

*y* = f(*x*)

在不同的*x*值处计算变化率是有用的，因为这能指示我们需要对*x*进行的变化，以获得*y*的相应变化。

由于我们在最小化函数，我们的目标是找到一个使 f(*x*)值尽可能低且具有零变化率的点；因此，这是一个全局最小值。根据函数的复杂性，这可能不一定可行，因为可能存在许多局部最小值或鞍点，优化算法可能会被困在其中。

> *在深度学习的背景下，我们优化的函数可能有许多局部最小值，这些局部最小值并不理想，还有许多被非常平坦区域包围的鞍点。*
> 
> *第 84 页，《深度学习》，2017 年。*

*因此，在深度学习的背景下，我们通常接受一个可能不一定对应全局最小值的次优解，只要它对应一个非常低的 f(*x*)值。*

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/calculus_in_machine_learning_4.png)

成本函数的折线图显示局部和全局最小值

摘自《深度学习》

如果我们处理的函数有多个输入，微积分还为我们提供了*偏导数*的概念；或者更简单地说，计算*y*相对于每一个输入*x**[i]*变化的变化率，同时保持其他输入不变的方法。

> *这就是为什么在梯度下降算法中，每个权重独立更新的原因：权重更新规则依赖于每个权重的 SSE 偏导数，由于每个权重都有不同的偏导数，因此每个权重都有单独的更新规则。*
> 
> *第 200 页，《深度学习》，2019 年。*

*因此，如果我们再次考虑误差函数的最小化，计算误差相对于每个特定权重的偏导数允许每个权重独立于其他权重进行更新。*

这也意味着梯度下降算法可能不会沿着误差表面沿直线下降。相反，每个权重将根据误差曲线的局部梯度进行更新。因此，一个权重可能比另一个权重更新得更多，以便梯度下降算法达到函数的最小值。

## **进一步阅读**

本节提供了更多关于该主题的资源，如果你想深入了解。

### **书籍**

+   [深度学习](https://www.amazon.com/Deep-Learning-Press-Essential-Knowledge/dp/0262537559/ref=sr_1_4?dchild=1&keywords=deep+learning&qid=1622968138&sr=8-4)，2019。

+   [深度学习](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618/ref=sr_1_1?dchild=1&keywords=deep+learning&qid=1622968138&sr=8-1)，2017。

## **总结**

在本教程中，你了解了微积分在机器学习中的核心作用。

具体来说，你学到了：

+   微积分在理解机器学习算法的内部机制中扮演着重要角色，例如，梯度下降算法根据变化率的计算来最小化误差函数。

+   微积分中变化率的概念也可以用来最小化更复杂的目标函数，这些函数不一定是凸形的。

+   偏导数的计算是微积分中的另一个重要概念，它使我们能够处理多个输入的函数。

你有任何问题吗？

在下面的评论中提问，我会尽力回答。*****