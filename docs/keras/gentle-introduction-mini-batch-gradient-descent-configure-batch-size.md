# 小批量梯度下降的简要介绍以及如何配置批量大小

> 原文： [`machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/`](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/)

随机梯度下降是用于训练深度学习模型的主要方法。

梯度下降有三种主要变体，可能会混淆哪一种使用。

在这篇文章中，您将发现一般应该使用的一种梯度下降以及如何配置它。

完成这篇文章后，你会知道：

*   什么梯度下降以及它如何从高层起作用。
*   什么批次，随机和小批量梯度下降以及每种方法的优点和局限性。
*   这种小批量梯度下降是首选方法，以及如何在您的应用程序上配置它。

让我们开始吧。

*   **Update Apr / 2018** ：添加了额外的参考，以支持批量大小为 32。

![A Gentle Introduction to Mini-Batch Gradient Descent and How to Configure Batch Size](img/e95bf53e34f94fbb07da8860454b1e2a.png)

小批量梯度下降和如何配置批量大小的温和介绍
照片由 [Brian Smithson](https://www.flickr.com/photos/smithser/6269720226/) ，保留一些权利。

## 教程概述

本教程分为 3 个部分;他们是：

1.  什么是梯度下降？
2.  对比 3 种类型的梯度下降
3.  如何配置 Mini-Batch Gradient Descent

## 什么是梯度下降？

梯度下降是一种优化算法，通常用于查找机器学习算法的权重或系数，例如人工神经网络和逻辑回归。

它的工作原理是让模型对训练数据做出预测，并使用预测误差来更新模型，以减少错误。

该算法的目标是找到最小化训练数据集上的模型的误差的模型参数（例如，系数或权重）。它通过对模型进行更改来实现此目的，该模型将其沿着梯度或误差斜率向下移动到最小误差值。这使算法的名称为“梯度下降”。

下面的伪代码草图总结了梯度下降算法：

```py
model = initialization(...)
n_epochs = ...
train_data = ...
for i in n_epochs:
	train_data = shuffle(train_data)
	X, y = split(train_data)
	predictions = predict(X, model)
	error = calculate_error(y, predictions)
	model = update_model(model, error)
```

有关更多信息，请参阅帖子：

*   [机器学习的梯度下降](http://machinelearningmastery.com/gradient-descent-for-machine-learning/)
*   [如何利用 Python 从零开始随机梯度下降实现线性回归](http://machinelearningmastery.com/implement-linear-regression-stochastic-gradient-descent-scratch-python/)

## 对比 3 种类型的梯度下降

梯度下降可以根据用于计算误差的训练模式的数量而变化;而这又用于更新模型。

用于计算误差的模式数包括用于更新模型的梯度的稳定性。我们将看到计算效率的梯度下降配置和误差梯度的保真度存在张力。

梯度下降的三种主要风格是批量，随机和小批量。

让我们仔细看看每一个。

### 什么是随机梯度下降？

随机梯度下降（通常缩写为 SGD）是梯度下降算法的变体，其计算误差并更新训练数据集中每个示例的模型。

每个训练样例的模型更新意味着随机梯度下降通常被称为[在线机器学习算法](https://en.wikipedia.org/wiki/Online_machine_learning)。

#### 上升空间

*   频繁更新可立即深入了解模型的表现和改进速度。
*   这种梯度下降的变体可能是最容易理解和实现的，特别是对于初学者。
*   增加的模型更新频率可以更快地学习某些问题。
*   噪声更新过程可以允许模型避免局部最小值（例如，早熟收敛）。

#### 缺点

*   如此频繁地更新模型在计算上比其他梯度下降配置更昂贵，在大型数据集上训练模型需要更长的时间。
*   频繁更新可能导致噪声梯度信号，这可能导致模型参数并且反过来模型误差跳跃（在训练时期上具有更高的方差）。
*   沿着误差梯度的噪声学习过程也可能使算法难以确定模型的最小误差。

### 什么是批量梯度下降？

批量梯度下降是梯度下降算法的变体，其计算训练数据集中每个示例的误差，但仅在评估了所有训练样本之后更新模型。

整个训练数据集的一个周期称为训练时期。因此，经常说批量梯度下降在每个训练时期结束时执行模型更新。

#### 上升空间

*   对模型的更新较少意味着这种梯度下降变体在计算上比随机梯度下降更有效。
*   降低的更新频率导致更稳定的误差梯度，并且可以在一些问题上导致更稳定的收敛。
*   预测误差的计算与模型更新的分离使算法成为基于并行处理的实现。

#### 缺点

*   更稳定的误差梯度可能导致模型过早收敛到不太理想的参数组。
*   训练时期结束时的更新需要在所有训练样例中累积预测误差的额外复杂性。
*   通常，批量梯度下降以这样的方式实现，即它需要存储器中的整个训练数据集并且可用于算法。
*   对于大型数据集，模型更新以及训练速度可能变得非常慢。

### 什么是 Mini-Batch Gradient Descent？

小批量梯度下降是梯度下降算法的变体，其将训练数据集分成小批量，用于计算模型误差和更新模型系数。

实现可以选择对小批量的梯度求和或者取梯度的平均值，这进一步减小了梯度的方差。

小批量梯度下降试图在随机梯度下降的稳健性和批量梯度下降的效率之间找到平衡。它是深度学习领域中最常用的梯度下降实现。

#### 上升空间

*   模型更新频率高于批量梯度下降，这允许更稳健的收敛，避免局部最小值。
*   与随机梯度下降相比，批量更新提供了计算上更有效的过程。
*   批量允许在内存和算法实现中不具有所有训练数据的效率。

#### 缺点

*   小批量需要为学习算法配置额外的“小批量大小”超参数。
*   错误信息必须在批量梯度下降等小批量训练示例中累积。

## 如何配置 Mini-Batch Gradient Descent

对于大多数应用，小批量梯度下降是梯度下降的推荐变体，特别是在深度学习中。

为简洁起见，通常称为“批量大小”的小批量大小通常被调整到正在执行实现的计算架构的一个方面。例如 2 的幂，适合 GPU 或 CPU 硬件的内存要求，如 32,64,128,256 等。

批量大小是学习过程中的滑块。

*   较小的值使学习过程在训练过程中以噪声成本快速收敛。
*   较大的值使学习过程缓慢收敛，并准确估计误差梯度。

**提示 1：批量大小的良好默认值可能是 32\.**

> ... [批量大小]通常在 1 到几百之间选择，例如， [批量大小] = 32 是一个很好的默认值，其值大于 10，利用了矩阵 - 矩阵乘积相对于矩阵向量乘积的加速。

- [深层架构基于梯度的训练的实用建议](https://arxiv.org/abs/1206.5533)，2012

**更新 2018** ：这是支持批量大小为 32 的另一篇论文，这里是引用（m 是批量大小）：

> 所呈现的结果证实，对于给定的计算成本，在大范围的实验中使用小批量尺寸实现了最佳的训练稳定性和泛化表现。在所有情况下，已经获得了最佳结果，批量大小 m = 32 或更小，通常小到 m = 2 或 m = 4。

- [重新审视深度神经网络的小批量训练](https://arxiv.org/abs/1804.07612)，2018 年。

**提示 2：在调整批量大小时，最好在不同批量大小的情况下查看模型验证错误与训练时间的学习曲线。**

> ...在选择了其他超参数（学习率除外）之后，通过比较训练曲线（训练和验证误差与训练时间量），可以分别优化其他超参数。

**提示 3：在调整所有其他超参数后调整批量大小和学习率。**

> ... [批量大小]和[学习率]可能会与其他超参数稍微交互，因此两者都应在最后重新优化。一旦选择[批量大小]，通常可以固定，而其他超参数可以进一步优化（动量超参数除外，如果使用的话）。

## 进一步阅读

如果您要深入了解，本节将提供有关该主题的更多资源。

### 相关文章

*   [机器学习的梯度下降](http://machinelearningmastery.com/gradient-descent-for-machine-learning/)
*   [如何利用 Python 从零开始随机梯度下降实现线性回归](http://machinelearningmastery.com/implement-linear-regression-stochastic-gradient-descent-scratch-python/)

### 补充阅读

*   [维基百科上的随机梯度下降](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
*   [维基百科上的在线机器学习](https://en.wikipedia.org/wiki/Online_machine_learning)
*   [梯度下降优化算法概述](http://sebastianruder.com/optimizing-gradient-descent/index.html)
*   [深层架构基于梯度的训练的实用建议](https://arxiv.org/abs/1206.5533)，2012
*   [随机优化的高效小批量训练](http://www.cs.cmu.edu/~muli/file/minibatch_sgd.pdf)，2014 年
*   [在深度学习中，为什么我们不使用整个训练集来计算梯度？ Quora 上的](https://www.quora.com/In-deep-learning-why-dont-we-use-the-whole-training-set-to-compute-the-gradient)
*   [大规模机器学习的优化方法](https://arxiv.org/abs/1606.04838)，2016

## 摘要

在这篇文章中，您发现了梯度下降算法以及您应该在实践中使用的版本。

具体来说，你学到了：

*   什么梯度下降以及它如何从高层起作用。
*   什么批次，随机和小批量梯度下降以及每种方法的优点和局限性。
*   这种小批量梯度下降是首选方法，以及如何在您的应用程序上配置它。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。