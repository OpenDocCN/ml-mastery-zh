# 如何训练最终的机器学习模型

> 原文： [`machinelearningmastery.com/train-final-machine-learning-model/`](https://machinelearningmastery.com/train-final-machine-learning-model/)

我们用来对新数据做出预测的机器学习模型称为最终模型。

在应用机器学习中可能会混淆如何训练最终模型。

初学者会向现场询问此错误，例如：

*   _ 如何通过交叉验证做出预测？_
*   _ 我从交叉验证中选择哪种模型？_
*   _ 我是否在训练数据集上准备好后使用该模型？_

这篇文章将澄清混乱。

在这篇文章中，您将了解如何最终确定机器学习模型，以便对新数据做出预测。

让我们开始吧。

![How to Train a Final Machine Learning Model](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/03/How-to-Train-a-Final-Machine-Learning-Model.jpg)

如何训练最终的机器学习模型
照片由[相机眼睛摄影](https://www.flickr.com/photos/camera_is_a_mirror_with_memory/16665301421/)，保留一些权利。

## 什么是最终模型？

最终机器学习模型是用于对新数据做出预测的模型。

也就是说，给定输入数据的新示例，您希望使用该模型来预测预期输出。这可以是分类（分配标签）或回归（实际值）。

例如，照片是 _ 狗 _ 还是 _ 猫 _ 的照片，还是明天的估计销售数量。

您的机器学习项目的目标是获得最佳的最终模型，其中“最佳”定义为：

*   **数据**：您提供的历史数据。
*   **时间**：你必须在项目上花费的时间。
*   **程序**：数据准备步骤，算法或算法，以及所选的算法配置。

在项目中，您可以收集数据，花费时间，发现数据准备过程，要使用的算法以及如何配置它。

最终的模型是这个过程的顶峰，你寻求的目的是为了开始实际做出预测。

## 训练/测试集的目的

为什么我们使用训练和测试装置？

创建训练并测试数据集的分割是快速评估算法在问题上的表现的一种方法。

训练数据集用于准备模型，以进行训练。

我们假装测试数据集是新数据，其中输出值被保留在算法中。我们从训练模型中收集来自测试数据集的输入的预测，并将它们与测试集的保留输出值进行比较。

比较测试数据集上的预测和保留输出允许我们计算测试数据集上模型的表现度量。这是在对看不见的数据做出预测时对问题进行训练的算法的技能估计。

### 让我们进一步解压缩

当我们评估算法时，我们实际上正在评估过程中的所有步骤，包括如何准备训练数据（例如缩放），算法的选择（例如 kNN）以及所选算法的配置方式（例如 k = 3） ）。

根据预测计算的绩效指标是对整个程序技能的估计。

我们从以下方面概括了绩效指标：

*   “ _**测试集**_ 的程序技巧

至

*   “ _**看不见的数据**_ 的程序技巧。

这是一个很大的飞跃，需要：

*   该程序足够强大，以至于技能估计接近我们对未见数据的实际预期。
*   表现测量的选择准确地捕获了我们对预测未见数据的测量感兴趣。
*   数据准备的选择对于新数据是很好理解和可重复的，并且如果需要将预测返回到其原始比例或与原始输入值相关，则可以是可逆的。
*   算法的选择对其预期用途和操作环境（例如复杂性或所选编程语言）有意义。

很多事情都依赖于测试集上整个过程的估计技能。

实际上，使用估计程序技能的训练/测试方法对看不见的数据通常具有很大的差异（除非我们有很多数据需要分割）。这意味着当它重复时，它会产生不同的结果，通常会产生非常不同的结果。

结果是我们可能完全不确定程序对看不见的数据的实际执行情况以及一个程序与另一个程序的比较。

通常，在时间允许的情况下，我们更愿意使用 k 折交叉验证。

## k 折交叉验证的目的

为什么我们使用 k-fold 交叉验证？

交叉验证是另一种估计方法对看不见的数据的技能的方法。就像使用训练测试一样。

交叉验证系统地在数据集的多个子集上创建和评估多个模型。

反过来，这提供了一系列绩效衡量标准。

*   我们可以计算这些度量的平均值，以了解该过程的平均执行情况。
*   我们可以计算这些测量的标准偏差，以了解该过程的技能在实践中有多大的变化。

当您尝试选择要使用的算法和数据准备过程时，这也有助于对一个过程与另一个过程进行更细致的比较。

此外，这些信息非常宝贵，因为您可以使用均值和点差来给出实际机器学习过程中预期表现的置信区间。

训练测试分裂和 k 折交叉验证都是重采样方法的示例。

## 为什么我们使用重采样方法？

应用机器学习的问题在于我们正在尝试对未知数进行建模。

在给定的预测性建模问题上，理想模型是在对新数据做出预测时表现最佳的模型。

我们没有新数据，所以我们必须假装统计技巧。

训练测试拆分和 k 折交叉验证称为重采样方法。重采样方法是对数据集进行采样和估计未知数量的统计程序。

在应用机器学习的情况下，我们感兴趣的是估计机器学习过程对看不见的数据的技能。更具体地说，由机器学习过程做出的预测的技巧。

一旦我们获得了估计的技能，我们就完成了重采样方法。

*   如果使用训练测试拆分，则意味着您可以丢弃拆分数据集和训练模型。
*   如果您正在使用 k-fold 交叉验证，那意味着您可以丢弃所有经过训练的模型。

他们已达到目的，不再需要。

您现在已准备好完成模型。

## 如何敲定模型？

通过对所有数据应用所选的机器学习过程来最终确定模型。

而已。

使用最终模型，您可以：

*   保存模型以供以后或操作使用。
*   对新数据做出预测。

交叉验证模型或训练测试数据集怎么样？

他们被丢弃了。他们不再需要了。他们的目的是帮助您选择最终确定的程序。

## 常见问题

本节列出了您可能遇到的一些常见问题。

### 为什么不在训练数据集上训练模型？

和

### 为什么不从交叉验证中保留最佳模型？

如果你愿意，你可以。

通过重复使用技能评估期间训练的模型，您可以节省时间和精力。

如果需要数天，数周或数月来训练模型，这可能是一件大事。

在训练所有可用数据时，您的模型可能会比仅用于估计模型表现的子集更好。

这就是我们希望在所有可用数据上训练最终模型的原因。

### 在所有数据上训练的模型的表现是否会有所不同？

我认为这个问题引发了大多数关于模型定型的误解。

换一种方式：

*   如果您在所有可用数据上训练模型，那么您如何知道模型的表现如何？

您已使用重采样程序回答了此问题。

如果设计得很好，您使用训练测试或 k 折交叉验证计算的表现指标可以适当地描述对所有可用历史数据进行训练的最终模型在一般情况下的表现。

如果您使用 k-fold 交叉验证，您将估计模型平均的“错误”（或相反，“正确”），以及该错误或正确性的预期传播。

这就是为什么精心设计的测试装置在应用机器学习中绝对至关重要。更强大的测试工具将使您能够更加依赖于估计的表现。

### 每次我训练模型时，我都会得到不同的表现分数;我应该选择得分最高的模特吗？

机器学习算法是随机的，并且可以预期在相同数据上的这种不同表现的行为。

重复采样方法（例如重复训练/测试或重复 k 折叠交叉验证）将有助于了解方法中存在多少差异。

如果这是一个真正的问题，您可以创建多个最终模型并从预测集合中取平均值以减少方差。

我在帖子中谈到了这个问题：

*   [在机器学习中拥抱随机性](http://machinelearningmastery.com/randomness-in-machine-learning/)

## 摘要

在这篇文章中，您了解了如何训练最终的机器学习模型以供操作使用。

您已经克服了最终确定模型的障碍，例如：

*   了解重采样程序的目标，例如训练测试拆分和 k 折交叉验证。
*   模型定型作为训练所有可用数据的新模型。
*   将估算绩效的关注与最终确定模型分开。

您是否有关于最终确定模型的其他问题或疑虑，我还没有解决？
在评论中提问，我会尽力帮助。