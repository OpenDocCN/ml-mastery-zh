# OpenCV 中的支持向量机

> 原文：[`machinelearningmastery.com/support-vector-machines-in-opencv/`](https://machinelearningmastery.com/support-vector-machines-in-opencv/)

支持向量机算法是最受欢迎的监督学习技术之一，并且它在 OpenCV 库中得到了实现。

本教程将介绍开始使用 OpenCV 中支持向量机所需的技能，我们将使用自定义数据集生成。在随后的教程中，我们将应用这些技能于图像分类和检测的具体应用。

在本教程中，你将学习如何在自定义二维数据集上应用 OpenCV 的支持向量机算法。

完成本教程后，你将了解：

+   支持向量机的一些最重要的特征。

+   如何在 OpenCV 中使用支持向量机算法处理自定义数据集。

**通过我的书[《OpenCV 中的机器学习》](https://machinelearning.samcart.com/products/machine-learning-opencv/)来** **启动你的项目**。它提供了**自学教程**和**可运行的代码**。

让我们开始吧。![](https://machinelearningmastery.com/wp-content/uploads/2023/03/svm_cover-scaled.jpg)

OpenCV 中的支持向量机

图片由[Lance Asper](https://unsplash.com/photos/O79h8KzusIc)提供，版权所有。

## **教程概述**

本教程分为两部分，它们是：

+   支持向量机工作原理的提醒

+   在 OpenCV 中发现 SVM 算法

## **支持向量机工作原理的提醒**

支持向量机（SVM）算法在[Jason Brownlee 的这个教程](https://machinelearningmastery.com/support-vector-machines-for-machine-learning/)中已经解释得很好，不过我们先从复习他教程中的一些最重要的点开始：

+   **为了简单起见，我们假设有两个独立的类别 0 和 1。一个超平面可以将这两个类别中的数据点分开，决策边界将输入空间分割以根据类别区分数据点。这个超平面的维度取决于输入数据点的维度。**

***   **如果给定一个新观察到的数据点，我们可以通过计算它位于超平面哪一侧来找出它所属的类别。**

***   **一个*边际*是决策边界与最近数据点之间的距离。它是通过仅考虑属于不同类别的最近数据点来确定的。它是这些最近数据点到决策边界的垂直距离。**

***   **最大的边际与最近数据点的距离特征化了最佳决策边界。这些最近的数据点被称为*支持向量*。**

***   **如果类别之间不能完全分开，因为它们可能分布得使得一些数据点在空间中混杂，那么需要放宽最大化边界的约束。通过引入一个称为 *C* 的可调参数，可以放宽边界约束。**

***   **参数 *C* 的值控制边界约束可以被违反的程度，值为 0 意味着完全不允许违反。增加 *C* 的目的是在最大化边界和减少误分类之间达到更好的折衷。**

***   **此外，SVM 使用核函数来计算输入数据点之间的相似性（或距离）度量。在最简单的情况下，当输入数据是线性可分且可以通过线性超平面分离时，核函数实现了一个点积操作。**

***   **如果数据点一开始不是线性可分的，*核技巧* 就会派上用场，其中核函数执行的操作旨在将数据转换到更高维的空间，使其变得线性可分。这类似于 SVM 在原始输入空间中找到一个非线性决策边界。**

**## **在 OpenCV 中发现 SVM 算法**

首先，我们考虑将 SVM 应用于一个简单的线性可分数据集，这样我们可以在继续更复杂的任务之前，直观地了解前面提到的几个概念。

为此，我们将生成一个包含 100 个数据点（由 `n_samples` 指定）的数据集，这些数据点均匀地分成 2 个高斯簇（由 `centers` 指定），标准差设置为 1.5（由 `cluster_std` 指定）。为了能够复制结果，我们还将定义一个 `random_state` 的值，我们将其设置为 15：

Python

```py
# Generate a dataset of 2D data points and their ground truth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=15)

# Plot the dataset
scatter(x[:, 0], x[:, 1], c=y_true)
show()
```

上面的代码应生成以下数据点的图。你可能会注意到，我们将颜色值设置为实际标签，以便区分属于两个不同类别的数据点：

![](https://machinelearningmastery.com/wp-content/uploads/2023/03/svm_1.png)

线性可分的数据点属于两个不同的类别

下一步是将数据集分成训练集和测试集，其中前者用于训练 SVM，后者用于测试：

Python

```py
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Plot the training and testing datasets
fig, (ax1, ax2) = subplots(1, 2)
ax1.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
ax1.set_title('Training data')
ax2.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
ax2.set_title('Testing data')
show()
```

![](https://machinelearningmastery.com/wp-content/uploads/2023/03/svm_2.png)

将数据点分为训练集和测试集

从上面的训练数据图像中，我们可以看到这两个类别明显可分，并且应该能够通过一个线性超平面轻松分开。因此，让我们继续创建和训练一个在 OpenCV 中使用线性核的 SVM，以找到这两个类别之间的最佳决策边界：

Python

```py
# Create a new SVM
svm = ml.SVM_create()

# Set the SVM kernel to linear
svm.setKernel(ml.SVM_LINEAR)

# Train the SVM on the set of training data
svm.train(x_train.astype(float32), ml.ROW_SAMPLE, y_train)
```

在这里，请注意 OpenCV 中 SVM 的 `train` 方法需要输入数据为 32 位浮点类型。

我们可以继续使用训练好的 SVM 对测试数据预测标签，并通过将预测与相应的真实标签进行比较来计算分类器的准确性：

Python

```py
# Predict the target labels of the testing data
_, y_pred = svm.predict(x_test.astype(float32))

# Compute and print the achieved accuracy
accuracy = (sum(y_pred[:, 0].astype(int) == y_test) / y_test.size) * 100
print('Accuracy:', accuracy, ‘%')
```

Python

```py
Accuracy: 100.0 %
```

预期地，所有测试数据点都被正确分类。让我们还可视化 SVM 算法在训练期间计算的决策边界，以更好地理解它是如何得出这一分类结果的。

与此同时，迄今为止的代码清单如下：

Python

```py
from sklearn.datasets import make_blobs
from sklearn import model_selection as ms
from numpy import float32
from matplotlib.pyplot import scatter, show, subplots

# Generate a dataset of 2D data points and their ground truth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=15)

# Plot the dataset
scatter(x[:, 0], x[:, 1], c=y_true)
show()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Plot the training and testing datasets
fig, (ax1, ax2) = subplots(1, 2)
ax1.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
ax1.set_title('Training data')
ax2.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
ax2.set_title('Testing data')
show()

# Create a new SVM
svm = ml.SVM_create()

# Set the SVM kernel to linear
svm.setKernel(ml.SVM_LINEAR)

# Train the SVM on the set of training data
svm.train(x_train.astype(float32), ml.ROW_SAMPLE, y_train)

# Predict the target labels of the testing data
_, y_pred = svm.predict(x_test.astype(float32))

# Compute and print the achieved accuracy
accuracy = (sum(y_pred[:, 0].astype(int) == y_test) / y_test.size) * 100
print('Accuracy:', accuracy, '%')
```

为了可视化决策边界，我们将创建许多二维点，构成一个矩形网格，该网格跨越用于测试的数据点占据的空间：

```py
x_bound, y_bound = meshgrid(arange(x_test[:, 0].min() - 1, x_test[:, 0].max() + 1, 0.05),
                            arange(x_test[:, 1].min() - 1, x_test[:, 1].max() + 1, 0.05))
```

接下来，我们将把构成矩形网格的数据点的 x 和 y 坐标组织成一个两列数组，并传递给 `predict` 方法，为每一个数据点生成一个类标签：

Python

```py
bound_points = column_stack((x_bound.reshape(-1, 1), y_bound.reshape(-1, 1))).astype(float32)
_, bound_pred = svm.predict(bound_points)
```

最后，我们可以通过轮廓图来可视化它们，覆盖用于测试的数据点，以确认 SVM 算法计算的决策边界确实是线性的：

Python

```py
contourf(x_bound, y_bound, bound_pred.reshape(x_bound.shape), cmap=cm.coolwarm)
scatter(x_test[:, 0], x_test[:, 1], c=y_test)
show()
```

![](https://machinelearningmastery.com/wp-content/uploads/2023/03/svm_3.png)

SVM 计算的线性决策边界

我们还可以从上图确认，在第一节中提到的，测试数据点已根据它们所在决策边界的一侧被分配了一个类标签。

此外，我们还可以突出显示被识别为支持向量并在决策边界确定中发挥关键作用的训练数据点：

Python

```py
support_vect = svm.getUncompressedSupportVectors()

scatter(x[:, 0], x[:, 1], c=y_true)
scatter(support_vect[:, 0], support_vect[:, 1], c='red')
show()
```

![](https://machinelearningmastery.com/wp-content/uploads/2023/03/svm_4.png)

突出显示的支持向量为红色

生成决策边界并可视化支持向量的完整代码清单如下：

Python

```py
from numpy import float32, meshgrid, arange, column_stack
from matplotlib.pyplot import scatter, show, contourf, cm

x_bound, y_bound = meshgrid(arange(x_test[:, 0].min() - 1, x_test[:, 0].max() + 1, 0.05),
                            arange(x_test[:, 1].min() - 1, x_test[:, 1].max() + 1, 0.05))

bound_points = column_stack((x_bound.reshape(-1, 1), y_bound.reshape(-1, 1))).astype(float32)
_, bound_pred = svm.predict(bound_points)

# Plot the testing set
contourf(x_bound, y_bound, bound_pred.reshape(x_bound.shape), cmap=cm.coolwarm)
scatter(x_test[:, 0], x_test[:, 1], c=y_test)
show()

support_vect = svm.getUncompressedSupportVectors()

scatter(x[:, 0], x[:, 1], c=y_true)
scatter(support_vect[:, 0], support_vect[:, 1], c='red')
show()
```

到目前为止，我们考虑了最简单的情况，即有两个可以明确区分的类。但是，如何区分空间中混合在一起的数据点所属的不太明显可分离的类别，比如以下情况：

Python

```py
# Generate a dataset of 2D data points and their ground truth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=8, random_state=15)
```

![](https://machinelearningmastery.com/wp-content/uploads/2023/03/svm_5.png)

属于两个不同类别的非线性可分数据点

![](https://machinelearningmastery.com/wp-content/uploads/2023/03/svm_6.png)

将非线性可分数据分割为训练集和测试集

在这种情况下，我们可能希望根据两个类别彼此重叠的程度探索不同的选项，例如 (1) 通过增加 *C* 参数的值放宽线性核的边界约束，以在最大化边界和减少误分类之间取得更好的折衷，或者 (2) 使用能够产生非线性决策边界的不同核函数，如径向基函数（RBF）。

在此过程中，我们需要设置 SVM 和正在使用的核函数的几个属性的值：

+   SVM_C_SVC：称为*C-支持向量分类*，此类 SVM 允许对具有不完全分离的类别进行 n 类分类（n $\geq$ 2）（即非线性可分）。使用 `setType` 方法设定。

+   C：处理非线性可分类时异常值的惩罚倍数。使用 `setC` 方法设定。

+   Gamma：决定了 RBF 核函数的半径。较小的 gamma 值导致更宽的半径，可以捕捉远离彼此的数据点的相似性，但可能导致过拟合。较大的 gamma 值导致较窄的半径，只能捕捉附近数据点的相似性，可能导致欠拟合。使用 `setGamma` 方法设定。

在这里，*C* 和 *gamma* 的值被随意设定，但您可以进行进一步的测试，以探索不同数值如何影响最终预测准确性。前述两个选项均使用以下代码达到 85%的预测准确性，但是通过不同的决策边界实现此准确性：

+   使用放宽边界约束的线性核函数：

Python

```py
svm.setKernel(ml.SVM_LINEAR)
svm.setType(ml.SVM_C_SVC)
svm.setC(10)
```

![](https://machinelearningmastery.com/wp-content/uploads/2023/03/svm_7.png)

使用放宽边界约束的线性核函数计算的决策边界

+   使用 RBF 核函数：

Python

```py
svm.setKernel(ml.SVM_RBF)
svm.setType(ml.SVM_C_SVC)
svm.setC(10)
svm.setGamma(0.1)
```

![](https://machinelearningmastery.com/wp-content/uploads/2023/03/svm_8.png)

使用 RBF 核函数计算的决策边界

SVM 参数的选择通常取决于任务和手头的数据，并需要进一步测试以进行相应的调整。

## **进一步阅读**

如果您想深入了解这个主题，本节提供了更多资源。

### **书籍**

+   [OpenCV 的机器学习](https://www.amazon.com/Machine-Learning-OpenCV-Intelligent-processing/dp/1783980281/ref=sr_1_1?crid=3VWMIM65XCS6K&keywords=machine+learning+for+opencv&qid=1678294085&sprefix=machine+learning+for+openc,aps,213&sr=8-1)，2017 年。

+   [使用 Python 掌握 OpenCV 4](https://www.amazon.com/Mastering-OpenCV-Python-practical-processing/dp/1789344913)，2019 年。

### **网站**

+   支持向量机简介，[`docs.opencv.org/4.x/d1/d73/tutorial_introduction_to_svm.html`](https://docs.opencv.org/4.x/d1/d73/tutorial_introduction_to_svm.html)

## **总结**

在本教程中，你学习了如何在自定义的二维数据集上应用 OpenCV 的支持向量机算法。

具体来说，你学到了：

+   支持向量机算法的几个最重要的特性。

+   如何在 OpenCV 中对自定义数据集使用支持向量机算法。

你有任何问题吗？

在下面的评论中提出你的问题，我会尽力回答。****************
