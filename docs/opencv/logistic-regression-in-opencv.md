# OpenCV 中的逻辑回归

> 原文：[`machinelearningmastery.com/logistic-regression-in-opencv/`](https://machinelearningmastery.com/logistic-regression-in-opencv/)

逻辑回归是一种简单但受欢迎的机器学习算法，用于二分类问题，其核心使用逻辑函数或 sigmoid 函数。它也已在 OpenCV 库中实现。

在本教程中，你将学习如何应用 OpenCV 的逻辑回归算法，首先从我们自己生成的自定义二分类数据集开始。然后我们将在随后的教程中将这些技能应用于特定的图像分类应用。

完成本教程后，你将了解到：

+   逻辑回归算法的一些最重要特征。

+   如何在 OpenCV 中对自定义数据集使用逻辑回归算法。

**通过我的书籍 [Machine Learning in OpenCV](https://machinelearning.samcart.com/products/machine-learning-opencv/) 启动你的项目**。它提供了**自学教程**和**可运行的代码**。

让我们开始吧。![](https://machinelearningmastery.com/wp-content/uploads/2023/12/logistic_cover-scaled.jpg)

OpenCV 中的逻辑回归

照片由 [Fabio Santaniello Bruun](https://unsplash.com/photos/aerial-phoography-of-road-near-river-and-green-leafed-trees-Ke-ENe3ByiQ) 提供。保留所有权利。

## **教程概述**

本教程分为两个部分；它们是：

+   逻辑回归的提醒

+   在 OpenCV 中探索逻辑回归

## **逻辑回归的提醒**

关于逻辑回归的话题已经在 Jason Brownlee 的这些教程中讲解得很好了 [[1](https://machinelearningmastery.com/logistic-regression-for-machine-learning/), [2](https://machinelearningmastery.com/logistic-regression-tutorial-for-machine-learning/), [3](https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/)]，但让我们先从回顾一些最重要的点开始：

+   **逻辑回归得名于其核心所使用的函数，即*逻辑函数*（也称为 sigmoid 函数）。**

***尽管名称中包含*回归*一词，逻辑回归实际上是一种用于二分类的方法，或者更简单地说，是解决两类值问题的技术。**

***逻辑回归可以被视为线性回归的扩展，因为它通过使用逻辑函数将特征的线性组合的实际值输出映射（或*压缩*）为范围在 [0, 1] 内的概率值。**

***   **在双类情景中，逻辑回归方法对默认类别的概率进行建模。举个简单的例子，假设我们试图根据花瓣数来区分类别 A 和 B，并且我们将默认类别定为 A。那么，对于一个未见过的输入 X，逻辑回归模型会给出 X 属于默认类别 A 的概率：**

**$$ P(X) = P(A = 1 | X) $$

+   **如果输入 X 的概率 P(X) > 0.5，则 X 被分类为默认类别 A。否则，它被分类为非默认类别 B。**

***   **逻辑回归模型由一组称为系数（或权重）的参数表示，这些参数通过训练数据学习得来。这些系数在训练过程中会被迭代调整，以最小化模型预测与实际类别标签之间的误差。**

***   **系数值可以通过梯度下降法或最大似然估计（MLE）技术在训练过程中进行估计。**

**## **探索 OpenCV 中的逻辑回归**

在深入更复杂的问题之前，让我们从一个简单的二分类任务开始。

正如我们在相关教程中通过其他机器学习算法（如 SVM 算法）所做的那样，我们将生成一个包含 100 个数据点（由`n_samples`指定）的数据集，这些数据点均分为两个高斯簇（由`centers`指定），标准差设置为 5（由`cluster_std`指定）。为了能够复制结果，我们将再次利用`random_state`参数，并将其设置为 15：

Python

```py
# Generate a dataset of 2D data points and their ground truth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=5, random_state=15)

# Plot the dataset
scatter(x[:, 0], x[:, 1], c=y_true)
show()
```

上面的代码应生成以下数据点的图示。你可能会注意到，我们将颜色值设置为真实标签，以便区分属于两个不同类别的数据点：

![](https://machinelearningmastery.com/wp-content/uploads/2023/12/logistic_1.png)

属于两种不同类别的数据点

下一步是将数据集分为训练集和测试集，其中前者用于训练逻辑回归模型，后者用于测试：

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

![](https://machinelearningmastery.com/wp-content/uploads/2023/12/logistic_2.png)

将数据点拆分为训练集和测试集

上面的图像表明，两个类别在训练和测试数据中明显可分。因此，我们期望这个二分类问题对于训练过的线性回归模型来说是一个简单的任务。让我们在 OpenCV 中创建并训练一个逻辑回归模型，以最终查看它在数据集测试部分的表现。

第一步是创建逻辑回归模型本身：

Python

```py
# Create an empty logistic regression model
lr = ml.LogisticRegression_create()
```

在下一步中，我们将选择训练方法，以便在训练过程中更新模型的系数。OpenCV 的实现允许我们在*批量梯度下降*方法和*迷你批量梯度下降*方法之间进行选择。

如果选择了*批量梯度下降*方法，模型的系数将在每次梯度下降算法迭代中使用整个训练数据集进行更新。如果我们处理的是非常大的数据集，那么这种更新模型系数的方法可能会非常耗费计算资源。

### 想开始使用 OpenCV 进行机器学习吗？

立即参加我的免费电子邮件速成课程（包含示例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

更新模型系数的更实际方法，特别是在处理大型数据集时，是选择*迷你批量梯度下降*方法，该方法将训练数据划分为较小的批量（称为迷你批量，因此得名），并通过逐个处理迷你批量来更新模型系数。

我们可以通过使用以下代码行检查 OpenCV 默认的训练方法：

Python

```py
# Check the default training method
print(‘Training Method:’, lr.getTrainMethod())
```

Python

```py
Training Method: 0
```

返回值 0 表示 OpenCV 中的批量梯度下降方法。如果我们想将其更改为迷你批量梯度下降方法，可以将`ml.LogisticRegression_MINI_BATCH`传递给`setTrainMethod`函数，然后设置迷你批量的大小：

Python

```py
# Set the training method to Mini-Batch Gradient Descent and the size of the mini-batch
lr.setTrainMethod(ml.LogisticRegression_MINI_BATCH)
lr.setMiniBatchSize(5)
```

将迷你批量大小设置为 5 意味着训练数据将被分成每个包含 5 个数据点的迷你批量，模型的系数将在处理完每个迷你批量后迭代更新。如果我们将迷你批量的大小设置为训练数据集中样本的总数，这实际上将导致批量梯度下降操作，因为每次迭代时都会一次性处理整个训练数据批量。

接下来，我们将定义在算法终止之前希望运行所选训练算法的迭代次数：

Python

```py
# Set the number of iterations
lr.setIterations(10)
```

我们现在可以在训练数据上训练逻辑回归模型：

Python

```py
# Train the logistic regressor on the set of training data
lr.train(x_train.astype(float32), ml.ROW_SAMPLE, y_train.astype(float32))
```

如前所述，训练过程旨在迭代地调整逻辑回归模型的系数，以最小化模型预测与实际类别标签之间的误差。

我们输入模型的每个训练样本包括两个特征值，分别表示为$x_1$和$x_2$。这意味着我们期望生成的模型由两个系数（每个输入特征一个）和一个定义偏差（或截距）的额外系数定义。

然后可以按照以下方式定义返回模型的概率值$\hat{y}$：

$$ \hat{y} = \sigma( \beta_0 + \beta_1 \; x_1 + \beta_2 \; x_2 ) $$

其中 $\beta_1$ 和 $\beta_2$ 表示模型系数，$\beta_0$ 为偏差，$\sigma$ 为应用于特征线性组合真实值的逻辑（或 sigmoid）函数。

让我们打印出学习到的系数值，看看是否能获取到我们期望的数量：

Python

```py
# Print the learned coefficients
print(lr.get_learnt_thetas())
```

Python

```py
[[-0.02413555 -0.34612912  0.08475047]]
```

我们发现我们按预期检索到了三个值，这意味着我们可以通过以下方式定义最佳模型，以区分我们正在处理的两类样本：

$$ \hat{y} = \sigma( -0.0241 – \; 0.3461 \; x_1 + 0.0848 \; x_2 ) $$

我们可以通过将特征值 $x_1$ 和 $x_2$ 插入上述模型，将一个新的、未见过的数据点分配到两个类别之一。如果模型返回的概率值 > 0.5，我们可以将其视为对类别 0（默认类别）的预测。否则，它就是对类别 1 的预测。

让我们继续尝试在数据集的测试部分上测试这个模型，以查看它对目标类别标签的预测效果：

Python

```py
# Predict the target labels of the testing data
_, y_pred = lr.predict(x_test.astype(float32))

# Compute and print the achieved accuracy
accuracy = (sum(y_pred[:, 0].astype(int) == y_test) / y_test.size) * 100
print('Accuracy:', accuracy, ‘%')
```

Python

```py
Accuracy: 95.0 %
```

我们可以绘制真实值与预测类别的图表，并打印出真实值和预测类别标签，以调查任何误分类情况：

Python

```py
# Plot the ground truth and predicted class labels
fig, (ax1, ax2) = subplots(1, 2)
ax1.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
ax1.set_title('Ground Truth Testing data')
ax2.scatter(x_test[:, 0], x_test[:, 1], c=y_pred)
ax2.set_title('Predicted Testing data’)
show()

# Print the ground truth and predicted class labels of the testing data
print('Ground truth class labels:', y_test, '\n',
      'Predicted class labels:', y_pred[:, 0].astype(int))
```

Python

```py
Ground truth class labels: [1 1 1 1 1 0 0 1 1 1 0 0 1 0 0 1 1 1 1 0] 
Predicted class labels: [1 1 1 1 1 0 0 1 0 1 0 0 1 0 0 1 1 1 1 0]
```

![](https://machinelearningmastery.com/wp-content/uploads/2023/12/logistic_3.png)

测试数据点属于真实类别和预测类别，其中红色圆圈突出显示了一个误分类的数据点

通过这种方式，我们可以看到一个样本在真实数据中原本属于类别 1，但在模型预测中被误分类为类别 0。

整个代码清单如下：

Python

```py
from cv2 import ml
from sklearn.datasets import make_blobs
from sklearn import model_selection as ms
from numpy import float32
from matplotlib.pyplot import scatter, show, subplots

# Generate a dataset of 2D data points and their ground truth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=5, random_state=15)

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

# Create an empty logistic regression model
lr = ml.LogisticRegression_create()

# Check the default training method
print('Training Method:', lr.getTrainMethod())

# Set the training method to Mini-Batch Gradient Descent and the size of the mini-batch
lr.setTrainMethod(ml.LogisticRegression_MINI_BATCH)
lr.setMiniBatchSize(5)

# Set the number of iterations
lr.setIterations(10)

# Train the logistic regressor on the set of training data
lr.train(x_train.astype(float32), ml.ROW_SAMPLE, y_train.astype(float32))

# Print the learned coefficients
print(lr.get_learnt_thetas())

# Predict the target labels of the testing data
_, y_pred = lr.predict(x_test.astype(float32))

# Compute and print the achieved accuracy
accuracy = (sum(y_pred[:, 0].astype(int) == y_test) / y_test.size) * 100
print('Accuracy:', accuracy, '%')

# Plot the ground truth and predicted class labels
fig, (ax1, ax2) = subplots(1, 2)
ax1.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
ax1.set_title('Ground truth testing data')
ax2.scatter(x_test[:, 0], x_test[:, 1], c=y_pred)
ax2.set_title('Predicted testing data')
show()

# Print the ground truth and predicted class labels of the testing data
print('Ground truth class labels:', y_test, '\n',
      'Predicted class labels:', y_pred[:, 0].astype(int))
```

在本教程中，我们考虑了为在 OpenCV 中实现的逻辑回归模型设置两个特定训练参数的值。这些参数定义了使用的训练方法以及训练过程中我们希望运行所选择训练算法的迭代次数。

然而，这些并不是可以为逻辑回归方法设置的唯一参数值。其他参数，如学习率和正则化类型，也可以修改以实现更好的训练准确度。因此，我们建议你探索这些参数，调查不同值如何影响模型的训练和预测准确性。

## **进一步阅读**

本节提供了更多关于该主题的资源，如果你想深入了解，可以参考这些资源。

### **书籍**

+   [《OpenCV 机器学习》](https://www.amazon.com/Machine-Learning-OpenCV-Intelligent-processing/dp/1783980281/ref=sr_1_1?crid=3VWMIM65XCS6K&keywords=machine+learning+for+opencv&qid=1678294085&sprefix=machine+learning+for+openc,aps,213&sr=8-1)，2017 年。

+   [《用 Python 掌握 OpenCV 4》](https://www.amazon.com/Mastering-OpenCV-Python-practical-processing/dp/1789344913)，2019 年。

### **网站**

+   逻辑回归，[`docs.opencv.org/3.4/dc/dd6/ml_intro.html#ml_intro_lr`](https://docs.opencv.org/3.4/dc/dd6/ml_intro.html#ml_intro_lr)

## **总结**

在本教程中，你学习了如何应用 OpenCV 的逻辑回归算法，从我们生成的自定义二分类数据集开始。

具体来说，你学到了：

+   逻辑回归算法的几个最重要特征。

+   如何在 OpenCV 中使用逻辑回归算法处理自定义数据集。

你有任何问题吗？

在下方评论中提问，我会尽力回答。**************
