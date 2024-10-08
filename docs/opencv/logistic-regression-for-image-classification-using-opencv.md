# 使用 OpenCV 的图像分类逻辑回归

> 原文：[`machinelearningmastery.com/logistic-regression-for-image-classification-using-opencv/`](https://machinelearningmastery.com/logistic-regression-for-image-classification-using-opencv/)

在一个[之前的教程](https://machinelearningmastery.com/logistic-regression-in-opencv/)，我们探讨了逻辑回归作为一个简单但流行的二分类机器学习算法，并在 OpenCV 库中实现了它。

到目前为止，我们已经看到逻辑回归如何应用于我们自己生成的自定义二分类数据集。

在本教程中，你将学习标准的逻辑回归算法，如何从本质上设计用于二分类问题，并修改为适应多分类问题，通过将其应用于图像分类任务。

完成本教程后，你将了解：

+   逻辑回归算法的几个重要特性。

+   如何将逻辑回归算法修改以解决多分类问题。

+   如何将逻辑回归应用于图像分类问题。

**快速启动你的项目**，参考我的书籍[《OpenCV 中的机器学习》](https://machinelearning.samcart.com/products/machine-learning-opencv/)。它提供了**自学教程**和**可运行的代码**。

让我们开始吧。 ![](https://machinelearningmastery.com/wp-content/uploads/2023/12/logistic_multi_cover-scaled.jpg)

使用 OpenCV 的图像分类逻辑回归

图片由[David Marcu](https://unsplash.com/photos/landscape-photography-of-mountain-hit-by-sun-rays-78A265wPiO4)提供，版权所有。

## **教程概述**

本教程分为三个部分，分别是：

+   回顾逻辑回归的定义

+   修改逻辑回归以解决多分类问题

+   将逻辑回归应用于多分类问题

## **回顾逻辑回归的定义**

在一个[之前的教程](https://machinelearningmastery.com/logistic-regression-in-opencv/)，我们开始探索 OpenCV 对逻辑回归算法的实现。到目前为止，我们已将其应用于我们生成的自定义二分类数据集，该数据集由聚集在两个簇中的二维点组成。

根据 Jason Brownlee 的逻辑回归教程，我们还回顾了逻辑回归的重要点。我们已经看到逻辑回归与线性回归紧密相关，因为它们都涉及特征的线性组合生成实值输出。然而，逻辑回归通过应用逻辑（或 sigmoid）函数扩展了这一过程。因此得名。它是将实值输出映射到[0, 1]范围内的概率值。这个概率值如果超过 0.5 的阈值则被分类为默认类别；否则，被分类为非默认类别。这使得逻辑回归本质上是一种*二分类*方法。

逻辑回归模型由与输入数据中的特征数量相等的系数以及一个额外的偏置值表示。这些系数和偏置值在训练过程中通过梯度下降或最大似然估计（MLE）技术进行学习。

## **为多类别分类问题修改逻辑回归**

如前一节所述，标准逻辑回归方法仅适用于两类问题，因为逻辑函数及其阈值处理将特征的线性组合的实值输出映射为类别 0 或类别 1。

因此，针对多类别分类问题（或涉及两个以上类别的问题）的逻辑回归需要对标准算法进行修改。

实现这一点的一种技术是将多类别分类问题拆分为多个二分类（或两类）子问题。然后可以将标准逻辑回归方法应用于每个子问题。这就是 OpenCV 实现多类别逻辑回归的方法：

> *… 逻辑回归支持二分类和多分类（对于多分类，它创建多个二分类分类器）。*
> 
> *–* [逻辑回归，OpenCV](https://docs.opencv.org/3.4/dc/dd6/ml_intro.html#ml_intro_lr)

这种类型的技术称为*一对一*方法，它涉及为数据集中每对唯一的类别训练一个独立的二分类器。在预测时，这些二分类器中的每一个都会对其所训练的两个类别中的一个投票，获得最多投票的类别被认为是预测类别。

还有其他技术可以通过逻辑回归实现多类别分类，例如通过*一对其余*方法。你可以在这些教程中找到更多信息 [[1](https://machinelearningmastery.com/multinomial-logistic-regression-with-python/), [2](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)]。

## **将逻辑回归应用于多类别分类问题**

为此，我们将使用[OpenCV 中的数字数据集](https://machinelearningmastery.com/?p=14607&preview=true)，尽管我们开发的代码也可以应用于其他多类数据集。

我们的第一步是加载 OpenCV 数字图像，将其分割成包含 0 到 9 的手写数字的许多子图像，并创建相应的真实标签，以便稍后量化训练的逻辑回归模型的准确性。对于这个特定的示例，我们将 80%的数据集图像分配给训练集，其余 20%的图像分配给测试集：

Python

```py
# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)

# Obtain training and testing datasets from the digits image
digits_train_imgs, digits_train_labels, digits_test_imgs, digits_test_labels = split_data(20, sub_imgs, 0.8)
```

接下来，我们将遵循类似于[前一个教程](https://machinelearningmastery.com/logistic-regression-in-opencv/)中的过程，我们在一个两类数据集上训练并测试了逻辑回归算法，改变了一些参数以适应更大的多类数据集。

第一阶段，再次是创建逻辑回归模型本身：

Python

```py
# Create an empty logistic regression model
lr_digits = ml.LogisticRegression_create()
```

我们可以再次确认 OpenCV 将批量梯度下降作为默认训练方法（由值 0 表示），然后将其更改为迷你批量梯度下降方法，指定迷你批次大小：

Python

```py
# Check the default training method
print('Training Method:', lr_digits.getTrainMethod())

# Set the training method to Mini-Batch Gradient Descent and the size of the mini-batch
lr_digits.setTrainMethod(ml.LogisticRegression_MINI_BATCH)
lr_digits.setMiniBatchSize(400)
```

不同的迷你批次大小肯定会影响模型的训练和预测准确性。

在这个示例中，我们对迷你批次大小的选择基于一种启发式方法，为了实用性，我们尝试了几种迷你批次大小，并确定了一个结果足够高的预测准确度的值（稍后我们将看到）。然而，你应该采取更系统的方法，以便对迷你批次大小做出更明智的决策，以便在计算成本和预测准确性之间提供更好的折衷。

接下来，我们将定义在选择的训练算法终止前我们希望运行的迭代次数：

Python

```py
# Set the number of iterations
lr.setIterations(10)
```

我们现在准备在训练数据上训练逻辑回归模型：

Python

```py
# Train the logistic regressor on the set of training data
lr_digits.train(digits_train_imgs.astype(float32), ml.ROW_SAMPLE, digits_train_labels.astype(float32))
```

在我们的[前一个教程](https://machinelearningmastery.com/logistic-regression-in-opencv/)中，我们打印出了学习到的系数，以了解如何定义最佳分离我们所用的两类样本的模型。

这次我们不会打印出学习到的系数，主要是因为系数太多了，因为我们现在处理的是高维输入数据。

我们将选择打印出学习到的系数数量（而不是系数本身）以及输入特征的数量，以便能够比较这两者：

Python

```py
# Print the number of learned coefficients, and the number of input features
print('Number of coefficients:', len(lr_digits.get_learnt_thetas()[0]))
print('Number of input features:', len(digits_train_imgs[0, :]))
```

Python

```py
Number of coefficients: 401
Number of input features: 400
```

确实，我们发现系数值的数量与输入特征一样多，加上一个额外的偏差值，正如我们预期的那样（我们处理的是$20\times 20$像素的图像，我们使用像素值本身作为输入特征，因此每张图像 400 个特征）。

我们可以通过在数据集的测试部分尝试这个模型来测试它对目标类别标签的预测效果：

Python

```py
# Predict the target labels of the testing data
_, y_pred = lr_digits.predict(digits_test_imgs.astype(float32))

# Compute and print the achieved accuracy
accuracy = (sum(y_pred[:, 0] == digits_test_labels[:, 0]) / digits_test_labels.size) * 100
print('Accuracy:', accuracy, '%')
```

Python

```py
Accuracy: 88.8 %
```

最后一步，让我们生成并绘制一个 [混淆](https://machinelearningmastery.com/confusion-matrix-machine-learning/) 矩阵，以更深入地了解哪些数字被互相混淆：

Python

```py
# Generate and plot confusion matrix
cm = confusion_matrix(digits_test_labels, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
show()
```

![](https://machinelearningmastery.com/wp-content/uploads/2023/12/logistic_multi_1.png)

混淆矩阵

通过这种方式，我们可以看到，性能最低的类别是 5 和 2，它们大多被误认为是 8。

完整的代码清单如下：

Python

```py
from cv2 import ml
from sklearn.datasets import make_blobs
from sklearn import model_selection as ms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from numpy import float32
from matplotlib.pyplot import show
from digits_dataset import split_images, split_data

# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)

# Obtain training and testing datasets from the digits image
digits_train_imgs, digits_train_labels, digits_test_imgs, digits_test_labels = split_data(20, sub_imgs, 0.8)

# Create an empty logistic regression model
lr_digits = ml.LogisticRegression_create()

# Check the default training method
print('Training Method:', lr_digits.getTrainMethod())

# Set the training method to Mini-Batch Gradient Descent and the size of the mini-batch
lr_digits.setTrainMethod(ml.LogisticRegression_MINI_BATCH)
lr_digits.setMiniBatchSize(400)

# Set the number of iterations
lr_digits.setIterations(10)

# Train the logistic regressor on the set of training data
lr_digits.train(digits_train_imgs.astype(float32), ml.ROW_SAMPLE, digits_train_labels.astype(float32))

# Print the number of learned coefficients, and the number of input features
print('Number of coefficients:', len(lr_digits.get_learnt_thetas()[0]))
print('Number of input features:', len(digits_train_imgs[0, :]))

# Predict the target labels of the testing data
_, y_pred = lr_digits.predict(digits_test_imgs.astype(float32))

# Compute and print the achieved accuracy
accuracy = (sum(y_pred[:, 0] == digits_test_labels[:, 0]) / digits_test_labels.size) * 100
print('Accuracy:', accuracy, '%')

# Generate and plot confusion matrix
cm = confusion_matrix(digits_test_labels, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
show()
```

在本教程中，我们将固有设计用于二分类的逻辑回归方法应用于多分类问题。我们使用像素值作为表示每张图像的输入特征，获得了 88.8% 的分类准确率。

那么，测试在从图像中提取的 HOG 描述符上训练逻辑回归算法是否会提高准确率怎么样？

### 想要开始使用 OpenCV 进行机器学习吗？

现在就参加我的免费电子邮件速成课程（包括示例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## **进一步阅读**

本节提供了更多资源，如果你想深入了解这个主题。

### **书籍**

+   [OpenCV 的机器学习](https://www.amazon.com/Machine-Learning-OpenCV-Intelligent-processing/dp/1783980281/ref=sr_1_1?crid=3VWMIM65XCS6K&keywords=machine+learning+for+opencv&qid=1678294085&sprefix=machine+learning+for+openc,aps,213&sr=8-1)，2017 年。

+   [掌握 OpenCV 4 的 Python](https://www.amazon.com/Mastering-OpenCV-Python-practical-processing/dp/1789344913)，2019 年。

### **网站**

+   逻辑回归，[`docs.opencv.org/3.4/dc/dd6/ml_intro.html#ml_intro_lr`](https://docs.opencv.org/3.4/dc/dd6/ml_intro.html#ml_intro_lr)

## **总结**

在本教程中，你学会了如何将固有设计用于二分类的标准逻辑回归算法，修改为适应多分类问题，并将其应用于图像分类任务。

具体来说，你学到了：

+   逻辑回归算法的几个重要特性。

+   逻辑回归算法如何修改以适应多分类问题。

+   如何将逻辑回归应用于图像分类问题。

你有任何问题吗？

在下面的评论中提出你的问题，我会尽力回答。
