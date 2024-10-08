# 使用 OpenCV 进行图像分类的随机森林

> 原文：[`machinelearningmastery.com/random-forest-for-image-classification-using-opencv/`](https://machinelearningmastery.com/random-forest-for-image-classification-using-opencv/)

随机森林算法是集成机器学习算法家族的一部分，是袋装决策树的一种流行变体。它也在 OpenCV 库中实现了。

在本教程中，您将学习如何应用 OpenCV 的随机森林算法进行图像分类，从相对简单的纸币数据集开始，然后在 OpenCV 的数字数据集上测试算法。

完成本教程后，您将了解：

+   随机森林算法的几个最重要的特性。

+   如何在 OpenCV 中使用随机森林算法进行图像分类。

**用我的书《OpenCV 中的机器学习》**[Machine Learning in OpenCV](https://machinelearning.samcart.com/products/machine-learning-opencv/) **启动您的项目**。它提供了**自学教程**和**可运行的代码**。

让我们开始吧。 ![](https://machinelearningmastery.com/wp-content/uploads/2023/07/forest_cover-scaled.jpg)

使用 OpenCV 进行图像分类的随机森林

照片由 [Jeremy Bishop](https://unsplash.com/photos/21vV_QxWr6U) 提供，保留部分权利。

## **教程概述**

本教程分为两个部分；它们是：

+   随机森林工作原理的提醒

+   将随机森林算法应用于图像分类

    +   纸币案例研究

    +   数字案例研究

## 随机森林工作原理的提醒

关于随机森林算法的主题已经在 Jason Brownlee 的这些教程中得到很好的解释[[1](https://machinelearningmastery.com/bagging-and-random-forest-ensemble-algorithms-for-machine-learning/)，[2](https://machinelearningmastery.com/implement-random-forest-scratch-python/)]，但让我们首先回顾一些最重要的点：

+   **随机森林是一种集成机器学习算法，称为*袋装*。它是*袋装决策树*的一种流行变体。**

***   **决策树是一个分支模型，由一系列决策节点组成，每个决策节点根据决策规则对数据进行划分。训练决策树涉及贪心地选择最佳分裂点（即最佳划分输入空间的点），通过最小化成本函数来完成。** 

***   **决策树通过贪心方法构建其决策边界，使其容易受到高方差的影响。这意味着训练数据集中的小变化可能导致非常不同的树结构，从而影响模型预测。如果决策树没有被修剪，它还会倾向于捕捉训练数据中的噪声和异常值。这种对训练数据的敏感性使得决策树容易过拟合。**

***   ***集成决策树* 通过结合来自多个决策树的预测来解决这种敏感性，每棵树都在通过替换抽样创建的训练数据的自助样本上进行训练。这种方法的局限性在于相同的贪婪方法训练每棵树，并且某些样本在训练期间可能被多次挑选，这使得树很可能共享相似（或相同）的分割点（因此，结果是相关的树）。**

***   **随机森林算法通过在训练数据的随机子集上训练每棵树来减轻这种相关性，这些子集是通过无替换地随机抽样数据集创建的。这样，贪婪算法只能考虑固定的子集来创建每棵树的分割点，这迫使树之间有所不同。**

***   **在分类问题中，森林中的每棵树都会产生一个预测输出，最终的类别标签是大多数树产生的输出。在回归问题中，最终的输出是所有树产生的输出的平均值。**

**## **将随机森林算法应用于图像分类**

### **纸币案例研究**

我们将首先使用 [这个教程](https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/) 中使用的纸币数据集。

纸币数据集是一个相对简单的数据集，涉及预测给定纸币的真实性。数据集包含 1,372 行，每行代表一个特征向量，包括从纸币照片中提取的四个不同测量值，以及其对应的类别标签（真实或虚假）。

每个特征向量中的值对应于以下内容：

1.  小波变换图像的方差（连续型）

1.  小波变换图像的偏度（连续型）

1.  小波变换图像的峰度（连续型）

1.  图像的熵（连续型）

1.  类别标签（整数）

数据集可以从 [UCI 机器学习库](https://archive.ics.uci.edu/ml/datasets/banknote+authentication) 下载。

### 想要开始使用 OpenCV 进行机器学习吗？

现在参加我的免费电子邮件速成课程（包括示例代码）。

点击注册并同时获取课程的免费 PDF 电子书版本。

如同 Jason 的教程中所示，我们将加载数据集，将其字符串数字转换为浮点数，并将其划分为训练集和测试集：

Python

```py
# Function to load the dataset
def load_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    return dataset

# Function to convert a string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float32(row[column].strip())

# Load the dataset from text file
data = load_csv('Data/data_banknote_authentication.txt')

# Convert the dataset string numbers to float
for i in range(len(data[0])):
    str_column_to_float(data, i)

# Convert list to array
data = array(data)

# Separate the dataset samples from the ground truth
samples = data[:, :4]
target = data[:, -1, newaxis].astype(int32)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = ms.train_test_split(samples, target, test_size=0.2, random_state=10)
```

OpenCV 库在 `ml` 模块中实现了 `RTrees_create` 函数，这将允许我们创建一个空的决策树：

Python

```py
# Create an empty decision tree
rtree = ml.RTrees_create()
```

森林中的所有树木将使用相同的参数值进行训练，尽管是在不同的训练数据子集上。默认参数值可以自定义，但让我们首先使用默认实现。我们将在下一节中回到自定义这些参数值：

Python

```py
# Train the decision tree
rtree.train(x_train, ml.ROW_SAMPLE, y_train)

# Predict the target labels of the testing data
_, y_pred = rtree.predict(x_test)

# Compute and print the achieved accuracy
accuracy = (sum(y_pred.astype(int32) == y_test) / y_test.size) * 100
print('Accuracy:', accuracy[0], '%')
```

Python

```py
Accuracy: 96.72727272727273 %
```

我们已经使用默认实现的随机森林算法在钞票数据集上获得了约**96.73%**的高准确率。

完整的代码列表如下：

```py
from csv import reader
from numpy import array, float32, int32, newaxis
from cv2 import ml
from sklearn import model_selection as ms

# Function to load the dataset
def load_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    return dataset

# Function to convert a string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float32(row[column].strip())

# Load the dataset from text file
data = load_csv('Data/data_banknote_authentication.txt')

# Convert the dataset string numbers to float
for i in range(len(data[0])):
    str_column_to_float(data, i)

# Convert list to array
data = array(data)

# Separate the dataset samples from the ground truth
samples = data[:, :4]
target = data[:, -1, newaxis].astype(int32)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = ms.train_test_split(samples, target, test_size=0.2, random_state=10)

# Create an empty decision tree
rtree = ml.RTrees_create()

# Train the decision tree
rtree.train(x_train, ml.ROW_SAMPLE, y_train)

# Predict the target labels of the testing data
_, y_pred = rtree.predict(x_test)

# Compute and print the achieved accuracy
accuracy = (sum(y_pred.astype(int32) == y_test) / y_test.size) * 100
print('Accuracy:', accuracy[0], '%')
```

### **数字案例研究**

考虑将随机森林应用于 OpenCV 的数字数据集中的图像。

数字数据集仍然相对简单。然而，我们将使用 HOG 方法从其图像中提取的特征向量将具有比钞票数据集中的特征向量更高的维度（81 个特征）。因此，我们可以认为数字数据集比钞票数据集更具挑战性。

我们将首先调查随机森林算法的默认实现如何应对高维数据。

Python

```py
from digits_dataset import split_images, split_data
from feature_extraction import hog_descriptors
from numpy import array, float32
from cv2 import ml

# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)

# Obtain training and testing datasets from the digits image
digits_train_imgs, digits_train_labels, digits_test_imgs, digits_test_labels = split_data(20, sub_imgs, 0.8)

# Convert the image data into HOG descriptors
digits_train_hog = hog_descriptors(digits_train_imgs)
digits_test_hog = hog_descriptors(digits_test_imgs)

# Create an empty decision tree
rtree_digits = ml.RTrees_create()

# Predict the target labels of the testing data
_, digits_test_pred = rtree_digits.predict(digits_test_hog)

# Compute and print the achieved accuracy
accuracy_digits = (sum(digits_test_pred.astype(int) == digits_test_labels) / digits_test_labels.size) * 100
print('Accuracy:', accuracy_digits[0], '%')
```

Python

```py
Accuracy: 81.0 %
```

我们发现默认实现返回的准确度为 81%。

从钞票数据集上获得的准确度下降可能表明，模型的默认实现可能无法学习我们现在处理的高维数据的复杂性。

让我们调查一下，通过更改以下内容是否可以提高准确度：

+   训练算法的终止标准，它考虑了森林中的树木数量，以及模型的估计性能，通过[袋外误差（OOB）](https://machinelearningmastery.com/bagging-and-random-forest-ensemble-algorithms-for-machine-learning/)来衡量。当前的终止标准可以通过`getTermCriteria`方法找到，并通过`setTermCriteria`方法设置。使用后者时，可以通过`TERM_CRITERIA_MAX_ITER`参数设置树的数量，而期望的准确度可以通过`TERM_CRITERIA_EPS`参数指定。

+   森林中每棵树可以达到的最大深度。当前深度可以通过`getMaxDepth`方法找到，并通过`setMaxDepth`方法设置。如果先满足上述终止条件，可能无法达到指定的树深度。

在调整上述参数时，请记住，增加树的数量可以提高模型捕捉训练数据中更复杂细节的能力；这也会线性增加预测时间，并使模型更容易过拟合。因此，谨慎调整参数。

如果我们在创建空决策树后添加以下几行代码，我们可以找到树深度以及终止标准的默认值：

Python

```py
print('Default tree depth:', rtree_digits.getMaxDepth())
print('Default termination criteria:', rtree_digits.getTermCriteria())
```

Python

```py
Default tree depth: 5
Default termination criteria: (3, 50, 0.1)
```

以这种方式，我们可以看到，默认情况下，森林中的每棵树的深度（或层级数）等于 5，而树的数量和期望的准确度分别设置为 50 和 0.1。`getTermCriteria`方法返回的第一个值指的是考虑的终止标准的`type`，其中值为 3 表示基于`TERM_CRITERIA_MAX_ITER`和`TERM_CRITERIA_EPS`的终止。

现在让我们尝试更改上述值，以研究它们对预测准确率的影响。代码列表如下：

Python

```py
from digits_dataset import split_images, split_data
from feature_extraction import hog_descriptors
from numpy import array, float32
from cv2 import ml, TERM_CRITERIA_MAX_ITER, TERM_CRITERIA_EPS

# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)

# Obtain training and testing datasets from the digits image
digits_train_imgs, digits_train_labels, digits_test_imgs, digits_test_labels = split_data(20, sub_imgs, 0.8)

# Convert the image data into HOG descriptors
digits_train_hog = hog_descriptors(digits_train_imgs)
digits_test_hog = hog_descriptors(digits_test_imgs)

# Create an empty decision tree
rtree_digits = ml.RTrees_create()

# Read the default parameter values
print('Default tree depth:', rtree_digits.getMaxDepth())
print('Default termination criteria:', rtree_digits.getTermCriteria())

# Change the default parameter values
rtree_digits.setMaxDepth(15)
rtree_digits.setTermCriteria((TERM_CRITERIA_MAX_ITER + TERM_CRITERIA_EPS, 100, 0.01))

# Train the decision tree
rtree_digits.train(digits_train_hog.astype(float32), ml.ROW_SAMPLE, digits_train_labels)

# Predict the target labels of the testing data
_, digits_test_pred = rtree_digits.predict(digits_test_hog)

# Compute and print the achieved accuracy
accuracy_digits = (sum(digits_test_pred.astype(int) == digits_test_labels) / digits_test_labels.size) * 100
print('Accuracy:', accuracy_digits[0], ‘%')
```

Python

```py
Accuracy: 94.1 %
```

我们可能会看到，新设置的参数值将预测准确率提高到了 94.1%。

这些参数值在这里是随意设置的，以说明这个例子。然而，始终建议采取更系统的方法来调整模型的参数，并调查每个参数对性能的影响。

## **进一步阅读**

本节提供了更多关于此主题的资源，如果你想更深入了解的话。

### **书籍**

+   [OpenCV 的机器学习](https://www.amazon.com/Machine-Learning-OpenCV-Intelligent-processing/dp/1783980281/ref=sr_1_1?crid=3VWMIM65XCS6K&keywords=machine+learning+for+opencv&qid=1678294085&sprefix=machine+learning+for+openc,aps,213&sr=8-1)，2017 年。

+   [使用 Python 掌握 OpenCV 4](https://www.amazon.com/Mastering-OpenCV-Python-practical-processing/dp/1789344913)，2019 年。

### **网站**

+   随机森林，[`www.stat.berkeley.edu/users/breiman/RandomForests/reg_home.htm`](https://www.stat.berkeley.edu/users/breiman/RandomForests/reg_home.htm)

## **总结**

在本教程中，你学会了如何应用 OpenCV 的随机森林算法进行图像分类，从一个相对*简单*的钞票数据集开始，然后在 OpenCV 的数字数据集上测试该算法。

具体来说，你学到了：

+   随机森林算法的一些最重要特征。

+   如何在 OpenCV 中使用随机森林算法进行图像分类。

你有任何问题吗？

在下面的评论中提问，我会尽力回答。************
