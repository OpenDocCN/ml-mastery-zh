# 使用 OpenCV 进行图像分割的常规贝叶斯分类器

> 原文：[`machinelearningmastery.com/normal-bayes-classifier-for-image-segmentation-using-opencv/`](https://machinelearningmastery.com/normal-bayes-classifier-for-image-segmentation-using-opencv/)

朴素贝叶斯算法是一种简单而强大的监督学习技术。其高斯变体已在 OpenCV 库中实现。

在本教程中，你将学习如何应用 OpenCV 的常规贝叶斯算法，首先是在一个自定义的二维数据集上，然后用于图像分割。

完成本教程后，你将知道：

+   应用贝叶斯定理到机器学习中的几个最重要的要点。

+   如何在 OpenCV 中对自定义数据集使用常规贝叶斯算法。

+   如何使用常规贝叶斯算法在 OpenCV 中进行图像分割。

**通过我的书 [Machine Learning in OpenCV](https://machinelearning.samcart.com/products/machine-learning-opencv/) **启动你的项目**。它提供了 **自学教程** 和 **实用代码**。

让我们开始吧。![](https://machinelearningmastery.com/wp-content/uploads/2023/04/bayes_cover-scaled.jpg)

使用 OpenCV 进行图像分割的常规贝叶斯分类器

图片由 [Fabian Irsara](https://unsplash.com/photos/_SwLdgVGfVY) 提供，部分权利保留。

## **教程概述**

本教程分为三部分，它们是：

+   机器学习中应用贝叶斯定理的提醒

+   发现 OpenCV 中的贝叶斯分类

+   使用常规贝叶斯分类器进行图像分割

## **机器学习中应用贝叶斯定理的提醒**

[这个教程](https://machinelearningmastery.com/bayes-theorem-for-machine-learning/)由 Jason Brownlee 编写，深入解释了机器学习中的贝叶斯定理，所以让我们先从复习他教程中的一些最重要的要点开始：

**贝叶斯定理在机器学习中非常有用，因为它提供了一个统计模型来形成数据与[假设](https://machinelearningmastery.com/what-is-a-hypothesis-in-machine-learning/)之间的关系。**

***   **贝叶斯定理表示为 $P(h | D) = P(D | h) * P(h) / P(D)$，该定理指出给定假设为真的概率（记作 $P(h | D)$，也称为*后验概率*）可以通过以下方式计算：**

***   *   **观察到数据的概率给定假设（记作 $P(D | h)$，也称为*似然*）。**

    ***   **假设为真而独立于数据的概率（记作 $P(h)$，也称为*先验概率*）。*****   **观察到数据独立于假设的概率（记作 $P(D)$，也称为*证据*）。******

*******   **贝叶斯定理假设构成输入数据 $D$ 的每个变量（或特征）都依赖于所有其他变量（或特征）。**

***   **在数据分类的背景下，可以将贝叶斯定理应用于计算给定数据样本的类别标签的条件概率问题：$P(class | data) = P(data | class) * P(class) / P(data)$，其中类别标签现在替代了假设。证据 $P(data)$ 是一个常数，可以省略。**

***   **如上所述问题的公式化，估计 $P(data | class)$ 的可能性可能会很困难，因为这要求数据样本的数量足够大，以包含每个类别的所有可能变量（或特征）组合。这种情况很少见，尤其是对于具有许多变量的高维数据。**

***   **上述公式可以简化为所谓的*朴素贝叶斯*，其中每个输入变量被单独处理：$P(class | X_1, X_2, \dots, X_n) = P(X_1 | class) * P(X_2 | class) * \dots * P(X_n | class) * P(class)$**

***   **朴素贝叶斯估计将公式从*依赖*条件概率模型更改为*独立*条件概率模型，其中输入变量（或特征）现在被假定为独立。这一假设在现实世界的数据中很少成立，因此得名*朴素*。**

**## **在 OpenCV 中发现贝叶斯分类**

假设我们处理的输入数据是连续的。在这种情况下，可以使用连续概率分布来建模，例如高斯（或正态）分布，其中每个类别的数据由其均值和标准差建模。

OpenCV 中实现的贝叶斯分类器是普通贝叶斯分类器（也常被称为*高斯朴素贝叶斯*），它假设来自每个类别的输入特征是正态分布的。

> *这个简单的分类模型假设来自每个类别的特征向量是正态分布的（尽管不一定是独立分布的）。*
> 
> *–* OpenCV，[机器学习概述](https://docs.opencv.org/4.x/dc/dd6/ml_intro.html)，2023。

要发现如何在 OpenCV 中使用普通贝叶斯分类器，让我们从在一个简单的二维数据集上测试它开始，就像我们在之前的教程中做过的那样。

### 想开始使用 OpenCV 进行机器学习吗？

现在立即参加我的免费电子邮件速成课程（包括示例代码）。

点击注册并获取课程的免费 PDF 电子书版本。

为此，让我们生成一个包含 100 个数据点的 dataset（由 `n_samples` 指定），这些数据点被均等地划分为 2 个高斯簇（由 `centers` 标识），标准差设为 1.5（由 `cluster_std` 指定）。我们还定义一个 `random_state` 值，以便能够复制结果：

Python

```py
# Generating a dataset of 2D data points and their ground truth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=15)

# Plotting the dataset
scatter(x[:, 0], x[:, 1], c=y_true)
show()
```

上述代码应生成以下数据点的图：

![](https://machinelearningmastery.com/wp-content/uploads/2023/04/bayes_1.png)

包含 2 个高斯簇的数据集的散点图

然后，我们将拆分数据集，将 80%的数据分配到训练集，其余 20%分配到测试集：

Python

```py
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = ms.train_test_split(x, y_true, test_size=0.2, random_state=10)
```

接下来，我们将创建标准贝叶斯分类器，并在将数据类型转换为 32 位浮点数后，对数据集进行训练和测试：

Python

```py
# Create a new Normal Bayes Classifier
norm_bayes = ml.NormalBayesClassifier_create()

# Train the classifier on the training data
norm_bayes.train(x_train.astype(float32), ml.ROW_SAMPLE, y_train)

# Generate a prediction from the trained classifier
ret, y_pred, y_probs = norm_bayes.predictProb(x_test.astype(float32))
```

通过使用`predictProb`方法，我们将获得每个输入向量的预测类别（每个向量存储在输入到标准贝叶斯分类器的数组的每一行）和输出概率。

在上述代码中，预测的类别存储在`y_pred`中，而`y_probs`是一个与类别数（此情况下为两个）相同列数的数组，保存每个输入向量属于每个考虑类别的概率值。分类器返回的每个输入向量的输出概率值的总和应该是 1。然而，这不一定是情况，因为分类器返回的概率值没有通过证据 $P(data)$ 进行标准化，如前节所述，我们已从分母中移除了。

> *相反，报告的是一种可能性，这基本上是条件概率方程的分子，即 p(C) p(M | C)。分母 p(M)不需要计算。*
> 
> – [OpenCV 的机器学习](https://www.amazon.com/Machine-Learning-OpenCV-Intelligent-processing/dp/1783980281/ref=sr_1_1?crid=3VWMIM65XCS6K&keywords=machine+learning+for+opencv&qid=1678294085&sprefix=machine+learning+for+openc,aps,213&sr=8-1)，2017 年。

尽管无论值是否被标准化，通过识别具有最高概率值的类别，可以找到每个输入向量的类别预测。

目前为止的代码列表如下：

Python

```py
from sklearn.datasets import make_blobs
from sklearn import model_selection as ms
from numpy import float32
from matplotlib.pyplot import scatter, show
from cv2 import ml

# Generate a dataset of 2D data points and their ground truth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=15)

# Plot the dataset
scatter(x[:, 0], x[:, 1], c=y_true)
show()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Create a new Normal Bayes Classifier
norm_bayes = ml.NormalBayesClassifier_create()

# Train the classifier on the training data
norm_bayes.train(x_train.astype(float32), ml.ROW_SAMPLE, y_train)

# Generate a prediction from the trained classifier
ret, y_pred, y_probs = norm_bayes.predictProb(x_test.astype(float32))

# Plot the class predictions
scatter(x_test[:, 0], x_test[:, 1], c=y_pred)
show()
```

我们可以看到，在这个简单数据集上训练的标准贝叶斯分类器生成的类别预测是正确的：

![](https://machinelearningmastery.com/wp-content/uploads/2023/04/bayes_2.png)

对测试样本生成的预测的散点图

## **使用标准贝叶斯分类器的图像分割**

贝叶斯分类器在许多应用中被广泛使用，其中包括皮肤分割，它将图像中的皮肤像素与非皮肤像素分开。

我们可以调整上述代码以对图像中的皮肤像素进行分割。为此，我们将使用[皮肤分割数据集](https://archive-beta.ics.uci.edu/dataset/229/skin+segmentation)，该数据集包含 50,859 个皮肤样本和 194,198 个非皮肤样本，用于训练标准贝叶斯分类器。数据集以 BGR 顺序呈现像素值及其对应的类别标签。

在加载数据集后，我们将把 BGR 像素值转换为 HSV（表示色调、饱和度和值），然后使用色调值来训练普通贝叶斯分类器。色调在图像分割任务中通常优于 RGB，因为它代表了未修改的真实颜色，比 RGB 更不容易受到光照变化的影响。在 HSV 颜色模型中，色调值是径向排列的，范围从 0 到 360 度：

Python

```py
from cv2 import ml,
from numpy import loadtxt, float32
from matplotlib.colors import rgb_to_hsv

# Load data from text file
data = loadtxt("Data/Skin_NonSkin.txt", dtype=int)

# Select the BGR values from the loaded data
BGR = data[:, :3]

# Convert to RGB by swapping the array columns
RGB = BGR.copy()
RGB[:, [2, 0]] = RGB[:, [0, 2]]

# Convert RGB values to HSV
HSV = rgb_to_hsv(RGB.reshape(RGB.shape[0], -1, 3) / 255)
HSV = HSV.reshape(RGB.shape[0], 3)

# Select only the hue values
hue = HSV[:, 0] * 360

# Select the labels from the loaded data
labels = data[:, -1]

# Create a new Normal Bayes Classifier
norm_bayes = ml.NormalBayesClassifier_create()

# Train the classifier on the hue values
norm_bayes.train(hue.astype(float32), ml.ROW_SAMPLE, labels)
```

**注意 1**：OpenCV 库提供了 `cvtColor` 方法来进行颜色空间转换，如 [本教程](https://machinelearningmastery.com/?p=14402&preview=true) 所示，但 `cvtColor` 方法要求源图像保持原始形状作为输入。而 Matplotlib 中的 `rgb_to_hsv` 方法则接受形状为 (…, 3) 的 NumPy 数组作为输入，其中数组值需要在 0 到 1 的范围内进行归一化。我们在这里使用后者，因为我们的训练数据由单独的像素组成，而不是以三通道图像的常见形式结构化的。

**注意 2**：普通贝叶斯分类器假设待建模的数据遵循高斯分布。虽然这不是一个严格的要求，但如果数据分布不同，分类器的性能可能会下降。我们可以通过绘制直方图来检查我们处理的数据的分布。例如，如果我们以皮肤像素的色调值为例，我们发现高斯曲线可以描述它们的分布：

Python

```py
from numpy import histogram
from matplotlib.pyplot import bar, title, xlabel, ylabel, show

# Choose the skin-labelled hue values
skin = x[labels == 1]

# Compute their histogram
hist, bin_edges = histogram(skin, range=[0, 360], bins=360)

# Display the computed histogram
bar(bin_edges[:-1], hist, width=4)
xlabel('Hue')
ylabel('Frequency')
title('Histogram of the hue values of skin pixels')
show()
```

![](https://machinelearningmastery.com/wp-content/uploads/2023/04/bayes_3.png)

检查数据的分布

一旦普通贝叶斯分类器经过训练，我们可以在一张图像上进行测试（我们可以考虑 [这张示例图像](https://unsplash.com/photos/gPZ8vbwdV5A) 进行测试）：

Python

```py
from cv2 import imread
from matplotlib.pyplot import show, imshow

# Load a test image
face_img = imread("Images/face.jpg")

# Reshape the image into a three-column array
face_BGR = face_img.reshape(-1, 3)

# Convert to RGB by swapping the array columns
face_RGB = face_BGR.copy()
face_RGB[:, [2, 0]] = face_RGB[:, [0, 2]]

# Convert from RGB to HSV
face_HSV = rgb_to_hsv(face_RGB.reshape(face_RGB.shape[0], -1, 3) / 255)
face_HSV = face_HSV.reshape(face_RGB.shape[0], 3)

# Select only the hue values
face_hue = face_HSV[:, 0] * 360

# Display the hue image
imshow(face_hue.reshape(face_img.shape[0], face_img.shape[1]))
show()

# Generate a prediction from the trained classifier
ret, labels_pred, output_probs = norm_bayes.predictProb(face_hue.astype(float32))

# Reshape array into the input image size and choose the skin-labelled pixels
skin_mask = labels_pred.reshape(face_img.shape[0], face_img.shape[1], 1) == 1

# Display the segmented image
imshow(skin_mask, cmap='gray')
show()
```

结果分割掩码显示了被标记为皮肤（类别标签为 1）的像素。

通过定性分析结果，我们可以看到大多数皮肤像素已被正确标记为皮肤。我们还可以看到一些头发丝（因此是非皮肤像素）被错误地标记为皮肤。如果我们查看它们的色调值，可能会发现这些值与皮肤区域的色调值非常相似，因此导致了错误标记。此外，我们还可以注意到使用色调值的有效性，这些值在面部区域的光照或阴影下仍相对恒定，与原始 RGB 图像中的表现一致：

![](https://machinelearningmastery.com/wp-content/uploads/2023/04/bayes_4.png)

原始图像（左）；色调值（中）；分割后的皮肤像素（右）

你能想到更多的测试方法来尝试普通贝叶斯分类器吗？

## **进一步阅读**

本节提供了更多资源，如果你想深入了解这个主题。

### **书籍**

+   [OpenCV 机器学习](https://www.amazon.com/Machine-Learning-OpenCV-Intelligent-processing/dp/1783980281/ref=sr_1_1?crid=3VWMIM65XCS6K&keywords=machine+learning+for+opencv&qid=1678294085&sprefix=machine+learning+for+openc,aps,213&sr=8-1), 2017.

+   [Python 与 OpenCV 4 实战](https://www.amazon.com/Mastering-OpenCV-Python-practical-processing/dp/1789344913), 2019.

## **总结**

在本教程中，你学习了如何应用 OpenCV 的正态贝叶斯算法，首先在自定义的二维数据集上，然后用于图像分割。

具体来说，你学到了：

+   应用贝叶斯定理到机器学习中的几个最重要的要点。

+   如何在 OpenCV 中使用正态贝叶斯算法处理自定义数据集。

+   如何在 OpenCV 中使用正态贝叶斯算法来进行图像分割。

你有任何问题吗？

在下方评论区留言你的问题，我会尽力回答。
