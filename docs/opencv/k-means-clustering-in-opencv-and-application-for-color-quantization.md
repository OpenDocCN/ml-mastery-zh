# OpenCV 中的 k-Means 聚类及其在颜色量化中的应用

> 原文：[`machinelearningmastery.com/k-means-clustering-in-opencv-and-application-for-color-quantization/`](https://machinelearningmastery.com/k-means-clustering-in-opencv-and-application-for-color-quantization/)

k-means 聚类算法是一种无监督机器学习技术，旨在将相似的数据分组到不同的集群中，以揭示数据中的模式，这些模式可能在肉眼观察下并不明显。

这可能是数据聚类中最广为人知的算法，并已在 OpenCV 库中实现。

在本教程中，您将学习如何应用 OpenCV 的 k-means 聚类算法进行图像的颜色量化。

完成本教程后，您将了解：

+   了解数据聚类在机器学习中的含义。

+   在 OpenCV 中将 k-means 聚类算法应用于包含不同数据集群的简单二维数据集。

+   如何在 OpenCV 中应用 k-means 聚类算法进行图像的颜色量化。

**通过我的书籍** [《OpenCV 中的机器学习》](https://machinelearning.samcart.com/products/machine-learning-opencv/) **来启动您的项目**。它提供了**自学教程**和**可运行的代码**。

让我们开始吧。 ![](https://machinelearningmastery.com/wp-content/uploads/2023/03/kmeans_cover-scaled.jpg)

使用 OpenCV 进行颜色量化的 k-Means 聚类

图片由 [Billy Huynh](https://unsplash.com/photos/W8KTS-mhFUE) 提供，保留部分权利。

## **教程概述**

本教程分为三部分，它们是：

+   聚类作为一种无监督机器学习任务

+   在 OpenCV 中发现 k-Means 聚类

+   使用 k-Means 进行颜色量化

## **聚类作为一种无监督机器学习任务**

聚类分析是一种无监督学习技术。

它涉及将数据自动分组到不同的组（或集群）中，其中每个集群中的数据相似，但与其他集群的数据不同。它旨在揭示数据中的模式，这些模式在聚类之前可能并不明显。

有许多不同的聚类算法，如 [本教程](https://machinelearningmastery.com/clustering-algorithms-with-python/) 所述，其中 k-means 聚类是最广为人知的算法之一。

k-means 聚类算法处理无标签的数据点。它试图将数据点分配到 *k* 个集群中，每个数据点属于离其最近的集群中心的集群，而每个集群的中心被视为属于该集群的数据点的均值。该算法要求用户提供 k 的值作为输入，因此需要事先知道或根据数据调整这个值。

## **在 OpenCV 中发现 k-Means 聚类**

在深入更复杂的任务之前，我们首先考虑将 k-means 聚类应用于包含不同数据集群的简单二维数据集。

为此，我们将生成一个由 100 个数据点（由`n_samples`指定）组成的数据集，这些数据点平均分配到 5 个高斯聚类（由`centers`标识），标准差设置为 1.5（由`cluster_std`确定）。为了能够复制结果，我们还定义一个`random_state`的值，我们将其设置为 10：

Python

```py
# Generating a dataset of 2D data points and their ground truth labels
x, y_true = make_blobs(n_samples=100, centers=5, cluster_std=1.5, random_state=10)

# Plotting the dataset
scatter(x[:, 0], x[:, 1])
show()
```

上面的代码应生成以下数据点的图表：

![](https://machinelearningmastery.com/wp-content/uploads/2023/03/kmeans_1-e1678295533468.png)

由 5 个高斯聚类组成的数据集的散点图

### 想要开始使用 OpenCV 进行机器学习吗？

现在就参加我的免费电子邮件速成课程（附样例代码）。

点击注册并获取课程的免费 PDF 电子书版本。

如果我们查看这个图表，我们可能已经可以直观地区分一个聚类与另一个聚类，这意味着这应该是一个足够简单的任务，适合 k-means 聚类算法。

在 OpenCV 中，k-means 算法不属于`ml`模块，但可以直接调用。为了能够使用它，我们需要为其输入参数指定值，如下所示：

+   输入，未标记的`data`。

+   所需的聚类数量`K`。

+   终止标准`TERM_CRITERIA_EPS`和`TERM_CRITERIA_MAX_ITER`，分别定义了所需的准确性和最大迭代次数，当达到这些标准时，算法迭代应停止。

+   `attempts`的数量，表示算法将执行的次数，每次使用不同的初始标记以寻找最佳聚类紧凑度。

+   聚类中心的初始化方式，无论是随机的、用户提供的，还是通过诸如 kmeans++ 的中心初始化方法，如参数`flags`所指定的。

OpenCV 中的 k-means 聚类算法返回：

+   每个聚类的`compactness`，计算为每个数据点到其对应聚类中心的平方距离之和。较小的紧凑度值表明数据点分布得更接近其对应的聚类中心，因此聚类更紧凑。

+   预测的聚类标签`y_pred`，将每个输入数据点与其对应的聚类关联起来。

+   每个数据点聚类的`centers`坐标。

现在我们将 k-means 聚类算法应用于之前生成的数据集。请注意，我们将输入数据转换为`float32`类型，这是 OpenCV 中`kmeans()`函数所期望的：

Python

```py
# Specify the algorithm's termination criteria
criteria = (TERM_CRITERIA_MAX_ITER + TERM_CRITERIA_EPS, 10, 1.0)

# Run the k-means clustering algorithm on the input data
compactness, y_pred, centers = kmeans(data=x.astype(float32), K=5, bestLabels=None, criteria=criteria, attempts=10, flags=KMEANS_RANDOM_CENTERS)

# Plot the data clusters, each having a different color, together with the corresponding cluster centers
scatter(x[:, 0], x[:, 1], c=y_pred)
scatter(centers[:, 0], centers[:, 1], c='red')
show()
```

上面的代码生成了以下图表，其中每个数据点根据其分配的聚类进行着色，聚类中心用红色标记：

![](https://machinelearningmastery.com/wp-content/uploads/2023/03/kmeans_2-e1678295496224.png)

使用 k-means 聚类识别聚类的数据集的散点图

完整的代码列表如下：

Python

```py
from cv2 import kmeans, TERM_CRITERIA_MAX_ITER, TERM_CRITERIA_EPS, KMEANS_RANDOM_CENTERS
from numpy import float32
from matplotlib.pyplot import scatter, show
from sklearn.datasets import make_blobs

# Generate a dataset of 2D data points and their ground truth labels
x, y_true = make_blobs(n_samples=100, centers=5, cluster_std=1.5, random_state=10)

# Plot the dataset
scatter(x[:, 0], x[:, 1])
show()

# Specify the algorithm's termination criteria
criteria = (TERM_CRITERIA_MAX_ITER + TERM_CRITERIA_EPS, 10, 1.0)

# Run the k-means clustering algorithm on the input data
compactness, y_pred, centers = kmeans(data=x.astype(float32), K=5, bestLabels=None, criteria=criteria, attempts=10, flags=KMEANS_RANDOM_CENTERS)

# Plot the data clusters, each having a different colour, together with the corresponding cluster centers
scatter(x[:, 0], x[:, 1], c=y_pred)
scatter(centers[:, 0], centers[:, 1], c='red')
show()
```

## **使用 k-means 的颜色量化**

k-means 聚类的一种应用是图像的颜色量化。

颜色量化指的是减少图像表示中使用的不同颜色数量的过程。

> *颜色量化对于在只能显示有限颜色数量的设备上显示多色图像至关重要，这通常是由于内存限制，并且能够高效地压缩某些类型的图像。*
> 
> *[*颜色量化*](https://en.wikipedia.org/wiki/Color_quantization)*，2023。**

*在这种情况下，我们将提供给 k-means 聚类算法的数据点是每个图像像素的 RGB 值。正如我们将看到的，我们将以 $M \times 3$ 数组的形式提供这些值，其中 $M$ 表示图像中的像素数量。*

让我们在[这张图片](https://unsplash.com/photos/rgP93cPsVEc)上尝试 k-means 聚类算法，我将其命名为 *bricks.jpg*：

![](https://machinelearningmastery.com/wp-content/uploads/2023/03/kmeans_4.jpg)

图像中突出的主色是红色、橙色、黄色、绿色和蓝色。然而，许多阴影和光斑为主色引入了额外的色调和颜色。

我们将首先使用 OpenCV 的 `imread` 函数读取图像。

[请记住](https://machinelearningmastery.com/?p=14402&preview=true)，OpenCV 以 BGR 而非 RGB 顺序加载此图像。在将其输入 k-means 聚类算法之前，不需要将其转换为 RGB，因为后者会根据像素值的顺序仍然分组相似的颜色。然而，由于我们使用 Matplotlib 来显示图像，因此我们会将其转换为 RGB，以便稍后可以正确显示量化结果：

Python

```py
# Read image
img = imread('Images/bricks.jpg')

# Convert it from BGR to RGB
img_RGB = cvtColor(img, COLOR_BGR2RGB)
```

正如我们之前提到的，下一步涉及将图像重塑为 $M \times 3$ 数组，然后我们可以应用 k-means 聚类算法，将结果数组值分配到多个聚类中，这些聚类对应于我们上面提到的主色数。

在下面的代码片段中，我还包括了一行代码，用于打印图像中总像素数的唯一 RGB 像素值。我们发现，从 14,155,776 像素中，我们有 338,742 个唯一 RGB 值，这个数字相当可观：

Python

```py
# Reshape image to an Mx3 array
img_data = img_RGB.reshape(-1, 3)

# Find the number of unique RGB values
print(len(unique(img_data, axis=0)), 'unique RGB values out of', img_data.shape[0], 'pixels')

# Specify the algorithm's termination criteria
criteria = (TERM_CRITERIA_MAX_ITER + TERM_CRITERIA_EPS, 10, 1.0)

# Run the k-means clustering algorithm on the pixel values
compactness, labels, centers = kmeans(data=img_data.astype(float32), K=5, bestLabels=None, criteria=criteria, attempts=10, flags=KMEANS_RANDOM_CENTERS)
```

此时，我们将应用聚类中心的实际 RGB 值到预测的像素标签上，并将结果数组重塑为原始图像的形状，然后显示它：

Python

```py
# Apply the RGB values of the cluster centers to all pixel labels
colours = centers[labels].reshape(-1, 3)

# Find the number of unique RGB values
print(len(unique(colours, axis=0)), 'unique RGB values out of', img_data.shape[0], 'pixels')

# Reshape array to the original image shape
img_colours = colours.reshape(img_RGB.shape)

# Display the quantized image
imshow(img_colours.astype(uint8))
show()
```

再次打印量化图像中的唯一 RGB 值，我们发现这些值已经减少到我们为 k-means 算法指定的聚类数量：

Python

```py
5 unique RGB values out of 14155776 pixels
```

如果我们查看颜色量化图像，会发现黄色和橙色砖块的像素被分组到同一个簇中，这可能是由于它们的 RGB 值相似。相比之下，其中一个簇则聚合了阴影区域的像素：

![](https://machinelearningmastery.com/wp-content/uploads/2023/03/kmeans_3-e1678295458313.png)

使用 5 个簇的 k-means 聚类进行颜色量化图像

现在尝试更改指定 k-means 聚类算法簇数的值，并调查其对量化结果的影响。

完整的代码清单如下：

Python

```py
from cv2 import kmeans, TERM_CRITERIA_MAX_ITER, TERM_CRITERIA_EPS, KMEANS_RANDOM_CENTERS, imread, cvtColor, COLOR_BGR2RGB
from numpy import float32, uint8, unique
from matplotlib.pyplot import show, imshow

# Read image
img = imread('Images/bricks.jpg')

# Convert it from BGR to RGB
img_RGB = cvtColor(img, COLOR_BGR2RGB)

# Reshape image to an Mx3 array
img_data = img_RGB.reshape(-1, 3)

# Find the number of unique RGB values
print(len(unique(img_data, axis=0)), 'unique RGB values out of', img_data.shape[0], 'pixels')

# Specify the algorithm's termination criteria
criteria = (TERM_CRITERIA_MAX_ITER + TERM_CRITERIA_EPS, 10, 1.0)

# Run the k-means clustering algorithm on the pixel values
compactness, labels, centers = kmeans(data=img_data.astype(float32), K=5, bestLabels=None, criteria=criteria, attempts=10, flags=KMEANS_RANDOM_CENTERS)

# Apply the RGB values of the cluster centers to all pixel labels
colours = centers[labels].reshape(-1, 3)

# Find the number of unique RGB values
print(len(unique(colours, axis=0)), 'unique RGB values out of', img_data.shape[0], 'pixels')

# Reshape array to the original image shape
img_colours = colours.reshape(img_RGB.shape)

# Display the quantized image
imshow(img_colours.astype(uint8))
show()
```

## **进一步阅读**

本节提供了更多资源，如果你想深入了解该主题。

### **书籍**

+   [《OpenCV 机器学习》](https://www.amazon.com/Machine-Learning-OpenCV-Intelligent-processing/dp/1783980281/ref=sr_1_1?crid=3VWMIM65XCS6K&keywords=machine+learning+for+opencv&qid=1678294085&sprefix=machine+learning+for+openc,aps,213&sr=8-1)，2017 年。

### **网站**

+   《10 种 Python 聚类算法》，[`machinelearningmastery.com/clustering-algorithms-with-python/`](https://machinelearningmastery.com/clustering-algorithms-with-python/)

+   OpenCV 中的 K-Means 聚类，[`docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html`](https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html)

+   k-means 聚类，[`en.wikipedia.org/wiki/K-means_clustering`](https://en.wikipedia.org/wiki/K-means_clustering)

## **总结**

在本教程中，你学习了如何应用 OpenCV 的 k-means 聚类算法进行图像的颜色量化。

具体来说，你学到了：

+   机器学习中的数据聚类是什么。

+   在 OpenCV 中应用 k-means 聚类算法于一个包含不同数据簇的简单二维数据集。

+   如何在 OpenCV 中应用 k-means 聚类算法进行图像的颜色量化。

你有任何问题吗？

在下面的评论中提问，我会尽力回答。
