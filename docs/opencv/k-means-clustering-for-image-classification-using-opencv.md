# 使用 OpenCV 进行图像分类的 k-Means 聚类

> 原文：[`machinelearningmastery.com/k-means-clustering-for-image-classification-using-opencv/`](https://machinelearningmastery.com/k-means-clustering-for-image-classification-using-opencv/)

在 [之前的教程](https://machinelearningmastery.com/?p=14752&preview=true)中，我们探讨了如何使用 k-means 聚类算法作为一种无监督机器学习技术，旨在将相似数据分组到不同的簇中，从而揭示数据中的模式。

到目前为止，我们已经看到如何将 k-means 聚类算法应用于一个包含不同簇的简单二维数据集，以及图像颜色量化的问题。

在本教程中，你将学习如何应用 OpenCV 的 k-means 聚类算法进行图像分类。

完成本教程后，你将会了解：

+   为什么 k-means 聚类可以应用于图像分类。

+   将 k-means 聚类算法应用于 OpenCV 中的数字数据集，以进行图像分类。

+   如何减少由于倾斜造成的数字变异，以提高 k-means 聚类算法在图像分类中的准确性。

**启动你的项目**，请参考我的书籍 [《OpenCV 中的机器学习》](https://machinelearning.samcart.com/products/machine-learning-opencv/)。它提供了**自学教程**和**实用代码**。

让我们开始吧。![](https://machinelearningmastery.com/wp-content/uploads/2023/03/kmeans_class_cover-scaled.jpg)

使用 OpenCV 进行图像分类的 k-Means 聚类

图片由 [Jeremy Thomas](https://unsplash.com/photos/E0AHdsENmDg) 提供，部分权利保留。

## **教程概述**

本教程分为两个部分，它们是：

+   k-Means 聚类作为一种无监督机器学习技术的回顾

+   应用 k-Means 聚类于图像分类

## **k-Means 聚类作为一种无监督机器学习技术的回顾**

在 [之前的教程](https://machinelearningmastery.com/?p=14752&preview=true)中，我们介绍了 k-means 聚类作为一种无监督学习技术。

我们已经看到该技术涉及自动将数据分组到不同的组（或簇）中，其中每个簇中的数据彼此相似，但与其他簇中的数据不同。其目标是揭示数据中的模式，这些模式在聚类之前可能并不明显。

我们已经将 k-means 聚类算法应用于一个包含五个簇的简单二维数据集，以标记每个簇中的数据点，并随后应用于颜色量化任务，在该任务中，我们使用此算法来减少表示图像的不同颜色数量。

在本教程中，我们将再次利用 k-means 聚类在数据中揭示隐藏结构的能力，将其应用于图像分类任务。

对于这样的任务，我们将使用在[之前的教程](https://machinelearningmastery.com/?p=14607&preview=true)中介绍的 OpenCV 数字数据集，我们将尝试以无监督的方式（即不使用实际标签信息）对类似手写数字的图像进行分组。

## **将 k-Means 聚类应用于图像分类**

我们首先需要加载 OpenCV 数字图像，将其分成许多包含从 0 到 9 的手写数字的子图像，并创建相应的实际标签，这将使我们能够量化 k-means 聚类算法的性能：

Python

```py
# Load the digits image and divide it into sub-images
img, sub_imgs = split_images('Images/digits.png', 20)

# Create the ground truth labels
imgs, labels_true, _, _ = split_data(20, sub_imgs, 1.0)
```

返回的`imgs`数组包含 5000 个子图像，以平铺的一维向量形式组织，每个图像由 400 个像素组成：

Python

```py
# Check the shape of the 'imgs' array
print(imgs.shape)
```

Python

```py
(5000, 400)
```

k-means 算法可以使用与我们在颜色量化示例中使用的输入参数相同的输入参数，唯一的例外是我们需要将`imgs`数组作为输入数据，并将`K`簇的值设置为 10（即我们可用的数字数量）：

Python

```py
# Specify the algorithm's termination criteria
criteria = (TERM_CRITERIA_MAX_ITER + TERM_CRITERIA_EPS, 10, 1.0)

# Run the k-means clustering algorithm on the image data
compactness, clusters, centers = kmeans(data=imgs.astype(float32), K=10, bestLabels=None, criteria=criteria, attempts=10, flags=KMEANS_RANDOM_CENTERS)
```

`kmeans`函数返回一个`centers`数组，该数组应包含每个簇的代表性图像。返回的`centers`数组形状为 10$\times$400，这意味着我们需要先将其重塑为 20$\times$20 像素的图像，然后再进行可视化：

Python

```py
# Reshape array into 20x20 images
imgs_centers = centers.reshape(-1, 20, 20)

# Visualise the cluster centers
fig, ax = subplots(2, 5)

for i, center in zip(ax.flat, imgs_centers):
    i.imshow(center)

show()
```

簇中心的代表性图像如下：

![](https://machinelearningmastery.com/wp-content/uploads/2023/03/kmeans_class_1.png)

k-means 算法找到的簇中心的代表性图像

k-means 算法生成的簇中心确实类似于 OpenCV 数字数据集中包含的手写数字，这一点非常值得注意。

你可能还会注意到，簇中心的顺序不一定按照 0 到 9 的数字顺序。这是因为 k-means 算法可以将相似的数据聚类在一起，但没有顺序的概念。然而，这在将预测标签与实际标签进行比较时也会带来问题。这是因为实际标签是根据图像中的数字生成的。然而，k-means 算法生成的簇标签不一定遵循相同的约定。为了解决这个问题，我们需要*重新排序*簇标签：

Python

```py
# Found cluster labels
labels = array([2, 0, 7, 5, 1, 4, 6, 9, 3, 8])

labels_pred = zeros(labels_true.shape, dtype='int')

# Re-order the cluster labels
for i in range(10):
    mask = clusters.ravel() == i
    labels_pred[mask] = labels[i]
```

现在我们准备计算算法的准确性，方法是找到与实际标签对应的预测标签的百分比：

Python

```py
# Calculate the algorithm's accuracy
accuracy = (sum(labels_true == labels_pred) / labels_true.size) * 100

# Print the accuracy
print("Accuracy: {0:.2f}%".format(accuracy[0]))
```

Python

```py
Accuracy: 54.80%
```

到目前为止的完整代码列表如下：

Python

```py
from cv2 import kmeans, TERM_CRITERIA_MAX_ITER, TERM_CRITERIA_EPS, KMEANS_RANDOM_CENTERS
from numpy import float32, array, zeros
from matplotlib.pyplot import show, imshow, subplots
from digits_dataset import split_images, split_data

# Load the digits image and divide it into sub-images
img, sub_imgs = split_images('Images/digits.png', 20)

# Create the ground truth labels
imgs, labels_true, _, _ = split_data(20, sub_imgs, 1.0)

# Check the shape of the 'imgs' array
print(imgs.shape)

# Specify the algorithm's termination criteria
criteria = (TERM_CRITERIA_MAX_ITER + TERM_CRITERIA_EPS, 10, 1.0)

# Run the k-means clustering algorithm on the image data
compactness, clusters, centers = kmeans(data=imgs.astype(float32), K=10, bestLabels=None, criteria=criteria, attempts=10, flags=KMEANS_RANDOM_CENTERS)

# Reshape array into 20x20 images
imgs_centers = centers.reshape(-1, 20, 20)

# Visualise the cluster centers
fig, ax = subplots(2, 5)

for i, center in zip(ax.flat, imgs_centers):
    i.imshow(center)

show()

# Cluster labels
labels = array([2, 0, 7, 5, 1, 4, 6, 9, 3, 8])

labels_pred = zeros(labels_true.shape, dtype='int')

# Re-order the cluster labels
for i in range(10):
    mask = clusters.ravel() == i
    labels_pred[mask] = labels[i]

# Calculate the algorithm's accuracy
accuracy = (sum(labels_true == labels_pred) / labels_true.size) * 100
```

现在，让我们打印出[混淆矩阵](https://machinelearningmastery.com/confusion-matrix-machine-learning/)，以深入了解哪些数字被误认为了其他数字：

Python

```py
from sklearn.metrics import confusion_matrix

# Print confusion matrix
print(confusion_matrix(labels_true, labels_pred))
```

Python

```py
[[399   0   2  28   2  23  27   0  13   6]
 [  0 351   0   1   0   1   0   0   1 146]
 [  4  57 315  26   3  14   4   4  38  35]
 [  2   4   3 241   9  12   3   1 141  84]
 [  0   8  58   0 261  27   3  93   0  50]
 [  3   4   0 150  27 190  12   1  53  60]
 [  6  13  83   4   0  13 349   0   0  32]
 [  0  22   3   1 178  10   0 228   0  58]
 [  0  15  16  85  15  18   3   8 260  80]
 [  2   4  23   7 228   8   1 161   1  65]]
```

混淆矩阵需要按如下方式解读：

![](https://machinelearningmastery.com/wp-content/uploads/2023/11/kmeans_class_2.png)

解读混淆矩阵

对角线上的值表示正确预测的数字数量，而非对角线上的值表示每个数字的误分类情况。我们可以看到表现最好的数字是*0*，其对角线值最高且误分类很少。表现最差的数字是*9*，因为它与许多其他数字（主要是 4）有最多的误分类。我们还可以看到，*7* 大多数被误认为 4，而*8* 则大多数被误认为*3*。

这些结果并不令人惊讶，因为如果我们查看数据集中的数字，可能会发现几个不同数字的曲线和倾斜使它们彼此相似。为了研究减少数字变化的效果，我们引入一个函数`deskew_image()`，该函数基于从图像矩计算出的倾斜度对图像应用仿射变换：

Python

```py
from cv2 import (kmeans, TERM_CRITERIA_MAX_ITER, TERM_CRITERIA_EPS, KMEANS_RANDOM_CENTERS, moments, warpAffine, INTER_CUBIC, WARP_INVERSE_MAP)
from numpy import float32, array, zeros
from matplotlib.pyplot import show, imshow, subplots
from digits_dataset import split_images, split_data
from sklearn.metrics import confusion_matrix

# Load the digits image and divide it into sub-images
img, sub_imgs = split_images('Images/digits.png', 20)

# Create the ground truth labels
imgs, labels_true, _, _ = split_data(20, sub_imgs, 1.0)

# De-skew all dataset images
imgs_deskewed = zeros(imgs.shape)

for i in range(imgs_deskewed.shape[0]):
    new = deskew_image(imgs[i, :].reshape(20, 20))
    imgs_deskewed[i, :] = new.reshape(1, -1)

# Specify the algorithm's termination criteria
criteria = (TERM_CRITERIA_MAX_ITER + TERM_CRITERIA_EPS, 10, 1.0)

# Run the k-means clustering algorithm on the de-skewed image data
compactness, clusters, centers = kmeans(data=imgs_deskewed.astype(float32), K=10, bestLabels=None, criteria=criteria, attempts=10, flags=KMEANS_RANDOM_CENTERS)

# Reshape array into 20x20 images
imgs_centers = centers.reshape(-1, 20, 20)

# Visualise the cluster centers
fig, ax = subplots(2, 5)

for i, center in zip(ax.flat, imgs_centers):
    i.imshow(center)

show()

# Cluster labels
labels = array([9, 5, 6, 4, 2, 3, 7, 8, 1, 0])

labels_pred = zeros(labels_true.shape, dtype='int')

# Re-order the cluster labels
for i in range(10):
    mask = clusters.ravel() == i
    labels_pred[mask] = labels[i]

# Calculate the algorithm's accuracy
accuracy = (sum(labels_true == labels_pred) / labels_true.size) * 100

# Print the accuracy
print("Accuracy: {0:.2f}%".format(accuracy[0]))

# Print confusion matrix
print(confusion_matrix(labels_true, labels_pred))

def deskew_image(img):

    # Calculate the image moments
    img_moments = moments(img)

    # Moment m02 indicates how much the pixel intensities are spread out along the vertical axis
    if abs(img_moments['mu02']) > 1e-2:

        # Calculate the image skew
        img_skew = (img_moments['mu11'] / img_moments['mu02'])

        # Generate the transformation matrix
        # (We are here tweaking slightly the approximation of vertical translation due to skew by making use of a
        # scaling factor of 0.6, because we empirically found that this value worked better for this application)
        m = float32([[1, img_skew, -0.6 * img.shape[0] * img_skew], [0, 1, 0]])

        # Apply the transformation matrix to the image
        img_deskew = warpAffine(src=img, M=m, dsize=img.shape, flags=INTER_CUBIC | WARP_INVERSE_MAP)

    else:

        # If the vertical spread of pixel intensities is small, return a copy of the original image
        img_deskew = img.copy()

    return img_deskew
```

Python

```py
Accuracy: 70.92%

[[400   1   5   1   2  58  27   1   1   4]
 [  0 490   1   1   0   1   2   0   1   4]
 [  5  27 379  28  10   2   3   4  30  12]
 [  1  27   7 360   7  44   2   9  31  12]
 [  1  29   3   0 225   0  13   1   0 228]
 [  5  12   1  14  24 270  11   0   7 156]
 [  8  40   6   0   6   8 431   0   0   1]
 [  0  39   2   0  48   0   0 377   4  30]
 [  2  32   3  21   8  77   2   0 332  23]
 [  5  13   1   5 158   5   2  28   1 282]]
```

去偏斜函数对某些数字具有如下效果：

![](https://machinelearningmastery.com/wp-content/uploads/2023/03/kmeans_class_3-scaled.jpg)

第一列展示了原始数据集图像，而第二列展示了修正倾斜后的图像。

值得注意的是，当减少数字的倾斜度时，准确率上升至 70.92%，而簇中心变得更能代表数据集中的数字：

![](https://machinelearningmastery.com/wp-content/uploads/2023/03/kmeans_class_4.png)

k-均值算法找到的簇中心的代表性图像

这个结果显示，倾斜是导致我们在未进行修正时准确率损失的重要因素。

你能想到其他可能引入的预处理步骤来提高准确率吗？

### 想开始学习使用 OpenCV 进行机器学习吗？

现在就参加我的免费电子邮件速成课程（包含示例代码）。

点击注册，还可以免费获得课程的 PDF 电子书版本。

## **进一步阅读**

本节提供了更多相关资源，如果你想深入了解的话。

### **书籍**

+   [Machine Learning for OpenCV](https://www.amazon.com/Machine-Learning-OpenCV-Intelligent-processing/dp/1783980281/ref=sr_1_1?crid=3VWMIM65XCS6K&keywords=machine+learning+for+opencv&qid=1678294085&sprefix=machine+learning+for+openc,aps,213&sr=8-1)，2017 年。

+   [Mastering OpenCV 4 with Python](https://www.amazon.com/Mastering-OpenCV-Python-practical-processing/dp/1789344913)，2019 年。

### **网站**

+   10 种 Python 聚类算法，[`machinelearningmastery.com/clustering-algorithms-with-python/`](https://machinelearningmastery.com/clustering-algorithms-with-python/)

+   OpenCV 中的 K-Means 聚类，[`docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html`](https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html)

+   k-means 聚类，[`en.wikipedia.org/wiki/K-means_clustering`](https://en.wikipedia.org/wiki/K-means_clustering)

## **总结**

在本教程中，你学习了如何应用 OpenCV 的 k-means 聚类算法进行图像分类。

具体来说，你学习了：

+   为什么 k-means 聚类可以应用于图像分类。

+   将 k-means 聚类算法应用于 OpenCV 中的数字数据集进行图像分类。

+   如何减少由于倾斜造成的数字变异，以提高 k-means 聚类算法在图像分类中的准确性。

你有什么问题吗？

在下面的评论中提问，我会尽力回答。
