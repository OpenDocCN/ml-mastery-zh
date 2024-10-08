# 使用 OpenCV 进行机器学习的图像向量表示

> 原文：[`machinelearningmastery.com/image-vector-representation-for-machine-learning-using-opencv/`](https://machinelearningmastery.com/image-vector-representation-for-machine-learning-using-opencv/)

在将图像输入机器学习算法之前，常见的预处理步骤之一是将其转换为特征向量。正如我们在本教程中将看到的，将图像转换为特征向量有几个优势，使其更加高效。

在将图像转换为特征向量的不同技术中，**方向梯度直方图**和**词袋模型**是与不同机器学习算法结合使用的两种最流行的技术。

在本教程中，你将发现方向梯度直方图（HOG）和词袋模型（BoW）技术用于图像向量表示。

完成本教程后，你将了解：

+   使用方向梯度直方图和词袋模型技术进行图像向量表示的优势是什么？

+   如何在 OpenCV 中使用方向梯度直方图技术。

+   如何在 OpenCV 中使用词袋模型技术。

**通过我的书** [《OpenCV 中的机器学习》](https://machinelearning.samcart.com/products/machine-learning-opencv/) **启动你的项目**。它提供了**自学教程**和**工作代码**。

让我们开始吧。![](https://machinelearningmastery.com/wp-content/uploads/2023/01/vector_cover-scaled.jpg)

使用 OpenCV 进行机器学习的图像向量表示

摄影：[John Fowler](https://unsplash.com/photos/RsRTIofe0HE)，版权所有。

## **教程概述**

本教程分为四个部分；它们是：

+   使用 HOG 或 BoW 进行图像向量表示的优势是什么？

+   方向梯度直方图技术

+   词袋模型技术

+   将技术付诸实践

## **使用 HOG 或 BoW 进行图像向量表示的优势是什么？**

在处理机器学习算法时，图像数据通常会经历一个数据预处理步骤，以使机器学习算法能够处理它。

在 OpenCV 中，例如，ml 模块要求图像数据以等长特征向量的形式输入到机器学习算法中。

> *每个训练样本是一个值向量（在计算机视觉中有时称为特征向量）。通常所有向量都有相同数量的组件（特征）；OpenCV 的 ml 模块假设如此。*
> 
> – [OpenCV](https://docs.opencv.org/4.x/dc/dd6/ml_intro.html)，2023。

一种组织图像数据的方法是将其展平为一维向量，其中向量的长度等于图像中的像素数。例如，一个$20\times 20$的像素图像会生成一个长度为 400 像素的一维向量。这个一维向量作为特征集输入到机器学习算法中，其中每个像素的强度值代表每个特征。

然而，虽然这是我们可以创建的最简单特征集，但它不是最有效的，特别是在处理较大的图像时，这会导致生成的输入特征过多，从而无法有效处理。

> *这可以显著影响在具有许多输入特征的数据上训练的机器学习算法的性能，通常被称为“*[维度诅咒](https://en.wikipedia.org/wiki/Curse_of_dimensionality)*。”*
> 
> [机器学习中的降维介绍](https://machinelearningmastery.com/dimensionality-reduction-for-machine-learning/)，2020 年。

我们希望减少表示每个图像的输入特征数量，以便机器学习算法能够更好地对输入数据进行泛化。用更技术的话来说，进行降维是理想的，这将图像数据从高维空间转换到低维空间。

一种方法是应用特征提取和表示技术，例如方向梯度直方图（HOG）或词袋模型（BoW），以更紧凑的方式表示图像，从而减少特征集中的冗余并减少处理它的计算需求。

将图像数据转换为特征向量的另一个优点是，图像的向量表示变得对光照、尺度或视角的变化更加鲁棒。

### 想开始使用 OpenCV 进行机器学习吗？

立即参加我的免费电子邮件速成课程（附样例代码）。

点击注册并获得免费的 PDF 电子书版本课程。

在接下来的部分中，我们将探讨使用 HOG 和 BoW 技术进行图像向量表示。

## **方向梯度直方图技术**

HOG 是一种特征提取技术，旨在通过边缘方向的分布来表示图像空间内物体的局部形状和外观。

简而言之，当 HOG 技术应用于图像时，会执行以下步骤：

1.  1.  使用例如 Prewitt 算子的图像梯度计算图像的水平和垂直方向的梯度。然后计算图像中每个像素的梯度的幅度和方向。

1.  1.  将图像划分为固定大小的不重叠单元格，并计算每个单元格的梯度直方图。每个图像单元格的直方图表示更紧凑且对噪声更具鲁棒性。单元格大小通常根据我们要捕捉的图像特征的大小进行设置。

1.  1.  将直方图在单元格块上拼接成一维特征向量并进行归一化。这使得描述符对光照变化更加鲁棒。

1.  1.  最后，将所有归一化的特征向量拼接在一起，表示单元格块，从而获得整个图像的最终特征向量表示。

OpenCV 中的 HOG 实现接受几个输入参数，这些参数对应于上述步骤，包括：

+   +   窗口大小（*winSize*）对应于要检测的最小对象大小。

    +   单元格大小（*cellSize*）通常捕捉感兴趣的图像特征的大小。

    +   块大小（*blockSize*）解决了光照变化的问题。

    +   块步幅（*blockStride*）控制相邻块的重叠程度。

    +   直方图的箱子数量（*nbins*）用于捕捉 0 到 180 度之间的梯度。

让我们创建一个函数 `hog_descriptors()`，使用 HOG 技术计算一组图像的特征向量：

Python

```py
def hog_descriptors(imgs):
    # Create a list to store the HOG feature vectors
    hog_features = []

    # Set parameter values for the HOG descriptor based on the image data in use
    winSize = (20, 20)
    blockSize = (10, 10)
    blockStride = (5, 5)
    cellSize = (10, 10)
    nbins = 9

    # Set the remaining parameters to their default values
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = False
    nlevels = 64

    # Create a HOG descriptor
    hog = HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    # Compute HOG descriptors for the input images and append the feature vectors to the list
    for img in imgs:
        hist = hog.compute(img.reshape(20, 20).astype(uint8))
        hog_features.append(hist)

    return array(hog_features)
```

**注意**：重要的是要注意，这里图像的重新形状对应于稍后将在本教程中使用的图像数据集。如果使用不同的数据集，请不要忘记相应调整此部分代码。

## **词袋模型技术**

BoW 技术已在 [这个教程](https://machinelearningmastery.com/gentle-introduction-bag-words-model/) 中介绍，应用于使用机器学习算法建模文本。

尽管如此，这项技术也可以应用于计算机视觉，在这种情况下，图像被视为可以提取特征的视觉词。因此，当应用于计算机视觉时，BoW 技术通常被称为视觉词袋技术。

简而言之，当将 BoW 技术应用于图像时，执行以下步骤：

1.  1.  使用诸如尺度不变特征变换（SIFT）或加速稳健特征（SURF）等算法从图像中提取特征描述符。理想情况下，提取的特征应对强度、尺度、旋转和仿射变换不变。

1.  1.  从特征描述符中生成代码字，其中每个代码字代表相似的图像块。生成这些代码字的一种方法是使用 k-means 聚类将相似的描述符聚集成簇，簇的中心代表视觉词，而簇的数量则代表词汇大小。

1.  1.  将特征描述符映射到词汇表中最近的簇，本质上是为每个特征描述符分配一个代码字。

1.  1.  将代码字分箱到直方图中，并使用此直方图作为图像的特征向量表示。

让我们创建一个函数 `bow_descriptors()`，该函数使用 SIFT 对一组图像应用 BoW 技术：

Python

```py
def bow_descriptors(imgs):
    # Create a SIFT descriptor
    sift = SIFT_create()

    # Create a BoW descriptor
    # The number of clusters equal to 50 (analogous to the vocabulary size) has been chosen empirically
    bow_trainer = BOWKMeansTrainer(50)
    bow_extractor = BOWImgDescriptorExtractor(sift, BFMatcher(NORM_L2))

    for img in imgs:
        # Reshape each RGB image and convert it to grayscale
        img = reshape(img, (32, 32, 3), 'F')
        img = cvtColor(img, COLOR_RGB2GRAY).transpose()

        # Extract the SIFT descriptors
        _, descriptors = sift.detectAndCompute(img, None)

        # Add the SIFT descriptors to the BoW vocabulary trainer
        if descriptors is not None:
            bow_trainer.add(descriptors)

    # Perform k-means clustering and return the vocabulary
    voc = bow_trainer.cluster()

    # Assign the vocabulary to the BoW descriptor extractor
    bow_extractor.setVocabulary(voc)

    # Create a list to store the BoW feature vectors
    bow_features = []

    for img in imgs:
        # Reshape each RGB image and convert it to grayscale
        img = reshape(img, (32, 32, 3), 'F')
        img = cvtColor(img, COLOR_RGB2GRAY).transpose()

        # Compute the BoW feature vector
        hist = bow_extractor.compute(img, sift.detect(img))

        # Append the feature vectors to the list
        if hist is not None:
            bow_features.append(hist[0])

    return array(bow_features)
```

**注意**：需要注意的是，这里图像的重塑方式对应于本教程后面将使用的图像数据集。如果你使用不同的数据集，请不要忘记相应调整这部分代码。

## **测试技术**

并不一定存在适用于所有情况的最佳技术，选择用于图像数据的技术通常需要进行控制实验。

在本教程中，作为一个示例，我们将对 OpenCV 附带的数字数据集应用 HOG 技术，对 CIFAR-10 数据集中的图像应用 BoW 技术。在本教程中，我们将仅考虑这两个数据集中的一个子集，以减少所需的处理时间。然而，相同的代码可以很容易地扩展到完整的数据集。

我们将从加载我们将使用的数据集开始。回顾一下，我们在本教程中已经看到如何从每个数据集中提取图像。`digits_dataset` 和 `cifar_dataset` 是我创建的 Python 脚本，分别包含加载数字和 CIFAR-10 数据集的代码：

Python

```py
from digits_dataset import split_images, split_data
from cifar_dataset import load_images

# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)

# Obtain a dataset from the digits image
digits_imgs, _, _, _ = split_data(20, sub_imgs, 0.8)

# Load a batch of images from the CIFAR dataset
cifar_imgs = load_images('Images/cifar-10-batches-py/data_batch_1')

# Consider only a subset of images
digits_subset = digits_imgs[0:100, :]
cifar_subset = cifar_imgs[0:100, :]
```

然后，我们可以将数据集传递给我们在本教程中创建的 `hog_descriptors()` 和 `bow_descriptors()` 函数：

Python

```py
digits_hog = hog_descriptors(digits_subset)
print('Size of HOG feature vectors:', digits_hog.shape)

cifar_bow = bow_descriptors(cifar_subset)
print('Size of BoW feature vectors:', cifar_bow.shape)
```

完整的代码列表如下：

Python

```py
from cv2 import (imshow, waitKey, HOGDescriptor, SIFT_create, BOWKMeansTrainer,
                 BOWImgDescriptorExtractor, BFMatcher, NORM_L2, cvtColor, COLOR_RGB2GRAY)
from digits_dataset import split_images, split_data
from cifar_dataset import load_images
from numpy import uint8, array, reshape

# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)

# Obtain a dataset from the digits image
digits_imgs, _, _, _ = split_data(20, sub_imgs, 0.8)

# Load a batch of images from the CIFAR dataset
cifar_imgs = load_images('Images/cifar-10-batches-py/data_batch_1')

# Consider only a subset of images
digits_subset = digits_imgs[0:100, :]
cifar_subset = cifar_imgs[0:100, :]

def hog_descriptors(imgs):
    # Create a list to store the HOG feature vectors
    hog_features = []

    # Set parameter values for the HOG descriptor based on the image data in use
    winSize = (20, 20)
    blockSize = (10, 10)
    blockStride = (5, 5)
    cellSize = (10, 10)
    nbins = 9

    # Set the remaining parameters to their default values
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = False
    nlevels = 64

    # Create a HOG descriptor
    hog = HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    # Compute HOG descriptors for the input images and append the feature vectors to the list
    for img in imgs:
        hist = hog.compute(img.reshape(20, 20).astype(uint8))
        hog_features.append(hist)

    return array(hog_features)

def bow_descriptors(imgs):
    # Create a SIFT descriptor
    sift = SIFT_create()

    # Create a BoW descriptor
    # The number of clusters equal to 50 (analogous to the vocabulary size) has been chosen empirically
    bow_trainer = BOWKMeansTrainer(50)
    bow_extractor = BOWImgDescriptorExtractor(sift, BFMatcher(NORM_L2))

    for img in imgs:
        # Reshape each RGB image and convert it to grayscale
        img = reshape(img, (32, 32, 3), 'F')
        img = cvtColor(img, COLOR_RGB2GRAY).transpose()

        # Extract the SIFT descriptors
        _, descriptors = sift.detectAndCompute(img, None)

        # Add the SIFT descriptors to the BoW vocabulary trainer
        if descriptors is not None:
            bow_trainer.add(descriptors)

    # Perform k-means clustering and return the vocabulary
    voc = bow_trainer.cluster()

    # Assign the vocabulary to the BoW descriptor extractor
    bow_extractor.setVocabulary(voc)

    # Create a list to store the BoW feature vectors
    bow_features = []

    for img in imgs:
        # Reshape each RGB image and convert it to grayscale
        img = reshape(img, (32, 32, 3), 'F')
        img = cvtColor(img, COLOR_RGB2GRAY).transpose()

        # Compute the BoW feature vector
        hist = bow_extractor.compute(img, sift.detect(img))

        # Append the feature vectors to the list
        if hist is not None:
            bow_features.append(hist[0])

    return array(bow_features)

digits_hog = hog_descriptors(digits_subset)
print('Size of HOG feature vectors:', digits_hog.shape)

cifar_bow = bow_descriptors(cifar_subset)
print('Size of BoW feature vectors:', cifar_bow.shape)
```

上面的代码返回以下输出：

Python

```py
Size of HOG feature vectors:  (100, 81)
Size of BoW feature vectors: (100, 50)
```

根据我们选择的参数值，我们可能会看到 HOG 技术为每个图像返回大小为 $1\times 81$ 的特征向量。这意味着每个图像现在由 81 维空间中的点表示。另一方面，BoW 技术为每个图像返回大小为 $1\times 50$ 的向量，其中向量长度由选择的 k-means 聚类数量决定，这也类似于词汇表大小。

因此，我们可以看到，通过应用 HOG 和 BoW 技术，我们不仅仅是将每个图像展平成一维向量，而是以更紧凑的方式表示了每个图像。

我们的下一步是看看如何利用这些数据使用不同的机器学习算法。

## **进一步阅读**

如果你想深入了解，本节提供了更多相关资源。

### **书籍**

+   [使用 Python 3 学习 OpenCV 4 计算机视觉](https://www.amazon.com/Learning-OpenCV-Computer-Vision-Python/dp/1789531616)，2020 年。

### **网站**

+   OpenCV，[`opencv.org/`](https://opencv.org/)

+   使用 OpenCV 解释的方向梯度直方图，[`learnopencv.com/histogram-of-oriented-gradients/`](https://learnopencv.com/histogram-of-oriented-gradients/)

+   计算机视觉中的词袋模型，[`en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision`](https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision)

## **总结**

在本教程中，你将了解方向梯度直方图和词袋技术在图像向量表示中的应用。

具体来说，你学到了：

+   使用方向梯度直方图和词袋技术进行图像向量表示的优势是什么？

+   如何在 OpenCV 中使用方向梯度直方图技术。

+   如何在 OpenCV 中使用词袋技术。

你有任何问题吗？

在下方的评论中提问，我会尽力回答。
