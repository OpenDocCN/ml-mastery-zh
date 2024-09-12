# 练习机器学习的图像数据集在 OpenCV 中

> 原文：[`machinelearningmastery.com/image-datasets-for-practicing-machine-learning-in-opencv/`](https://machinelearningmastery.com/image-datasets-for-practicing-machine-learning-in-opencv/)

在你开始机器学习之旅的最初阶段，公开可用的数据集减轻了自己创建数据集的担忧，让你能够专注于学习使用机器学习算法。如果数据集大小适中，不需要过多的预处理，可以更快地练习使用算法，然后再转向更具挑战性的问题，这也会很有帮助。

我们将查看的两个数据集是 OpenCV 提供的较简单的数字数据集和更具挑战性但广泛使用的 CIFAR-10 数据集。在我们探索 OpenCV 的机器学习算法时，我们将使用这两个数据集中的任何一个。

在本教程中，你将学习如何下载和提取 OpenCV 数字和 CIFAR-10 数据集，以练习在 OpenCV 中进行机器学习。

完成本教程后，你将了解：

+   如何下载和提取 OpenCV 数字数据集。

+   如何下载和提取 CIFAR-10 数据集，而不依赖其他 Python 包（如 TensorFlow）。

**启动你的项目**，可以参考我的书籍 [《OpenCV 中的机器学习》](https://machinelearning.samcart.com/products/machine-learning-opencv/)。它提供了**自学教程**和**可运行的代码**。

让我们开始吧。![](https://machinelearningmastery.com/wp-content/uploads/2023/01/datasets_cover-scaled.jpg)

练习机器学习的图像数据集在 OpenCV 中

图片由 [OC Gonzalez](https://unsplash.com/photos/xg8z_KhSorQ) 拍摄，版权所有。

## **教程概览**

本教程分为三个部分；它们是：

+   数字数据集

+   CIFAR-10 数据集

+   加载数据集

## **数字数据集**

[OpenCV 提供的图像，digits.png](https://github.com/opencv/opencv/tree/master/samples/data)，由 20$\times$20 像素的子图像组成，每个子图像展示了从 0 到 9 的一个数字，可以拆分成数据集。总的来说，*digits* 图像包含 5,000 个手写数字。

OpenCV 提供的数字数据集不一定代表更复杂数据集所面临的现实挑战，主要因为其图像内容变化非常有限。然而，它的简单性和易用性将允许我们以较低的预处理和计算成本快速测试几种机器学习算法。

为了能够从完整的数字图像中提取数据集，我们的第一步是将其拆分成构成它的多个子图像。为此，我们来创建以下 `split_images` 函数：

Python

```py
from cv2 import imread, IMREAD_GRAYSCALE
from numpy import hsplit, vsplit, array

def split_images(img_name, img_size):

    # Load the full image from the specified file
    img = imread(img_name, IMREAD_GRAYSCALE)

    # Find the number of sub-images on each row and column according to their size
    num_rows = img.shape[0] / img_size
    num_cols = img.shape[1] / img_size

    # Split the full image horizontally and vertically into sub-images
    sub_imgs = [hsplit(row, num_cols) for row in vsplit(img, num_rows)]

    return img, array(sub_imgs)
```

`split_images` 函数接受全图像的路径以及子图像的像素大小作为输入。由于我们处理的是正方形子图像，我们将其大小用一个维度表示，该维度等于 20。

该函数随后应用 OpenCV 的 `imread` 方法将图像的灰度版本加载到 NumPy 数组中。然后，使用 `hsplit` 和 `vsplit` 方法分别对 NumPy 数组进行水平和垂直切割。

`split_images` 函数返回的子图像数组大小为 (50, 100, 20, 20)。

一旦我们提取了子图像数组，我们将把它分割成训练集和测试集。我们还需要为这两个数据分割创建真实标签，以便在训练过程中使用并评估测试结果。

以下 `split_data` 函数用于这些目的：

Python

```py
from numpy import float32, arange, repeat, newaxis

def split_data(img_size, sub_imgs, ratio):

    # Compute the partition between the training and testing data
    partition = int(sub_imgs.shape[1] * ratio)

    # Split dataset into training and testing sets
    train = sub_imgs[:, :partition, :, :]
    test = sub_imgs[:, partition:sub_imgs.shape[1], :, :]

    # Flatten each image into a one-dimensional vector
    train_imgs = train.reshape(-1, img_size ** 2)
    test_imgs = test.reshape(-1, img_size ** 2)

    # Create the ground truth labels
    labels = arange(10)
    train_labels = repeat(labels, train_imgs.shape[0] / labels.shape[0])[:, newaxis]
    test_labels = repeat(labels, test_imgs.shape[0] / labels.shape[0])[:, newaxis]

    return train_imgs, train_labels, test_imgs, test_labels
```

`split_data` 函数以子图像数组和数据集训练部分的分割比例作为输入。然后，该函数计算 `partition` 值，将子图像数组沿其列分割成训练集和测试集。这个 `partition` 值随后用于将第一组列分配给训练数据，将剩余的列分配给测试数据。

为了在 *digits.png* 图像上可视化这种分割，效果如下所示：

![](https://machinelearningmastery.com/wp-content/uploads/2023/01/kNN_1.png)

将子图像分割为训练数据集和测试数据集

你还会注意到，我们将每个 20$\times$20 的子图像展平为一个长度为 400 像素的一维向量，使得在包含训练和测试图像的数组中，每一行现在存储的是展平后的 20$/times$20 像素图像。

`split_data` 函数的最后一部分创建了值在 0 到 9 之间的真实标签，并根据我们拥有的训练和测试图像的数量重复这些值。

## **CIFAR-10 数据集**

CIFAR-10 数据集并未随 OpenCV 提供，但我们将考虑它，因为它比 OpenCV 的数字数据集更好地代表了现实世界的挑战。

CIFAR-10 数据集总共包含 60,000 张 32$\times$32 RGB 图像。它包括属于 10 个不同类别的各种图像，如飞机、猫和船。数据集文件已经分割成 5 个 pickle 文件，其中包含 1,000 张训练图像及其标签，另一个文件包含 1,000 张测试图像及其标签。

让我们继续从 [这个链接](https://www.cs.toronto.edu/~kriz/cifar.html) 下载 CIFAR-10 数据集用于 Python (**注意**：不使用 TensorFlow/Keras 的原因是展示如何在不依赖额外 Python 包的情况下工作)。请注意你在硬盘上保存并解压数据集的路径。

以下代码加载数据集文件，并返回训练和测试图像以及标签：

Python

```py
from pickle import load
from numpy import array, newaxis

def load_images(path):

    # Create empty lists to store the images and labels
    imgs = []
    labels = []

    # Iterate over the dataset's files
    for batch in range(5):

        # Specify the path to the training data
        train_path_batch = path + 'data_batch_' + str(batch + 1)

        # Extract the training images and labels from the dataset files
        train_imgs_batch, train_labels_batch = extract_data(train_path_batch)

        # Store the training images
        imgs.append(train_imgs_batch)
        train_imgs = array(imgs).reshape(-1, 3072)

        # Store the training labels
        labels.append(train_labels_batch)
        train_labels = array(labels).reshape(-1, 1)

    # Specify the path to the testing data
    test_path_batch = path + 'test_batch'

    # Extract the testing images and labels from the dataset files
    test_imgs, test_labels = extract_data(test_path_batch)
    test_labels = array(test_labels)[:, newaxis]

    return train_imgs, train_labels, test_imgs, test_labels

def extract_data(path):

    # Open pickle file and return a dictionary
    with open(path, 'rb') as fo:
        dict = load(fo, encoding='bytes')

    # Extract the dictionary values
    dict_values = list(dict.values())

    # Extract the images and labels
    imgs = dict_values[2]
    labels = dict_values[1]

    return imgs, labels
```

需要记住的是，使用更大且更多样化的数据集（如 CIFAR-10）进行不同模型的测试相比于使用较简单的数据集（如 digits 数据集）的妥协在于，前者的训练可能会更耗时。

## **加载数据集**

让我们试着调用我们之前创建的函数。

我将属于 digits 数据集的代码与属于 CIFAR-10 数据集的代码分开成两个不同的 Python 脚本，分别命名为 `digits_dataset.py` 和 `cifar_dataset.py`：

Python

```py
from digits_dataset import split_images, split_data
from cifar_dataset import load_images

# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)

# Obtain training and testing datasets from the digits image
digits_train_imgs, digits_train_labels, digits_test_imgs, digits_test_labels = split_data(20, sub_imgs, 0.8)

# Obtain training and testing datasets from the CIFAR-10 dataset
cifar_train_imgs, cifar_train_labels, cifar_test_imgs, cifar_test_labels = load_images('Images/cifar-10-batches-py/')
```

**注意**：不要忘记将上面代码中的路径更改为你保存*数据文件*的位置。

在接下来的教程中，我们将看到如何使用不同的机器学习技术处理这些数据集，首先了解如何将数据集图像转换为特征向量，作为使用它们进行机器学习的预处理步骤之一。

### 想要开始使用 OpenCV 进行机器学习吗？

现在就参加我的免费电子邮件速成课程（包括示例代码）。

点击注册，并获得课程的免费 PDF Ebook 版本。

## **进一步阅读**

本节提供了更多关于该主题的资源，如果你想深入了解。

### **书籍**

+   [Mastering OpenCV 4 with Python](https://www.amazon.com/Mastering-OpenCV-Python-practical-processing/dp/1789344913)，2019 年。

### **网站**

+   OpenCV, [`opencv.org/`](https://opencv.org/)

## **总结**

在本教程中，你学习了如何下载和提取 OpenCV digits 和 CIFAR-10 数据集，以便在 OpenCV 中练习机器学习。

具体来说，你学到了：

+   如何下载和提取 OpenCV digits 数据集。

+   如何下载和提取 CIFAR-10 数据集，而无需依赖其他 Python 包（如 TensorFlow）。

你有任何问题吗？

在下面的评论中提出你的问题，我会尽力回答。
