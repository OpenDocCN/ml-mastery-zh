# 使用 OpenCV 进行图像分类和检测的支持向量机

> 原文：[`machinelearningmastery.com/support-vector-machines-for-image-classification-and-detection-using-opencv/`](https://machinelearningmastery.com/support-vector-machines-for-image-classification-and-detection-using-opencv/)

在[之前的教程](https://machinelearningmastery.com/?p=14889&preview=true)中，我们探讨了将支持向量机算法作为 OpenCV 库中最受欢迎的监督学习技术之一。

到目前为止，我们已经看到如何将支持向量机应用于我们生成的自定义数据集，该数据集由两个类的二维点组成。

在本教程中，你将学习如何将 OpenCV 的支持向量机算法应用于解决图像分类和检测问题。

完成本教程后，你将了解到：

+   支持向量机的几个重要特征。

+   如何将支持向量机应用于图像分类和检测问题。

**通过我的书籍[《OpenCV 中的机器学习》](https://machinelearning.samcart.com/products/machine-learning-opencv/)**来**启动你的项目**。它提供了**自学教程**和**实用代码**。

让我们开始吧。![](https://machinelearningmastery.com/wp-content/uploads/2023/03/svm_class_detect_cover-scaled.jpg)

使用 OpenCV 进行图像分类和检测的支持向量机

图片由[Patrick Ryan](https://unsplash.com/photos/3kUIaB2EPp8)提供，部分版权保留。

## **教程概述**

本教程分为三个部分；它们是：

+   支持向量机工作原理回顾

+   将 SVM 算法应用于图像分类

+   使用 SVM 算法进行图像检测

## **支持向量机工作原理回顾**

在[之前的教程](https://machinelearningmastery.com/?p=14889&preview=true)中，我们介绍了如何在 OpenCV 库中使用支持向量机（SVM）算法。到目前为止，我们已将其应用于我们生成的自定义数据集，该数据集由两个类的二维点组成。

我们已经看到，SVM 旨在通过计算一个决策边界来将数据点分为不同的类，该边界最大化到每个类的最近数据点（称为支持向量）的间隔。通过调整一个名为*C*的参数，可以放宽最大化间隔的约束，该参数控制最大化间隔和减少训练数据中的错误分类之间的权衡。

SVM 算法可能会使用不同的核函数，这取决于输入数据是否是线性可分的。在非线性可分的数据情况下，可能使用非线性核将数据转换到更高维空间，以便在其中线性可分。这类似于 SVM 在原始输入空间中找到一个非线性的决策边界。

## **将 SVM 算法应用于图像分类**

我们将使用[OpenCV 中的数字数据集](https://machinelearningmastery.com/?p=14607&preview=true)来完成这个任务，尽管我们开发的代码也可以用于其他数据集。

### 想要开始使用 OpenCV 进行机器学习吗？

现在就参加我的免费电子邮件速成课程（附有示例代码）。

点击注册，同时获取课程的免费 PDF 电子书版本。

我们的第一步是加载 OpenCV 数字图像，将其分成多个包含手写数字 0 到 9 的子图像，并创建相应的真实标签，以便我们以后可以量化训练好的 SVM 分类器的准确性。对于这个具体的例子，我们将把 80% 的数据集图像分配给训练集，剩下的 20% 图像分配给测试集：

Python

```py
# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)

# Obtain training and testing datasets from the digits image
digits_train_imgs, digits_train_labels, digits_test_imgs, digits_test_labels = split_data(20, sub_imgs, 0.8)
```

我们的下一步是在 OpenCV 中创建一个使用 RBF 核的 SVM。正如我们在[之前的教程](https://machinelearningmastery.com/?p=14889&preview=true)中所做的那样，我们必须设置与 SVM 类型和核函数相关的几个参数值。我们还将包括终止标准，以停止 SVM 优化问题的迭代过程：

Python

```py
# Create a new SVM
svm_digits = ml.SVM_create()

# Set the SVM kernel to RBF
svm_digits.setKernel(ml.SVM_RBF)
svm_digits.setType(ml.SVM_C_SVC)
svm_digits.setGamma(0.5)
svm_digits.setC(12)
svm_digits.setTermCriteria((TERM_CRITERIA_MAX_ITER + TERM_CRITERIA_EPS, 100, 1e-6))
```

我们将首先将每张图像转换为其 HOG 描述符，如[本教程](https://machinelearningmastery.com/?p=14553&preview=true)中所述，而不是直接在原始图像数据上训练和测试 SVM。HOG 技术旨在通过利用图像的局部形状和外观来获得更紧凑的表示。对 HOG 描述符进行分类器训练可以提高其区分不同类别的能力，同时减少数据处理的计算开销：

Python

```py
# Converting the image data into HOG descriptors
digits_train_hog = hog_descriptors(digits_train_imgs)
digits_test_hog = hog_descriptors(digits_test_imgs)
```

我们可以最终对 HOG 描述符上的 SVM 进行训练，并继续预测测试数据的标签，基于此我们可以计算分类器的准确性：

Python

```py
# Predict labels for the testing data
_, digits_test_pred = svm_digits.predict(digits_test_hog.astype(float32))

# Compute and print the achieved accuracy
accuracy_digits = (sum(digits_test_pred.astype(int) == digits_test_labels) / digits_test_labels.size) * 100
print('Accuracy:', accuracy_digits[0], '%')
```

```py
Accuracy: 97.1 %
```

对于这个具体的例子，*C* 和 *gamma* 的值是经验性地设置的。然而，建议采用调优技术，如 *网格搜索* 算法，来研究是否有更好的超参数组合可以进一步提升分类器的准确性。

完整的代码清单如下：

Python

```py
from cv2 import ml, TERM_CRITERIA_MAX_ITER, TERM_CRITERIA_EPS
from numpy import float32
from digits_dataset import split_images, split_data
from feature_extraction import hog_descriptors

# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)

# Obtain training and testing datasets from the digits image
digits_train_imgs, digits_train_labels, digits_test_imgs, digits_test_labels = split_data(20, sub_imgs, 0.8)

# Create a new SVM
svm_digits = ml.SVM_create()

# Set the SVM kernel to RBF
svm_digits.setKernel(ml.SVM_RBF)
svm_digits.setType(ml.SVM_C_SVC)
svm_digits.setGamma(0.5)
svm_digits.setC(12)
svm_digits.setTermCriteria((TERM_CRITERIA_MAX_ITER + TERM_CRITERIA_EPS, 100, 1e-6))

# Converting the image data into HOG descriptors
digits_train_hog = hog_descriptors(digits_train_imgs)
digits_test_hog = hog_descriptors(digits_test_imgs)

# Train the SVM on the set of training data
svm_digits.train(digits_train_hog.astype(float32), ml.ROW_SAMPLE, digits_train_labels)

# Predict labels for the testing data
_, digits_test_pred = svm_digits.predict(digits_test_hog.astype(float32))

# Compute and print the achieved accuracy
accuracy_digits = (sum(digits_test_pred.astype(int) == digits_test_labels) / digits_test_labels.size) * 100
print('Accuracy:', accuracy_digits[0], '%')
```

## **使用 SVM 算法进行图像检测**

可以将我们上面开发的图像分类思路扩展到图像检测中，后者指的是在图像中识别和定位感兴趣的对象。

我们可以通过在更大图像中的不同位置重复我们在前一部分开发的图像分类来实现这一点（我们将把这个更大的图像称为 *测试图像*）。

对于这个具体的例子，我们将创建一个由 OpenCV 数字数据集中随机选择的子图像拼接而成的 *拼贴画*，然后尝试检测感兴趣的数字出现情况。

首先创建测试图像。我们将通过从整个数据集中随机选择 25 个等间距的子图像，打乱它们的顺序，并将它们组合成一个$100\times 100$像素的图像来实现：

```py
# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)

# Obtain training and testing datasets from the digits image
digits_train_imgs, _, digits_test_imgs, _ = split_data(20, sub_imgs, 0.8)

# Create an empty list to store the random numbers
rand_nums = []

# Seed the random number generator for repeatability
seed(10)

# Choose 25 random digits from the testing dataset
for i in range(0, digits_test_imgs.shape[0], int(digits_test_imgs.shape[0] / 25)):

    # Generate a random integer
    rand = randint(i, int(digits_test_imgs.shape[0] / 25) + i - 1)

    # Append it to the list
    rand_nums.append(rand)

# Shuffle the order of the generated random integers
shuffle(rand_nums)

# Read the image data corresponding to the random integers
rand_test_imgs = digits_test_imgs[rand_nums, :]

# Initialize an array to hold the test image
test_img = zeros((100, 100), dtype=uint8)

# Start a sub-image counter
img_count = 0

# Iterate over the test image
for i in range(0, test_img.shape[0], 20):
    for j in range(0, test_img.shape[1], 20):

        # Populate the test image with the chosen digits
        test_img[i:i + 20, j:j + 20] = rand_test_imgs[img_count].reshape(20, 20)

        # Increment the sub-image counter
        img_count += 1

# Display the test image
imshow(test_img, cmap='gray')
show()
```

结果测试图像如下所示：

![](https://machinelearningmastery.com/wp-content/uploads/2023/03/svm_class_detect_1.png)

图像检测的测试图像

接下来，我们将像前一节那样训练一个新创建的 SVM。然而，鉴于我们现在处理的是检测问题，真实标签不应对应图像中的数字，而应区分训练集中正样本和负样本。

比如说，我们有兴趣检测测试图像中两个*0*数字的出现。因此，数据集中训练部分中的*0*图像被视为*正样本*，并通过类标签 1 区分。所有其他属于剩余数字的图像被视为*负样本*，并通过类标签 0 区分。

一旦生成了真实标签，我们可以开始在训练数据集上创建和训练 SVM：

Python

```py
# Generate labels for the positive and negative samples
digits_train_labels = ones((digits_train_imgs.shape[0], 1), dtype=int)
digits_train_labels[int(digits_train_labels.shape[0] / 10):digits_train_labels.shape[0], :] = 0

# Create a new SVM
svm_digits = ml.SVM_create()

# Set the SVM kernel to RBF
svm_digits.setKernel(ml.SVM_RBF)
svm_digits.setType(ml.SVM_C_SVC)
svm_digits.setGamma(0.5)
svm_digits.setC(12)
svm_digits.setTermCriteria((TERM_CRITERIA_MAX_ITER + TERM_CRITERIA_EPS, 100, 1e-6))

# Convert the training images to HOG descriptors
digits_train_hog = hog_descriptors(digits_train_imgs)

# Train the SVM on the set of training data
svm_digits.train(digits_train_hog, ml.ROW_SAMPLE, digits_train_labels)
```

我们将要添加到上面代码列表中的最终代码执行以下操作：

1.  按预定义的步幅遍历测试图像。

1.  从测试图像中裁剪出与特征数字的子图像（即，20 $\times$ 20 像素）等大小的图像块，并在每次迭代时进行处理。

1.  提取每个图像块的 HOG 描述符。

1.  将 HOG 描述符输入到训练好的 SVM 中，以获得标签预测。

1.  每当检测到时，存储图像块坐标。

1.  在原始测试图像上为每个检测绘制边界框。

Python

```py
# Create an empty list to store the matching patch coordinates
positive_patches = []

# Define the stride to shift with
stride = 5

# Iterate over the test image
for i in range(0, test_img.shape[0] - 20 + stride, stride):
    for j in range(0, test_img.shape[1] - 20 + stride, stride):

        # Crop a patch from the test image
        patch = test_img[i:i + 20, j:j + 20].reshape(1, 400)

        # Convert the image patch into HOG descriptors
        patch_hog = hog_descriptors(patch)

        # Predict the target label of the image patch
        _, patch_pred = svm_digits.predict(patch_hog.astype(float32))

        # If a match is found, store its coordinate values
        if patch_pred == 1:
            positive_patches.append((i, j))

# Convert the list to an array
positive_patches = array(positive_patches)

# Iterate over the match coordinates and draw their bounding box
for i in range(positive_patches.shape[0]):

    rectangle(test_img, (positive_patches[i, 1], positive_patches[i, 0]),
              (positive_patches[i, 1] + 20, positive_patches[i, 0] + 20), 255, 1)

# Display the test image
imshow(test_img, cmap='gray')
show()
```

完整的代码列表如下：

Python

```py
from cv2 import ml, TERM_CRITERIA_MAX_ITER, TERM_CRITERIA_EPS, rectangle
from numpy import float32, zeros, ones, uint8, array
from matplotlib.pyplot import imshow, show
from digits_dataset import split_images, split_data
from feature_extraction import hog_descriptors
from random import randint, seed, shuffle

# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)

# Obtain training and testing datasets from the digits image
digits_train_imgs, _, digits_test_imgs, _ = split_data(20, sub_imgs, 0.8)

# Create an empty list to store the random numbers
rand_nums = []

# Seed the random number generator for repeatability
seed(10)

# Choose 25 random digits from the testing dataset
for i in range(0, digits_test_imgs.shape[0], int(digits_test_imgs.shape[0] / 25)):

    # Generate a random integer
    rand = randint(i, int(digits_test_imgs.shape[0] / 25) + i - 1)

    # Append it to the list
    rand_nums.append(rand)

# Shuffle the order of the generated random integers
shuffle(rand_nums)

# Read the image data corresponding to the random integers
rand_test_imgs = digits_test_imgs[rand_nums, :]

# Initialize an array to hold the test image
test_img = zeros((100, 100), dtype=uint8)

# Start a sub-image counter
img_count = 0

# Iterate over the test image
for i in range(0, test_img.shape[0], 20):
    for j in range(0, test_img.shape[1], 20):

        # Populate the test image with the chosen digits
        test_img[i:i + 20, j:j + 20] = rand_test_imgs[img_count].reshape(20, 20)

        # Increment the sub-image counter
        img_count += 1

# Display the test image
imshow(test_img, cmap='gray')
show()

# Generate labels for the positive and negative samples
digits_train_labels = ones((digits_train_imgs.shape[0], 1), dtype=int)
digits_train_labels[int(digits_train_labels.shape[0] / 10):digits_train_labels.shape[0], :] = 0

# Create a new SVM
svm_digits = ml.SVM_create()

# Set the SVM kernel to RBF
svm_digits.setKernel(ml.SVM_RBF)
svm_digits.setType(ml.SVM_C_SVC)
svm_digits.setGamma(0.5)
svm_digits.setC(12)
svm_digits.setTermCriteria((TERM_CRITERIA_MAX_ITER + TERM_CRITERIA_EPS, 100, 1e-6))

# Convert the training images to HOG descriptors
digits_train_hog = hog_descriptors(digits_train_imgs)

# Train the SVM on the set of training data
svm_digits.train(digits_train_hog, ml.ROW_SAMPLE, digits_train_labels)

# Create an empty list to store the matching patch coordinates
positive_patches = []

# Define the stride to shift with
stride = 5

# Iterate over the test image
for i in range(0, test_img.shape[0] - 20 + stride, stride):
    for j in range(0, test_img.shape[1] - 20 + stride, stride):

        # Crop a patch from the test image
        patch = test_img[i:i + 20, j:j + 20].reshape(1, 400)

        # Convert the image patch into HOG descriptors
        patch_hog = hog_descriptors(patch)

        # Predict the target label of the image patch
        _, patch_pred = svm_digits.predict(patch_hog.astype(float32))

        # If a match is found, store its coordinate values
        if patch_pred == 1:
            positive_patches.append((i, j))

# Convert the list to an array
positive_patches = array(positive_patches)

# Iterate over the match coordinates and draw their bounding box
for i in range(positive_patches.shape[0]):

    rectangle(test_img, (positive_patches[i, 1], positive_patches[i, 0]),
              (positive_patches[i, 1] + 20, positive_patches[i, 0] + 20), 255, 1)

# Display the test image
imshow(test_img, cmap='gray')
show()
```

结果图像显示我们成功检测到了测试图像中出现的两个*0*数字：

![](https://machinelearningmastery.com/wp-content/uploads/2023/03/svm_class_detect_2.png)

检测*0*数字的两个出现

我们考虑了一个简单的示例，但相同的思想可以轻松地适应更具挑战性的实际问题。如果你打算将上述代码适应更具挑战性的问题：

+   记住，感兴趣的对象可能会在图像中以不同的大小出现，因此可能需要进行多尺度检测任务。

+   在生成正负样本来训练你的 SVM 时，避免遇到类别不平衡问题。本教程中考虑的示例是变化非常小的图像（我们仅限于 10 个数字，没有尺度、光照、背景等方面的变化），任何数据集不平衡似乎对检测结果几乎没有影响。然而，现实中的挑战通常不会如此简单，类别之间的不平衡分布可能会导致性能较差。

## **进一步阅读**

如果你想深入了解，本节提供了更多资源。

### **书籍**

+   [OpenCV 的机器学习](https://www.amazon.com/Machine-Learning-OpenCV-Intelligent-processing/dp/1783980281/ref=sr_1_1?crid=3VWMIM65XCS6K&keywords=machine+learning+for+opencv&qid=1678294085&sprefix=machine+learning+for+openc,aps,213&sr=8-1)，2017 年。

+   [掌握 OpenCV 4 与 Python](https://www.amazon.com/Mastering-OpenCV-Python-practical-processing/dp/1789344913)，2019 年。

### **网站**

+   支持向量机介绍，[`docs.opencv.org/4.x/d1/d73/tutorial_introduction_to_svm.html`](https://docs.opencv.org/4.x/d1/d73/tutorial_introduction_to_svm.html)

## **总结**

在本教程中，你学习了如何应用 OpenCV 的支持向量机算法来解决图像分类和检测问题。

具体来说，你学习了：

+   支持向量机的一些重要特征。

+   如何将支持向量机应用于图像分类和检测问题。

你有任何问题吗？

在下面的评论中提问，我会尽力回答。
