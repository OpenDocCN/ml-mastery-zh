# OpenCV 中的图像特征提取：边缘和角点

> 原文：[`machinelearningmastery.com/opencv_edges_and_corners/`](https://machinelearningmastery.com/opencv_edges_and_corners/)

在计算机视觉和图像处理的世界里，从图像中提取有意义的特征是非常重要的。这些特征作为各种下游任务的重要输入，如目标检测和分类。找出这些特征有多种方式。最简单的方法是计数像素。但在 OpenCV 中，有许多例程可以帮助你从图像中提取特征。在这篇文章中，你将看到 OpenCV 如何帮助发现一些高级特征。

完成本教程后，你将了解：

+   角点和边缘可以从图像中提取

+   在 OpenCV 中提取角点和边缘的常见算法有哪些

**通过我的书** [《OpenCV 中的机器学习》](https://machinelearning.samcart.com/products/machine-learning-opencv/) **来启动你的项目**。它提供了**自学教程**和**可运行代码**。

让我们开始吧！[](../Images/53a71141b9fec057868923016859fea3.png)

OpenCV 中的图像特征提取：边缘和角点

图片由[Michael Barth](https://unsplash.com/photos/gray-building-under-calm-sky-7Yp3v4Ol1jI)提供，版权所有。

## **概述**

本文分为三部分；它们是：

+   理解图像特征提取

+   OpenCV 中的 Canny 边缘检测

+   OpenCV 中的 Harris 角点检测

## **先决条件**

对于本教程，我们假设你已经熟悉：

+   [使用 OpenCV 读取和显示图像](https://machinelearningmastery.com/?p=14402&preview=true)

## 理解图像特征提取

图像特征提取涉及到识别和表示图像中的独特结构。读取图像的像素当然是一种方法，但这属于低级特征。图像的高级特征可以是边缘、角点，甚至是更复杂的纹理和形状。

特征是图像的特征属性。通过这些独特的特征，你可以区分不同的图像。这是计算机视觉中的第一步。通过提取这些特征，你可以创建比单纯的像素更紧凑和有意义的表示。这有助于进一步的分析和处理。

在接下来的内容中，你将学习到两个基本但非常常见的特征提取算法。这两种算法都以 numpy 数组格式返回基于像素的分类。

## OpenCV 中的 Canny 边缘检测

多年来，已经开发了许多图像特征提取算法。这些算法不是机器学习模型，而是更接近于确定性算法。这些算法各自针对特定的特征。

OpenCV 提供了一整套丰富的工具和函数用于图像特征提取。我们首先从 Canny 边缘检测开始。

在图像中寻找边缘可能是最简单的特征提取。其目标是识别哪个像素在边缘上。边缘定义为像素强度的梯度。换句话说，如果有突兀的颜色变化，则认为它是一个边缘。但其中还有更多细节，因此噪声被排除。

让我们考虑以下图像，并将其保存为 `image.jpg` 在本地目录中：

+   [`unsplash.com/photos/VSLPOL9PwB8`](https://unsplash.com/photos/VSLPOL9PwB8)

一个寻找和说明边缘的示例如下：

Python

```py
import cv2
import numpy as np

# Load the image
img = cv2.imread('image.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect edges using Canny method
edges = cv2.Canny(gray, 150, 300)

# Display the image with corners
img[edges == 255] = (255,0,0)
cv2.imshow('Canny Edges', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

在上面，图像被转换为灰度图像，然后调用了 `cv2.Canny()` 函数。许多特征提取算法需要灰度图像，因为它们通常设计为在单一颜色通道上工作。

`cv2.Canny()` 函数的参数需要两个数值，分别用于最小和最大阈值。这些阈值用于**滞后阈值处理**以将像素合并成边缘。最大阈值越高，结果中仅保留更强的边缘。然而，最小阈值越高，你将会看到更多的“断裂边缘”。

此函数返回一个与图像像素维度匹配的 numpy 数组，其值要么为 0（不在边缘上），要么为 255（在边缘上）。上面的代码将这些像素着色为蓝色。结果如下：

![](img/270cb779b61c2c5d48a8b0b85fa7f47f.png)

Canny 边缘检测的结果

原始照片由 [Gleren Meneghin](https://unsplash.com/photos/VSLPOL9PwB8) 提供，部分版权保留。

你应该能看到上面的蓝色线条标记了门和窗户，并且勾勒出了每一块砖。你可以调整这两个阈值以查看不同的结果。

## OpenCV 中的 Harris 角点检测

Harris 角点检测是一种用于识别强度显著变化的方法，这些变化通常对应于图像中物体的角点。OpenCV 提供了该技术的简单高效实现，使我们能够检测角点，这些角点作为图像分析和匹配的显著特征。

从图像中提取角点可以分为三个步骤：

1.  将图像转换为灰度图像，因为 Harris 角点检测算法仅在单一颜色通道上工作

1.  运行 `cv2.cornerHarris(image, blockSize, ksize, k)` 并获取每个像素的分数

1.  通过比较分数与图像最大值，确定哪个像素在角落处

`cornerHarris()` 函数的参数包括邻域大小 `blockSize` 和核大小 `ksize`。这两个值都是小的正整数，但后者必须是奇数。最后一个参数 `k` 是一个正浮点值，用于控制角点检测的敏感度。该值过大可能会导致算法将角点误认为边缘。你可能需要对其值进行实验。

一个示例代码，运行 Harris 角点检测在上述相同图像上：

```py
import cv2
import numpy as np

# Load the image
img = cv2.imread('image.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect corners using the Harris method
dst = cv2.cornerHarris(gray, 3, 5, 0.1)

# Create a boolean bitmap of corner positions
corners = dst > 0.05 * dst.max()

# Find the coordinates from the boolean bitmap
coord = np.argwhere(corners)

# Draw circles on the coordinates to mark the corners
for y, x in coord:
    cv2.circle(img, (x,y), 3, (0,0,255), -1)

# Display the image with corners
cv2.imshow('Harris Corners', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

生成的图像如下：

![](img/4217f16f4749ab12366b58e4c60f9e84.png)

Harris 角点检测的结果

原始照片由 [Gleren Meneghin](https://unsplash.com/photos/VSLPOL9PwB8)提供，保留部分版权。

红点是通过 `cv2.circle()` 函数在上面的 for 循环中绘制的。它们仅用于说明。关键思想是，算法为图像的每个像素提供一个分数，以表示该像素被认为是角点、边缘还是“平坦”的（即，两者都不是）。你需要通过将分数与整个图像中的最大值进行比较来控制结论的敏感度，具体见

```py
corners = dst &gt; 0.05 * dst.max()

```

结果是一个布尔型 numpy 数组 `corners`，它随后通过 `np.argwhere()` 函数被转换为坐标数组。

从上面的图像可以看出，Harris 角点检测并不完美，但如果角点足够明显，它是可以被检测到的。

### 想要开始使用 OpenCV 进行机器学习？

现在就报名参加我的免费电子邮件速成课程（包含示例代码）。

点击注册，还可获得课程的免费 PDF 电子书版本。

## **进一步阅读**

本节提供了更多关于该主题的资源，如果你想深入了解，可以参考。

### **书籍**

+   [掌握 OpenCV 4 与 Python](https://www.amazon.com/Mastering-OpenCV-Python-practical-processing/dp/1789344913)，2019。

### **网站**

+   OpenCV，[`opencv.org/`](https://opencv.org/)

+   OpenCV 特征检测与描述，[`docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html`](https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html)

+   OpenCV Canny 边缘检测，[`docs.opencv.org/4.x/da/d22/tutorial_py_canny.html`](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)

## **总结**

在本教程中，你学习了如何在图像上应用 OpenCV 的 Canny 边缘检测和 Harris 角点检测算法。

具体来说，你学到了：

+   这些是基于像素的算法，它们将每个像素分类为边缘或非边缘、角点或非角点。

+   如何使用 OpenCV 函数将这些算法应用于图像并解读结果

如果你有任何问题，请在下面的评论中提出。
