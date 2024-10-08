# OpenCV 中的图像特征提取：关键点和描述向量

> 原文：[`machinelearningmastery.com/opencv_sift_surf_orb_keypoints/`](https://machinelearningmastery.com/opencv_sift_surf_orb_keypoints/)

在 [上一篇文章](https://machinelearningmastery.com/opencv_edges_and_corners/) 中，你学习了一些 OpenCV 中的基本特征提取算法。特征以分类像素的形式提取。这确实从图像中抽象出特征，因为你不需要考虑每个像素的不同颜色通道，而是考虑一个单一的值。在这篇文章中，你将学习一些其他特征提取算法，它们可以更简洁地告诉你关于图像的信息。

完成本教程后，你将了解：

+   图像中的关键点是什么

+   在 OpenCV 中用于提取关键点的常见算法有哪些

**通过我的书籍** [《OpenCV 中的机器学习》](https://machinelearning.samcart.com/products/machine-learning-opencv/) **启动你的项目**。它提供了 **自学教程** 和 **有效的代码**。

让我们开始吧。![](img/024d2c09a7b79bfc4ab5d65e42396879.png)

OpenCV 中的图像特征提取：关键点和描述向量

图片由 [Silas Köhler](https://unsplash.com/photos/black-skeleton-keys-C1P4wHhQbjM) 提供，部分权利保留。

## **概述**

本文分为两个部分；它们是：

+   使用 SIFT 和 SURF 进行 OpenCV 中的关键点检测

+   使用 ORB 进行 OpenCV 中的关键点检测

## **先决条件**

对于本教程，我们假设你已经熟悉：

+   [使用 OpenCV 读取和显示图像](https://machinelearningmastery.com/?p=14402&preview=true)

## 使用 SIFT 和 SURF 进行 OpenCV 中的关键点检测

尺度不变特征变换（SIFT）和加速稳健特征（SURF）是用于检测和描述图像中局部特征的强大算法。它们被称为尺度不变和鲁棒，是因为与 Harris 角点检测相比，其结果在图像发生某些变化后仍然是可以预期的。

SIFT 算法对图像应用高斯模糊，并计算多个尺度下的差异。直观上，如果整个图像是单一的平面颜色，这种差异将为零。因此，这个算法被称为关键点检测，它识别图像中像素值变化最显著的地方，例如角点。

SIFT 算法为每个关键点推导出某些“方向”值，并输出表示方向值直方图的向量。

运行 SIFT 算法的速度比较慢，因此有一个加速版本，即 SURF。详细描述 SIFT 和 SURF 算法会比较冗长，但幸运的是，你不需要了解太多就可以在 OpenCV 中使用它。

让我们通过以下图像看一个例子：

+   [`unsplash.com/photos/VSLPOL9PwB8`](https://unsplash.com/photos/VSLPOL9PwB8)

与之前的帖子类似，SIFT 和 SURF 算法假设图像为灰度图像。这次，你需要首先创建一个检测器，并将其应用于图像：

```py
import cv2

# Load the image and convery to grayscale
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize SIFT and SURF detectors
sift = cv2.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()

# Detect key points and compute descriptors
keypoints_sift, descriptors_sift = sift.detectAndCompute(img, None)
keypoints_surf, descriptors_surf = surf.detectAndCompute(img, None)
```

**注意：** 你可能会发现运行上述代码时在你的 OpenCV 安装中遇到困难。为了使其运行，你可能需要从头编译自己的 OpenCV 模块。这是因为 SIFT 和 SURF 已经申请了专利，所以 OpenCV 认为它们是“非自由”的。由于 SIFT 专利已过期（SURF 仍在有效期内），如果你下载一个更新版本的 OpenCV，你可能会发现 SIFT 运行正常。

SIFT 或 SURF 算法的输出是一个关键点列表和一个描述符的 numpy 数组。描述符数组是 Nx128，对于 N 个关键点，每个由长度为 128 的向量表示。每个关键点是一个具有多个属性的对象，例如方向角度。

默认情况下可以检测到许多关键点，因为这有助于关键点的最佳用途之一——寻找失真图像之间的关联。

为了减少输出中检测到的关键点数量，你可以在 SIFT 中设置更高的“对比度阈值”和更低的“边缘阈值”（默认值分别为 0.03 和 10），或者在 SURF 中增加“Hessian 阈值”（默认值为 100）。这些可以通过 `sift.setContrastThreshold(0.03)`、`sift.setEdgeThreshold(10)` 和 `surf.setHessianThreshold(100)` 进行调整。

要在图像上绘制关键点，你可以使用 `cv2.drawKeypoints()` 函数，并将所有关键点的列表应用于它。完整代码，使用仅 SIFT 算法并设置非常高的阈值以保留少量关键点，如下所示：

```py
import cv2

# Load the image and convery to grayscale
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()
sift.setContrastThreshold(0.25)
sift.setEdgeThreshold(5)

# Detect key points and compute descriptors
keypoints, descriptors = sift.detectAndCompute(img, None)
for x in keypoints:
    print("({:.2f},{:.2f}) = size {:.2f} angle {:.2f}".format(x.pt[0], x.pt[1], x.size, x.angle))

img_kp = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Keypoints", img_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

创建的图像如下：

![](img/b24e32936e21c3ce59b89b1263f1df22.png)

SIFT 算法检测到的关键点（放大查看）

原始照片由 [Gleren Meneghin](https://unsplash.com/photos/VSLPOL9PwB8) 提供，部分权利保留。

`cv2.drawKeypoints()` 函数不会修改你的原始图像，而是返回一个新图像。在上面的图片中，你可以看到关键点被绘制为与其“大小”成比例的圆圈，并有一个表示方向的描边。门上的“17”号以及邮件槽上都有关键点，但实际上还有更多。从上面的 for 循环中，你可以看到一些关键点重叠，因为发现了多个方向角度。

在图像上显示关键点时，你使用了返回的关键点对象。然而，如果你想进一步处理关键点，例如运行聚类算法，你可能会发现存储在 `descriptors` 中的特征向量很有用。但请注意，你仍然需要关键点列表中的信息，例如坐标，以匹配特征向量。

## 使用 OpenCV 中的 ORB 进行关键点检测

由于 SIFT 和 SURF 算法已经申请了专利，因此有开发无需许可的免费替代品的动力。这是 OpenCV 开发者自己开发的产品。

ORB 代表定向 FAST 和旋转 BRIEF。它是两个其他算法 FAST 和 BRIEF 的组合，并进行了修改以匹配 SIFT 和 SURF 的性能。你无需了解算法细节就可以使用它，输出结果也是一个关键点对象的列表，如下所示：

```py
import cv2

# Load the image and convery to grayscale
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize ORB detector
orb = cv2.ORB_create(30)

# Detect key points and compute descriptors
keypoints, descriptors = orb.detectAndCompute(img, None)
for x in keypoints:
    print("({:.2f},{:.2f}) = size {:.2f} angle {:.2f}".format(
            x.pt[0], x.pt[1], x.size, x.angle))

img_kp = cv2.drawKeypoints(img, keypoints, None,
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Keypoints", img_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

在上面的操作中，你设置了 ORB 以在创建探测器时生成前 30 个关键点。默认情况下，这个数字是 500。

探测器返回的仍然是关键点列表和描述符的 numpy 数组（每个关键点的特征向量）。然而，现在每个关键点的描述符长度为 32，而不是 128。

生成的关键点如下：

![](img/dac0ab34ca23dab6ba83dbb937ec3667.png)

ORB 算法检测的关键点

原始照片由 [Gleren Meneghin](https://unsplash.com/photos/VSLPOL9PwB8) 提供，部分权利保留。

你可以看到，关键点大致生成在相同的位置。结果并不完全相同，因为存在重叠的关键点（或偏移了非常小的距离），ORB 算法容易达到 30 的最大数量。此外，不同算法之间的大小不可比较。

### 想要开始使用 OpenCV 进行机器学习吗？

现在立即获取我的免费电子邮件速成课程（附样例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## **进一步阅读**

本节提供了更多资源，如果你想深入了解这个话题。

### **书籍**

+   [Mastering OpenCV 4 with Python](https://www.amazon.com/Mastering-OpenCV-Python-practical-processing/dp/1789344913)，2019 年。

### **网站**

+   OpenCV，[`opencv.org/`](https://opencv.org/)

+   OpenCV 特征检测与描述，[`docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html`](https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html)

## **总结**

在本教程中，你学习了如何应用 OpenCV 的关键点检测算法，SIFT、SURF 和 ORB。

具体来说，你学到了：

+   图像中的关键点是什么

+   如何使用 OpenCV 函数查找关键点及其相关描述向量。

如果你有任何问题，请在下方留言。
