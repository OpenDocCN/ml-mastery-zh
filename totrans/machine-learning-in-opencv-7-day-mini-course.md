# OpenCV 中的机器学习（7 天迷你课程）

> 原文：[`machinelearningmastery.com/machine-learning-in-opencv-7-day-mini-course/`](https://machinelearningmastery.com/machine-learning-in-opencv-7-day-mini-course/)

机器学习是处理许多任务的一个惊人工具。OpenCV 是一个用于图像处理的优秀库。如果我们能将它们结合起来，那就太好了。

在这个 7 部分的速成课程中，你将通过示例学习如何利用机器学习和 OpenCV 的图像处理 API 实现一些目标。这个迷你课程是针对那些已经对 Python 编程感到舒适、了解机器学习基本概念并有一定图像处理背景的从业人员设计的。让我们开始吧。

![](img/e9a40e7e62698f06dc995c46f87d7492.png)

OpenCV 中的机器学习（7 天迷你课程）

图片由 [Nomadic Julien](https://unsplash.com/photos/people-walking-on-street-during-daytime-uBfK5i6j1B8) 提供。保留部分权利。

## 这个迷你课程适合谁？

在我们开始之前，让我们确保你在正确的地方。下面的列表提供了一些关于此课程设计对象的通用指南。如果你不完全符合这些要点，不要惊慌，你可能只需要在某个领域稍作复习以跟上课程进度。

+   **会写一点代码的开发者**。这意味着你能够使用 Python 完成任务，并且知道如何在工作站上设置生态系统（这是一个先决条件）。这并不意味着你是一个代码天才，但意味着你不怕安装软件包和编写脚本。

+   知道一点机器学习的开发者。这意味着你了解一些常见的机器学习算法，如回归或神经网络。这并不意味着你是机器学习博士，仅仅是你知道这些领域的标志性知识或知道在哪里查找它们。

+   **了解一点图像处理的开发者**。这意味着你知道如何读取图像文件，如何操作像素，以及如何裁剪子图像。最好使用 OpenCV。这并不意味着你是图像处理专家，但你理解数字图像是像素数组。

这个迷你课程不是关于机器学习、OpenCV 或数字图像处理的教科书。而是一个项目指南，逐步带你从一个具有最低限度知识的开发者成长为一个可以自信使用 OpenCV 中的机器学习的开发者。

## 迷你课程概述

这个迷你课程分为 7 部分。

每一课的设计时间大约是 30 分钟。你可能会更快完成某些部分，也可能选择深入研究并花费更多时间。

你可以根据自己的节奏完成每个部分。一个舒适的时间表可能是每隔一天完成一节课，共七天。强烈推荐。

接下来的 7 节课中你将涵盖的主题如下：

+   **第 1 课**：OpenCV 介绍

+   **第 2 课**：使用 OpenCV 读取和显示图像

+   **第 3 课**：寻找圆形

+   **课程 4**：提取子图像

+   **课程 5**：匹配硬币

+   **课程 6**：构建硬币分类器

+   **课程 7**：在 OpenCV 中使用 DNN 模块

这将会非常有趣。

不过你需要做一些工作，包括阅读、研究和编程。你想学习机器学习和计算机视觉，对吗？

**在评论中发布你的结果**；我会为你加油！

坚持下去；不要放弃。

## 课程 01：OpenCV 简介

OpenCV 是一个流行的开源图像处理库。它在 Python、C++、Java 和 Matlab 中都有 API 绑定。它提供了数千个函数，并实现了许多先进的图像处理算法。如果你使用 Python，OpenCV 的一个常见替代品是 PIL（Python Imaging Library，或其继任者 Pillow）。与 PIL 相比，OpenCV 提供了更丰富的功能集，并且通常速度更快，因为它是用 C++ 实现的。

这个迷你课程是为了在 OpenCV 中应用机器学习。在本课程的后续课程中，你还需要 TensorFlow/Keras 和 Python 中的 tf2onnx 库。

在这节课中，你的目标是安装 OpenCV。

对于基础的 Python 环境，你可以使用 `pip` 安装软件包。要使用 `pip` 安装 OpenCV、TensorFlow 和 tf2onnx，你可以使用：

```py
sudo pip install opencv-python tensorflow tf2onnx
```

OpenCV 在 PyPI 中被称为 `opencv-python` 包，但它仅包含“免费”算法和主要模块。还有一个名为 `opencv-contrib-python` 的包，其中包含了“额外”模块。这些额外的模块较不稳定且未经充分测试。如果你更愿意安装后者，你应该使用以下命令：

```py
sudo pip install opencv-contrib-python tensorflow tf2onnx
```

然而，如果你使用 Anaconda 或 miniconda 环境，软件包的名称仅为 `opencv`，你可以使用 `conda install` 命令进行安装。

要检查你的 OpenCV 安装是否正常工作，你可以简单地运行一个小脚本并检查其版本：

```py
import cv2
print(cv2.version.opencv_version)
```

想了解更多关于 OpenCV 的信息，你可以从其 [在线文档](https://docs.opencv.org/4.x/) 开始。

### 你的任务

重复上述代码，以确保你已正确安装 OpenCV。你能通过在代码中添加几行来打印 TensorFlow 模块的版本吗？

在下一课中，你将使用 OpenCV 读取和显示图像。

## 课程 02：使用 OpenCV 读取和显示图像

OpenCV，即开源计算机视觉库，是一个强大的图像处理和计算机视觉任务工具。但在深入复杂算法之前，让我们先掌握基础：读取和显示图像。

使用 OpenCV 读取图像是使用 `cv2.imread()` 函数。它接受图像文件的路径并返回一个 NumPy 数组。这个数组通常是三维的，形状为高度×宽度×通道，每个元素是无符号 8 位整数。在 OpenCV 中，“通道”通常是 BGR（蓝-绿-红）。但如果你更愿意以灰度图像加载，你可以添加一个额外的参数，如：

```py
import cv2

image = cv2.imread("path/filename.jpg", cv2.IMREAD_GRAYSCALE)
print(image.shape)
```

上面的代码将打印数组的维度。虽然我们通常将图像描述为宽度×高度，但数组维度描述为高度×宽度。如果图像以灰度形式读取，则只有一个通道，因此输出将是一个二维数组。

如果你移除了第二个参数，使其仅为 `cv2.imread("path/filename.jpg")`，则数组的形状应为高度×宽度×3，表示 BGR 的三个通道。

要显示图像，可以使用 OpenCV 的 `cv2.imshow()` 函数。这将创建一个窗口来显示图像。但是，除非你要求 OpenCV 等待你与窗口互动，否则此窗口将不会显示。通常，你可以使用：

```py
...
cv2.imshow("My image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

`cv2.imshow()` 将窗口标题作为第一个参数。要显示的图像应为 BGR 通道顺序。`cv2.waitKey()` 函数将等待你按键的时间为函数参数中指定的毫秒数。如果为零，它将无限期等待。按键将以整数形式返回其代码点，在这种情况下，你可以忽略它。作为一个好的做法，你应该在程序结束前关闭窗口。

将所有内容组合起来：

```py
import cv2

image = cv2.imread("path/filename.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("My image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 你的任务

修改上面的代码，将路径指向你磁盘中的一张图像并试一下。你如何修改上面的代码以等待 Esc 键被按下，但忽略所有其他按键？（提示：Esc 键的代码点是 27）

在下一课中，你将看到如何在图像中寻找模式。

## 课程 03：寻找圆形

由于数字图像表示为矩阵，你可以设计你的算法并检查图像的每个像素以识别图像中是否存在某种模式。多年来，发明了许多巧妙的算法，你可以在任何数字图像处理教科书中学习其中的一些。

在这个迷你课程中，你将解决一个简单的问题：给定一张包含许多硬币的图像，识别并计数某种特定类型的硬币。硬币是圆形的。要识别图像中的圆形，一种有前景的算法是使用霍夫圆变换。

霍夫变换是一种利用图像的**梯度**信息的算法。因此，它作用于灰度图像而不是彩色图像。要将彩色图像转换为灰度图像，你可以使用 OpenCV 的 `cv2.cvtColor()` 函数。由于霍夫变换基于梯度信息，它对图像噪声非常敏感。应用高斯模糊是一个常见的预处理步骤，用于减少霍夫变换的噪声。在代码中，对于你读取的 BGR 图像，你可以应用以下操作：

```py
...
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (25,25), 1)
```

这里使用 $25\times 25$ 的内核应用高斯模糊。根据图像中的噪声水平，你可以使用较小或较大的内核。

霍夫圆变换用于从图像中寻找圆，使用以下函数：

```py
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                           param1=80, param2=60, minRadius=90, maxRadius=150)
```

有很多参数。第一个参数是灰度图像，第二个参数是要使用的算法。其余的参数如下：

+   `dp`：图像分辨率与累加器分辨率的比例。通常使用 1.0 到 2.0。

+   `minDist`：检测到的圆心之间的最小距离。值越小，假阳性越多。

+   `param1` 这是 Canny 边缘检测器的阈值

+   `param2`：当使用算法 `cv2.HOUGH_GRADIENT` 时，这是累加器阈值。值越小，假阳性越多。

+   `minRadius` 和 `maxRadius`：检测的最小和最大圆半径

从 `cv2.HoughCircles()` 函数返回的值是一个 NumPy 数组，表示为包含中心坐标和半径的行。

让我们尝试 [一个示例图像](https://machinelearningmastery.com/wp-content/uploads/2024/01/coins-1.jpg)：

![](https://machinelearningmastery.com/wp-content/uploads/2024/01/coins-1.jpg)

硬币

下载图像，保存为 `coins-1.jpg`，并运行以下代码：

```py
import cv2
import numpy as np

image_path = "coins-1.jpg"
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (25,25), 1)
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT,
                           dp=1, minDist=100,
                           param1=80, param2=60, minRadius=90, maxRadius=150)
if circles is not None:
    for c in np.uint16(np.round(circles[0])):
        cv2.circle(img, (c[0], c[1]), c[2], (255,0,0), 10)
        cv2.circle(img, (c[0], c[1]), 2, (0,0,255), 20)
cv2.imshow("Circles", img)
cv2.waitKey()
cv2.destroyAllWindows()
```

![](img/2b42283093765e74280c9447a37dfed3.png)

检测到的圆用蓝色表示，圆心用红色标出

上面的代码首先将检测到的圆数据四舍五入并转换为整数。然后在原始图像上绘制这些圆。从上面的示例中，你可以看到霍夫圆变换如何帮助你找到图像中的硬币。

### 进一步阅读

+   P. E. Hart. “霍夫变换的发明”。IEEE 信号处理杂志，26(6)，2009 年 11 月，第 18–22 页。DOI: 10.1109/msp.2009.934181。

+   R. O. Duda 和 P. E. Hart. “使用霍夫变换检测图像中的直线和曲线”。通讯 ACM，15，1 月 11–15。DOI: 10.1145/361237.361242。

### 你的任务

检测对你提供给 `cv2.HoughCircles()` 函数的参数非常敏感。尝试修改参数并查看结果。你也可以尝试为不同的图片找到最佳参数，尤其是不同光照条件或不同分辨率的图片。

在下一节中，你将看到如何根据检测到的圆从图像中提取硬币。

## 第 04 课：提取子图像

使用 OpenCV 读取的图像是一个形状为高度×宽度×通道的 NumPy 数组。要提取图像的一部分，你可以简单地使用 NumPy 切片语法。例如，从 BGR 彩色图像中，你可以用以下代码提取红色通道：

```py
red = img[:, :, 2]
```

因此，要提取图像的一部分，你可以使用

```py
subimg = img[y0:y1, x0:x1]
```

这样你会得到较大图像的一个矩形部分。请记住，在矩阵中，你首先从上到下计算垂直元素（像素），然后从左到右计算水平元素。因此，你应在切片语法中首先描述 $y$-坐标范围。

让我们修改前一节中的代码，提取我们找到的每个硬币：

```py
import cv2
import numpy as np

image_path = "coins-1.jpg"
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (25,25), 1)
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                           param1=80, param2=60, minRadius=90, maxRadius=150)
for c in np.uint16(np.round(circles[0])):
    x, y, r = c
    subimg = img[y-r:y+r, x-r:x+r]
    cv2.imshow("Coin", subimg)
    cv2.waitKey(0)
cv2.destroyAllWindows()
```

这段代码为霍夫变换找到的每个圆提取一个正方形子图像。然后在窗口中显示该子图像，并等待你按键后再显示下一个。

![](img/270da6ee70892a159503a892bffba9d8.png)

OpenCV 窗口显示检测到的硬币

### 你的任务

运行此代码。你会发现每个检测到的圆圈可能大小不同，提取的子图像也是如此。在显示它们之前，你如何将子图像调整为一致的大小？

在下一课中，你将学习如何将提取的子图像与参考图像进行比较。

## 课时 05：匹配硬币

我们的任务是从图像中识别并计数便士硬币。你可以在[维基百科上找到美国便士的图像。](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Lincoln-Cent-Reverse-sheild.png/240px-Lincoln-Cent-Reverse-sheild.png)

以此为参考图像，如何将识别出的硬币与参考图像进行比较？这比听起来更复杂。使用过的硬币可能会生锈、暗淡或有划痕。图片中的硬币可能会旋转。比较像素并判断它们是否为同一硬币并不容易。

更好的方法是使用关键点匹配算法。OpenCV 中有几种关键点算法。我们来尝试使用 ORB，它是 OpenCV 团队的一个发明。从上述链接下载参考图像为`penny.png`，你可以使用以下代码提取关键点和关键点描述符：

```py
import cv2

reference_path = "penny.png"
sample = cv2.imread(reference_path)
orb = cv2.ORB_create(nfeatures=500)
kp, ref_desc = orb.detectAndCompute(sample, None)
```

元组`kp`是关键点对象，但不像`ref_desc`数组那样重要，后者是**关键点描述符**。这是一个形状为$K\times 32$的 NumPy 数组，其中$K$为检测到的关键点数。每个关键点的 ORB 描述符是一个 32 个整数的向量。

如果你从另一张图像中获取描述符，你可以将其与已有的描述符进行比较。你不应该期望描述符完全匹配。相反，你可以应用[**Lowe 比率测试**](https://stackoverflow.com/questions/51197091/)来决定关键点是否匹配：

```py
...
bf = cv2.BFMatcher()
kp, desc = orb.detectAndCompute(coin_gray, None)
matches = bf.knnMatch(ref_desc, desc, k=2)
count = len([1 for m, n in matches if m.distance < 0.80*n.distance])
```

在这里，你使用`cv2.BFMatcher()`的蛮力匹配器运行 kNN 算法，以获取每个参考关键点到候选图像中关键点的两个最近邻。然后比较向量距离（距离较短表示匹配较好）。Lowe 比率测试用于判断匹配是否足够好。你可以尝试不同于 0.8 的常数。我们计算良好匹配的数量，如果找到足够的良好匹配，我们就认为硬币被识别出来了。

以下是完整的代码，我们将使用 Matplotlib 显示并识别每个找到的硬币。由于 Matplotlib 期望的是 RGB 通道的彩色图像而不是 BGR 通道，你需要在使用`plt.imshow()`显示图像之前，使用`cv2.cvtColor()`将图像转换为 RGB。

```py
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "coins-1.jpg"
reference_path = "penny.png"

img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (25,25), 1)
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                           param1=80, param2=60, minRadius=90, maxRadius=150)

sample = cv2.imread(reference_path)
orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher()
kp, ref_desc = orb.detectAndCompute(sample, None)

plt.figure(2)
N = len(circles[0])
rows = math.ceil(N / 4)
for i, c in enumerate(np.uint16(np.around(circles[0]))):
    x, y, r = c
    coin = img[y-r:y+r, x-r:x+r]
    coin_gray = cv2.cvtColor(coin, cv2.COLOR_BGR2GRAY)
    kp, desc = orb.detectAndCompute(coin_gray, None)
    matches = bf.knnMatch(ref_desc, desc, k=2)
    count = len([1 for m, n in matches if m.distance < 0.80*n.distance])
    plt.subplot(rows, 4, i+1)
    plt.imshow(cv2.cvtColor(coin, cv2.COLOR_BGR2RGB))
    plt.title(f"{count}")
    plt.axis('off')
plt.show()
```

![](img/79dcb6d1ab1c805135ec1de815afb83d.png)

检测到的硬币和匹配的关键点数量

你可以看到，匹配的关键点数量无法提供一个明确的度量来帮助识别便士硬币与其他硬币。你能想到其他可能有用的算法吗？

### 进一步阅读

+   [特征匹配。OpenCV。](https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html)

+   [Lowe 的比率测试是如何工作的？Stack Overflow。](https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work)

### 你的任务

上面的代码使用了 ORB 关键点。你也可以尝试使用 SIFT 关键点。如何修改上面的代码？你会看到关键点匹配数量的变化吗？另外，由于 ORB 特征是向量。你能构建一个逻辑回归分类器来识别好的关键点，从而不需要依赖参考图像吗？

在下一节课中，你将处理一个更好的硬币识别器。

## 课程 06：构建一个硬币分类器

提供一枚硬币的图像并确定它是否为美国便士硬币，对人类来说很容易，但对计算机来说却不那么简单。已知进行这种分类的最佳方法是使用机器学习。你应该首先从图像中提取特征向量，然后运行机器学习算法作为分类器，以判断它是否匹配。

决定使用哪种特征本身就是一个困难的问题。但是如果你使用卷积神经网络，你可以让机器学习算法自动找出特征。然而，训练神经网络需要数据。幸运的是，你不需要很多数据。让我们看看你如何构建一个。

首先，你可以在这里查看一些硬币的图片：

+   [`machinelearningmastery.com/wp-content/uploads/2024/01/coins-2.jpg`](https://machinelearningmastery.com/wp-content/uploads/2024/01/coins-2.jpg)

+   [`machinelearningmastery.com/wp-content/uploads/2024/01/coins-3.jpg`](https://machinelearningmastery.com/wp-content/uploads/2024/01/coins-3.jpg)

![](img/3fa52e51d85e30572209207a0015d874.png)

要提取的硬币图片作为数据集用于训练神经网络

将它们保存为`coins-2.jpg`和`coins-3.jpg`，然后使用霍夫圆变换提取硬币的图像，并将它们保存到一个名为`dataset`的目录中：

```py
import cv2
import numpy as np

idx = 1
for image_path in ["coins-2.jpg", "coins-3.jpg"]:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (25,25), 1)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT,
                               dp=1, minDist=100,
                               param1=80, param2=60, minRadius=90, maxRadius=150)

    for c in np.uint16(np.around(circles[0])):
        x, y, r = c
        cv2.imwrite(f"dataset/{idx}.jpg", img[y-r:y+r, x-r:x+r])
        idx += 1
```

图像数量不多。你可以手动将每张图像标记为便士硬币（正样本）或非便士硬币（负样本）。一种方法是将正样本移动到子目录`dataset/pos`中，将负样本移动到`dataset/neg`中。为了方便你，你可以在[这里](https://machinelearningmastery.com/wp-content/uploads/2024/01/coins.zip)找到标记过的图像的压缩文件副本。

使用这些，我们来构建一个卷积神经网络来解决这个二分类问题，使用 Keras 和 TensorFlow。

在数据准备阶段，你将读取每个正样本和负样本图像。为了简化卷积神经网络，你将输入大小固定为$256\times 256$像素，通过先调整图像大小来实现。为了增加数据集的变化性，你可以将每张图像旋转 90 度、180 度和 270 度，并将其添加到数据集中（因为图像样本都是正方形的，这很简单）。然后，你可以利用 scikit-learn 中的`train_test_split()`函数将数据集分为训练集和测试集，比例为 7:3。

为了创建模型，你可以使用多个 Conv2D 层和 MaxPooling 的分类架构，然后在输出层后跟 Dense 层。请注意这是一个二分类模型。因此，在最终输出层，你应该使用 sigmoid 激活函数。

在训练时，你可以简单地使用大量的迭代（例如，`epochs=200`）并设置早期停止，以避免担心过拟合。你应该监控在测试集上评估的损失，以确保不会出现过拟合。在代码中，你可以这样训练模型并将其保存为`penny.h5`：

```py
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential

images = []
labels = []
for filename in glob.glob("dataset/pos/*"):
    img = cv2.imread(filename)
    img = cv2.resize(img, (256,256))
    images.append(img)
    labels.append(1)
    for _ in range(3):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        images.append(img)
        labels.append(1)
for filename in glob.glob("dataset/neg/*"):
    img = cv2.imread(filename)
    img = cv2.resize(img, (256,256))
    images.append(img)
    labels.append(0)
    for _ in range(3):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        images.append(img)
        labels.append(0)

images = np.array(images)
labels = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3)

model = Sequential([
    Conv2D(16, (5,5), input_shape=(256,256,3), padding="same", activation="relu"),
    MaxPooling2D((2,2)),
    Conv2D(32, (5,5), activation="relu"),
    MaxPooling2D((2,2)),
    Conv2D(64, (5,5), activation="relu"),
    MaxPooling2D((2,2)),
    Conv2D(128, (5,5), activation="relu"),
    Flatten(),
    Dense(256, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Training
earlystopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
model.compile(loss="binary_crossentropy", optimizer="adagrad", metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=200, batch_size=32, callbacks=[earlystopping])
model.save("penny.h5")
```

观察其输出，你应该很容易看到准确率在几次迭代后超过 90%。

### 你的任务

运行上述代码并创建一个训练好的模型`penny.h5`，你将在下一课中使用它。

你可以修改模型设计，看看是否能提高准确率。你可以尝试的一些想法包括：使用不同数量的 Conv2D-MaxPooling 层、不同的层大小（例如 16-32-64-128），或使用 ReLU 以外的激活函数。

在下一课中，你将把你在 Keras 中创建的模型转换为 OpenCV 使用的格式。

## 第 07 课：在 OpenCV 中使用 DNN 模块

假设你在上一课中已经构建了一个卷积神经网络，你现在可以将其与 OpenCV 一起使用。如果你首先将其转换为 ONNX 格式，OpenCV 会更容易使用你的模型。为此，你需要 Python 的 tf2onnx 模块。安装完成后，你可以使用以下命令转换模型：

```py
python -m tf2onnx.convert --keras penny.h5 --output penny.onnx
```

由于你将 Keras 模型保存为`penny.h5`，此命令将创建文件`penny.onnx`。

使用 ONNX 模型文件，你现在可以使用 OpenCV 的`cv2.dnn`模块。使用方法如下：

```py
import cv2

net = cv2.dnn.readNetFromONNX("penny.onnx")
net.setInput(blob)
output = float(net.forward())
```

也就是说，你使用 OpenCV 创建一个神经网络对象，分配输入，并使用`forward()`运行模型以获取输出，这个输出是 0 到 1 之间的浮点值，具体取决于你设计模型的方式。作为神经网络的惯例，即使你只提供一个输入样本，输入也是**批处理**的。因此，在将图像发送到神经网络之前，你应该为图像添加一个批处理维度。

现在让我们看看如何实现硬币计数的目标。你可以从修改第 05 课的代码开始，如下所示：

```py
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "coins-1.jpg"
model_path = "penny.onnx"

img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (25,25), 1)
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT,
                           dp=1, minDist=100,
                           param1=80, param2=60, minRadius=90, maxRadius=150)

plt.figure(2)
N = len(circles[0])
rows = math.ceil(N / 4)
net = cv2.dnn.readNetFromONNX("penny2.onnx")
for i, c in enumerate(np.uint16(np.around(circles[0]))):
    x, y, r = c
    coin = img[y-r:y+r, x-r:x+r]
    coin = cv2.resize(coin, (256,256))
    blob = coin[np.newaxis, ...]
    net.setInput(blob)
    score = float(net.forward())
    plt.subplot(rows, 4, i+1)
    plt.imshow(cv2.cvtColor(coin, cv2.COLOR_BGR2RGB))
    plt.title(f"{score:.2f}")
    plt.axis('off')
plt.show()
```

你使用了卷积神经网络来读取 sigmoidal 输出，而不是使用 ORB 并计算匹配的好关键点。输出如下：

![](img/771be27b59b976bf3e3a956b4f503330.png)

神经网络检测到的硬币及其匹配得分

你可以看到这个模型的效果相当好。所有硬币都被识别，得分接近 1。负样本效果不如预期（可能是因为我们提供的负样本不够）。让我们使用 0.9 作为得分截断值，并重写程序以进行计数：

```py
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "coins-1.jpg"
model_path = "penny.onnx"

img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (25,25), 1)
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                           param1=80, param2=60, minRadius=90, maxRadius=150)
positive = 0
negative = 0
net = cv2.dnn.readNetFromONNX(model_path)
for i, c in enumerate(np.uint16(np.around(circles[0]))):
    x, y, r = c
    coin = img[y-r:y+r, x-r:x+r]
    coin = cv2.resize(coin, (256,256))
    blob = coin[np.newaxis, ...]
    net.setInput(blob)
    score = float(net.forward())
    if score >= 0.9:
        positive += 1
    else:
        negative += 1
print(f"{positive} out of {positive+negative} coins identified are pennies")
```

### 你的任务

运行上述代码进行测试。你可以尝试使用另一张图片，如下所示：

+   [`machinelearningmastery.com/wp-content/uploads/2024/01/coins-4.jpg`](https://machinelearningmastery.com/wp-content/uploads/2024/01/coins-4.jpg)

你能修改上面的代码，通过不断从你的摄像头读取图像来报告计数吗？

这是最后一节课。

## 完结！（*看看你走了多远*)

你完成了，干得好！

花点时间回顾一下你走了多远。

+   你发现了 OpenCV 作为一个机器学习库，除了其图像处理能力之外。

+   你利用 OpenCV 提取图像特征作为数值向量，这是任何机器学习算法的基础。

+   你构建了一个神经网络模型，并将其转换为与 OpenCV 兼容。

+   最后，你构建了一个硬币计数程序。虽然不完美，但它展示了如何将 OpenCV 与机器学习结合起来。

不要轻视这一点，你在短时间内取得了长足的进步。这只是你计算机视觉和机器学习之旅的开始。继续练习和提升你的技能。

## 总结

**你在这个迷你课程中的表现如何？**

你喜欢这个速成课程吗？

**你有任何问题吗？有没有什么难点？**

告诉我。请在下方留言。
