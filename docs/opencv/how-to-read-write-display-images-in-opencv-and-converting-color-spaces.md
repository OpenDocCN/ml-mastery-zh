# 如何在 OpenCV 中读取、写入、显示图像以及转换颜色空间

> 原文：[`machinelearningmastery.com/how-to-read-write-display-images-in-opencv-and-converting-color-spaces/`](https://machinelearningmastery.com/how-to-read-write-display-images-in-opencv-and-converting-color-spaces/)

在处理图像时，一些最基本的操作包括从磁盘读取图像、显示图像、访问其像素值以及在颜色空间之间转换。

本教程解释了这些基本操作，从描述数字图像如何通过其空间坐标和强度值来构建开始。

在本教程中，你将熟悉在处理图像时至关重要的最基本的 OpenCV 操作。

完成本教程后，你将了解到：

+   数字图像如何通过其空间坐标和强度值来构建。

+   如何在 OpenCV 中读取和显示图像。

+   如何访问图像的像素值。

+   图像如何从一种颜色空间转换到另一种颜色空间。

**启动你的项目**，可以参考我的书籍 [Machine Learning in OpenCV](https://machinelearning.samcart.com/products/machine-learning-opencv/)。它提供了**自学教程**和**可运行的代码**。

让我们开始吧。 ![](https://machinelearningmastery.com/wp-content/uploads/2022/12/image_basics_cover-scaled.jpg)

使用 OpenCV 读取和显示图像，以及在颜色空间之间转换

图片由 [Andrew Ridley](https://unsplash.com/photos/jR4Zf-riEjI) 提供，部分权利保留。

## **教程概述**

本教程分为三个部分，它们是：

+   图像的构建

+   在 OpenCV 中读取和显示图像

+   颜色空间转换

## **图像的构建**

数字图像由像素组成，每个像素由其*空间坐标*和*强度*或*灰度级*值来表征。

本质上，图像可以通过一个二维函数 *I*(*x*, *y*) 来描述，其中 *x* 和 *y* 表示上述空间坐标，*I* 在任何图像位置 (*x*, *y*) 的值表示像素强度。在数字图像中，空间坐标以及强度值都是有限的、离散的量。

我们刚刚描述的数字图像类型称为*灰度*图像，这是因为它由一个单一的通道组成，像素值仅包含强度信息。像素强度通常由范围为 [0, 255] 的整数值表示，这意味着每个像素可以取 256 个离散值中的任何一个。

另一方面，RGB 图像由三个通道组成，即*红色*、*绿色*和*蓝色*。

RGB 颜色模型并不是唯一存在的模型，但它可能是许多计算机视觉应用中最常用的。它是一个加色模型，指的是通过混合（或添加）不同颜色源的光谱来创建颜色的过程。

由于 RGB 图像由三个通道组成，因此我们需要三个函数来描述它：*I**[R]*(*x*, *y*), *I**[G]*(*x*, *y*) 和 *I**[B]*(*x*, *y*)，分别对应红色、绿色和蓝色通道。因此，在 RGB 图像中，每个像素值由三个强度值的三元组表示。

## **在 OpenCV 中读取和显示图像**

首先，导入 Python 中 OpenCV 库的`imread`方法：

Python

```py
from cv2 import imread
```

然后继续读取一张 RGB 图像。为此，我下载了[这张图片](https://unsplash.com/photos/N04FIfHhv_k)并将其保存到磁盘，文件名为*Dog.jpg*，保存在一个名为*Images*的文件夹中。

Python

```py
img = imread('Images/Dog.jpg')
```

`imread`方法返回一个包含图像像素值的 NumPy 数组`img`。我们可以如下检查数组的数据类型和维度：

Python

```py
print('Datatype:', img.dtype, '\nDimensions:', img.shape)
```

Python

```py
Datatype: uint8 
Dimensions: (4000, 6000, 3)
```

返回的信息告诉我们数组的数据类型是 uint8，这意味着我们处理的是 8 位无符号整数值。这表示图像中每个通道的像素可以取任意 2⁸ = 256 个值，范围从 0 到 255。这与我们上面审查的图像格式完全一致。我们还了解到数组的维度是 4000 × 6000 × 3，分别对应图像的行数、列数和通道数。

该图像是一个 3 维的 NumPy 数组。因此，你可以使用 NumPy 语法来操作这个数组。

现在尝试访问位于图像左上角的第一个像素的值。请记住，Python 中的数组是从零开始索引的，因此该像素的坐标是(0, 0)。

Python

```py
print(img[0, 0])
```

Python

```py
[173 186 232]
```

从输出中可以看到，如预期的那样，每个像素携带三个值，每个值对应图像中的三个通道之一。我们将在下一部分发现这三个值分别对应哪个特定通道。

**注意**：一个重要的点是，如果`imread`方法无法加载输入图像（因为提供的图像名称不存在或其路径无效），它不会自行生成错误，而是返回一个`NoneType`对象。因此，可以在继续运行进一步的代码之前，包含以下检查，以确保`img`值的有效性：

Python

```py
if img is not None:
    ...
```

接下来，我们将使用 Matplotlib 包和 OpenCV 的 `imshow` 方法显示图像。后者的第一个参数是包含图像的窗口的名称，第二个参数是要显示的图像。我们还将在图像显示后调用 OpenCV 的 `waitkey` 函数，该函数会等待指定的毫秒数的键盘事件。如果输入值为 0，则 `waitkey` 函数将无限等待，允许我们看到显示的窗口，直到生成键盘事件。

+   **使用 Matplotlib：**

**Python

```py
import matplotlib.pyplot as plt

plt.imshow(img)
plt.title('Displaying image using Matplotlib')
plt.show()
```

![](https://machinelearningmastery.com/wp-content/uploads/2022/12/image_basics_1.png)

使用 Matplotlib 显示 BGR 图像。

+   **使用 OpenCV：**

**Python

```py
from cv2 import imshow, waitKey

imshow('Displaying image using OpenCV', img) 
waitKey(0)
```

![](https://machinelearningmastery.com/wp-content/uploads/2022/12/image_basics_2.png)

使用 OpenCV 显示 BGR 图像。

如果你对 Matplotlib 生成的输出感到惊讶并想知道发生了什么原因，这主要是因为 OpenCV 以 BGR 而不是 RGB 顺序读取和显示图像。

> *OpenCV 的初始开发者选择了 BGR 颜色格式（而不是 RGB 格式），因为当时 BGR 颜色格式在软件提供商和相机制造商中非常流行。*
> 
> *[掌握 OpenCV 4 与 Python](https://www.amazon.com/Mastering-OpenCV-Python-practical-processing/dp/1789344913)，2019 年。*

*另一方面，Matplotlib 使用 RGB 颜色格式，因此需要先将 BGR 图像转换为 RGB 才能正确显示。*

### 想开始使用 OpenCV 进行机器学习吗？

现在就获取我的免费电子邮件速成课程（包含示例代码）。

点击注册并获取免费的 PDF 电子书版本课程。

使用 OpenCV，你还可以将 NumPy 数组写入文件中，如下所示：

```py
from cv2 import imwrite
imwrite("output.jpg", img)
```

当你使用 OpenCV 的 `imwrite()` 函数写入图像时，必须确保 NumPy 数组的格式符合 OpenCV 的要求，即它是一个具有 BGR 通道顺序的 uint8 类型的 3 维数组，格式为行 × 列 × 通道。*

## **颜色空间之间的转换**

从一种颜色空间转换到另一种颜色空间可以通过 OpenCV 的 `cvtColor` 方法实现，该方法将源图像作为输入参数，并使用颜色空间转换代码。

为了在 BGR 和 RGB 颜色空间之间转换，我们可以使用以下代码：

Python

```py
from cv2 import cvtColor, COLOR_BGR2RGB

img_rgb = cvtColor(img, COLOR_BGR2RGB)
```

如果我们需要重新尝试使用 Matplotlib 显示图像，我们现在可能会看到它正确显示：

Python

```py
plt.imshow(img_rgb)
plt.show()
```

![](https://machinelearningmastery.com/wp-content/uploads/2022/12/image_basics_3.png)

将 BGR 图像转换为 RGB 并使用 Matplotlib 显示它。

如果我们还需要访问新转换的 RGB 图像的第一个像素的值：

Python

```py
print(img_rgb[0, 0])
```

Python

```py
[232 186 173]
```

比较这些值与我们之前为 BGR 图像打印的值 [173 186 232]，我们可能会注意到第一个和第三个值现在已交换。这告诉我们的是，值的顺序与图像通道的顺序相对应。

BGR 转 RGB 并不是通过这种方法实现的唯一颜色转换。事实上，还有许多颜色空间转换代码可供选择，例如 `COLOR_RGB2HSV`，它在 RGB 和 HSV（色相、饱和度、明度）颜色空间之间进行转换。

另一个常见的转换是将 RGB 转换为灰度图像，正如我们之前提到的，得到的结果应该是一个单通道图像。我们来试一下：

Python

```py
from cv2 import COLOR_RGB2GRAY

img_gray = cvtColor(img_rgb, COLOR_RGB2GRAY)

imshow(‘Grayscale Image', img_gray)
waitKey(0)
```

![](https://machinelearningmastery.com/wp-content/uploads/2022/12/image_basics_4.png)

如何将 RGB 图像转换为灰度图像并使用 OpenCV 显示它。

转换似乎已成功完成，但我们也来尝试访问坐标（0, 0）处第一个像素的值：

Python

```py
print(img_gray[0, 0])
```

Python

```py
198
```

正如预期的那样，只打印出一个对应于像素强度值的单一数字。

值得注意的是，这并不是将图像转换为灰度图像的唯一方法。实际上，如果我们要处理的应用程序只需要使用灰度图像（而不是 RGB 图像），我们也可以选择直接以灰度图像的形式读取图像：

Python

```py
from cv2 import IMREAD_GRAYSCALE

img_gray = imread('Images/Dog.jpg', IMREAD_GRAYSCALE)

imshow(‘Grayscale Image', img_gray)
waitKey(0)
```

**注意**：这里的 OpenCV 文档警告说，使用 `IMREAD_GRAYSCALE` 会利用编解码器的内部灰度转换（如果可用），这可能会导致与 `cvtColor()` 的输出不同。

`imread` 方法还支持其他几种标志值，其中两个是 `IMREAD_COLOR` 和 `IMREAD_UNCHANGED`。`IMREAD_COLOR` 标志是默认选项，它将图像转换为 BGR 颜色，忽略任何透明度。而 `IMREAD_UNCHANGED` 则读取可能包含 alpha 通道的图像。

## **进一步阅读**

本节提供了更多关于该主题的资源，如果你想深入了解。

### **书籍**

+   [掌握 OpenCV 4 与 Python](https://www.amazon.com/Mastering-OpenCV-Python-practical-processing/dp/1789344913)，2019 年。

### **网站**

+   OpenCV，[`opencv.org/`](https://opencv.org/)

+   OpenCV 颜色转换代码，[`docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#func-members`](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#func-members)

## **总结**

在本教程中，你熟悉了在处理图像时必不可少的最基本的 OpenCV 操作。

具体来说，你学到了：

+   数字图像如何从空间坐标和强度值的角度进行构造。

+   如何在 OpenCV 中读取和显示图像。

+   如何访问图像的像素值。

+   图像如何从一种颜色空间转换到另一种颜色空间。

你有任何问题吗？

请在下面的评论中提出你的问题，我将尽力回答。*****
