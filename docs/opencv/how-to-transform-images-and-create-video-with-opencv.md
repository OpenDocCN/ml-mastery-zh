# 如何使用 OpenCV 转换图像并创建视频

> 原文：[`machinelearningmastery.com/how-to-transform-images-and-create-video-with-opencv/`](https://machinelearningmastery.com/how-to-transform-images-and-create-video-with-opencv/)

当你使用 OpenCV 时，你最常处理的是图像。然而，你可能会发现从多个图像创建动画是有用的。展示快速连续的图像可能会给你不同的见解，或者通过引入时间轴让你更容易可视化你的工作。

在这篇文章中，你将看到如何在 OpenCV 中创建视频剪辑。作为示例，你还将学习一些基本的图像处理技术来创建这些图像。特别是，你将学习：

+   如何将图像作为 numpy 数组进行操作

+   如何使用 OpenCV 函数操作图像

+   如何在 OpenCV 中创建视频文件

**快速启动你的项目**，请参阅我的书籍 [《OpenCV 中的机器学习》](https://machinelearning.samcart.com/products/machine-learning-opencv/)。它提供了**自学教程**和**可用代码**。

让我们开始吧。![](img/131a94182e4a8bf20db4d9fd07f527c8.png)

如何使用 OpenCV 转换图像并创建视频

图片由 [KAL VISUALS](https://unsplash.com/photos/man-in-black-shirt-holding-video-camera-I-nd-LSCY04) 提供。一些权利保留。

## 概述

本文分为两个部分；它们是：

+   Ken Burns 效果

+   编写视频

## Ken Burns 效果

你将通过参考其他帖子创建大量图像。也许这是为了可视化你机器学习项目的一些进展，或者展示计算机视觉技术如何操作你的图像。为了简化问题，你将对输入图像进行最简单的操作：裁剪。

本文的任务是创建**Ken Burns 效果**。这是一种以电影制作人 Ken Burns 命名的平移和缩放技术：

> Ken Burns 效果不是在屏幕上显示一张大的静态照片，而是裁剪到一个细节，然后在图像上平移。
> 
> — 维基百科，“Ken Burns 效果”

让我们看看如何使用 OpenCV 在 Python 代码中创建 Ken Burns 效果。我们从一张图像开始，例如下面这张可以从维基百科下载的鸟类图片：

+   [`upload.wikimedia.org/wikipedia/commons/b/b7/Hooded_mountain_tanager_%28Buthraupis_montana_cucullata%29_Caldas.jpg`](https://upload.wikimedia.org/wikipedia/commons/b/b7/Hooded_mountain_tanager_%28Buthraupis_montana_cucullata%29_Caldas.jpg)

![](img/f165edf1ba488d20835a3b5d9463f747.png)

一张*Buthraupis montana cucullata*的图片。照片由 Charles J. Sharp 提供。（CC-BY-SA）

这张图片的分辨率是 4563×3042 像素。用 OpenCV 打开这张图片很简单：

```py
import cv2

imgfile = "Hooded_mountain_tanager_(Buthraupis_montana_cucullata)_Caldas.jpg"

img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
cv2.imshow("bird", img)
cv2.waitKey(0)
```

OpenCV 读取的图像 `img` 实际上是一个形状为 (3042, 4563, 3) 的 numpy 数组，数据类型为 `uint8`（8 位无符号整数），因为这是一个彩色图像，每个像素以 BGR 值在 0 到 255 之间表示。

Ken Burns 效果是缩放和平移。视频中的每一帧都是原始图像的裁剪（然后放大以填充屏幕）。给定一个 numpy 数组来裁剪图像是很简单的，因为 numpy 已经为你提供了切片语法：

```py
cropped = img[y0:y1, x0:x1]
```

图像是一个三维 numpy 数组。前两个维度分别表示高度和宽度（与设置矩阵坐标的方式相同）。因此，你可以使用 numpy 切片语法在垂直方向上获取像素 $y_0$ 到 $y_1$，在水平方向上获取像素 $x_0$ 到 $x_1$（请记住，在矩阵中，坐标是从上到下和从左到右编号的）。

裁剪一张图片意味着将尺寸为 $W\times H$ 的图片裁剪成一个更小的尺寸 $W’\times H’$。为了制作视频，你需要创建固定尺寸的帧。裁剪后的尺寸 $W’\times H’$ 需要调整大小。此外，为了避免失真，裁剪后的图像还需要保持预定义的长宽比。

要调整图像大小，你可以定义一个新的 numpy 数组，然后逐一计算并填充像素值。有很多方法可以计算像素值，例如使用线性插值或简单地复制最近的像素。如果你尝试实现调整大小操作，你会发现它并不困难，但仍然相当繁琐。因此，更简单的方法是使用 OpenCV 的原生函数，如下所示：

```py
resized = cv2.resize(cropped, dsize=target_dim, interpolation=cv2.INTER_LINEAR)
```

函数 `cv2.resize()` 接受一张图像和目标尺寸（以像素宽度和高度的元组形式），并返回一个新的 numpy 数组。你可以指定调整大小的算法。上面的代码使用线性插值，在大多数情况下效果很好。

这些基本上是你可以在 OpenCV 中操作图像的所有方法，即：

+   直接操作 numpy 数组。这对于你想在像素级别进行操作的简单任务效果很好。

+   使用 OpenCV 函数。这更适用于复杂任务，在这些任务中，你需要考虑整个图像，或者操作每个像素效率过低。

利用这些，你可以创建你的 Ken Burns 动画。流程如下：

1.  给定一张图像（最好是高分辨率的），你需要通过指定起始和结束的焦点坐标来定义平移。你还需要定义起始和结束的缩放比例。

1.  你有一个预定义的视频时长和帧率（FPS）。视频中的总帧数是时长乘以 FPS。

1.  对于每一帧，计算裁剪坐标。然后将裁剪后的图像调整到视频的目标分辨率。

1.  准备好所有帧后，你可以写入视频文件。

让我们从常量开始：假设我们要创建一个两秒钟的 720p 视频（分辨率为 1280×720），以 25 FPS 制作（这虽然较低，但视觉效果可接受）。平移将从图像的左侧 40% 和顶部 60% 处的中心开始，并在图像的左侧 50% 和顶部 50% 处的中心结束。缩放将从原始图像的 70% 开始，然后缩放到 100%。

```py
imgfile = "Hooded_mountain_tanager_(Buthraupis_montana_cucullata)_Caldas.jpg"
video_dim = (1280, 720)
fps = 25
duration = 2.0
start_center = (0.4, 0.6)
end_center = (0.5, 0.5)
start_scale = 0.7
end_scale = 1.0
```

你需要多次裁剪图像以创建帧（准确地说，有 2×25=50 帧）。因此，创建一个裁剪函数是有益的：

```py
def crop(img, x, y, w, h):
    x0, y0 = max(0, x-w//2), max(0, y-h//2)
    x1, y1 = x0+w, y0+h
    return img[y0:y1, x0:x1]
```

这个裁剪函数接受一个图像、像素坐标中的初步中心位置，以及宽度和高度（以像素为单位）。裁剪会确保不会超出图像边界，因此使用了两个 `max()` 函数。裁剪是通过 numpy 切片语法完成的。

如果你认为当前时间点在整个持续时间的 $\alpha$% 位置，你可以使用仿射变换来计算确切的缩放级别和全景的位置。就全景中心的相对位置（以原始宽度和高度的百分比计算）而言，仿射变换给出

```py
rx = end_center[0]*alpha + start_center[0]*(1-alpha)
ry = end_center[1]*alpha + start_center[1]*(1-alpha)
```

其中 `alpha` 在 0 和 1 之间。同样，缩放级别为

```py
scale = end_scale*alpha + start_scale*(1-alpha)
```

给定原始图像的大小和缩放比例，你可以通过乘法计算裁剪图像的大小。但由于图像的纵横比可能与视频不同，你应该调整裁剪尺寸以适应视频的纵横比。假设图像的 numpy 数组是 `img`，而上述计算的缩放级别是 `scale`，裁剪后的大小可以计算为：

```py
orig_shape = img.shape[:2]

if orig_shape[1]/orig_shape[0] > video_dim[0]/video_dim[1]:
    h = int(orig_shape[0]*scale)
    w = int(h * video_dim[0] / video_dim[1])
else:
    w = int(orig_shape[1]*scale)
    h = int(w * video_dim[1] / video_dim[0])
```

上述是为了比较图像和视频之间的纵横比（宽度除以高度），并使用缩放级别来计算更有限的边缘，并根据目标纵横比计算另一边缘。

一旦你知道需要多少帧，你可以使用 for 循环来创建每一帧，每一帧具有不同的仿射参数 `alpha`，这些参数可以通过 numpy 函数 `linspace()` 获得。完整代码如下：

```py
import cv2
import numpy as np

imgfile = "Hooded_mountain_tanager_(Buthraupis_montana_cucullata)_Caldas.jpg"
video_dim = (1280, 720)
fps = 25
duration = 2.0
start_center = (0.4, 0.6)
end_center = (0.5, 0.5)
start_scale = 0.7
end_scale = 1.0

img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
orig_shape = img.shape[:2]

def crop(img, x, y, w, h):
    x0, y0 = max(0, x-w//2), max(0, y-h//2)
    x1, y1 = x0+w, y0+h
    return img[y0:y1, x0:x1]

num_frames = int(fps * duration)
frames = []
for alpha in np.linspace(0, 1, num_frames):
    rx = end_center[0]*alpha + start_center[0]*(1-alpha)
    ry = end_center[1]*alpha + start_center[1]*(1-alpha)
    x = int(orig_shape[1]*rx)
    y = int(orig_shape[0]*ry)
    scale = end_scale*alpha + start_scale*(1-alpha)
    # determined how to crop based on the aspect ratio of width/height
    if orig_shape[1]/orig_shape[0] > video_dim[0]/video_dim[1]:
        h = int(orig_shape[0]*scale)
        w = int(h * video_dim[0] / video_dim[1])
    else:
        w = int(orig_shape[1]*scale)
        h = int(w * video_dim[1] / video_dim[0])
    # crop, scale to video size, and save the frame
    cropped = crop(img, x, y, w, h)
    scaled = cv2.resize(cropped, dsize=video_dim, interpolation=cv2.INTER_LINEAR)
    frames.append(scaled)

# write to MP4 file
vidwriter = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, video_dim)
for frame in frames:
    vidwriter.write(frame)
vidwriter.release()
```

最后的几行是如何使用 OpenCV 写入视频。你创建一个具有指定 FPS 和分辨率的 `VideoWriter` 对象。然后你逐帧写入视频，最后释放对象以关闭写入的文件。

创建的视频类似于 [这个](https://machinelearningmastery.com/wp-content/uploads/2023/10/output.mp4)。预览如下：

<https://machinelearningmastery.com/wp-content/uploads/2023/10/output.webm?_=1>

[`machinelearningmastery.com/wp-content/uploads/2023/10/output.webm`](https://machinelearningmastery.com/wp-content/uploads/2023/10/output.webm)

创建视频的预览。查看此内容需要支持的浏览器。

## 写入视频

从上一节的示例中，你可以看到我们如何创建一个 `VideoWriter` 对象：

```py
vidwriter = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, video_dim)
```

与图像文件（如 JPEG 或 PNG）的写入方式不同，OpenCV 创建的视频格式不是从文件名中推断的。它是第二个参数来指定视频格式，即 **FourCC**，这是一个由四个字符组成的代码。你可以从以下网址的列表中找到 FourCC 代码和对应的视频格式：

+   [`fourcc.org/codecs.php`](https://fourcc.org/codecs.php)

然而，并非所有 FourCC 代码都可以使用。这是因为 OpenCV 使用 FFmpeg 工具来创建视频。你可以使用以下命令找到支持的视频格式列表：

```py
ffmpeg -codecs
```

确保`ffmpeg`命令与 OpenCV 使用的相同。另外请注意，上述命令的输出仅告诉你 ffmpeg 支持的格式，而不是相应的 FourCC 代码。你需要在其他地方查找代码，例如从上述 URL 中。

要检查是否可以使用特定的 FourCC 代码，你必须试用并查看 OpenCV 是否抛出异常：

```py
try:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter('temp.mkv', fourcc, 30, (640, 480))
    assert writer.isOpened()
    print("Supported")
except:
    print("Not supported")
```

### 想要开始使用 OpenCV 进行机器学习吗？

现在就来参加我的免费电子邮件速成课程（包含示例代码）。

点击注册，还可以获得课程的免费 PDF 电子书版本。

## 总结

在这篇文章中，你学习了如何在 OpenCV 中创建视频。创建的视频由一系列帧（即无音频）组成。每一帧都是固定大小的图像。例如，你学习了如何对图片应用 Ken Burns 效果，其中你特别应用了：

+   使用 numpy 切片语法裁剪图像的技巧

+   使用 OpenCV 函数调整图像大小的技巧

+   使用仿射变换来计算缩放和平移的参数，并创建视频帧。

最后，你使用`VideoWriter`对象将帧写入视频文件中。
