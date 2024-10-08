# 如何使用 OpenCV 读取和显示视频

> 原文：[`machinelearningmastery.com/how-to-read-and-display-videos-using-opencv/`](https://machinelearningmastery.com/how-to-read-and-display-videos-using-opencv/)

数字视频是数字图像的近亲，因为它们由许多数字图像快速连续地显示组成，以创建运动视觉数据的效果。

OpenCV 库提供了几种处理视频的方法，如从不同来源读取视频数据和访问其多个属性。

在本教程中，你将熟悉处理视频时最基本的 OpenCV 操作。

完成本教程后，你将了解到：

+   数字视频如何作为数字图像的近亲进行制定。

+   如何从摄像头读取组成视频的图像帧。

+   如何从保存的视频文件中读取组成视频的图像帧。

**启动你的项目**，使用我的书籍 [《OpenCV 中的机器学习》](https://machinelearning.samcart.com/products/machine-learning-opencv/)。它提供了**自学教程**和**工作代码**。

让我们开始吧。![](https://machinelearningmastery.com/wp-content/uploads/2023/01/video_basics_cover-scaled.jpg)

使用 OpenCV 读取和显示视频

照片由 [Thomas William](https://unsplash.com/photos/4qGbMEZb56c) 拍摄，版权所有。

## **教程概述**

本教程分为三个部分，它们是：

+   视频是如何制定的？

+   从摄像头读取和显示图像帧

+   从视频文件中读取和显示图像帧

## **视频是如何制定的？**

我们已经看到，数字图像由像素组成，每个像素由其图像空间中的*空间坐标*和其*强度*或*灰度级*值来表征。

我们还提到，包含单一通道的灰度图像可以由 2D 函数 *I*(*x*, *y*) 描述，其中 *x* 和 *y* 表示上述空间坐标，*I* 在任何图像位置 (*x*, *y*) 的值表示像素强度。

一个 RGB 图像可以通过这三种 2D 函数描述，*I**[R]*(*x*, *y*), *I**[G]*(*x*, *y*), 和 *I**[B]*(*x*, *y*)，分别对应其红色、绿色和蓝色通道。

在描述数字视频时，我们将添加一个额外的维度，即*t*，表示*时间*。这样做的原因是数字视频由数字图像在一段时间内快速连续地显示组成。在视频的上下文中，我们将这些图像称为*图像帧*。图像帧连续显示的速率称为*帧率*，以*每秒帧数*（FPS）来测量。

因此，如果我们需要从某个特定时间点 *t* 中的*灰度*视频中挑选一帧图像，我们将通过函数 *I*(*x*, *y, t*) 来描述它，这现在包含了时间维度。

同样地，如果我们需要从特定时间实例*t*的*RGB*视频中选择图像帧，我们将通过三个函数来描述它：*I**[R]*(*x*, *y, t*), *I**[G]*(*x*, *y, t*), 和 *I**[B]*(*x*, *y, t*)，分别对应其红、绿和蓝通道。

我们的公式告诉我们，数字视频中包含的数据是*时间依赖*的，这意味着数据随时间变化。

简单来说，这意味着在时间实例*t*处具有坐标(*x*, *y*)的像素的强度值可能与另一个时间实例(*t* + 1)处的强度值不同。这种强度值的变化可能来自于被记录的物理场景的变化，但也可能来自视频数据中的噪声（例如来自相机传感器本身）。

## **从相机读取和显示图像帧**

要从连接到计算机的相机或存储在硬盘上的视频文件中读取图像帧，我们的第一步是创建一个`VideoCapture`对象来处理。所需的参数是`int`类型的索引值，对应于要读取的相机，或者是视频文件的名称。

让我们首先从相机中抓取图像帧开始。

如果您的计算机内置或连接了网络摄像头，则可以通过值`0`来索引它。如果您有其他连接的相机希望读取，则可以使用值`1`、`2`等，具体取决于可用的相机数量。

Python

```py
from cv2 import VideoCapture

capture = VideoCapture(0)
```

在尝试读取和显示图像帧之前，检查是否成功建立了与相机的连接是明智的。可以使用`capture.isOpened()`方法来实现这一目的，如果连接未能建立则返回`False`：

Python

```py
if not capture.isOpened():
    print("Error establishing connection")
```

如果相机成功连接，我们可以通过使用`capture.read()`方法来读取图像帧，如下所示：

Python

```py
ret, frame = capture.read()
```

该方法返回`frame`中的下一个图像帧，以及一个布尔值`ret`，如果成功抓取到图像帧则为`True`，反之为`False`，例如相机已断开连接时可能返回空图像。

以与静态图像相同的方式显示抓取的图像框架，使用`imshow`方法：

Python

```py
from cv2 import imshow

if ret:
    imshow('Displaying image frames from a webcam', frame)
```

请记住，在使用 OpenCV 时，每个图像帧都以 BGR 颜色格式读取。

在完整的代码列表中，我们将把上述代码放入一个`while`循环中，该循环将继续从相机中抓取图像帧，直到用户终止它。为了让用户终止`while`循环，我们将包含以下两行代码：

Python

```py
from cv2 import waitKey

if waitKey(25) == 27:
    break
```

这里，`waitKey`函数停止并等待指定的毫秒数以响应键盘事件。它返回按键的代码，如果在指定时间内没有生成键盘事件，则返回-1。在我们的特定情况下，我们指定了 25 毫秒的时间窗口，并检查 ASCII 码`27`，这对应于按下`Esc`键。当按下`Esc`键时，`while`循环通过`break`命令终止。

我们将要包括的最后几行代码用于停止视频捕捉，释放内存，并关闭用于图像显示的窗口：

Python

```py
from cv2 import destroyAllWindows

capture.release()
destroyAllWindows()
```

在某些笔记本电脑上，当使用视频捕捉时，你会看到内置摄像头旁边的小 LED 灯亮起。你需要停止视频捕捉才能关闭该 LED 灯。即使你的程序正在从摄像头读取，也没有关系。你必须在其他程序可以使用你的摄像头之前也停止视频捕捉。

完整的代码列表如下：

Python

```py
from cv2 import VideoCapture, imshow, waitKey, destroyAllWindows

# Create video capture object
capture = VideoCapture(0)

# Check that a camera connection has been established
if not capture.isOpened():
    print("Error establishing connection")

while capture.isOpened():

    # Read an image frame
    ret, frame = capture.read()

    # If an image frame has been grabbed, display it
    if ret:
        imshow('Displaying image frames from a webcam', frame)

    # If the Esc key is pressed, terminate the while loop
    if waitKey(25) == 27:
        break

# Release the video capture and close the display window
capture.release()
destroyAllWindows()
```

### 想开始使用 OpenCV 进行机器学习吗？

立即参加我的免费电子邮件速成课程（附示例代码）。

点击注册并获取课程的免费 PDF 电子书版本。

## **从视频文件读取和显示图像帧**

另一种方法是从存储在硬盘上的视频文件中读取图像帧。OpenCV 支持许多视频格式。为此，我们将修改我们的代码以指定视频文件的路径，而不是摄像头的索引。

我下载了[这个视频](https://www.videvo.net/video/rising-up-over-a-river/452789/)，将其重命名为*Iceland.mp4*，并将其保存到名为*Videos*的本地文件夹中。

我可以从显示在本地驱动器上的视频属性中看到，该视频包括 1920 x 1080 像素的图像帧，并且以 25 帧每秒的帧率运行。

为了读取该视频的图像帧，我们将修改以下代码行如下：

Python

```py
capture = VideoCapture('Videos/Iceland.mp4')
```

还可以获取捕捉对象的多个属性，例如图像帧的宽度和高度以及帧率：

Python

```py
from cv2 import CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS

frame_width = capture.get(CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(CAP_PROP_FRAME_HEIGHT)
fps = capture.get(CAP_PROP_FPS)
```

完整的代码列表如下：

Python

```py
from cv2 import (VideoCapture, imshow, waitKey, destroyAllWindows,
CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS)

# Create video capture object
capture = VideoCapture('Videos/Iceland2.mp4')

# Check that a camera connection has been established
if not capture.isOpened():
    print("Error opening video file")

else:
    # Get video properties and print them
    frame_width = capture.get(CAP_PROP_FRAME_WIDTH)
    frame_height = capture.get(CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(CAP_PROP_FPS)

    print("Image frame width: ", int(frame_width))
    print("Image frame height: ", int(frame_height))
    print("Frame rate: ", int(fps))

while capture.isOpened():

    # Read an image frame
    ret, frame = capture.read()

    # If an image frame has been grabbed, display it
    if ret:
        imshow('Displaying image frames from video file', frame)

    # If the Esc key is pressed, terminate the while loop
    if waitKey(25) == 27:
        break

# Release the video capture and close the display window
capture.release()
destroyAllWindows()
```

视频具有时间维度。但在 OpenCV 中，你一次处理一个**帧**。这可以使视频处理与图像处理保持一致，从而使你可以将一种技术从一个应用到另一个。

我们可以在`while`循环内部包括其他代码行，以处理每个图像帧，这些图像帧在`capture.read()`方法抓取后获得。一个例子是将每个 BGR 图像帧转换为灰度图像，为此我们可以使用与[我们用于转换静态图像](https://machinelearningmastery.com/reading-and-displaying-images-and-converting-between-color-spaces-using-opencv)相同的`cvtColor`方法。

Python

```py
from cv2 import COLOR_BGR2GRAY

frame = cvtColor(frame, COLOR_BGR2GRAY)
```

你能想到对图像帧应用哪些其他变换吗？

## **进一步阅读**

本节提供了更多关于此主题的资源，如果你想深入了解。

### **书籍**

+   [掌握 OpenCV 4 与 Python](https://www.amazon.com/Mastering-OpenCV-Python-practical-processing/dp/1789344913)，2019 年。

### **网站**

+   OpenCV，[`opencv.org/`](https://opencv.org/)

## **总结**

在本教程中，你将熟悉处理视频时必需的最基本的 OpenCV 操作。

具体来说，你学到了：

+   数字视频如何作为数字图像的近亲进行编制。

+   如何从摄像机中读取构成视频的图像帧。

+   如何从保存的视频文件中读取构成视频的图像帧。

你有任何问题吗？

在下面的评论中提出你的问题，我会尽力回答。
