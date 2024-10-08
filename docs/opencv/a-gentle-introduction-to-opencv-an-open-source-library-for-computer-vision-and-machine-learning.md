# 温和介绍 OpenCV：开源计算机视觉与机器学习库

> 原文：[`machinelearningmastery.com/a-gentle-introduction-to-opencv-an-open-source-library-for-computer-vision-and-machine-learning/`](https://machinelearningmastery.com/a-gentle-introduction-to-opencv-an-open-source-library-for-computer-vision-and-machine-learning/)

如果你对图像和视频处理感兴趣，并希望在计算机视觉应用中引入机器学习，那么 OpenCV 是你需要获取的库。

OpenCV 是一个庞大的开源库，可以与多种编程语言接口，包括 Python，并被许多个人和商业实体广泛使用。

在本教程中，你将熟悉 OpenCV 库及其重要性。

完成本教程后，你将知道：

+   OpenCV 库是什么。

+   它的用途是什么，谁在使用它。

+   如何在 Python 中安装和导入 OpenCV。

**通过我的书籍** [《OpenCV 中的机器学习》](https://machinelearning.samcart.com/products/machine-learning-opencv/) **来启动你的项目**。它提供了**自学教程**和**有效的代码**。

让我们开始吧。![](https://machinelearningmastery.com/wp-content/uploads/2022/12/opencv_intro_cover-scaled.jpg)

温和介绍 OpenCV：开源计算机视觉与机器学习库

照片由[Greg Rakozy](https://unsplash.com/photos/oMpAz-DN-9I)拍摄，部分权利保留。

## **教程概览**

本教程分为四部分，它们是：

+   OpenCV 是什么？

+   OpenCV 的用途是什么？

+   谁在使用 OpenCV？

+   如何在 Python 中安装和导入 OpenCV？

## **OpenCV 是什么？**

OpenCV 代表*开源计算机视觉* *库；顾名思义，它是一个开源的*计算机视觉和机器学习软件库。

它拥有 Apache 2.0 许可证，允许用户使用、修改和分发软件。这使得商业实体特别愿意在其产品中使用此库。

OpenCV 库是用 C++原生编写的，支持 Windows、Linux、Android 和 MacOS，并提供 C++、Python、Java 和 MATLAB 接口。

它主要针对实时计算机视觉应用。

## **OpenCV 的用途是什么？**

OpenCV 是一个庞大的库，包含超过 2500 个优化的算法，可以用于许多不同的计算机视觉应用，如：

+   **人脸检测和识别。**

***   **物体识别。 *****   **物体跟踪。 *****   **图像配准与拼接。 *****   **增强现实。 **********

******以及许多其他内容。

在这一系列教程中，你将发现 OpenCV 库在将机器学习应用于计算机视觉应用中的具体作用。

OpenCV 库中实现的一些流行机器学习算法包括：

+   **K 最近邻**

***   **支持向量机*****   **决策树******

******以及包括 TensorFlow 和 PyTorch 在内的多个深度学习框架的支持。

## **谁在使用 OpenCV****？**

OpenCV 网站估计库的下载量超过 1800 万次，用户社区由超过 4.7 万名用户组成。

许多知名公司也在使用 OpenCV 库。

OpenCV 网站提到了一些知名公司，如 Google、Yahoo、Microsoft、Intel 和 Toyota，它们在工作中使用该库。

这些公司使用 OpenCV 库的应用范围也非常广泛：

> *OpenCV 的应用范围包括从拼接街景图像、检测以色列监控视频中的入侵、监控中国矿山设备、帮助 Willow Garage 的机器人导航和拾取物体、检测欧洲游泳池溺水事故、在西班牙和纽约运行互动艺术、检查土耳其跑道上的碎片、检查全球工厂中产品的标签到日本的快速人脸检测。*
> 
> *–* [OpenCV](https://opencv.org/about/)，2022 年。

这显示了 OpenCV 库的广泛应用。

## **OpenCV 在 Python 中如何安装和导入****？**

如前所述，OpenCV 库是用 C++ 编写的，但其功能仍可以从 Python 中调用。

这是通过绑定生成器创建 C++ 和 Python 之间的桥梁实现的。

可以通过以下单行命令从 Python 包索引（PyPi）安装 OpenCV 库：

Python

```py
pip install opencv-python
```

导入 OpenCV 以利用其功能，就像调用以下命令一样简单：

Python

```py
import cv2
```

在我们浏览库的过程中，我们将频繁使用 `import` 命令。

我们将从最基础的开始，学习如何将图像和视频读取为 NumPy 数组，显示它们，访问其像素值，以及在颜色空间之间转换。

### 想要开始使用 OpenCV 进行机器学习吗？

立即参加我的免费电子邮件速成课程（附示例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

那么，让我们开始吧！

## **进一步阅读**

如果你想深入了解本主题，这一部分提供了更多资源。

### **书籍**

+   [掌握 OpenCV 4 和 Python](https://www.amazon.com/Mastering-OpenCV-Python-practical-processing/dp/1789344913)，2019 年。

### **网站**

+   OpenCV，[`opencv.org/`](https://opencv.org/)

## **总结**

在本教程中，你将熟悉 OpenCV 库以及它的重要性。

具体来说，你学到了：

+   什么是 OpenCV 库。

+   它用于什么，谁在使用它。

+   如何在 Python 中安装和导入 OpenCV。

你有任何问题吗？

在下面的评论中提出你的问题，我会尽力回答。************
