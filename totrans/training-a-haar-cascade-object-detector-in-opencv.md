# 在 OpenCV 中训练 Haar 级联目标检测器

> 原文：[`machinelearningmastery.com/training-a-haar-cascade-object-detector-in-opencv/`](https://machinelearningmastery.com/training-a-haar-cascade-object-detector-in-opencv/)

在 OpenCV 中使用 Haar 级联分类器很简单。你只需要提供一个 XML 文件中的训练模型即可创建分类器。然而，从零开始训练并不那么直接。在本教程中，你将看到训练应该是怎样的。特别是，你将学习：

+   在 OpenCV 中训练 Haar 级联分类器的工具有哪些

+   如何准备训练数据

+   如何进行训练

**通过我的书籍** [《OpenCV 中的机器学习》](https://machinelearning.samcart.com/products/machine-learning-opencv/) **快速启动你的项目**。它提供了**自学教程**和**有效的代码**。

让我们开始吧！[](../Images/dfc96e66aab3732163b1d9403e8efe6f.png)

在 OpenCV 中训练 Haar 级联目标检测器

图片由 [Adrià Crehuet Cano](https://unsplash.com/photos/children-playing-soccer-LIhB1_mAGhY) 提供。保留部分权利。

## 概述

本文分为五部分，内容包括：

+   OpenCV 中训练 Cascade 分类器的问题

+   环境设置

+   Cascade 分类器训练概述

+   准备训练数据

+   训练 Haar Cascade 分类器

## OpenCV 中训练 Cascade 分类器的问题

OpenCV 已经存在多年，并有许多版本。在写作时，OpenCV 5 正在开发中，推荐的版本是 OpenCV 4，准确来说是 4.8.0。

在 OpenCV 3 和 OpenCV 4 之间进行了大量清理。最显著的是大量代码被重写。变化是显著的，并且许多函数也发生了变化。这包括训练 Haar 级联分类器的工具。

Cascade 分类器不是一个简单的模型像 SVM 那样容易训练。它是一个使用 AdaBoost 的集成模型。因此，训练涉及多个步骤。OpenCV 3 有一个命令行工具来帮助进行这种训练，但在 OpenCV 4 中，该工具已被破坏，修复尚未提供。

因此，只能使用 OpenCV 3 训练 Haar 级联分类器。幸运的是，训练后你可以丢弃它，并在将模型保存到 XML 文件后恢复到 OpenCV 4。这就是你将在本文中做的事情。

你不能在 Python 中同时拥有 OpenCV 3 和 OpenCV 4。因此，建议为训练创建一个单独的环境。在 Python 中，你可以使用`venv`模块创建虚拟环境，这实际上是创建一个单独的已安装模块集合。另一种选择是使用 Anaconda 或 Pyenv，它们在相同的理念下有不同的架构。在上述所有选择中，Anaconda 环境被认为是最简单的。

### 想要开始使用 OpenCV 进行机器学习吗？

现在就来参加我的免费电子邮件速成课程（附带示例代码）。

点击注册，并获得课程的免费 PDF 电子书版本。

## 环境设置

如果你使用 Anaconda，会更简单，你可以使用以下命令创建并使用一个新环境，并将其命名为“cvtrain”：

```py
conda create -n cvtrain python 'opencv>=3,<4'
conda activate cvtrain
```

如果你发现命令`opencv_traincascade`可用，那么你就准备好了：

```py
$ opencv_traincascade
Usage: opencv_traincascade
  -data <cascade_dir_name>
  -vec <vec_file_name>
  -bg <background_file_name>
  [-numPos <number_of_positive_samples = 2000>]
  [-numNeg <number_of_negative_samples = 1000>]
  [-numStages <number_of_stages = 20>]
  [-precalcValBufSize <precalculated_vals_buffer_size_in_Mb = 1024>]
  [-precalcIdxBufSize <precalculated_idxs_buffer_size_in_Mb = 1024>]
  [-baseFormatSave]
  [-numThreads <max_number_of_threads = 16>]
  [-acceptanceRatioBreakValue <value> = -1>]
--cascadeParams--
  [-stageType <BOOST(default)>]
  [-featureType <{HAAR(default), LBP, HOG}>]
  [-w <sampleWidth = 24>]
  [-h <sampleHeight = 24>]
--boostParams--
  [-bt <{DAB, RAB, LB, GAB(default)}>]
  [-minHitRate <min_hit_rate> = 0.995>]
  [-maxFalseAlarmRate <max_false_alarm_rate = 0.5>]
  [-weightTrimRate <weight_trim_rate = 0.95>]
  [-maxDepth <max_depth_of_weak_tree = 1>]
  [-maxWeakCount <max_weak_tree_count = 100>]
--haarFeatureParams--
  [-mode <BASIC(default) | CORE | ALL
--lbpFeatureParams--
--HOGFeatureParams--
```

如果你使用`pyenv`或`venv`，则需要更多步骤。首先，创建一个环境并安装 OpenCV（你应该注意到与 Anaconda 生态系统中包的名称不同）：

```py
# create an environment and install opencv 3
pyenv virtualenv 3.11 cvtrain
pyenv activate cvtrain
pip install 'opencv-python>=3,<4'
```

这允许你使用 OpenCV 运行 Python 程序，但你没有用于训练的命令行工具。要获取这些工具，你需要按照以下步骤从源代码编译它们：

1.  下载 OpenCV 源代码并切换到 3.4 分支

    ```py
    # download OpenCV source code and switch to 3.4 branch
    git clone https://github.com/opencv/opencv
    cd opencv
    git checkout 3.4
    cd ..
    ```

1.  创建与仓库目录分开的构建目录：

    ```py
    mkdir build
    cd build
    ```

1.  使用`cmake`工具准备构建目录，并参考 OpenCV 仓库：

    ```py
    cmake ../opencv
    ```

1.  运行`make`进行编译（你可能需要先在系统中安装开发者库）

    ```py
    make
    ls bin
    ```

1.  所需的工具将位于`bin/`目录中，如上面的最后一条命令所示

所需的命令行工具是`opencv_traincascade`和`opencv_createsamples`。本文其余部分假设你已经有了这些工具。

## 级联分类器训练概述

你将使用 OpenCV 工具训练一个**级联分类器**。该分类器是一个使用 AdaBoost 的集成模型。简单来说，多个较小的模型被创建，其中每个模型在分类上较弱。结合起来，它成为一个强大的分类器，具有良好的精确度和召回率。

每个**弱分类器**都是一个二元分类器。要训练它们，你需要一些正样本和负样本。负样本很简单：你提供一些随机图片给 OpenCV，让 OpenCV 选择一个矩形区域（最好这些图片中没有目标物体）。然而，正样本则作为图像和包含物体的边界框提供。

一旦提供了这些数据集，OpenCV 将从中提取 Haar 特征，并使用它们来训练多个分类器。Haar 特征是通过将正样本或负样本划分为矩形区域得到的。如何进行划分涉及到一些随机性。因此，OpenCV 需要时间来找到最好的方式来推导用于分类任务的 Haar 特征。

在 OpenCV 中，你只需提供以 OpenCV 可以读取的格式（如 JPEG 或 PNG）的图像文件中的训练数据。对于负样本，它只需要一个包含文件名的纯文本文件。对于正样本，则需要一个“信息文件”，这是一个包含文件名、图像中物体数量以及相应边界框的纯文本文件。

训练用的正样本应为二进制格式。OpenCV 提供了一个工具`opencv_createsamples`，可以从“信息文件”生成二进制格式。然后将这些正样本与负样本一起提供给另一个工具`opencv_traincascade`，以进行训练并生成 XML 格式的模型输出。这是你可以加载到 OpenCV Haar 级联分类器中的 XML 文件。

## 准备训练数据

让我们考虑创建一个**猫脸**检测器。要训练这样的检测器，你首先需要数据集。一种可能性是位于以下位置的 Oxford-IIIT Pet Dataset：

+   [`www.robots.ox.ac.uk/~vgg/data/pets/`](https://www.robots.ox.ac.uk/~vgg/data/pets/)

这是一个 800MB 的数据集，在计算机视觉数据集的标准下算是一个小数据集。图像以 Pascal VOC 格式进行标注。简而言之，每张图像都有一个对应的 XML 文件，格式如下：

XHTML

```py
<?xml version="1.0"?>
<annotation>
  <folder>OXIIIT</folder>
  <filename>Abyssinian_100.jpg</filename>
  <source>
    <database>OXFORD-IIIT Pet Dataset</database>
    <annotation>OXIIIT</annotation>
    <image>flickr</image>
  </source>
  <size>
    <width>394</width>
    <height>500</height>
    <depth>3</depth>
  </size>
  <segmented>0</segmented>
  <object>
    <name>cat</name>
    <pose>Frontal</pose>
    <truncated>0</truncated>
    <occluded>0</occluded>
    <bndbox>
      <xmin>151</xmin>
      <ymin>71</ymin>
      <xmax>335</xmax>
      <ymax>267</ymax>
    </bndbox>
    <difficult>0</difficult>
  </object>
</annotation>
```

XML 文件告诉你它所指的是哪个图像文件（如上例中的`Abyssinian_100.jpg`），以及它包含什么对象，边界框在`<bndbox></bndbox>`标签之间。

要从 XML 文件中提取边界框，你可以使用以下函数：

```py
import xml.etree.ElementTree as ET

def read_voc_xml(xmlfile: str) -> dict:
    root = ET.parse(xmlfile).getroot()
    boxes = {"filename": root.find("filename").text,
             "objects": []}
    for box in root.iter('object'):
        bb = box.find('bndbox')
        obj = {
            "name": box.find('name').text,
            "xmin": int(bb.find("xmin").text),
            "ymin": int(bb.find("ymin").text),
            "xmax": int(bb.find("xmax").text),
            "ymax": int(bb.find("ymax").text),
        }
        boxes["objects"].append(obj)

    return boxes
```

上述函数返回的字典示例如下：

```py
{'filename': 'yorkshire_terrier_160.jpg',
'objects': [{'name': 'dog', 'xmax': 290, 'xmin': 97, 'ymax': 245, 'ymin': 18}]}
```

有了这些，就可以轻松创建用于训练的数据集：在 Oxford-IIT Pet 数据集中，照片要么是猫，要么是狗。你可以将所有的狗照片作为负样本。然后，所有的猫照片将是带有适当边界框集的正样本。

OpenCV 期望的正样本“信息文件”是一个文本文件，每行的格式如下：

```py
filename N x0 y0 w0 h0 x1 y1 w1 h1 ...
```

文件名后的数字是该图像中边界框的数量。每个边界框都是一个正样本。后面的内容是边界框。每个框由其左上角的像素坐标和框的宽度和高度指定。为了获得 Haar 级联分类器的最佳结果，边界框应与模型预期的长宽比一致。

假设你下载的宠物数据集位于`dataset/`目录中，你应该会看到文件按如下方式组织：

```py
dataset
|-- annotations
|   |-- README
|   |-- list.txt
|   |-- test.txt
|   |-- trainval.txt
|   |-- trimaps
|   |   |-- Abyssinian_1.png
|   |   |-- Abyssinian_10.png
|   |   ...
|   |   |-- yorkshire_terrier_98.png
|   |   `-- yorkshire_terrier_99.png
|   `-- xmls
|       |-- Abyssinian_1.xml
|       |-- Abyssinian_10.xml
|       ...
|       |-- yorkshire_terrier_189.xml
|       `-- yorkshire_terrier_190.xml
`-- images
    |-- Abyssinian_1.jpg
    |-- Abyssinian_10.jpg
    ...
    |-- yorkshire_terrier_98.jpg
    `-- yorkshire_terrier_99.jpg
```

有了这些，使用以下程序可以轻松创建正样本的“信息文件”和负样本文件列表：

```py
import pathlib
import xml.etree.ElementTree as ET

import numpy as np

def read_voc_xml(xmlfile: str) -> dict:
    """read the Pascal VOC XML and return (filename, object name, bounding box)
    where bounding box is a vector of (xmin, ymin, xmax, ymax). The pixel
    coordinates are 1-based.
    """
    root = ET.parse(xmlfile).getroot()
    boxes = {"filename": root.find("filename").text,
             "objects": []
            }
    for box in root.iter('object'):
        bb = box.find('bndbox')
        obj = {
            "name": box.find('name').text,
            "xmin": int(bb.find("xmin").text),
            "ymin": int(bb.find("ymin").text),
            "xmax": int(bb.find("xmax").text),
            "ymax": int(bb.find("ymax").text),
        }
        boxes["objects"].append(obj)

    return boxes

# Read Pascal VOC and write data
base_path = pathlib.Path("dataset")
img_src = base_path / "images"
ann_src = base_path / "annotations" / "xmls"

negative = []
positive = []
for xmlfile in ann_src.glob("*.xml"):
    # load xml
    ann = read_voc_xml(str(xmlfile))
    if ann['objects'][0]['name'] == 'dog':
        # negative sample (dog)
        negative.append(str(img_src / ann['filename']))
    else:
        # positive sample (cats)
        bbox = []
        for obj in ann['objects']:
            x = obj['xmin']
            y = obj['ymin']
            w = obj['xmax'] - obj['xmin']
            h = obj['ymax'] - obj['ymin']
            bbox.append(f"{x} {y} {w} {h}")
        line = f"{str(img_src/ann['filename'])} {len(bbox)} {' '.join(bbox)}"
        positive.append(line)

# write the output to `negative.dat` and `postiive.dat`
with open("negative.dat", "w") as fp:
    fp.write("\n".join(negative))

with open("positive.dat", "w") as fp:
    fp.write("\n".join(positive))
```

该程序扫描数据集中的所有 XML 文件，然后提取每张猫照片中的边界框。列表`negative`将保存狗照片的路径。列表`positive`将保存猫照片的路径以及上述格式的边界框，每行作为一个字符串。在循环结束后，这两个列表会被写入磁盘，分别作为`negative.dat`和`positive.dat`文件。

`negative.dat`的内容很简单。`positive.dat`的内容如下：

```py
dataset/images/Siamese_102.jpg 1 154 92 194 176
dataset/images/Bengal_152.jpg 1 84 8 187 201
dataset/images/Abyssinian_195.jpg 1 8 6 109 115
dataset/images/Russian_Blue_135.jpg 1 228 90 103 117
dataset/images/Persian_122.jpg 1 60 16 230 228
```

在运行训练之前的步骤是将`positive.dat`转换为二进制格式。这可以通过以下命令行完成：

```py
opencv_createsamples -info positive.dat -vec positive.vec -w 30 -h 30
```

该命令应在与`positive.dat`相同的目录中运行，以便可以找到数据集图像。此命令的输出将是`positive.vec`，也称为“vec 文件”。在执行此操作时，你需要使用`-w`和`-h`参数指定窗口的宽度和高度。这是为了将边界框裁剪的图像调整为此像素大小，然后写入 vec 文件。这还应与运行训练时指定的窗口大小匹配。

## 训练 Haar 级联分类器

训练分类器需要时间。它分多个阶段完成。每个阶段都要写入中间文件，所有阶段完成后，你将获得保存在 XML 文件中的训练模型。OpenCV 期望将所有这些生成的文件存储在一个目录中。

运行训练过程确实很简单。假设创建一个新的目录`cat_detect`来存储生成的文件。目录创建完成后，可以使用命令行工具`opencv_traincascade`运行训练：

Shell

```py
# need to create the data dir first
mkdir cat_detect
# then run the training
opencv_traincascade -data cat_detect -vec positive.vec -bg negative.dat -numPos 900 -numNeg 2000 -numStages 10 -w 30 -h 30
```

注意使用`positive.vec`作为正样本和`negative.dat`作为负样本。还要注意，`-w`和`-h`参数与之前在`opencv_createsamples`命令中使用的相同。其他命令行参数解释如下：

+   `-data <dirname>`：存储训练分类器的目录。该目录应已存在

+   `-vec <filename>`：正样本的 vec 文件

+   `-bg <filename>`：负样本列表，也称为“背景”图像

+   `-numPos <N>`：每个阶段训练中使用的正样本数量

+   `-numNeg <N>`：每个阶段训练中使用的负样本数量

+   `-numStages <N>`：要训练的级联阶段数量

+   `-w <width>`和`-h <height>`：对象的像素大小。必须与使用`opencv_createsamples`工具创建训练样本时的大小相同

+   `-minHitRate <rate>`：每个阶段所需的最小真实正例率。训练一个阶段不会终止，直到达到此要求。

+   `-maxFalseAlarmRate <rate>`：每个阶段的最大假正例率。训练一个阶段不会终止，直到达到此要求。

+   `-maxDepth <N>`：弱分类器的最大深度

+   `-maxWeakCount <N>`：每个阶段的弱分类器最大数量

这些参数并非全部必需。但你应该尝试不同的组合，看看是否能训练出更好的检测器。

在训练期间，你会看到以下屏幕：

```py
$ opencv_traincascade -data cat_detect -vec positive.vec -bg negative.dat -numPos 900 -numNeg 2000 -numStages 10 -w 30 -h 30
PARAMETERS:
cascadeDirName: cat_detect
vecFileName: positive.vec
bgFileName: negative.dat
numPos: 900
numNeg: 2000
numStages: 10
precalcValBufSize[Mb] : 1024
precalcIdxBufSize[Mb] : 1024
acceptanceRatioBreakValue : -1
stageType: BOOST
featureType: HAAR
sampleWidth: 30
sampleHeight: 30
boostType: GAB
minHitRate: 0.995
maxFalseAlarmRate: 0.5
weightTrimRate: 0.95
maxDepth: 1
maxWeakCount: 100
mode: BASIC
Number of unique features given windowSize [30,30] : 394725

===== TRAINING 0-stage =====
<BEGIN
POS count : consumed   900 : 900
NEG count : acceptanceRatio    2000 : 1
Precalculation time: 3
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|        1|
+----+---------+---------+
|   4|        1|   0.8925|
+----+---------+---------+
|   5| 0.998889|   0.7785|
...
|  19| 0.995556|    0.503|
+----+---------+---------+
|  20| 0.995556|    0.492|
+----+---------+---------+
END>
...
Training until now has taken 0 days 2 hours 55 minutes 44 seconds.

===== TRAINING 9-stage =====
<BEGIN
POS count : consumed   900 : 948
NEG count : acceptanceRatio    2000 : 0.00723552
Precalculation time: 4
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|        1|
+----+---------+---------+
|   4|        1|        1|
+----+---------+---------+
|   5| 0.997778|   0.9895|
...
|  50| 0.995556|   0.5795|
+----+---------+---------+
|  51| 0.995556|   0.4895|
+----+---------+---------+
END>
Training until now has taken 0 days 3 hours 25 minutes 12 seconds.
```

你应该注意到训练运行的$N$个阶段编号为 0 到$N-1$。某些阶段可能需要更长时间进行训练。开始时，会显示训练参数以明确其正在做什么。然后在每个阶段中，会逐行打印一个表格。表格显示三列：特征数量`N`、命中率`HR`（真实正例率）和误报率`FA`（假正例率）。

在阶段 0 之前，您应该看到打印的 `minHitRate` 为 0.995 和 `maxFalseAlarmRate` 为 0.5。因此，每个阶段将找到许多 Haar 特征，直到分类器能够保持命中率在 0.995 以上，同时虚警率低于 0.5。理想情况下，您希望命中率为 1，虚警率为 0。由于 Haar cascade 是一个集成方法，如果大多数结果正确，则会得到正确的预测。大致上，您可以认为具有 $n$ 个阶段、命中率为 $p$ 和虚警率为 $q$ 的分类器，其总体命中率为 $p^n$，总体虚警率为 $q^n$。在上述设置中，$n=10$，$p>0.995$，$q<0.5$。因此，总体虚警率将低于 0.1%，总体命中率高于 95%。

这个训练命令在现代计算机上完成需要超过 3 小时。输出将命名为 `cascade.xml`，保存在输出目录下。您可以使用以下示例代码检查结果：

```py
import cv2

image = 'dataset/images/Abyssinian_88.jpg'
model = 'cat_detect/cascade.xml'

classifier = cv2.CascadeClassifier(model)
img = cv2.imread(image)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform object detection
objects = classifier.detectMultiScale(gray,
                                      scaleFactor=1.1, minNeighbors=5,
                                      minSize=(30, 30))

# Draw rectangles around detected objects
for (x, y, w, h) in objects:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the result
cv2.imshow('Object Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

结果将取决于您的模型训练得有多好，也取决于您传递给 `detectMultiScale()` 的参数。有关如何设置这些参数，请参见上一篇文章。

上述代码在数据集中运行检测器，您可能会看到如下结果：

![](img/4b31463ae32fac80dacce33d99039295.png)

使用训练好的 Haar cascade 对象检测器的示例输出

您会看到一些误报，但猫的脸已被检测到。改善质量的方法有多种。例如，您使用的训练数据集没有使用方形边界框，而您在训练和检测中使用了方形形状。调整数据集可能会有所改善。同样，您在训练命令行中使用的其他参数也会影响结果。然而，您应该意识到，Haar cascade 检测器非常快速，但使用的阶段越多，速度会变得越慢。

## 进一步阅读

本节提供了更多相关资源，供您深入了解该主题。

### 书籍

+   [掌握 OpenCV 4 与 Python](https://www.amazon.com/Mastering-OpenCV-Python-practical-processing/dp/1789344913)，2019 年。

### 网站

+   OpenCV [级联分类器训练教程](https://docs.opencv.org/4.x/dc/d88/tutorial_traincascade.html)

+   [常见问题解答：OpenCV Haar 训练](https://www.computer-vision-software.com/blog/2009/11/faq-opencv-haartraining/)

+   [教程：OpenCV Haar 训练](https://web.archive.org/web/20220804065334/http://note.sonots.com/SciSoftware/haartraining.html)

## 总结

在这篇文章中，您学习了如何在 OpenCV 中训练 Haar cascade 对象检测器。具体来说，您学习了：

+   如何为 Haar cascade 训练准备数据

+   如何在命令行中运行训练过程

+   如何使用 OpenCV 3.x 训练检测器，并在 OpenCV 4.x 中使用训练好的模型
