# 使用 matplotlib、Seaborn 和 Bokeh 在 Python 中进行数据可视化

> 原文：[`machinelearningmastery.com/data-visualization-in-python-with-matplotlib-seaborn-and-bokeh/`](https://machinelearningmastery.com/data-visualization-in-python-with-matplotlib-seaborn-and-bokeh/)

数据可视化是所有 AI 和机器学习应用的重要方面。通过不同的图形表示，你可以获得数据的关键洞察。在本教程中，我们将讨论 Python 中的数据可视化几种选项。我们将使用 MNIST 数据集和 Tensorflow 库进行数据处理和操作。为了说明创建各种类型图表的方法，我们将使用 Python 的图形库，即 matplotlib、Seaborn 和 Bokeh。

完成本教程后，你将了解：

+   如何在 matplotlib 中可视化图像

+   如何在 matplotlib、Seaborn 和 Bokeh 中制作散点图

+   如何在 matplotlib、Seaborn 和 Bokeh 中制作多线图

**启动你的项目**，阅读我的新书 [《Python 机器学习》](https://machinelearningmastery.com/python-for-machine-learning/)，其中包括*逐步教程*和所有示例的*Python 源代码*文件。

让我们开始吧。![从飞机上拍摄的伊斯坦布尔的照片](https://machinelearningmastery.com/wp-content/uploads/2022/03/IMG_0570-scaled.jpg)

使用 matplotlib、Seaborn 和 Bokeh 在 Python 中进行数据可视化

照片由 Mehreen Saeed 拍摄，部分权利保留。

## 教程概述

本教程分为七个部分，它们是：

+   散点数据的准备

+   matplotlib 中的图形

+   matplotlib 和 Seaborn 中的散点图

+   Bokeh 中的散点图

+   线图数据的准备

+   在 matplotlib、Seaborn 和 Bokeh 中绘制线图

+   更多关于可视化的内容

## 散点数据的准备

在这篇文章中，我们将使用 matplotlib、Seaborn 和 Bokeh。它们都是需要安装的外部库。要使用 `pip` 安装它们，请运行以下命令：

```py
pip install matplotlib seaborn bokeh
```

为了演示目的，我们还将使用 MNIST 手写数字数据集。我们将从 TensorFlow 中加载它，并对其运行 PCA 算法。因此，我们还需要安装 TensorFlow 和 pandas：

```py
pip install tensorflow pandas
```

之后的代码将假设已执行以下导入：

```py
# Importing from tensorflow and keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras import utils
from tensorflow import dtypes, tensordot
from tensorflow import convert_to_tensor, linalg, transpose
# For math operations
import numpy as np
# For plotting with matplotlib
import matplotlib.pyplot as plt
# For plotting with seaborn
import seaborn as sns  
# For plotting with bokeh
from bokeh.plotting import figure, show
from bokeh.models import Legend, LegendItem
# For pandas dataframe
import pandas as pd
```

我们从 `keras.datasets` 库中加载 MNIST 数据集。为了简化起见，我们将仅保留包含前三个数字的数据子集。我们现在还将忽略测试集。

```py
...
# load dataset
(x_train, train_labels), (_, _) = mnist.load_data()
# Choose only the digits 0, 1, 2
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
# Shape of training data
total_examples, img_length, img_width = x_train.shape
# Print the statistics
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)
```

输出

```py
Training data has  18623 images
Each image is of size  28 x 28
```

### 想开始学习 Python 进行机器学习吗？

现在就参加我的免费 7 天电子邮件速成课程（包含示例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## matplotlib 中的图形

Seaborn 确实是 matplotlib 的一个附加库。因此，即使使用 Seaborn，你也需要了解 matplotlib 如何处理图表。

Matplotlib 称其画布为图形。你可以将图形划分为几个称为子图的部分，以便将两个可视化并排放置。

例如，让我们使用 matplotlib 可视化 MNIST 数据集的前 16 张图像。我们将使用`subplots()`函数创建 2 行 8 列的图像。`subplots()`函数将为每个单元创建**坐标轴**对象。然后，我们将使用`imshow()`方法在每个坐标轴对象上显示每张图像。最后，将使用`show()`函数显示图像：

```py
img_per_row = 8
fig,ax = plt.subplots(nrows=2, ncols=img_per_row,
                      figsize=(18,4),
                      subplot_kw=dict(xticks=[], yticks=[]))
for row in [0, 1]:
    for col in range(img_per_row):
        ax[row, col].imshow(x_train[row*img_per_row + col].astype('int'))   
plt.show()
```

![训练数据集前 16 张图像显示在 2 行 8 列中](https://machinelearningmastery.com/wp-content/uploads/2022/03/output_8_0.png)

训练数据集的前 16 张图像显示在 2 行 8 列中

在这里，我们可以看到 matplotlib 的一些特性。matplotlib 有一个默认的图形和默认的坐标轴。matplotlib 的`pyplot`子模块下定义了许多函数，用于在默认坐标轴上绘图。如果我们想在特定坐标轴上绘图，可以使用坐标轴对象下的绘图函数。操作图形是过程性的。这意味着 matplotlib 内部记住了一个数据结构，我们的操作会改变它。`show()`函数仅显示一系列操作的结果。因此，我们可以逐步调整图形中的许多细节。在上面的示例中，我们通过将`xticks`和`yticks`设置为空列表来隐藏了“刻度”（即坐标轴上的标记）。

## matplotlib 和 Seaborn 中的散点图

在机器学习项目中，我们常用的一种可视化方式是散点图。

例如，我们对 MNIST 数据集应用 PCA，并提取每张图像的前三个成分。在下面的代码中，我们从数据集中计算特征向量和特征值，然后沿着特征向量的方向投影每张图像的数据，并将结果存储在`x_pca`中。为了简单起见，我们在计算特征向量之前没有将数据标准化为零均值和单位方差。这一省略不影响我们可视化的目的。

```py
...
# Convert the dataset into a 2D array of shape 18623 x 784
x = convert_to_tensor(np.reshape(x_train, (x_train.shape[0], -1)),
                      dtype=dtypes.float32)
# Eigen-decomposition from a 784 x 784 matrix
eigenvalues, eigenvectors = linalg.eigh(tensordot(transpose(x), x, axes=1))
# Print the three largest eigenvalues
print('3 largest eigenvalues: ', eigenvalues[-3:])
# Project the data to eigenvectors
x_pca = tensordot(x, eigenvectors, axes=1)
```

打印出的特征值如下：

```py
3 largest eigenvalues:  tf.Tensor([5.1999642e+09 1.1419439e+10 4.8231231e+10], shape=(3,), dtype=float32)
```

数组`x_pca`的形状为 18623 x 784。我们考虑最后两列作为 x 和 y 坐标，并在图中标出每一行的点。我们还可以根据每个点对应的数字进一步为其上色。

以下代码使用 matplotlib 生成散点图。图是通过坐标轴对象的`scatter()`函数创建的，该函数将 x 和 y 坐标作为前两个参数。`scatter()`方法的`c`参数指定将成为其颜色的值。`s`参数指定其大小。代码还创建了一个图例，并为图形添加了标题。

```py
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(x_pca[:, -1], x_pca[:, -2], c=train_labels, s=5)
legend_plt = ax.legend(*scatter.legend_elements(),
                       loc="lower left", title="Digits")
ax.add_artist(legend_plt)
plt.title('First Two Dimensions of Projected Data After Applying PCA')
plt.show()
```

![使用 Matplotlib 生成的 2D 散点图](https://machinelearningmastery.com/wp-content/uploads/2022/03/output_13_0.png)

使用 matplotlib 生成的 2D 散点图

将上述内容综合起来，以下是使用 matplotlib 生成 2D 散点图的完整代码：

```py
from tensorflow.keras.datasets import mnist
from tensorflow import dtypes, tensordot
from tensorflow import convert_to_tensor, linalg, transpose
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
(x_train, train_labels), (_, _) = mnist.load_data()
# Choose only the digits 0, 1, 2
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
# Verify the shape of training data
total_examples, img_length, img_width = x_train.shape
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)

# Convert the dataset into a 2D array of shape 18623 x 784
x = convert_to_tensor(np.reshape(x_train, (x_train.shape[0], -1)),
                      dtype=dtypes.float32)
# Eigen-decomposition from a 784 x 784 matrix
eigenvalues, eigenvectors = linalg.eigh(tensordot(transpose(x), x, axes=1))
# Print the three largest eigenvalues
print('3 largest eigenvalues: ', eigenvalues[-3:])
# Project the data to eigenvectors
x_pca = tensordot(x, eigenvectors, axes=1)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(x_pca[:, -1], x_pca[:, -2], c=train_labels, s=5)
legend_plt = ax.legend(*scatter.legend_elements(),
                       loc="lower left", title="Digits")
ax.add_artist(legend_plt)
plt.title('First Two Dimensions of Projected Data After Applying PCA')
plt.show()
```

Matplotlib 还允许生成 3D 散点图。为此，首先需要创建一个具有 3D 投影的坐标轴对象。然后使用 `scatter3D()` 函数创建 3D 散点图，第一个三个参数为 x、y 和 z 坐标。下面的代码使用沿着与三个最大特征值对应的特征向量投影的数据。此代码创建了一个颜色条，而不是图例：

```py
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection='3d')
plt_3d = ax.scatter3D(x_pca[:, -1], x_pca[:, -2], x_pca[:, -3], c=train_labels, s=1)
plt.colorbar(plt_3d)
plt.show()
```

![使用 Matplotlib 生成的 3D 散点图](https://machinelearningmastery.com/wp-content/uploads/2022/03/output_15_0.png)

使用 matplotlib 生成的 3D 散点图

`scatter3D()` 函数仅将点放置到 3D 空间中。之后，我们仍然可以修改图形的显示方式，例如每个坐标轴的标签和背景颜色。但在 3D 图形中，一个常见的调整是**视口**，即我们查看 3D 空间的角度。视口由坐标轴对象中的 `view_init()` 函数控制：

```py
ax.view_init(elev=30, azim=-60)
```

视口由仰角（即相对于水平面角度）和方位角（即水平面上的旋转）控制。默认情况下，matplotlib 使用 30 度的仰角和 -60 度的方位角，如上所示。

综合所有内容，以下是使用 matplotlib 创建 3D 散点图的完整代码：

```py
from tensorflow.keras.datasets import mnist
from tensorflow import dtypes, tensordot
from tensorflow import convert_to_tensor, linalg, transpose
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
(x_train, train_labels), (_, _) = mnist.load_data()
# Choose only the digits 0, 1, 2
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
# Verify the shape of training data
total_examples, img_length, img_width = x_train.shape
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)

# Convert the dataset into a 2D array of shape 18623 x 784
x = convert_to_tensor(np.reshape(x_train, (x_train.shape[0], -1)),
                      dtype=dtypes.float32)
# Eigen-decomposition from a 784 x 784 matrix
eigenvalues, eigenvectors = linalg.eigh(tensordot(transpose(x), x, axes=1))
# Print the three largest eigenvalues
print('3 largest eigenvalues: ', eigenvalues[-3:])
# Project the data to eigenvectors
x_pca = tensordot(x, eigenvectors, axes=1)

# Create the plot
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection='3d')
ax.view_init(elev=30, azim=-60)
plt_3d = ax.scatter3D(x_pca[:, -1], x_pca[:, -2], x_pca[:, -3], c=train_labels, s=1)
plt.colorbar(plt_3d)
plt.show()
```

在 Seaborn 中创建散点图也很简单。`scatterplot()` 方法会自动创建图例，并在绘制点时对不同的类别使用不同的符号。默认情况下，图形会在 matplotlib 的“当前坐标轴”上创建，除非通过 `ax` 参数指定坐标轴对象。

```py
fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(x_pca[:, -1], x_pca[:, -2],
                style=train_labels, hue=train_labels,
                palette=["red", "green", "blue"])
plt.title('First Two Dimensions of Projected Data After Applying PCA')
plt.show()
```

![使用 Seaborn 生成的 2D 散点图](https://machinelearningmastery.com/wp-content/uploads/2022/03/output_17_0.png)

使用 Seaborn 生成的 2D 散点图

Seaborn 相对于 matplotlib 的好处有两点：首先，我们有一个精美的默认样式。例如，如果我们比较上述两个散点图的点样式，Seaborn 的点周围有边框，以防止许多点混在一起。实际上，如果我们在调用任何 matplotlib 函数之前运行以下代码：

```py
sns.set(style = "darkgrid")
```

我们仍然可以使用 matplotlib 函数，但通过使用 Seaborn 的样式可以得到更好的图形。其次，如果我们使用 pandas DataFrame 来保存数据，使用 Seaborn 会更方便。例如，让我们将 MNIST 数据从张量转换为 pandas DataFrame：

```py
df_mnist = pd.DataFrame(x_pca[:, -3:].numpy(), columns=["pca3","pca2","pca1"])
df_mnist["label"] = train_labels
print(df_mnist)
```

现在，DataFrame 看起来如下：

```py
             pca3        pca2         pca1  label
0     -537.730103  926.885254  1965.881592      0
1      167.375885 -947.360107  1070.359375      1
2      553.685425 -163.121826  1754.754272      2
3     -642.905579 -767.283020  1053.937988      1
4     -651.812988 -586.034424   662.468201      1
...           ...         ...          ...    ...
18618  415.358948 -645.245972   853.439209      1
18619  754.555786    7.873116  1897.690552      2
18620 -321.809357  665.038086  1840.480225      0
18621  643.843628  -85.524895  1113.795166      2
18622   94.964279 -549.570984   561.743042      1

[18623 rows x 4 columns]
```

然后，我们可以使用以下代码重现 Seaborn 的散点图：

```py
fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(data=df_mnist, x="pca1", y="pca2",
                style="label", hue="label",
                palette=["red", "green", "blue"])
plt.title('First Two Dimensions of Projected Data After Applying PCA')
plt.show()
```

我们不会将数组作为坐标传递给 `scatterplot()` 函数，而是使用 `data` 参数中的列名。

以下是使用 Seaborn 生成散点图的完整代码，数据存储在 pandas 中：

```py
from tensorflow.keras.datasets import mnist
from tensorflow import dtypes, tensordot
from tensorflow import convert_to_tensor, linalg, transpose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
(x_train, train_labels), (_, _) = mnist.load_data()
# Choose only the digits 0, 1, 2
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
# Verify the shape of training data
total_examples, img_length, img_width = x_train.shape
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)

# Convert the dataset into a 2D array of shape 18623 x 784
x = convert_to_tensor(np.reshape(x_train, (x_train.shape[0], -1)),
                      dtype=dtypes.float32)
# Eigen-decomposition from a 784 x 784 matrix
eigenvalues, eigenvectors = linalg.eigh(tensordot(transpose(x), x, axes=1))
# Print the three largest eigenvalues
print('3 largest eigenvalues: ', eigenvalues[-3:])
# Project the data to eigenvectors
x_pca = tensordot(x, eigenvectors, axes=1)

# Making pandas DataFrame
df_mnist = pd.DataFrame(x_pca[:, -3:].numpy(), columns=["pca3","pca2","pca1"])
df_mnist["label"] = train_labels

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(data=df_mnist, x="pca1", y="pca2",
                style="label", hue="label",
                palette=["red", "green", "blue"])
plt.title('First Two Dimensions of Projected Data After Applying PCA')
plt.show()
```

Seaborn 作为一些 matplotlib 函数的封装，并没有完全取代 matplotlib。例如，Seaborn 不支持 3D 绘图，我们仍然需要使用 matplotlib 函数来实现这些目的。

## Bokeh 中的散点图

matplotlib 和 Seaborn 创建的图表是静态图像。如果你需要放大、平移或切换图表的某部分显示，应该使用 Bokeh。

在 Bokeh 中创建散点图也很简单。以下代码生成一个散点图并添加一个图例。Bokeh 库中的`show()`方法会打开一个新浏览器窗口来显示图像。你可以通过缩放、缩放、滚动等方式与图表互动，使用渲染图旁边工具栏中显示的选项。你还可以通过点击图例来隐藏部分散点。

```py
colormap = {0: "red", 1:"green", 2:"blue"}
my_scatter = figure(title="First Two Dimensions of Projected Data After Applying PCA", 
                    x_axis_label="Dimension 1",
                    y_axis_label="Dimension 2")
for digit in [0, 1, 2]:
    selection = x_pca[train_labels == digit]
    my_scatter.scatter(selection[:,-1].numpy(), selection[:,-2].numpy(),
                       color=colormap[digit], size=5,
                       legend_label="Digit "+str(digit))
my_scatter.legend.click_policy = "hide"
show(my_scatter)
```

Bokeh 将以 HTML 和 JavaScript 生成图表。你控制图表的所有操作都由一些 JavaScript 函数处理。其输出如下所示：

![使用 Bokeh 在新浏览器窗口生成的 2D 散点图。注意右侧的各种选项，用于与图表互动。](https://machinelearningmastery.com/wp-content/uploads/2022/03/bokeh_scatter.png)

使用 Bokeh 在新浏览器窗口生成的 2D 散点图。注意右侧的各种选项，用于与图表互动。

以下是使用 Bokeh 生成上述散点图的完整代码：

```py
from tensorflow.keras.datasets import mnist
from tensorflow import dtypes, tensordot
from tensorflow import convert_to_tensor, linalg, transpose
import numpy as np
from bokeh.plotting import figure, show

# Load dataset
(x_train, train_labels), (_, _) = mnist.load_data()
# Choose only the digits 0, 1, 2
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
# Verify the shape of training data
total_examples, img_length, img_width = x_train.shape
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)

# Convert the dataset into a 2D array of shape 18623 x 784
x = convert_to_tensor(np.reshape(x_train, (x_train.shape[0], -1)),
                      dtype=dtypes.float32)
# Eigen-decomposition from a 784 x 784 matrix
eigenvalues, eigenvectors = linalg.eigh(tensordot(transpose(x), x, axes=1))
# Print the three largest eigenvalues
print('3 largest eigenvalues: ', eigenvalues[-3:])
# Project the data to eigenvectors
x_pca = tensordot(x, eigenvectors, axes=1)

# Create scatter plot in Bokeh
colormap = {0: "red", 1:"green", 2:"blue"}
my_scatter = figure(title="First Two Dimensions of Projected Data After Applying PCA",
                    x_axis_label="Dimension 1",
                    y_axis_label="Dimension 2")
for digit in [0, 1, 2]:
    selection = x_pca[train_labels == digit]
    my_scatter.scatter(selection[:,-1].numpy(), selection[:,-2].numpy(),
                       color=colormap[digit], size=5, alpha=0.5,
                       legend_label="Digit "+str(digit))
my_scatter.legend.click_policy = "hide"
show(my_scatter)
```

如果你在 Jupyter Notebook 中渲染 Bokeh 图表，你可能会看到图表在新浏览器窗口中生成。要将图表放在 Jupyter Notebook 中，你需要在运行 Bokeh 函数之前，告诉 Bokeh 你在笔记本环境下，方法是运行以下代码：

```py
from bokeh.io import output_notebook
output_notebook()
```

此外，请注意我们在循环中创建三个数字的散点图，每次一个数字。这是为了使图例可互动，因为每次调用`scatter()`时都会创建一个新对象。如果我们一次性创建所有散点，如下所示，点击图例会隐藏和显示所有内容，而不是仅显示一个数字的点。

```py
colormap = {0: "red", 1:"green", 2:"blue"}
colors = [colormap[i] for i in train_labels]
my_scatter = figure(title="First Two Dimensions of Projected Data After Applying PCA", 
           x_axis_label="Dimension 1", y_axis_label="Dimension 2")
scatter_obj = my_scatter.scatter(x_pca[:, -1].numpy(), x_pca[:, -2].numpy(), color=colors, size=5)
legend = Legend(items=[
    LegendItem(label="Digit 0", renderers=[scatter_obj], index=0),
    LegendItem(label="Digit 1", renderers=[scatter_obj], index=1),
    LegendItem(label="Digit 2", renderers=[scatter_obj], index=2),
    ])
my_scatter.add_layout(legend)
my_scatter.legend.click_policy = "hide"
show(my_scatter)
```

## 准备线图数据

在我们继续展示如何可视化线图数据之前，让我们生成一些示例数据。下面是一个使用 Keras 库的简单分类器，我们训练它来学习手写数字分类。`fit()`方法返回的历史对象是一个包含训练阶段所有学习历史的字典。为了简化，我们将使用 10 个 epochs 来训练模型。

```py
epochs = 10
y_train = utils.to_categorical(train_labels)
input_dim = img_length*img_width
# Create a Sequential model
model = Sequential()
# First layer for reshaping input images from 2D to 1D
model.add(Reshape((input_dim, ), input_shape=(img_length, img_width)))
# Dense layer of 8 neurons
model.add(Dense(8, activation='relu'))
# Output layer
model.add(Dense(total_classes, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.33, epochs=epochs, batch_size=10, verbose=0)
print('Learning history: ', history.history)
```

上述代码将生成一个包含`loss`、`accuracy`、`val_loss`和`val_accuracy`键的字典，如下所示：

输出

```py
Learning history:  {'loss': [0.5362154245376587, 0.08184114843606949, ...],
'accuracy': [0.9426144361495972, 0.9763565063476562, ...],
'val_loss': [0.09874073415994644, 0.07835448533296585, ...],
'val_accuracy': [0.9716889262199402, 0.9788480401039124, ...]}
```

## matplotlib、Seaborn 和 Bokeh 中的线图

我们来看看各种选项，用于可视化训练分类器获得的学习历史。

在 matplotlib 中创建多线图就像下面这样简单。我们从历史记录中获取训练和验证准确性的值列表，默认情况下，matplotlib 会将其视为顺序数据（即 x 坐标是从 0 开始的整数）。

```py
plt.plot(history.history['accuracy'], label="Training accuracy")
plt.plot(history.history['val_accuracy'], label="Validation accuracy")
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

![使用 Matplotlib 的多线图](https://machinelearningmastery.com/wp-content/uploads/2022/03/lineplot1.png)

使用 Matplotlib 的多线图

创建多线图的完整代码如下：

```py
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
(x_train, train_labels), (_, _) = mnist.load_data()
# Choose only the digits 0, 1, 2
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
# Verify the shape of training data
total_examples, img_length, img_width = x_train.shape
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)

# Prepare for classifier network
epochs = 10
y_train = utils.to_categorical(train_labels)
input_dim = img_length*img_width
# Create a Sequential model
model = Sequential()
# First layer for reshaping input images from 2D to 1D
model.add(Reshape((input_dim, ), input_shape=(img_length, img_width)))
# Dense layer of 8 neurons
model.add(Dense(8, activation='relu'))
# Output layer
model.add(Dense(total_classes, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.33, epochs=epochs, batch_size=10, verbose=0)
print('Learning history: ', history.history)

# Plot accuracy in Matplotlib
plt.plot(history.history['accuracy'], label="Training accuracy")
plt.plot(history.history['val_accuracy'], label="Validation accuracy")
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

同样，我们也可以在 Seaborn 中做同样的事情。正如我们在散点图的例子中看到的，我们可以将数据作为值序列明确传递给 Seaborn，或者通过 pandas DataFrame 传递。让我们使用 pandas DataFrame 绘制训练损失和验证损失：

```py
# Create pandas DataFrame
df_history = pd.DataFrame(history.history)
print(df_history)

# Plot using Seaborn
my_plot = sns.lineplot(data=df_history[["loss","val_loss"]])
my_plot.set_xlabel('Epochs')
my_plot.set_ylabel('Loss')
plt.legend(labels=["Training", "Validation"])
plt.title('Training and Validation Loss')
plt.show()
```

它将打印以下表格，即我们从历史记录中创建的 DataFrame：

输出

```py
       loss  accuracy  val_loss  val_accuracy
0  0.536215  0.942614  0.098741      0.971689
1  0.081841  0.976357  0.078354      0.978848
2  0.064002  0.978841  0.080637      0.972991
3  0.055695  0.981726  0.064659      0.979987
4  0.054693  0.984371  0.070817      0.983729
5  0.053512  0.985173  0.069099      0.977709
6  0.053916  0.983089  0.068139      0.979662
7  0.048681  0.985093  0.064914      0.977709
8  0.052084  0.982929  0.080508      0.971363
9  0.040484  0.983890  0.111380      0.982590
```

它生成的图表如下：

![使用 Seaborn 的多线图](https://machinelearningmastery.com/wp-content/uploads/2022/03/lineplot2.png)

使用 Seaborn 的多线图

默认情况下，Seaborn 会从 DataFrame 中理解列标签，并将其用作图例。在上面的例子中，我们为每个图提供了新的标签。此外，线图的 x 轴默认取自 DataFrame 的索引，在我们的例子中是从 0 到 9 的整数。

生成 Seaborn 图表的完整代码如下：

```py
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
(x_train, train_labels), (_, _) = mnist.load_data()
# Choose only the digits 0, 1, 2
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
# Verify the shape of training data
total_examples, img_length, img_width = x_train.shape
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)

# Prepare for classifier network
epochs = 10
y_train = utils.to_categorical(train_labels)
input_dim = img_length*img_width
# Create a Sequential model
model = Sequential()
# First layer for reshaping input images from 2D to 1D
model.add(Reshape((input_dim, ), input_shape=(img_length, img_width)))
# Dense layer of 8 neurons
model.add(Dense(8, activation='relu'))
# Output layer
model.add(Dense(total_classes, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.33, epochs=epochs, batch_size=10, verbose=0)

# Prepare pandas DataFrame
df_history = pd.DataFrame(history.history)
print(df_history)

# Plot loss in seaborn
my_plot = sns.lineplot(data=df_history[["loss","val_loss"]])
my_plot.set_xlabel('Epochs')
my_plot.set_ylabel('Loss')
plt.legend(labels=["Training", "Validation"])
plt.title('Training and Validation Loss')
plt.show()
```

正如你所预期的，如果我们想精确控制 x 和 y 坐标，我们还可以将参数 `x` 和 `y` 与 `data` 一起传递给 `lineplot()`，就像我们在上面的 Seaborn 散点图示例中一样。

Bokeh 也可以生成多线图，如下代码所示。正如我们在散点图例子中看到的，我们需要明确提供 x 和 y 坐标，并且一次绘制一条线。同样，`show()` 方法会打开一个新的浏览器窗口来显示图表，你可以与之互动。

```py
p = figure(title="Training and validation accuracy",
           x_axis_label="Epochs", y_axis_label="Accuracy")
epochs_array = np.arange(epochs)
p.line(epochs_array, df_history['accuracy'], legend_label="Training",
       color="blue", line_width=2)
p.line(epochs_array, df_history['val_accuracy'], legend_label="Validation",
       color="green")
p.legend.click_policy = "hide"
p.legend.location = 'bottom_right'
show(p)
```

![使用 Bokeh 的多线图。注意右侧工具栏上的用户交互选项。](https://machinelearningmastery.com/wp-content/uploads/2022/03/lineplot3.png)

使用 Bokeh 的多线图。注意右侧工具栏上的用户交互选项。

制作 Bokeh 图表的完整代码如下：

```py
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show

# Load dataset
(x_train, train_labels), (_, _) = mnist.load_data()
# Choose only the digits 0, 1, 2
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
# Verify the shape of training data
total_examples, img_length, img_width = x_train.shape
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)

# Prepare for classifier network
epochs = 10
y_train = utils.to_categorical(train_labels)
input_dim = img_length*img_width
# Create a Sequential model
model = Sequential()
# First layer for reshaping input images from 2D to 1D
model.add(Reshape((input_dim, ), input_shape=(img_length, img_width)))
# Dense layer of 8 neurons
model.add(Dense(8, activation='relu'))
# Output layer
model.add(Dense(total_classes, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.33, epochs=epochs, batch_size=10, verbose=0)

# Prepare pandas DataFrame
df_history = pd.DataFrame(history.history)
print(df_history)

# Plot accuracy in Bokeh
p = figure(title="Training and validation accuracy",
           x_axis_label="Epochs", y_axis_label="Accuracy")
epochs_array = np.arange(epochs)
p.line(epochs_array, df_history['accuracy'], legend_label="Training",
       color="blue", line_width=2)
p.line(epochs_array, df_history['val_accuracy'], legend_label="Validation",
       color="green")
p.legend.click_policy = "hide"
p.legend.location = 'bottom_right'
show(p)
```

## 更多可视化内容

我们之前介绍的每一个工具都有更多功能，让我们控制可视化中的细节。重要的是查看它们各自的文档，以了解如何完善你的图表。同样重要的是查看文档中的示例代码，以学习如何使你的可视化更好。

不提供过多细节，这里有一些你可能想要添加到可视化中的想法：

+   添加辅助线，例如在时间序列数据上标记训练集和验证集。matplotlib 的 `axvline()` 函数可以在图上画竖直线来实现这个目的。

+   添加注释，例如箭头和文本标签，以识别图中的关键点。查看 matplotlib axes 对象中的 `annotate()` 函数。

+   控制透明度水平，以防止图形元素重叠。我们上面介绍的所有绘图函数都允许使用 `alpha` 参数，提供一个介于 0 和 1 之间的值，表示我们可以看到图形的透明程度。

+   如果数据以这种方式更好地呈现，我们可以在某些轴上显示对数刻度。通常称为对数图或半对数图。

在我们结束本文之前，以下是一个示例，展示了如何在 matplotlib 中创建并排可视化，其中一个使用了 Seaborn：

```py
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow import dtypes, tensordot
from tensorflow import convert_to_tensor, linalg, transpose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
(x_train, train_labels), (_, _) = mnist.load_data()
# Choose only the digits 0, 1, 2
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
# Verify the shape of training data
total_examples, img_length, img_width = x_train.shape
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)

# Convert the dataset into a 2D array of shape 18623 x 784
x = convert_to_tensor(np.reshape(x_train, (x_train.shape[0], -1)),
                      dtype=dtypes.float32)
# Eigen-decomposition from a 784 x 784 matrix
eigenvalues, eigenvectors = linalg.eigh(tensordot(transpose(x), x, axes=1))
# Print the three largest eigenvalues
print('3 largest eigenvalues: ', eigenvalues[-3:])
# Project the data to eigenvectors
x_pca = tensordot(x, eigenvectors, axes=1)

# Prepare for classifier network
epochs = 10
y_train = utils.to_categorical(train_labels)
input_dim = img_length*img_width
# Create a Sequential model
model = Sequential()
# First layer for reshaping input images from 2D to 1D
model.add(Reshape((input_dim, ), input_shape=(img_length, img_width)))
# Dense layer of 8 neurons
model.add(Dense(8, activation='relu'))
# Output layer
model.add(Dense(total_classes, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.33, epochs=epochs, batch_size=10, verbose=0)

# Prepare pandas DataFrame
df_history = pd.DataFrame(history.history)
print(df_history)

# Plot side-by-side
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
# left plot
scatter = ax[0].scatter(x_pca[:, -1], x_pca[:, -2], c=train_labels, s=5)
legend_plt = ax[0].legend(*scatter.legend_elements(),
                         loc="lower left", title="Digits")
ax[0].add_artist(legend_plt)
ax[0].set_title('First Two Dimensions of Projected Data After Applying PCA')
# right plot
my_plot = sns.lineplot(data=df_history[["loss","val_loss"]], ax=ax[1])
my_plot.set_xlabel('Epochs')
my_plot.set_ylabel('Loss')
ax[1].legend(labels=["Training", "Validation"])
ax[1].set_title('Training and Validation Loss')
plt.show()
```

![](img/3bb3fd2a032ee31d47de2d67b0c35b6f.png)

使用 matplotlib 和 Seaborn 创建的并排可视化

Bokeh 的等效方法是分别创建每个子图，然后在显示时指定布局：

```py
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow import dtypes, tensordot
from tensorflow import convert_to_tensor, linalg, transpose
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.layouts import row

# Load dataset
(x_train, train_labels), (_, _) = mnist.load_data()
# Choose only the digits 0, 1, 2
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
# Verify the shape of training data
total_examples, img_length, img_width = x_train.shape
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)

# Convert the dataset into a 2D array of shape 18623 x 784
x = convert_to_tensor(np.reshape(x_train, (x_train.shape[0], -1)),
                      dtype=dtypes.float32)
# Eigen-decomposition from a 784 x 784 matrix
eigenvalues, eigenvectors = linalg.eigh(tensordot(transpose(x), x, axes=1))
# Print the three largest eigenvalues
print('3 largest eigenvalues: ', eigenvalues[-3:])
# Project the data to eigenvectors
x_pca = tensordot(x, eigenvectors, axes=1)

# Prepare for classifier network
epochs = 10
y_train = utils.to_categorical(train_labels)
input_dim = img_length*img_width
# Create a Sequential model
model = Sequential()
# First layer for reshaping input images from 2D to 1D
model.add(Reshape((input_dim, ), input_shape=(img_length, img_width)))
# Dense layer of 8 neurons
model.add(Dense(8, activation='relu'))
# Output layer
model.add(Dense(total_classes, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.33, epochs=epochs, batch_size=10, verbose=0)

# Prepare pandas DataFrame
df_history = pd.DataFrame(history.history)
print(df_history)

# Create scatter plot in Bokeh
colormap = {0: "red", 1:"green", 2:"blue"}
my_scatter = figure(title="First Two Dimensions of Projected Data After Applying PCA",
                    x_axis_label="Dimension 1",
                    y_axis_label="Dimension 2",
                    width=500, height=400)
for digit in [0, 1, 2]:
    selection = x_pca[train_labels == digit]
    my_scatter.scatter(selection[:,-1].numpy(), selection[:,-2].numpy(),
                       color=colormap[digit], size=5, alpha=0.5,
                       legend_label="Digit "+str(digit))
my_scatter.legend.click_policy = "hide"

# Plot accuracy in Bokeh
p = figure(title="Training and validation accuracy",
           x_axis_label="Epochs", y_axis_label="Accuracy",
           width=500, height=400)
epochs_array = np.arange(epochs)
p.line(epochs_array, df_history['accuracy'], legend_label="Training",
       color="blue", line_width=2)
p.line(epochs_array, df_history['val_accuracy'], legend_label="Validation",
       color="green")
p.legend.click_policy = "hide"
p.legend.location = 'bottom_right'

show(row(my_scatter, p))
```

![](img/1a58106cdf91b36d6e7e7ec780cba930.png)

侧边并排绘制在 Bokeh 中创建的图

## 进一步阅读

如果你希望更深入地了解这个主题，本节提供更多资源。

### 书籍

+   [Think Python：如何像计算机科学家一样思考](https://greenteapress.com/thinkpython/html/index.html)，Allen B. Downey 著

+   [Python 3 编程：Python 语言完全介绍](https://www.amazon.com/dp/B001OFK2DK/)，Mark Summerfield 著

+   [Python 编程：计算机科学导论](https://www.amazon.com/dp/1590282418/)，John Zelle 著

+   [Python 数据分析](https://www.amazon.com/dp/1491957662)，第二版，Wes McKinney 著

### 文章

+   [Python 中数据可视化方法的简明介绍](https://machinelearningmastery.com/data-visualization-methods-in-python/)

+   [如何使用 Seaborn 数据可视化机器学习](https://machinelearningmastery.com/seaborn-data-visualization-for-machine-learning/)

### API 参考

+   [matplotlib.pyplot.scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html#matplotlib.pyplot.scatter)

+   [matplotlib.pyplot.plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html)

+   [seaborn.scatterplot](https://seaborn.pydata.org/generated/seaborn.scatterplot.html)

+   [seaborn.lineplot](https://seaborn.pydata.org/generated/seaborn.lineplot.html)

+   [Bokeh 使用基本图元进行绘图](https://docs.bokeh.org/en/latest/docs/user_guide/plotting.html)

+   [Bokeh 散点图](https://docs.bokeh.org/en/latest/docs/reference/plotting/figure.html#bokeh.plotting.Figure.scatter)

+   [Bokeh 折线图](https://docs.bokeh.org/en/latest/docs/first_steps/first_steps_1.html)

## 摘要

在这个教程中，你将发现 Python 中数据可视化的各种选项。

具体来说，你学会了：

+   如何在不同行和列创建子图

+   如何使用 matplotlib 渲染图像

+   如何使用 matplotlib 生成 2D 和 3D 散点图

+   如何使用 Seaborn 和 Bokeh 创建 2D 图

+   如何使用 matplotlib、Seaborn 和 Bokeh 创建多行图

对于本帖讨论的数据可视化选项，您是否有任何问题？请在下面的评论中提问，我将尽力回答。
