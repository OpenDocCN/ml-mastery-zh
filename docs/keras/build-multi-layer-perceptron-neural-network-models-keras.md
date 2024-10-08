# 如何用 Keras 构建多层感知机神经网络模型

> 原文： [`machinelearningmastery.com/build-multi-layer-perceptron-neural-network-models-keras/`](https://machinelearningmastery.com/build-multi-layer-perceptron-neural-network-models-keras/)

用于深度学习的 Keras Python 库专门用于构建一系列层的模型。

在这篇文章中，您将了解用来创建神经网络的简单组件和使用 Keras 的简单深度学习模型。

您可以通过[我新书](https://machinelearningmastery.com/deep-learning-with-python/)中的一些代码了解如何为一系列预测性建模问题开发深度学习模型，书中包含 18 个分步教程和 9 个项目。

让我们开始吧。

*   **2017 年 3 月更新**：更新了 Keras 2.0.2，TensorFlow 1.0.1 和 Theano 0.9.0 的示例。

![How To Build Multi-Layer Perceptron Neural Network Models with Keras](img/1ff32d002f02ee2ec7be1f308faffc6b.png)

如何用 Keras 建立多层感知机神经网络模型
照片由 [George Rex](https://www.flickr.com/photos/rogersg/13316350364/) ，保留一些权利。

## Keras 中的神经网络模型

Keras 库的重点是模型。

最简单的模型用 Sequential 类定义，它是一个层的线性栈。

您可以创建一个 Sequential 模型类然后定义模型中的所有层，例如：

```py
from keras.models import Sequential
model = Sequential(...)
```

更有用的书写方式是创建一个 Sequential 模型然后按照您希望执行的计算顺序添加层，例如：

```py
from keras.models import Sequential
model = Sequential()
model.add(...)
model.add(...)
model.add(...)
```

## 模型输入

模型中的第一层必须指定输入的形状。

这是输入属性的数量，由 input_dim 参数定义。这个参数需要是一个整数。

例如，您可以按照 Dense 类型层的 8 个输入定义输入，如下所示：

```py
Dense(16, input_dim=8)
```

## 模型层

不同类型的层是一些共同的属性，特别是它们的初始化权重和激活函数。

### 权重初始化

用于层的初始化类型在 init 参数中指定。

一些常见的层初始化类型包括：

*   “_ uniform _”：将权重初始化为 0 到 0.05 之间的小的均匀随机值。
*   “_ normal _”：将权重初始化为小高斯随机值（均值为 0，标准差 0.05）。
*   “_ zero _”：所有权重都设置为零。

您可以在[初始化用法](http://keras.io/initializations/)页面上看到完整的初始化方法列表。

### 激活功能

Keras 支持一系列标准神经元激活功能，例如：softmax，rectifier，tanh 和 sigmoid。

您通常会通过激活参数指定每层使用的激活函数的类型，该函数采用字符串值。

您可以在[激活用法](http://keras.io/activations/)页面上查看 Keras 支持的完整激活方式列表。

有趣的是，您还可以创建一个 Activation 对象，并在层之后将其直接添加到模型中，以将该激活应用于层的输出。

### 层类型

标准神经网络有大量的核心层类型。

您可以选择的一些常见且有用的层类型是：

*   ** Dense**：完全连接的层和多层感知机模型上使用的最常见的层。
*   **Dropout** ：将 dropout 应用于模型，将输入的一小部分设置为零，以减少过拟合。
*   ** Merge**：将多个模型的输入组合到一个模型中。

您可以在[核心层](http://keras.io/layers/core/)页面上了解核心 Keras 层的完整列表

## 模型编译

一旦定义了模型，就需要编译它。

这将创建底层后端（Theano 或 TensorFlow）使用的有效结构，以便在训练期间有效地执行您的模型。

您可以使用 compile（）函数编译模型，它需要定义三个重要属性：

1.  模型优化器。
2.  损失函数。
3.  指标。

```py
model.compile(optimizer=, loss=, metrics=)
```

### 1.模型优化器

优化程序是用于更新模型中权重的搜索技术。

您可以创建优化器对象并通过优化器参数将其传递给编译函数。这允许您使用自己的参数（例如学习率）配置优化过程。例如：

```py
sgd = SGD(...)
model.compile(optimizer=sgd)
```

您还可以通过为优化程序参数指定优化程序的名称来使用优化程序的默认参数。例如：

```py
model.compile(optimizer='sgd')
```

您可能想要选择的一些流行的梯度下降优化器包括：

*   **SGD** ：随机梯度下降，支持动量。
*   **RMSprop** ：Geoff Hinton 提出的自适应学习率优化方法。
*   **Adam** ：自适应力矩估计（Adam），也使用自适应学习率。

您可以在 [Usage of optimizers](http://keras.io/optimizers/) 页面上了解 Keras 支持的所有优化器。

您可以在 [Sebastian Ruder 的帖子](http://sebastianruder.com/optimizing-gradient-descent/index.html)[梯度下降优化算法部分](http://sebastianruder.com/optimizing-gradient-descent/index.html#gradientdescentoptimizationalgorithms)上了解有关不同梯度下降方法的更多信息。梯度下降优化算法概述。

### 2.模型损失函数

损失函数，也称为目标函数，由用于导航权重空间的优化器来评估模型。

您可以通过 loss 参数指定要用于编译函数的 loss 函数的名称。一些常见的例子包括：

*   '`mse`'：表示均方误差。
*   '`binary_crossentropy`'：用于二进制对数损失（logloss）。
*   '`categorical_crossentropy`'：用于多类对数损失（logloss）。

您可以在[目标用途](http://keras.io/objectives/)页面上了解更多关于 Keras 支持的损失函数。

### 3.模型指标

在训练期间，模型评估度量标准。

目前仅支持一个指标即准确率。

## 模特训练

使用 fit（）函数在 NumPy 数组上训练模型，例如

```py
model.fit(X, y, epochs=, batch_size=)
```

训练都指定了训练的次数和训练数据的批量大小。

*   Epochs（nb_epoch）是模型指定给训练数据集的次数。
*   Batch Size（batch_size）是在执行权重更新之前向模型显示的训练实例的数量。

拟合函数还允许在训练期间对模型进行一些基本评估。您可以设置 validation_split 值以返回每次训练的训练数据集的一小部分以便评估验证，或提供要评估的数据的（X，y）的 validation_data 元组。

拟合模型返回历史对象，其中包含为每个时期的模型计算的详细信息和指标。这可用于绘制模型表现。

## 模型预测

训练完模型后，可以使用它来预测测试数据或新数据。

您可以从训练模型中计算出许多不同的输出类型，每种输出类型都是使用模型对象上的不同函数调用计算的。例如：

*   _model.evaluate（）_：计算输入数据的损失值。
*   _model.predict（）_：为输入数据生成网络输出。
*   _model.predict_classes（）_：为输入数据生成类输出。
*   _model.predict_proba（）_：为输入数据生成类概率。

例如，在分类问题上，您将使用 predict_classes（）函数对测试数据或新数据实例做出预测。

## 总结模型

一旦您对您的模型感到满意，说明您已经完成了模型。

您可能希望输出模型的摘要。例如，您可以通过调用摘要函数来显示模型的摘要，例如：

```py
model.summary()
```

您还可以使用 get_config（）函数检索模型配置的摘要，例如：

```py
model.get_config()
```

最后，您可以直接创建模型结构的图像。例如：

```py
from keras.utils.vis_utils import plot_model
plot(model, to_file='model.png')
```

## 资源

您可以使用以下资源了解有关如何在 Keras 中创建简单神经网络和深度学习模型的更多信息：

*   [开始使用 Keras 顺序模型](http://keras.io/getting-started/sequential-model-guide/)。
*   [关于 Keras 模型](http://keras.io/models/about-keras-models/)。
*   [顺序模型 API](http://keras.io/models/sequential/) 。

## 总结

在这篇文章中，您已经了解了可用于创建人工神经网络和深度学习模型的 Keras API。

具体来说，您了解了 Keras 模型的生命周期，包括：

*   构建模型。
*   创建和添加层，及对层进行初始化权重和激活函数的设置。
*   编译模型，包括优化方法，损失函数和度量。
*   训练模型，包括训练迭代次数和训练数据批量的大小
*   模型预测。
*   总结模型。

如果您对 Keras for Deep Learning 或本文有任何疑问，请在评论中提问，我会尽力回答。
