# 如何在 Python 中对长短期记忆网络使用`TimeDistributed`层

> 原文： [`machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/`](https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/)

长期短期网络或 LSTM 是一种流行且功能强大的循环神经网络或 RNN。

它们可能很难配置并应用于任意序列预测问题，即使使用定义良好且“易于使用”的接口，如 Python 中的 Keras 深度学习库中提供的那些接口也是如此。

在 Keras 中遇到这种困难的一个原因是使用了 TimeDistributed 包装层，并且需要一些 LSTM 层来返回序列而不是单个值。

在本教程中，您将发现为序列预测配置 LSTM 网络的不同方法，TimeDistributed 层所扮演的角色以及如何使用它。

完成本教程后，您将了解：

*   如何设计一对一的 LSTM 用于序列预测。
*   如何在没有 TimeDistributed Layer 的情况下设计用于序列预测的多对一 LSTM。
*   如何使用 TimeDistributed Layer 设计多对多 LSTM 以进行序列预测。

让我们开始吧。

![How to Use the TimeDistributed Layer for Long Short-Term Memory Networks in Python](img/c6a3f825141c0b28d96fff209cf03362.jpg)

如何在 Python 中使用 TimeDistributed Layer for Long Short-Term Memory Networks
[jans canon](https://www.flickr.com/photos/43158397@N02/5774000092/) 的照片，保留一些权利。

## 教程概述

本教程分为 5 个部分;他们是：

1.  TimeDistributed Layer
2.  序列学习问题
3.  用于序列预测的一对一 LSTM
4.  用于序列预测的多对一 LSTM（没有 TimeDistributed）
5.  用于序列预测的多对多 LSTM（具有 TimeDistributed）

### 环境

本教程假定安装了 SciPy，NumPy 和 Pandas 的 Python 2 或 Python 3 开发环境。

本教程还假设 scikit-learn 和 Keras v2.0 +与 Theano 或 TensorFlow 后端一起安装。

有关设置 Python 环境的帮助，请参阅帖子：

*   [如何使用 Anaconda 设置用于机器学习和深度学习的 Python 环境](http://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## TimeDistributed Layer

LSTM 功能强大，但难以使用且难以配置，尤其适合初学者。

另一个复杂因素是 [TimeDistributed](https://keras.io/layers/wrappers/#timedistributed) Layer（以及之前的`TimeDistributedDense`层），它被隐式描述为层包装器：

> 这个包装器允许我们将一个层应用于输入的每个时间片。

你应该如何以及何时使用 LSTM 的这个包装器？

当您在 Keras GitHub 问题和 StackOverflow 上搜索有关包装层的讨论时，这种混淆更加复杂。

例如，在问题“[何时以及如何使用 TimeDistributedDense](https://github.com/fchollet/keras/issues/1029) ”中，“fchollet（Keras'作者）解释说：

> TimeDistributedDense 对 3D 张量的每个时间步应用相同的 Dense（完全连接）操作。

如果您已经了解 TimeDistributed 层的用途以及何时使用它，那么这是完全合理的，但对初学者来说根本没有帮助。

本教程旨在消除使用带有 LSTM 的 TimeDistributed 包装器的混乱，以及可以检查，运行和使用的工作示例，以帮助您进行具体的理解。

## 序列学习问题

我们将使用一个简单的序列学习问题来演示 TimeDistributed 层。

在这个问题中，序列[0.0,0.2,0.4,0.6,0.8]将一次作为输入一个项目给出，并且必须依次作为输出返回，一次一个项目。

可以把它想象成一个简单的打印程序。我们给出 0.0 作为输入，我们期望看到 0.0 作为输出，对序列中的每个项重复。

我们可以直接生成这个序列如下：

```py
from numpy import array
length = 5
seq = array([i/float(length) for i in range(length)])
print(seq)
```

运行此示例将打印生成的序列：

```py
[ 0\.   0.2  0.4  0.6  0.8]
```

该示例是可配置的，如果您愿意，您可以稍后自己玩更长/更短的序列。请在评论中告诉我您的结果。

## 用于序列预测的一对一 LSTM

在我们深入研究之前，重要的是要表明这种序列学习问题可以分段学习。

也就是说，我们可以将问题重新构造为序列中每个项目的输入 - 输出对的数据集。给定 0，网络应输出 0，给定 0.2，网络必须输出 0.2，依此类推。

这是问题的最简单的公式，并且要求将序列分成输入 - 输出对，并且序列一次一步地预测并聚集在网络外部。

输入输出对如下：

```py
X, 	y
0.0,	0.0
0.2,	0.2
0.4,	0.4
0.6,	0.6
0.8,	0.8
```

LSTM 的输入必须是三维的。我们可以将 2D 序列重塑为具有 5 个样本，1 个时间步长和 1 个特征的 3D 序列。我们将输出定义为具有 1 个特征的 5 个样本。

```py
X = seq.reshape(5, 1, 1)
y = seq.reshape(5, 1)
```

我们将网络模型定义为具有 1 个输入和 1 个时间步长。第一个隐藏层将是一个有 5 个单位的 LSTM。输出层是一个带有 1 个输出的全连接层。

该模型将适用于有效的 ADAM 优化算法和均方误差损失函数。

批量大小设置为时期中的样本数量，以避免必须使 LSTM 有状态并手动管理状态重置，尽管这可以很容易地完成，以便在每个样本显示到网络后更新权重。

完整的代码清单如下：

```py
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# prepare sequence
length = 5
seq = array([i/float(length) for i in range(length)])
X = seq.reshape(len(seq), 1, 1)
y = seq.reshape(len(seq), 1)
# define LSTM configuration
n_neurons = length
n_batch = length
n_epoch = 1000
# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result:
	print('%.1f' % value)
```

首先运行该示例将打印已配置网络的结构。

我们可以看到 LSTM 层有 140 个参数。这是根据输入数量（1）和输出数量（隐藏层中 5 个单位为 5）计算的，如下所示：

```py
n = 4 * ((inputs + 1) * outputs + outputs²)
n = 4 * ((1 + 1) * 5 + 5²)
n = 4 * 35
n = 140
```

我们还可以看到，完全连接的层只有 6 个参数用于输入数量（5 个用于前一层的 5 个输入），输出数量（1 个用于层中的 1 个神经元）和偏差。

```py
n = inputs * outputs + outputs
n = 5 * 1 + 1
n = 6
```

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm_1 (LSTM)                (None, 1, 5)              140
_________________________________________________________________
dense_1 (Dense)              (None, 1, 1)              6
=================================================================
Total params: 146.0
Trainable params: 146
Non-trainable params: 0.0
_________________________________________________________________
```

网络正确地学习预测问题。

```py
0.0
0.2
0.4
0.6
0.8
```

## 用于序列预测的多对一 LSTM（没有 TimeDistributed）

在本节中，我们开发了一个 LSTM 来一次输出序列，尽管没有 TimeDistributed 包装层。

LSTM 的输入必须是三维的。我们可以将 2D 序列重塑为具有 1 个样本，5 个时间步长和 1 个特征的 3D 序列。我们将输出定义为具有 5 个特征的 1 个样本。

```py
X = seq.reshape(1, 5, 1)
y = seq.reshape(1, 5)
```

您可以立即看到，必须稍微调整问题定义，以便在没有 TimeDistributed 包装器的情况下支持网络进行序列预测。具体来说，输出一个向量而不是一次一步地构建输出序列。差异可能听起来很微妙，但了解 TimeDistributed 包装器的作用非常重要。

我们将模型定义为具有 5 个时间步长的一个输入。第一个隐藏层将是一个有 5 个单位的 LSTM。输出层是一个完全连接的层，有 5 个神经元。

```py
# create LSTM
model = Sequential()
model.add(LSTM(5, input_shape=(5, 1)))
model.add(Dense(length))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
```

接下来，我们将模型仅适用于训练数据集中的单个样本的 500 个迭代和批量大小为 1。

```py
# train LSTM
model.fit(X, y, epochs=500, batch_size=1, verbose=2)
```

综合这些，下面提供了完整的代码清单。

```py
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# prepare sequence
length = 5
seq = array([i/float(length) for i in range(length)])
X = seq.reshape(1, length, 1)
y = seq.reshape(1, length)
# define LSTM configuration
n_neurons = length
n_batch = 1
n_epoch = 500
# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(length, 1)))
model.add(Dense(length))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result[0,:]:
	print('%.1f' % value)
```

首先运行该示例将打印已配置网络的摘要。

我们可以看到 LSTM 层有 140 个参数，如上一节所述。

LSTM 单元已经瘫痪，每个单元都输出一个值，提供 5 个值的向量作为完全连接层的输入。时间维度或序列信息已被丢弃并折叠成 5 个值的向量。

我们可以看到完全连接的输出层有 5 个输入，预计输出 5 个值。我们可以解释如下要学习的 30 个权重：

```py
n = inputs * outputs + outputs
n = 5 * 5 + 5
n = 30
```

网络摘要报告如下：

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm_1 (LSTM)                (None, 5)                 140
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 30
=================================================================
Total params: 170.0
Trainable params: 170
Non-trainable params: 0.0
_________________________________________________________________
```

该模型适合，在最终确定和打印预测序列之前打印损失信息。

序列被正确再现，但是作为单个部分而不是逐步通过输入数据。我们可能使用 Dense 层作为第一个隐藏层而不是 LSTM，因为 LSTM 的这种使用并没有充分利用它们完整的序列学习和处理能力。

```py
0.0
0.2
0.4
0.6
0.8
```

## 用于序列预测的多对多 LSTM（具有 TimeDistributed）

在本节中，我们将使用 TimeDistributed 层来处理 LSTM 隐藏层的输出。

使用 TimeDistributed 包装层时要记住两个关键点：

*   **输入必须（至少）为 3D** 。这通常意味着您需要在 TimeDistributed wrapped Dense 层之前配置最后一个 LSTM 层以返回序列（例如，将“return_sequences”参数设置为“True”）。
*   **输出为 3D** 。这意味着如果 TimeDistributed 包裹的 Dense 层是输出层并且您正在预测序列，则需要将 y 数组的大小调整为 3D 向量。

我们可以将输出的形状定义为具有 1 个样本，5 个时间步长和 1 个特征，就像输入序列一样，如下所示：

```py
y = seq.reshape(1, length, 1)
```

我们可以通过将“`return_sequences`”参数设置为 true 来定义 LSTM 隐藏层以返回序列而不是单个值。

```py
model.add(LSTM(n_neurons, input_shape=(length, 1), return_sequences=True))
```

这具有每个 LSTM 单元返回 5 个输出序列的效果，输出数据中的每个时间步长一个输出，而不是如前一示例中的单个输出值。

我们还可以使用输出层上的 TimeDistributed 来包装具有单个输出的完全连接的 Dense 层。

```py
model.add(TimeDistributed(Dense(1)))
```

输出层中的单个输出值是关键。它强调我们打算从输入中的每个时间步的序列输出一个时间步。碰巧我们将一次处理输入序列的 5 个时间步。

TimeDistributed 通过一次一个步骤将相同的 Dense 层（相同的权重）应用于 LSTM 输出来实现此技巧。这样，输出层只需要一个连接到每个 LSTM 单元（加上一个偏置）。

因此，需要增加训练时期的数量以考虑较小的网络容量。我将它从 500 加倍到 1000，以匹配第一个一对一的例子。

将它们放在一起，下面提供了完整的代码清单。

```py
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
# prepare sequence
length = 5
seq = array([i/float(length) for i in range(length)])
X = seq.reshape(1, length, 1)
y = seq.reshape(1, length, 1)
# define LSTM configuration
n_neurons = length
n_batch = 1
n_epoch = 1000
# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(length, 1), return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result[0,:,0]:
	print('%.1f' % value)
```

运行该示例，我们可以看到已配置网络的结构。

我们可以看到，与前面的示例一样，LSTM 隐藏层中有 140 个参数。

完全连接的输出层是一个非常不同的故事。实际上，它完全符合一对一的例子。一个神经元，对于前一层中的每个 LSTM 单元具有一个权重，加上一个用于偏置输入。

这有两个重要的事情：

*   允许在定义问题时构建和学习问题，即一个输出到一个输出，保持每个时间步的内部过程分开。
*   通过要求更少的权重来简化网络，使得一次只处理一个时间步长。

将一个更简单的完全连接层应用于从前一层提供的序列中的每个时间步，以构建输出序列。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm_1 (LSTM)                (None, 5, 5)              140
_________________________________________________________________
time_distributed_1 (TimeDist (None, 5, 1)              6
=================================================================
Total params: 146.0
Trainable params: 146
Non-trainable params: 0.0
_________________________________________________________________
```

同样，网络学习序列。

```py
0.0
0.2
0.4
0.6
0.8
```

我们可以将时间步长问题框架和 TimeDistributed 层视为在第一个示例中实现一对一网络的更紧凑方式。它甚至可能在更大规模上更有效（空间或时间）。

## 进一步阅读

以下是您可能希望深入研究的 TimeDistributed 层的一些资源和讨论。

*   [Keras API 中的 TimeDistributed Layer](https://keras.io/layers/wrappers/#timedistributed)
*   [GitHub 上的 TimeDistributed](https://github.com/fchollet/keras/blob/master/keras/layers/wrappers.py#L56) 代码
*   [StackExchange 上'Keras'](http://datascience.stackexchange.com/questions/10836/the-difference-between-dense-and-timedistributeddense-of-keras)的'密集'和'TimeDistributedDense'之间的区别
*   [何时以及如何在 GitHub 上使用 TimeDistributedDense](https://github.com/fchollet/keras/issues/1029)

## 摘要

在本教程中，您了解了如何为序列预测开发 LSTM 网络以及 TimeDistributed 层的作用。

具体来说，你学到了：

*   如何设计一对一的 LSTM 用于序列预测。
*   如何在没有 TimeDistributed Layer 的情况下设计用于序列预测的多对一 LSTM。
*   如何使用 TimeDistributed Layer 设计多对多 LSTM 以进行序列预测。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。