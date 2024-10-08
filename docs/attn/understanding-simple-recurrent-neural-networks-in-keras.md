# 在 Keras 中理解简单的递归神经网络

> 原文：[`machinelearningmastery.com/understanding-simple-recurrent-neural-networks-in-keras/`](https://machinelearningmastery.com/understanding-simple-recurrent-neural-networks-in-keras/)

本教程适用于希望了解递归神经网络（RNN）工作原理及如何通过 Keras 深度学习库使用它们的任何人。虽然 Keras 库提供了解决问题和构建应用所需的所有方法，但了解一切如何工作也很重要。本文展示了 RNN 模型中的计算步骤。接下来，将开发用于时间序列预测的完整端到端系统。

完成本教程后，您将了解：

+   RNN 的结构

+   当给定输入时，RNN 如何计算输出

+   如何为 Keras 中的 SimpleRNN 准备数据

+   如何训练一个 SimpleRNN 模型

**使用我的书[使用注意力构建 Transformer 模型](https://machinelearningmastery.com/transformer-models-with-attention/)快速启动项目**。它提供了**自学教程**和**可运行代码**，帮助您构建一个完全可工作的 Transformer 模型，可以

*将句子从一种语言翻译成另一种语言*...

让我们开始吧。

![Umstead state park](https://machinelearningmastery.com/wp-content/uploads/2021/09/IMG_9433-scaled.jpg)

在 Keras 中理解简单的递归神经网络。照片由 Mehreen Saeed 提供，部分权利保留。

## 教程概述

本教程分为两部分；它们是：

1.  RNN 的结构

    1.  不同层的不同权重和偏置与 RNN 的关联

    1.  在给定输入时计算输出的计算方式

1.  用于时间序列预测的完整应用程序

## 先决条件

假设您在开始实施之前对 RNN 有基本的了解。[递归神经网络及其动力学的简介](https://machinelearningmastery.com/an-introduction-to-recurrent-neural-networks-and-the-math-that-powers-them)为您快速概述了 RNN。

现在让我们直接进入实施部分。

## 导入部分

要开始实现 RNN，请添加导入部分。

Python

```py
from pandas import read_csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
```

### 想要开始构建带注意力的 Transformer 模型吗？

现在立即参加我的免费 12 天电子邮件速成课程（附有示例代码）。

点击注册，并获得课程的免费 PDF 电子书版本。

## Keras SimpleRNN

下面的函数返回一个包含 `SimpleRNN` 层和 `Dense` 层的模型，用于学习序列数据。`input_shape` 参数指定了 `(time_steps x features)`。我们将简化一切，并使用单变量数据，即只有一个特征；时间步骤将在下面讨论。

Python

```py
def create_RNN(hidden_units, dense_units, input_shape, activation):
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, 
                        activation=activation[0]))
    model.add(Dense(units=dense_units, activation=activation[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

demo_model = create_RNN(2, 1, (3,1), activation=['linear', 'linear'])
```

对象 `demo_model` 通过 `SimpleRNN` 层创建了两个隐藏单元，并通过 `Dense` 层创建了一个密集单元。`input_shape` 设置为 3×1，并且在两个层中都使用了 `linear` 激活函数以保持简单性。需要注意的是，线性激活函数 $f(x) = x$ 对输入不做任何更改。网络结构如下：

如果我们有 $m$ 个隐藏单元（在上面的案例中 $m=2$），那么：

+   输入：$x \in R$

+   隐藏单元：$h \in R^m$

+   输入单元的权重：$w_x \in R^m$

+   隐藏单元的权重：$w_h \in R^{mxm}$

+   隐藏单元的偏置：$b_h \in R^m$

+   密集层的权重：$w_y \in R^m$

+   密集层的偏置：$b_y \in R$

查看上述权重。注意：由于权重是随机初始化的，所以这里展示的结果可能与您的结果不同。重要的是要了解每个使用的对象的结构及其如何与其他对象交互以产生最终输出。

Python

```py
wx = demo_model.get_weights()[0]
wh = demo_model.get_weights()[1]
bh = demo_model.get_weights()[2]
wy = demo_model.get_weights()[3]
by = demo_model.get_weights()[4]

print('wx = ', wx, ' wh = ', wh, ' bh = ', bh, ' wy =', wy, 'by = ', by)
```

输出

```py
wx =  [[ 0.18662322 -1.2369459 ]]  wh =  [[ 0.86981213 -0.49338293]
 [ 0.49338293  0.8698122 ]]  bh =  [0\. 0.]  wy = [[-0.4635998]
 [ 0.6538409]] by =  [0.]
```

现在让我们进行一个简单的实验，看看 SimpleRNN 和 Dense 层如何生成输出。保持这个图形在视野中。

![循环神经网络的层次结构](https://machinelearningmastery.com/wp-content/uploads/2021/09/rnnCode1.png)

循环神经网络的层次结构

我们将输入 `x` 三个时间步，并让网络生成一个输出。计算时间步 1、2 和 3 的隐藏单元的值。$h_0$ 初始化为零向量。输出 $o_3$ 是从 $h_3$ 和 $w_y$ 计算的。由于我们使用线性单元，不需要激活函数。

Python

```py
x = np.array([1, 2, 3])
# Reshape the input to the required sample_size x time_steps x features 
x_input = np.reshape(x,(1, 3, 1))
y_pred_model = demo_model.predict(x_input)

m = 2
h0 = np.zeros(m)
h1 = np.dot(x[0], wx) + h0 + bh
h2 = np.dot(x[1], wx) + np.dot(h1,wh) + bh
h3 = np.dot(x[2], wx) + np.dot(h2,wh) + bh
o3 = np.dot(h3, wy) + by

print('h1 = ', h1,'h2 = ', h2,'h3 = ', h3)

print("Prediction from network ", y_pred_model)
print("Prediction from our computation ", o3)
```

输出

```py
h1 =  [[ 0.18662322 -1.23694587]] h2 =  [[-0.07471441 -3.64187904]] h3 =  [[-1.30195881 -6.84172557]]
Prediction from network  [[-3.8698118]]
Prediction from our computation  [[-3.86981216]]
```

## 在 Sunspots 数据集上运行 RNN

现在我们理解了 SimpleRNN 和 Dense 层是如何组合在一起的。让我们在一个简单的时间序列数据集上运行一个完整的 RNN。我们需要按照以下步骤进行：

1.  从给定的 URL 读取数据集

1.  将数据分割为训练集和测试集

1.  为 Keras 格式准备输入数据

1.  创建一个 RNN 模型并对其进行训练。

1.  对训练集和测试集进行预测，并打印两个集合上的均方根误差。

1.  查看结果

### 步骤 1、2：读取数据并分割为训练集和测试集

下面的函数从给定的 URL 读取训练和测试数据，并将其分割成给定百分比的训练和测试数据。它使用 scikit-learn 中的 `MinMaxScaler` 将数据缩放到 0 到 1 之间，并返回训练和测试数据的单维数组。

Python

```py
# Parameter split_percent defines the ratio of training examples
def get_train_test(url, split_percent=0.8):
    df = read_csv(url, usecols=[1], engine='python')
    data = np.array(df.values.astype('float32'))
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data).flatten()
    n = len(data)
    # Point for splitting data into train and test
    split = int(n*split_percent)
    train_data = data[range(split)]
    test_data = data[split:]
    return train_data, test_data, data

sunspots_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv'
train_data, test_data, data = get_train_test(sunspots_url)
```

### 步骤 3：为 Keras 调整数据形状

下一步是为 Keras 模型训练准备数据。输入数组应该被整形为：`total_samples x time_steps x features`。

有许多种方法可以准备时间序列数据进行训练。我们将创建具有非重叠时间步长的输入行。下图显示了时间步长 = 2 的示例。这里，时间步长表示用于预测时间序列数据下一个值的先前时间步数。

![如何为太阳黑子示例准备数据](https://machinelearningmastery.com/wp-content/uploads/2021/09/rnnCode2.png)

太阳黑子示例的数据准备方法

以下函数 `get_XY()` 以一维数组为输入，将其转换为所需的输入 `X` 和目标 `Y` 数组。我们将使用 12 个 `time_steps` 作为太阳黑子数据集的时间步长，因为太阳黑子的周期通常为 12 个月。你可以尝试其他 `time_steps` 的值。

Python

```py
# Prepare the input X and target Y
def get_XY(dat, time_steps):
    # Indices of target array
    Y_ind = np.arange(time_steps, len(dat), time_steps)
    Y = dat[Y_ind]
    # Prepare X
    rows_x = len(Y)
    X = dat[range(time_steps*rows_x)]
    X = np.reshape(X, (rows_x, time_steps, 1))    
    return X, Y

time_steps = 12
trainX, trainY = get_XY(train_data, time_steps)
testX, testY = get_XY(test_data, time_steps)
```

### 第 4 步：创建 RNN 模型并训练

对于此步骤，你可以重用之前定义的 `create_RNN()` 函数。

Python

```py
model = create_RNN(hidden_units=3, dense_units=1, input_shape=(time_steps,1), 
                   activation=['tanh', 'tanh'])
model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)
```

### 第 5 步：计算并打印均方根误差

函数 `print_error()` 计算实际值与预测值之间的均方误差。

Python

```py
def print_error(trainY, testY, train_predict, test_predict):    
    # Error of predictions
    train_rmse = math.sqrt(mean_squared_error(trainY, train_predict))
    test_rmse = math.sqrt(mean_squared_error(testY, test_predict))
    # Print RMSE
    print('Train RMSE: %.3f RMSE' % (train_rmse))
    print('Test RMSE: %.3f RMSE' % (test_rmse))    

# make predictions
train_predict = model.predict(trainX)
test_predict = model.predict(testX)
# Mean square error
print_error(trainY, testY, train_predict, test_predict)
```

输出

```py
Train RMSE: 0.058 RMSE
Test RMSE: 0.077 RMSE
```

### 第 6 步：查看结果

以下函数绘制了实际目标值和预测值。红色的线条将训练数据和测试数据点分开。

Python

```py
# Plot the result
def plot_result(trainY, testY, train_predict, test_predict):
    actual = np.append(trainY, testY)
    predictions = np.append(train_predict, test_predict)
    rows = len(actual)
    plt.figure(figsize=(15, 6), dpi=80)
    plt.plot(range(rows), actual)
    plt.plot(range(rows), predictions)
    plt.axvline(x=len(trainY), color='r')
    plt.legend(['Actual', 'Predictions'])
    plt.xlabel('Observation number after given time steps')
    plt.ylabel('Sunspots scaled')
    plt.title('Actual and Predicted Values. The Red Line Separates The Training And Test Examples')
plot_result(trainY, testY, train_predict, test_predict)
```

生成了以下图表：

![](https://machinelearningmastery.com/wp-content/uploads/2021/09/rnnCode3.png)

## 综合代码

下面是本教程的全部代码。尝试在你的环境中运行这些代码，并实验不同的隐藏单元和时间步长。你可以在网络中添加第二个 `SimpleRNN` 观察其表现。你也可以使用 `scaler` 对象将数据重新缩放到正常范围。

Python

```py
# Parameter split_percent defines the ratio of training examples
def get_train_test(url, split_percent=0.8):
    df = read_csv(url, usecols=[1], engine='python')
    data = np.array(df.values.astype('float32'))
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data).flatten()
    n = len(data)
    # Point for splitting data into train and test
    split = int(n*split_percent)
    train_data = data[range(split)]
    test_data = data[split:]
    return train_data, test_data, data

# Prepare the input X and target Y
def get_XY(dat, time_steps):
    Y_ind = np.arange(time_steps, len(dat), time_steps)
    Y = dat[Y_ind]
    rows_x = len(Y)
    X = dat[range(time_steps*rows_x)]
    X = np.reshape(X, (rows_x, time_steps, 1))    
    return X, Y

def create_RNN(hidden_units, dense_units, input_shape, activation):
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))
    model.add(Dense(units=dense_units, activation=activation[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def print_error(trainY, testY, train_predict, test_predict):    
    # Error of predictions
    train_rmse = math.sqrt(mean_squared_error(trainY, train_predict))
    test_rmse = math.sqrt(mean_squared_error(testY, test_predict))
    # Print RMSE
    print('Train RMSE: %.3f RMSE' % (train_rmse))
    print('Test RMSE: %.3f RMSE' % (test_rmse))    

# Plot the result
def plot_result(trainY, testY, train_predict, test_predict):
    actual = np.append(trainY, testY)
    predictions = np.append(train_predict, test_predict)
    rows = len(actual)
    plt.figure(figsize=(15, 6), dpi=80)
    plt.plot(range(rows), actual)
    plt.plot(range(rows), predictions)
    plt.axvline(x=len(trainY), color='r')
    plt.legend(['Actual', 'Predictions'])
    plt.xlabel('Observation number after given time steps')
    plt.ylabel('Sunspots scaled')
    plt.title('Actual and Predicted Values. The Red Line Separates The Training And Test Examples')

sunspots_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv'
time_steps = 12
train_data, test_data, data = get_train_test(sunspots_url)
trainX, trainY = get_XY(train_data, time_steps)
testX, testY = get_XY(test_data, time_steps)

# Create model and train
model = create_RNN(hidden_units=3, dense_units=1, input_shape=(time_steps,1), 
                   activation=['tanh', 'tanh'])
model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)

# make predictions
train_predict = model.predict(trainX)
test_predict = model.predict(testX)

# Print error
print_error(trainY, testY, train_predict, test_predict)

#Plot result
plot_result(trainY, testY, train_predict, test_predict)
```

## 进一步阅读

本节提供了更多相关资源，如果你想深入了解。

### 书籍

+   [深度学习基础](https://www.amazon.com/Deep-Learning-Essentials-hands-fundamentals/dp/1785880365) 由 Wei Di、Anurag Bhardwaj 和 Jianing Wei 编著。

+   [深度学习](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618/ref=as_li_ss_tl?dchild=1&keywords=deep+learning&qid=1606171954&s=books&sr=1-1&linkCode=sl1&tag=inspiredalgor-20&linkId=0a0c58945768a65548b639df6d1a98ed&language=en_US) 由 Ian Goodfellow、Joshua Bengio 和 Aaron Courville 编著。

### 文章

+   [关于 BPTT 的维基百科文章](https://en.wikipedia.org/wiki/Backpropagation_through_time)

+   [递归神经网络算法深度学习巡礼](https://machinelearningmastery.com/recurrent-neural-network-algorithms-for-deep-learning/)

+   [时间反向传播的温和介绍](https://machinelearningmastery.com/gentle-introduction-backpropagation-time/)

+   [如何为长短期记忆网络准备单变量时间序列数据](https://machinelearningmastery.com/prepare-univariate-time-series-data-long-short-term-memory-networks/)

## 总结

在本教程中，你发现了递归神经网络及其各种架构。

具体来说，你学到了：

+   RNN 的结构

+   RNN 如何从先前的输入中计算输出

+   如何使用 RNN 实现时间序列预测的端到端系统

对于本文中讨论的 RNNs，你有任何问题吗？请在下方评论中提出你的问题，我会尽力回答。
