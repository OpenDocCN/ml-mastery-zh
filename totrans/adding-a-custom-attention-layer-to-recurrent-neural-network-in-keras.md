# 向 Keras 中的循环神经网络添加自定义注意力层

> 原文：[`machinelearningmastery.com/adding-a-custom-attention-layer-to-recurrent-neural-network-in-keras/`](https://machinelearningmastery.com/adding-a-custom-attention-layer-to-recurrent-neural-network-in-keras/)

过去几年来，深度学习网络已经获得了巨大的流行。"注意力机制"被整合到深度学习网络中以提高其性能。在网络中添加注意力组件已经显示出在机器翻译、图像识别、文本摘要等任务中显著的改进。

本教程展示了如何向使用循环神经网络构建的网络添加自定义注意力层。我们将演示如何使用一个非常简单的数据集进行时间序列预测的端到端应用。本教程旨在帮助任何希望了解如何向深度学习网络添加用户定义层，并利用这个简单示例构建更复杂应用程序的人士。

完成本教程后，您将了解：

+   在 Keras 中创建自定义注意力层需要哪些方法

+   如何在使用 SimpleRNN 构建的网络中加入新层

**使用我的书《使用注意力构建 Transformer 模型》启动您的项目**。它提供了**自学教程**和**完整的工作代码**，指导您构建一个完全工作的 Transformer 模型，可以

*将句子从一种语言翻译成另一种语言*...

让我们开始吧。

![向循环神经网络添加自定义注意力层 <br> 照片由 ](https://machinelearningmastery.com/wp-content/uploads/2021/10/yahya-ehsan-L895sqROaGw-unsplash-scaled.jpg)

在 Keras 中向循环神经网络添加自定义注意力层

照片由 Yahya Ehsan 拍摄，部分权利保留。

## 教程概述

本教程分为三部分；它们是：

+   为时间序列预测准备一个简单的数据集

+   如何使用 SimpleRNN 构建的网络进行时间序列预测

+   向 SimpleRNN 网络添加自定义注意力层

## 先决条件

假设您熟悉以下主题。您可以点击下面的链接进行概览。

+   [什么是注意力？](https://machinelearningmastery.com/what-is-attention/)

+   [从零开始理解注意力机制](https://machinelearningmastery.com/the-attention-mechanism-from-scratch/)

+   [RNN 简介及其数学基础](https://machinelearningmastery.com/an-introduction-to-recurrent-neural-networks-and-the-math-that-powers-them/)

+   [理解 Keras 中的简单循环神经网络](https://machinelearningmastery.com/understanding-simple-recurrent-neural-networks-in-keras/)

## 数据集

本文的重点是了解如何向深度学习网络添加自定义注意力层。为此，让我们以斐波那契数列为例，简单说明一下。斐波那契数列的前 10 个数如下所示：

0, 1, 1, 2, 3, 5, 8, 13, 21, 34, …

给定前‘t’个数，能否让机器准确地重构下一个数？这意味着除了最后两个数外，所有之前的输入都将被丢弃，并对最后两个数执行正确的操作。

在本教程中，您将使用`t`个时间步来构建训练示例，并将`t+1`时刻的值作为目标。例如，如果`t=3`，则训练示例和相应的目标值如下所示：

![](https://machinelearningmastery.com/wp-content/uploads/2021/10/fib.png)

### 想要开始使用注意力构建 Transformer 模型吗？

现在就参加我的免费 12 天电子邮件速成课程（包含示例代码）。

点击此处注册，并免费获得课程的 PDF 电子书版本。

## SimpleRNN 网络

在这一部分，您将编写生成数据集的基本代码，并使用 SimpleRNN 网络来预测斐波那契数列的下一个数字。

### 导入部分

首先，让我们编写导入部分：

```py
from pandas import read_csv
import numpy as np
from keras import Model
from keras.layers import Layer
import keras.backend as K
from keras.layers import Input, Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.metrics import mean_squared_error
```

### 准备数据集

下面的函数生成 n 个斐波那契数列的序列（不包括起始两个值）。如果将`scale_data`设置为 True，则还会使用 scikit-learn 中的 MinMaxScaler 将值缩放到 0 到 1 之间。让我们看看`n=10`时的输出。

```py
def get_fib_seq(n, scale_data=True):
    # Get the Fibonacci sequence
    seq = np.zeros(n)
    fib_n1 = 0.0
    fib_n = 1.0 
    for i in range(n):
            seq[i] = fib_n1 + fib_n
            fib_n1 = fib_n
            fib_n = seq[i] 
    scaler = []
    if scale_data:
        scaler = MinMaxScaler(feature_range=(0, 1))
        seq = np.reshape(seq, (n, 1))
        seq = scaler.fit_transform(seq).flatten()        
    return seq, scaler

fib_seq = get_fib_seq(10, False)[0]
print(fib_seq)
```

```py
[ 1\.  2\.  3\.  5\.  8\. 13\. 21\. 34\. 55\. 89.]
```

接下来，我们需要一个函数`get_fib_XY()`，将序列重新格式化为 Keras 输入层使用的训练示例和目标值。当给定参数`time_steps`时，`get_fib_XY()`将每行数据集构建为具有`time_steps`列的数据。此函数不仅从斐波那契序列构建训练集和测试集，还使用`scale_data`参数将训练示例进行洗牌并重新调整到所需的 TensorFlow 格式，即`total_samples x time_steps x features`。同时，如果`scale_data`设置为`True`，函数还返回一个`scaler`对象，用于将值缩放到 0 到 1 之间。

让我们生成一个小的训练集，看看它的样子。我们设置了`time_steps=3`和`total_fib_numbers=12`，大约 70%的示例用于测试。请注意，训练和测试示例已通过`permutation()`函数进行了洗牌。

```py
def get_fib_XY(total_fib_numbers, time_steps, train_percent, scale_data=True):
    dat, scaler = get_fib_seq(total_fib_numbers, scale_data)    
    Y_ind = np.arange(time_steps, len(dat), 1)
    Y = dat[Y_ind]
    rows_x = len(Y)
    X = dat[0:rows_x]
    for i in range(time_steps-1):
        temp = dat[i+1:rows_x+i+1]
        X = np.column_stack((X, temp))
    # random permutation with fixed seed   
    rand = np.random.RandomState(seed=13)
    idx = rand.permutation(rows_x)
    split = int(train_percent*rows_x)
    train_ind = idx[0:split]
    test_ind = idx[split:]
    trainX = X[train_ind]
    trainY = Y[train_ind]
    testX = X[test_ind]
    testY = Y[test_ind]
    trainX = np.reshape(trainX, (len(trainX), time_steps, 1))    
    testX = np.reshape(testX, (len(testX), time_steps, 1))
    return trainX, trainY, testX, testY, scaler

trainX, trainY, testX, testY, scaler = get_fib_XY(12, 3, 0.7, False)
print('trainX = ', trainX)
print('trainY = ', trainY)
```

```py
trainX =  [[[ 8.]
  [13.]
  [21.]]

 [[ 5.]
  [ 8.]
  [13.]]

 [[ 2.]
  [ 3.]
  [ 5.]]

 [[13.]
  [21.]
  [34.]]

 [[21.]
  [34.]
  [55.]]

 [[34.]
  [55.]
  [89.]]]
trainY =  [ 34\.  21\.   8\.  55\.  89\. 144.]
```

### 设置网络

现在让我们设置一个包含两个层的小型网络。第一层是`SimpleRNN`层，第二层是`Dense`层。以下是模型的摘要。

```py
# Set up parameters
time_steps = 20
hidden_units = 2
epochs = 30

# Create a traditional RNN network
def create_RNN(hidden_units, dense_units, input_shape, activation):
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))
    model.add(Dense(units=dense_units, activation=activation[1]))
    model.compile(loss='mse', optimizer='adam')
    return model

model_RNN = create_RNN(hidden_units=hidden_units, dense_units=1, input_shape=(time_steps,1), 
                   activation=['tanh', 'tanh'])
model_RNN.summary()
```

```py
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
simple_rnn_3 (SimpleRNN)     (None, 2)                 8         
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 3         
=================================================================
Total params: 11
Trainable params: 11
Non-trainable params: 0
```

### 训练网络并评估

接下来的步骤是添加代码以生成数据集、训练网络并评估它。这一次，我们将数据缩放到 0 到 1 之间。由于`scale_data`参数的默认值为`True`，我们不需要传递该参数。

```py
# Generate the dataset
trainX, trainY, testX, testY, scaler  = get_fib_XY(1200, time_steps, 0.7)

model_RNN.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=2)

# Evalute model
train_mse = model_RNN.evaluate(trainX, trainY)
test_mse = model_RNN.evaluate(testX, testY)

# Print error
print("Train set MSE = ", train_mse)
print("Test set MSE = ", test_mse)
```

作为输出，你将看到训练进度以及均方误差的以下值：

```py
Train set MSE =  5.631405292660929e-05
Test set MSE =  2.623497312015388e-05
```

## 在网络中添加自定义注意力层

在 Keras 中，通过子类化`Layer`类很容易创建一个实现注意力的自定义层。Keras 指南列出了通过子类化创建新层的明确步骤。你将在这里使用这些指南。单个层对应的所有权重和偏置由此类封装。你需要编写`__init__`方法，并覆盖以下方法：

+   `build()`: Keras 指南建议在知道输入大小后通过此方法添加权重。此方法“惰性”创建权重。内置函数`add_weight()`可用于添加注意力层的权重和偏置。

+   `call()`: `call()` 方法实现了输入到输出的映射。在训练期间，它应该实现前向传播。

### 注意力层的调用方法

注意力层的`call()`方法必须计算对齐分数、权重和上下文。你可以通过斯特凡尼亚在 [从零开始理解注意力机制](https://machinelearningmastery.com/the-attention-mechanism-from-scratch/) 文章中详细了解这些参数。你将在`call()`方法中实现巴赫达瑙注意力机制。

从 Keras 的 `Layer` 类继承一个层并通过 `add_weights()` 方法添加权重的好处在于权重会自动调整。Keras 对 `call()` 方法的操作/计算进行“逆向工程”，并在训练期间计算梯度。在添加权重时，指定`trainable=True`非常重要。如果需要，你还可以为自定义层添加一个`train_step()`方法并指定自己的权重训练方法。

下面的代码实现了自定义注意力层。

```py
# Add attention layer to the deep learning network
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)

    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context
```

### 带有注意力层的 RNN 网络

现在让我们将一个注意力层添加到之前创建的 RNN 网络中。`create_RNN_with_attention()`函数现在在网络中指定了一个 RNN 层、一个注意力层和一个稠密层。确保在指定 SimpleRNN 时设置`return_sequences=True`，这将返回所有先前时间步的隐藏单元输出。

让我们来看一下带有注意力的模型摘要。

```py
def create_RNN_with_attention(hidden_units, dense_units, input_shape, activation):
    x=Input(shape=input_shape)
    RNN_layer = SimpleRNN(hidden_units, return_sequences=True, activation=activation)(x)
    attention_layer = attention()(RNN_layer)
    outputs=Dense(dense_units, trainable=True, activation=activation)(attention_layer)
    model=Model(x,outputs)
    model.compile(loss='mse', optimizer='adam')    
    return model    

model_attention = create_RNN_with_attention(hidden_units=hidden_units, dense_units=1, 
                                  input_shape=(time_steps,1), activation='tanh')
model_attention.summary()
```

```py
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 20, 1)]           0         
_________________________________________________________________
simple_rnn_2 (SimpleRNN)     (None, 20, 2)             8         
_________________________________________________________________
attention_1 (attention)      (None, 2)                 22        
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 3         
=================================================================
Total params: 33
Trainable params: 33
Non-trainable params: 0
_________________________________________________________________
```

### 使用注意力的深度学习网络进行训练和评估

现在是时候训练和测试你的模型，并查看它在预测序列下一个斐波那契数上的表现如何。

```py
model_attention.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=2)

# Evalute model
train_mse_attn = model_attention.evaluate(trainX, trainY)
test_mse_attn = model_attention.evaluate(testX, testY)

# Print error
print("Train set MSE with attention = ", train_mse_attn)
print("Test set MSE with attention = ", test_mse_attn)
```

你将看到训练进度作为输出，以及以下内容：

```py
Train set MSE with attention =  5.3511179430643097e-05
Test set MSE with attention =  9.053358553501312e-06
```

即使对于这个简单的例子，测试集上的均方误差在使用注意力层后更低。通过调优超参数和模型选择，你可以获得更好的结果。尝试在更复杂的问题上以及通过增加网络层来验证这一点。你还可以使用`scaler`对象将数字缩放回原始值。

你可以进一步通过使用 LSTM 代替 SimpleRNN，或者通过卷积和池化层构建网络。如果你愿意，还可以将其改为编码-解码网络。

## 统一的代码

如果你想尝试，本教程的整个代码如下粘贴。请注意，由于此算法的随机性质，你的输出可能与本教程中给出的不同。

```py
from pandas import read_csv
import numpy as np
from keras import Model
from keras.layers import Layer
import keras.backend as K
from keras.layers import Input, Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.metrics import mean_squared_error

# Prepare data
def get_fib_seq(n, scale_data=True):
    # Get the Fibonacci sequence
    seq = np.zeros(n)
    fib_n1 = 0.0
    fib_n = 1.0 
    for i in range(n):
            seq[i] = fib_n1 + fib_n
            fib_n1 = fib_n
            fib_n = seq[i] 
    scaler = []
    if scale_data:
        scaler = MinMaxScaler(feature_range=(0, 1))
        seq = np.reshape(seq, (n, 1))
        seq = scaler.fit_transform(seq).flatten()        
    return seq, scaler

def get_fib_XY(total_fib_numbers, time_steps, train_percent, scale_data=True):
    dat, scaler = get_fib_seq(total_fib_numbers, scale_data)    
    Y_ind = np.arange(time_steps, len(dat), 1)
    Y = dat[Y_ind]
    rows_x = len(Y)
    X = dat[0:rows_x]
    for i in range(time_steps-1):
        temp = dat[i+1:rows_x+i+1]
        X = np.column_stack((X, temp))
    # random permutation with fixed seed   
    rand = np.random.RandomState(seed=13)
    idx = rand.permutation(rows_x)
    split = int(train_percent*rows_x)
    train_ind = idx[0:split]
    test_ind = idx[split:]
    trainX = X[train_ind]
    trainY = Y[train_ind]
    testX = X[test_ind]
    testY = Y[test_ind]
    trainX = np.reshape(trainX, (len(trainX), time_steps, 1))    
    testX = np.reshape(testX, (len(testX), time_steps, 1))
    return trainX, trainY, testX, testY, scaler

# Set up parameters
time_steps = 20
hidden_units = 2
epochs = 30

# Create a traditional RNN network
def create_RNN(hidden_units, dense_units, input_shape, activation):
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))
    model.add(Dense(units=dense_units, activation=activation[1]))
    model.compile(loss='mse', optimizer='adam')
    return model

model_RNN = create_RNN(hidden_units=hidden_units, dense_units=1, input_shape=(time_steps,1), 
                   activation=['tanh', 'tanh'])

# Generate the dataset for the network
trainX, trainY, testX, testY, scaler  = get_fib_XY(1200, time_steps, 0.7)
# Train the network
model_RNN.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=2)

# Evalute model
train_mse = model_RNN.evaluate(trainX, trainY)
test_mse = model_RNN.evaluate(testX, testY)

# Print error
print("Train set MSE = ", train_mse)
print("Test set MSE = ", test_mse)

# Add attention layer to the deep learning network
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)

    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

def create_RNN_with_attention(hidden_units, dense_units, input_shape, activation):
    x=Input(shape=input_shape)
    RNN_layer = SimpleRNN(hidden_units, return_sequences=True, activation=activation)(x)
    attention_layer = attention()(RNN_layer)
    outputs=Dense(dense_units, trainable=True, activation=activation)(attention_layer)
    model=Model(x,outputs)
    model.compile(loss='mse', optimizer='adam')    
    return model    

# Create the model with attention, train and evaluate
model_attention = create_RNN_with_attention(hidden_units=hidden_units, dense_units=1, 
                                  input_shape=(time_steps,1), activation='tanh')
model_attention.summary()    

model_attention.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=2)

# Evalute model
train_mse_attn = model_attention.evaluate(trainX, trainY)
test_mse_attn = model_attention.evaluate(testX, testY)

# Print error
print("Train set MSE with attention = ", train_mse_attn)
print("Test set MSE with attention = ", test_mse_attn)
```

## 进一步阅读

如果你想深入了解，本节提供了更多相关资源。

### 书籍

+   [深度学习基础](https://www.amazon.com/Deep-Learning-Essentials-hands-fundamentals/dp/1785880365) 作者：韦迪，安拉格·巴德瓦吉，和简宁·韦。

+   [深度学习](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618/ref=as_li_ss_tl?dchild=1&keywords=deep+learning&qid=1606171954&s=books&sr=1-1&linkCode=sl1&tag=inspiredalgor-20&linkId=0a0c58945768a65548b639df6d1a98ed&language=en_US) 作者：伊恩·古德费洛、约书亚·本吉奥、和亚伦·库尔维尔。

### 论文

+   [通过联合学习对齐和翻译进行神经机器翻译](https://arxiv.org/abs/1409.0473)，2014 年。

### 文章

+   [深度学习循环神经网络算法导览。](https://machinelearningmastery.com/recurrent-neural-network-algorithms-for-deep-learning/)

+   [什么是注意力机制？](https://machinelearningmastery.com/what-is-attention/)

+   [从零开始的注意力机制。](https://machinelearningmastery.com/the-attention-mechanism-from-scratch/)

+   [介绍 RNN 及其数学原理。](https://machinelearningmastery.com/an-introduction-to-recurrent-neural-networks-and-the-math-that-powers-them/)

+   [理解 Keras 中简单循环神经网络。](https://machinelearningmastery.com/understanding-simple-recurrent-neural-networks-in-keras/)

+   [如何在 Keras 中开发带有注意力的编码-解码模型](https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/)

## 摘要

在本教程中，你学会了如何向使用 Keras 构建的深度学习网络添加自定义注意力层。

具体来说，你学到了：

+   如何重写 Keras 的`Layer`类。

+   方法`build()`用于向注意力层添加权重。

+   方法`call()`用于指定注意力层输入到输出的映射。

+   如何向使用 SimpleRNN 构建的深度学习网络添加自定义注意力层。

你在本文讨论的循环神经网络有任何问题吗？请在下方评论中提问，我会尽力回答。
