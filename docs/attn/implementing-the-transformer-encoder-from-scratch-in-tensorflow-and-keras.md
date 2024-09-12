# 在 TensorFlow 和 Keras 中从头开始实现 Transformer 编码器

> 原文：[`machinelearningmastery.com/implementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras/`](https://machinelearningmastery.com/implementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras/)

在看完如何实现 [缩放点积注意力](https://machinelearningmastery.com/how-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras) 并将其集成到 Transformer 模型的 [多头注意力](https://machinelearningmastery.com/how-to-implement-multi-head-attention-from-scratch-in-tensorflow-and-keras) 后，让我们进一步实现完整的 Transformer 模型，通过应用其编码器来达到我们的最终目标，即将该模型应用于自然语言处理（NLP）。

在本教程中，您将学习如何在 TensorFlow 和 Keras 中从头开始实现 Transformer 编码器。

完成本教程后，您将了解：

+   组成 Transformer 编码器的层。

+   如何从头开始实现 Transformer 编码器。

用我的书 [使用注意力构建 Transformer 模型](https://machinelearningmastery.com/transformer-models-with-attention/) 来**启动您的项目**。它提供了**自学教程**和**可工作的代码**，帮助您构建一个完全工作的 Transformer 模型，能够

*将句子从一种语言翻译到另一种语言*...

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2022/03/encoder_cover-scaled.jpg)

在 TensorFlow 和 Keras 中从头开始实现 Transformer 编码器

照片由 [ian dooley](https://unsplash.com/photos/DuBNA1QMpPA) 提供，部分权利保留。

## **教程概述**

本教程分为三个部分；它们是：

+   Transformer 架构总结

    +   Transformer 编码器

+   从头开始实现 Transformer 编码器

    +   全连接前馈神经网络和层归一化

    +   编码器层

    +   Transformer 编码器

+   测试代码

## **先决条件**

对于本教程，我们假设您已经熟悉以下内容：

+   [Transformer 模型](https://machinelearningmastery.com/the-transformer-model/)

+   [缩放点积注意力](https://machinelearningmastery.com/how-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras)

+   [多头注意力](https://machinelearningmastery.com/how-to-implement-multi-head-attention-from-scratch-in-tensorflow-and-keras)

+   [Transformer 位置编码](https://machinelearningmastery.com/the-transformer-positional-encoding-layer-in-keras-part-2/)

## **Transformer 架构总结**

[回顾](https://machinelearningmastery.com/the-transformer-model/) 已经看到 Transformer 架构遵循编码器-解码器结构。左侧的编码器负责将输入序列映射到连续表示的序列；右侧的解码器接收编码器的输出以及前一个时间步的解码器输出以生成输出序列。

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png)

Transformer 架构的编码器-解码器结构

摘自 “[Attention Is All You Need](https://arxiv.org/abs/1706.03762)“

在生成输出序列时，Transformer 不依赖于递归和卷积。

你已经看到 Transformer 的解码器部分在其架构上与编码器有许多相似之处。在本教程中，你将重点关注组成 Transformer 编码器的组件。

### **Transformer 编码器**

Transformer 编码器由 $N$ 个相同的层堆叠而成，每层进一步包含两个主要子层：

+   第一个子层包括一个多头注意力机制，该机制将查询、键和值作为输入。

+   第二个子层包括一个全连接前馈网络。

![](https://machinelearningmastery.com/wp-content/uploads/2021/10/transformer_1.png)

Transformer 架构的编码器模块

摘自 “[Attention Is All You Need](https://arxiv.org/abs/1706.03762)“

每个这些两个子层后面都有层归一化，其中将子层输入（通过残差连接）和输出送入。每一步层归一化的输出如下：

LayerNorm（子层输入 + 子层输出）

为了方便这种操作——涉及子层输入和输出之间的加法，Vaswani 等人设计了模型中的所有子层和嵌入层以产生维度为 $d_{\text{model}}$ = 512 的输出。

另外，[回顾](https://machinelearningmastery.com/how-to-implement-multi-head-attention-from-scratch-in-tensorflow-and-keras) 将查询、键和值作为 Transformer 编码器的输入。

在这里，查询、键和值在经过嵌入和位置编码增强后，携带相同的输入序列，其中查询和键的维度为 $d_k$，而值的维度为 $d_v$。

此外，Vaswani 等人还通过在每个子层的输出（在层归一化步骤之前）以及位置编码输入编码器之前应用 dropout 来引入正则化。

现在，让我们看看如何从头开始在 TensorFlow 和 Keras 中实现 Transformer 编码器。

### 想要开始构建具有注意力机制的 Transformer 模型吗？

现在就可以立即领取我的免费 12 天电子邮件速成课程（包括示例代码）。

点击注册并获得免费的 PDF 电子书版课程。

## **从零开始实现 Transformer 编码器**

### **全连接前馈神经网络和层归一化**

我们首先创建如上图所示的*Feed Forward*和*Add & Norm*层的类。

Vaswani 等人告诉我们，全连接前馈网络由两个线性变换组成，中间夹有一个 ReLU 激活。第一个线性变换产生维度为$d_{ff}$ = 2048 的输出，而第二个线性变换产生维度为$d_{\text{model}}$ = 512 的输出。

为此，我们首先创建一个名为`FeedForward`的类，它继承自 Keras 中的`Layer`基类，并初始化稠密层和 ReLU 激活：

Python

```py
class FeedForward(Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.fully_connected1 = Dense(d_ff)  # First fully connected layer
        self.fully_connected2 = Dense(d_model)  # Second fully connected layer
        self.activation = ReLU()  # ReLU activation layer
        ...
```

我们将向其中添加一个类方法`call()`，它接收一个输入，并通过两个具有 ReLU 激活的全连接层，返回一个维度为 512 的输出：

Python

```py
...
def call(self, x):
    # The input is passed into the two fully-connected layers, with a ReLU in between
    x_fc1 = self.fully_connected1(x)

    return self.fully_connected2(self.activation(x_fc1))
```

下一步是创建另一个类`AddNormalization`，它同样继承自 Keras 中的`Layer`基类，并初始化一个层归一化层：

Python

```py
class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = LayerNormalization()  # Layer normalization layer
        ...
```

在其中，包含以下类方法，它将其子层的输入和输出进行求和，然后对结果应用层归一化：

Python

```py
...
def call(self, x, sublayer_x):
    # The sublayer input and output need to be of the same shape to be summed
    add = x + sublayer_x

    # Apply layer normalization to the sum
    return self.layer_norm(add)
```

### **编码器层**

接下来，你将实现编码器层，Transformer 编码器将完全复制这个层$N$次。

为此，我们首先创建一个名为`EncoderLayer`的类，并初始化它所包含的所有子层：

Python

```py
class EncoderLayer(Layer):
    def __init__(self, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()
        ...
```

在这里，你可能会注意到你已经初始化了之前创建的`FeedForward`和`AddNormalization`类的实例，并将它们的输出分配给各自的变量`feed_forward`和`add_norm`（1 和 2）。`Dropout`层是不言自明的，其中`rate`定义了输入单元被设为 0 的频率。你在[上一篇教程](https://machinelearningmastery.com/how-to-implement-multi-head-attention-from-scratch-in-tensorflow-and-keras)中创建了`MultiHeadAttention`类，如果你将代码保存到了一个单独的 Python 脚本中，请不要忘记`import`它。我将我的代码保存到名为*multihead_attention.py*的 Python 脚本中，因此我需要包括代码行*from multihead_attention import MultiHeadAttention.*。

现在让我们继续创建实现所有编码器子层的类方法`call()`：

Python

```py
...
def call(self, x, padding_mask, training):
    # Multi-head attention layer
    multihead_output = self.multihead_attention(x, x, x, padding_mask)
    # Expected output shape = (batch_size, sequence_length, d_model)

    # Add in a dropout layer
    multihead_output = self.dropout1(multihead_output, training=training)

    # Followed by an Add & Norm layer
    addnorm_output = self.add_norm1(x, multihead_output)
    # Expected output shape = (batch_size, sequence_length, d_model)

    # Followed by a fully connected layer
    feedforward_output = self.feed_forward(addnorm_output)
    # Expected output shape = (batch_size, sequence_length, d_model)

    # Add in another dropout layer
    feedforward_output = self.dropout2(feedforward_output, training=training)

    # Followed by another Add & Norm layer
    return self.add_norm2(addnorm_output, feedforward_output)
```

除了输入数据之外，`call()`方法还可以接收填充掩码。作为之前教程中提到的简要提醒，*填充*掩码是必要的，以抑制输入序列中的零填充与实际输入值一起处理。

同一个类方法可以接收一个`training`标志，当设置为`True`时，仅在训练期间应用 Dropout 层。

### **Transformer 编码器**

最后一步是创建一个名为`Encoder`的 Transformer 编码器类：

Python

```py
class Encoder(Layer):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model)
        self.dropout = Dropout(rate)
        self.encoder_layer = [EncoderLayer(h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]
        ...
```

Transformer 编码器在此之后会接收一个经过单词嵌入和位置编码处理的输入序列。为了计算位置编码，让我们使用 Mehreen Saeed 在[本教程](https://machinelearningmastery.com/the-transformer-positional-encoding-layer-in-keras-part-2/)中描述的`PositionEmbeddingFixedWeights`类。

就像您在前面的部分中所做的那样，在这里，您还将创建一个名为`call()`的类方法，该方法将单词嵌入和位置编码应用于输入序列，并将结果馈送到$N$个编码器层：

Python

```py
...
def call(self, input_sentence, padding_mask, training):
    # Generate the positional encoding
    pos_encoding_output = self.pos_encoding(input_sentence)
    # Expected output shape = (batch_size, sequence_length, d_model)

    # Add in a dropout layer
    x = self.dropout(pos_encoding_output, training=training)

    # Pass on the positional encoded values to each encoder layer
    for i, layer in enumerate(self.encoder_layer):
        x = layer(x, padding_mask, training)

    return x
```

完整的 Transformer 编码器的代码清单如下：

Python

```py
from tensorflow.keras.layers import LayerNormalization, Layer, Dense, ReLU, Dropout
from multihead_attention import MultiHeadAttention
from positional_encoding import PositionEmbeddingFixedWeights

# Implementing the Add & Norm Layer
class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = LayerNormalization()  # Layer normalization layer

    def call(self, x, sublayer_x):
        # The sublayer input and output need to be of the same shape to be summed
        add = x + sublayer_x

        # Apply layer normalization to the sum
        return self.layer_norm(add)

# Implementing the Feed-Forward Layer
class FeedForward(Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.fully_connected1 = Dense(d_ff)  # First fully connected layer
        self.fully_connected2 = Dense(d_model)  # Second fully connected layer
        self.activation = ReLU()  # ReLU activation layer

    def call(self, x):
        # The input is passed into the two fully-connected layers, with a ReLU in between
        x_fc1 = self.fully_connected1(x)

        return self.fully_connected2(self.activation(x_fc1))

# Implementing the Encoder Layer
class EncoderLayer(Layer):
    def __init__(self, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()

    def call(self, x, padding_mask, training):
        # Multi-head attention layer
        multihead_output = self.multihead_attention(x, x, x, padding_mask)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        multihead_output = self.dropout1(multihead_output, training=training)

        # Followed by an Add & Norm layer
        addnorm_output = self.add_norm1(x, multihead_output)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in another dropout layer
        feedforward_output = self.dropout2(feedforward_output, training=training)

        # Followed by another Add & Norm layer
        return self.add_norm2(addnorm_output, feedforward_output)

# Implementing the Encoder
class Encoder(Layer):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model)
        self.dropout = Dropout(rate)
        self.encoder_layer = [EncoderLayer(h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]

    def call(self, input_sentence, padding_mask, training):
        # Generate the positional encoding
        pos_encoding_output = self.pos_encoding(input_sentence)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        x = self.dropout(pos_encoding_output, training=training)

        # Pass on the positional encoded values to each encoder layer
        for i, layer in enumerate(self.encoder_layer):
            x = layer(x, padding_mask, training)

        return x
```

## **测试代码**

您将使用 Vaswani 等人（2017 年）在论文[注意力机制全是你需要的](https://arxiv.org/abs/1706.03762)中指定的参数值进行工作：

Python

```py
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_ff = 2048  # Dimensionality of the inner fully connected layer
d_model = 512  # Dimensionality of the model sub-layers' outputs
n = 6  # Number of layers in the encoder stack

batch_size = 64  # Batch size from the training process
dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers
...
```

至于输入序列，暂时您将使用虚拟数据，直到在单独的教程中[训练完整的 Transformer 模型](https://machinelearningmastery.com/training-the-transformer-model)时，您将使用实际句子：

Python

```py
...
enc_vocab_size = 20 # Vocabulary size for the encoder
input_seq_length = 5  # Maximum length of the input sequence

input_seq = random.random((batch_size, input_seq_length))
...
```

接下来，您将创建`Encoder`类的一个新实例，将其输出分配给`encoder`变量，随后输入参数，并打印结果。暂时将填充掩码参数设置为`None`，但在实现完整的 Transformer 模型时会回到这里：

Python

```py
...
encoder = Encoder(enc_vocab_size, input_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
print(encoder(input_seq, None, True))
```

将所有内容联系在一起得到以下代码清单：

Python

```py
from numpy import random

enc_vocab_size = 20 # Vocabulary size for the encoder
input_seq_length = 5  # Maximum length of the input sequence
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_ff = 2048  # Dimensionality of the inner fully connected layer
d_model = 512  # Dimensionality of the model sub-layers' outputs
n = 6  # Number of layers in the encoder stack

batch_size = 64  # Batch size from the training process
dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers

input_seq = random.random((batch_size, input_seq_length))

encoder = Encoder(enc_vocab_size, input_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
print(encoder(input_seq, None, True))
```

运行此代码会生成形状为（*批量大小*，*序列长度*，*模型维度*）的输出。请注意，由于输入序列的随机初始化和密集层参数值的不同，您可能会看到不同的输出。

Python

```py
tf.Tensor(
[[[-0.4214715  -1.1246173  -0.8444572  ...  1.6388322  -0.1890367
    1.0173352 ]
  [ 0.21662089 -0.61147404 -1.0946581  ...  1.4627445  -0.6000164
   -0.64127874]
  [ 0.46674493 -1.4155326  -0.5686513  ...  1.1790234  -0.94788337
    0.1331717 ]
  [-0.30638126 -1.9047263  -1.8556844  ...  0.9130118  -0.47863355
    0.00976158]
  [-0.22600567 -0.9702025  -0.91090447 ...  1.7457147  -0.139926
   -0.07021569]]
...

 [[-0.48047638 -1.1034104  -0.16164204 ...  1.5588069   0.08743562
   -0.08847156]
  [-0.61683714 -0.8403657  -1.0450369  ...  2.3587787  -0.76091915
   -0.02891812]
  [-0.34268388 -0.65042275 -0.6715749  ...  2.8530657  -0.33631966
    0.5215888 ]
  [-0.6288677  -1.0030932  -0.9749813  ...  2.1386387   0.0640307
   -0.69504136]
  [-1.33254    -1.2524267  -0.230098   ...  2.515467   -0.04207756
   -0.3395423 ]]], shape=(64, 5, 512), dtype=float32)
```

## **进一步阅读**

如果您希望深入了解此主题，本节提供了更多资源。

### **书籍**

+   [Python 高级深度学习](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X)，2019

+   [自然语言处理的 Transformer](https://www.amazon.com/Transformers-Natural-Language-Processing-architectures/dp/1800565798)，2021

### **论文**

+   [注意力机制全是你需要的](https://arxiv.org/abs/1706.03762)，2017

## **总结**

在本教程中，你学会了如何从零开始在 TensorFlow 和 Keras 中实现 Transformer 编码器。

具体来说，你学到了：

+   形成 Transformer 编码器的一部分的层

+   如何从零开始实现 Transformer 编码器

你有什么问题吗？

在下面的评论中提出你的问题，我会尽力回答。
