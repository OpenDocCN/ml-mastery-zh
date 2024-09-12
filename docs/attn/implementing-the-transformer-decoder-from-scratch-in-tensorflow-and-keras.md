# 在 TensorFlow 和 Keras 中从零开始实现 Transformer 解码器

> 原文：[`machinelearningmastery.com/implementing-the-transformer-decoder-from-scratch-in-tensorflow-and-keras/`](https://machinelearningmastery.com/implementing-the-transformer-decoder-from-scratch-in-tensorflow-and-keras/)

Transformer 编码器和解码器之间存在许多相似之处，例如它们实现了多头注意力机制、层归一化以及作为最终子层的全连接前馈网络。在实现了[Transformer 编码器](https://machinelearningmastery.com/implementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras)之后，我们现在将继续应用我们的知识来实现 Transformer 解码器，作为实现完整 Transformer 模型的进一步步骤。您的最终目标是将完整模型应用于自然语言处理（NLP）。

在本教程中，您将学习如何在 TensorFlow 和 Keras 中从零开始实现 Transformer 解码器。

完成本教程后，您将了解：

+   构成 Transformer 解码器的层

+   如何从零开始实现 Transformer 解码器

使用我的书籍[使用注意力构建 Transformer 模型](https://machinelearningmastery.com/transformer-models-with-attention/)**启动您的项目**。它提供了具有**工作代码**的**自学教程**，引导您构建一个完全工作的 Transformer 模型，可以

*将一种语言的句子翻译成另一种语言*...

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2022/03/decoder_cover-scaled.jpg)

在 TensorFlow 和 Keras 中从零开始实现 Transformer 解码器

照片由 [François Kaiser](https://unsplash.com/photos/8Ceyil3gIog) 拍摄，部分权利保留。

## **教程概述**

本教程分为三个部分，它们是：

+   Transformer 架构回顾

    +   Transformer 解码器

+   在 TensorFlow 和 Keras 中从零开始实现 Transformer 解码器

    +   解码器层

    +   Transformer 解码器

+   测试代码

## **先决条件**

本教程假设您已经熟悉以下内容：

+   [Transformer 模型](https://machinelearningmastery.com/the-transformer-model/)

+   [缩放点积注意力机制](https://machinelearningmastery.com/?p=13364&preview=true)

+   [多头注意力机制](https://machinelearningmastery.com/?p=13351&preview=true)

+   [Transformer 位置编码](https://machinelearningmastery.com/the-transformer-positional-encoding-layer-in-keras-part-2/)

+   [Transformer 编码器](https://machinelearningmastery.com/?p=13389&preview=true)

## **Transformer 架构回顾**

[回忆](https://machinelearningmastery.com/the-transformer-model/)已经看到，Transformer 架构遵循编码器-解码器结构。编码器在左侧负责将输入序列映射到连续表示的序列；解码器在右侧接收编码器输出以及前一时间步的解码器输出，生成输出序列。

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png)

Transformer 架构的编码器-解码器结构

取自“[注意力机制全是你需要的](https://arxiv.org/abs/1706.03762)“

在生成输出序列时，Transformer 不依赖于循环和卷积。

您已经看到 Transformer 的解码器部分在架构上与编码器有许多相似之处。本教程将探索这些相似之处。

## **Transformer 解码器**

类似于[Transformer 编码器](https://machinelearningmastery.com/implementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras)，Transformer 解码器也由 $N$ 个相同层的堆叠组成。然而，Transformer 解码器还实现了一个额外的多头注意力块，总共有三个主要子层：

+   第一子层包括一个多头注意力机制，接收查询（queries）、键（keys）和值（values）作为输入。

+   第二子层包括第二个多头注意力机制。

+   第三子层包括一个全连接的前馈网络。

![](https://machinelearningmastery.com/wp-content/uploads/2021/10/transformer_2.png)

Transformer 架构的解码器块

取自“[注意力机制全是你需要的](https://arxiv.org/abs/1706.03762)“

这三个子层中的每一个后面都跟着层归一化，层归一化步骤的输入是其对应的子层输入（通过残差连接）和输出。

在解码器端，进入第一个多头注意力块的查询、键和值也代表相同的输入序列。然而，这一次是将*目标*序列嵌入并增强了位置信息，然后才提供给解码器。另一方面，第二个多头注意力块接收编码器输出作为键和值，并接收第一个解码器注意力块的归一化输出作为查询。在这两种情况下，查询和键的维度保持等于$d_k$，而值的维度保持等于$d_v$。

Vaswani 等人还通过对每个子层的输出（在层归一化步骤之前）以及传入解码器的位置编码应用 dropout 来在解码器端引入正则化。

现在让我们来看一下如何从头开始在 TensorFlow 和 Keras 中实现 Transformer 解码器。

### 想开始构建带有注意力机制的 Transformer 模型吗？

立即参加我的免费 12 天电子邮件速成课程（包括示例代码）。

点击注册并获取免费的 PDF 电子书版本课程。

## **从头开始实现 Transformer 解码器**

### **解码器层**

由于你在[实现 Transformer 编码器](https://machinelearningmastery.com/implementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras)时已经实现了所需的子层，因此你将创建一个解码器层类，直接利用这些子层：

Python

```py
from multihead_attention import MultiHeadAttention
from encoder import AddNormalization, FeedForward

class DecoderLayer(Layer):
    def __init__(self, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.multihead_attention1 = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.multihead_attention2 = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout3 = Dropout(rate)
        self.add_norm3 = AddNormalization()
        ...
```

请注意，由于不同子层的代码已经保存到多个 Python 脚本（即 *multihead_attention.py* 和 *encoder.py*）中，因此需要导入它们才能使用所需的类。

正如你在 Transformer 编码器中所做的那样，你现在将创建 `call()` 类方法，来实现所有解码器子层：

Python

```py
...
def call(self, x, encoder_output, lookahead_mask, padding_mask, training):
    # Multi-head attention layer
    multihead_output1 = self.multihead_attention1(x, x, x, lookahead_mask)
    # Expected output shape = (batch_size, sequence_length, d_model)

    # Add in a dropout layer
    multihead_output1 = self.dropout1(multihead_output1, training=training)

    # Followed by an Add & Norm layer
    addnorm_output1 = self.add_norm1(x, multihead_output1)
    # Expected output shape = (batch_size, sequence_length, d_model)

    # Followed by another multi-head attention layer
    multihead_output2 = self.multihead_attention2(addnorm_output1, encoder_output, encoder_output, padding_mask)

    # Add in another dropout layer
    multihead_output2 = self.dropout2(multihead_output2, training=training)

    # Followed by another Add & Norm layer
    addnorm_output2 = self.add_norm1(addnorm_output1, multihead_output2)

    # Followed by a fully connected layer
    feedforward_output = self.feed_forward(addnorm_output2)
    # Expected output shape = (batch_size, sequence_length, d_model)

    # Add in another dropout layer
    feedforward_output = self.dropout3(feedforward_output, training=training)

    # Followed by another Add & Norm layer
    return self.add_norm3(addnorm_output2, feedforward_output)
```

多头注意力子层还可以接收填充掩码或前瞻掩码。简要提醒一下在[之前的教程](https://machinelearningmastery.com/how-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras)中提到的内容，*填充*掩码是必要的，以防止输入序列中的零填充被处理与实际输入值一起处理。*前瞻*掩码防止解码器关注后续单词，这样对特定单词的预测只能依赖于前面单词的已知输出。

相同的 `call()` 类方法也可以接收一个 `training` 标志，以便仅在训练期间应用 Dropout 层，当标志的值设置为 `True` 时。

### **Transformer 解码器**

Transformer 解码器将你刚刚实现的解码器层复制 $N$ 次。

你将创建以下 `Decoder()` 类来实现 Transformer 解码器：

Python

```py
from positional_encoding import PositionEmbeddingFixedWeights

class Decoder(Layer):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model)
        self.dropout = Dropout(rate)
        self.decoder_layer = [DecoderLayer(h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)
        ...
```

与 Transformer 编码器一样，解码器侧第一个多头注意力块的输入接收经过词嵌入和位置编码处理后的输入序列。为此，初始化一个 `PositionEmbeddingFixedWeights` 类的实例（在[这个教程](https://machinelearningmastery.com/the-transformer-positional-encoding-layer-in-keras-part-2/)中介绍），并将其输出分配给 `pos_encoding` 变量。

最后一步是创建一个类方法 `call()`，该方法将词嵌入和位置编码应用于输入序列，并将结果与编码器输出一起馈送到 $N$ 个解码器层：

Python

```py
...
def call(self, output_target, encoder_output, lookahead_mask, padding_mask, training):
    # Generate the positional encoding
    pos_encoding_output = self.pos_encoding(output_target)
    # Expected output shape = (number of sentences, sequence_length, d_model)

    # Add in a dropout layer
    x = self.dropout(pos_encoding_output, training=training)

    # Pass on the positional encoded values to each encoder layer
    for i, layer in enumerate(self.decoder_layer):
        x = layer(x, encoder_output, lookahead_mask, padding_mask, training)

    return x
```

完整 Transformer 解码器的代码清单如下：

Python

```py
from tensorflow.keras.layers import Layer, Dropout
from multihead_attention import MultiHeadAttention
from positional_encoding import PositionEmbeddingFixedWeights
from encoder import AddNormalization, FeedForward

# Implementing the Decoder Layer
class DecoderLayer(Layer):
    def __init__(self, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.multihead_attention1 = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.multihead_attention2 = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout3 = Dropout(rate)
        self.add_norm3 = AddNormalization()

    def call(self, x, encoder_output, lookahead_mask, padding_mask, training):
        # Multi-head attention layer
        multihead_output1 = self.multihead_attention1(x, x, x, lookahead_mask)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        multihead_output1 = self.dropout1(multihead_output1, training=training)

        # Followed by an Add & Norm layer
        addnorm_output1 = self.add_norm1(x, multihead_output1)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Followed by another multi-head attention layer
        multihead_output2 = self.multihead_attention2(addnorm_output1, encoder_output, encoder_output, padding_mask)

        # Add in another dropout layer
        multihead_output2 = self.dropout2(multihead_output2, training=training)

        # Followed by another Add & Norm layer
        addnorm_output2 = self.add_norm1(addnorm_output1, multihead_output2)

        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output2)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in another dropout layer
        feedforward_output = self.dropout3(feedforward_output, training=training)

        # Followed by another Add & Norm layer
        return self.add_norm3(addnorm_output2, feedforward_output)

# Implementing the Decoder
class Decoder(Layer):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model)
        self.dropout = Dropout(rate)
        self.decoder_layer = [DecoderLayer(h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]

    def call(self, output_target, encoder_output, lookahead_mask, padding_mask, training):
        # Generate the positional encoding
        pos_encoding_output = self.pos_encoding(output_target)
        # Expected output shape = (number of sentences, sequence_length, d_model)

        # Add in a dropout layer
        x = self.dropout(pos_encoding_output, training=training)

        # Pass on the positional encoded values to each encoder layer
        for i, layer in enumerate(self.decoder_layer):
            x = layer(x, encoder_output, lookahead_mask, padding_mask, training)

        return x
```

## **测试代码**

您将使用文献《[Attention Is All You Need](https://arxiv.org/abs/1706.03762)》（Vaswani 等人，2017 年）中指定的参数值：

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

至于输入序列，暂时您将使用虚拟数据，直到您在单独的教程中[训练完整的 Transformer 模型](https://machinelearningmastery.com/training-the-transformer-model)，届时您将使用实际的句子：

Python

```py
...
dec_vocab_size = 20 # Vocabulary size for the decoder
input_seq_length = 5  # Maximum length of the input sequence

input_seq = random.random((batch_size, input_seq_length))
enc_output = random.random((batch_size, input_seq_length, d_model))
...
```

接下来，您将创建`Decoder`类的新实例，将其输出分配给`decoder`变量，随后传入输入参数并打印结果。目前，您将把填充和前瞻掩码设置为`None`，但在实现完整的 Transformer 模型时将返回到这些设置：

Python

```py
...
decoder = Decoder(dec_vocab_size, input_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
print(decoder(input_seq, enc_output, None, True)
```

将所有内容综合起来，得到以下代码清单：

Python

```py
from numpy import random

dec_vocab_size = 20  # Vocabulary size for the decoder
input_seq_length = 5  # Maximum length of the input sequence
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_ff = 2048  # Dimensionality of the inner fully connected layer
d_model = 512  # Dimensionality of the model sub-layers' outputs
n = 6  # Number of layers in the decoder stack

batch_size = 64  # Batch size from the training process
dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers

input_seq = random.random((batch_size, input_seq_length))
enc_output = random.random((batch_size, input_seq_length, d_model))

decoder = Decoder(dec_vocab_size, input_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
print(decoder(input_seq, enc_output, None, True))
```

运行此代码会生成形状为（*批大小*，*序列长度*，*模型维度*）的输出。请注意，由于输入序列的随机初始化和密集层参数值的不同，您可能会看到不同的输出。

Python

```py
tf.Tensor(
[[[-0.04132953 -1.7236308   0.5391184  ... -0.76394725  1.4969798
    0.37682498]
  [ 0.05501875 -1.7523409   0.58404493 ... -0.70776534  1.4498456
    0.32555297]
  [ 0.04983566 -1.8431275   0.55850077 ... -0.68202156  1.4222856
    0.32104644]
  [-0.05684051 -1.8862512   0.4771412  ... -0.7101341   1.431343
    0.39346313]
  [-0.15625843 -1.7992781   0.40803364 ... -0.75190556  1.4602519
    0.53546077]]
...

 [[-0.58847624 -1.646842    0.5973466  ... -0.47778523  1.2060764
    0.34091905]
  [-0.48688865 -1.6809179   0.6493542  ... -0.41274604  1.188649
    0.27100053]
  [-0.49568555 -1.8002801   0.61536175 ... -0.38540334  1.2023914
    0.24383534]
  [-0.59913146 -1.8598882   0.5098136  ... -0.3984461   1.2115746
    0.3186561 ]
  [-0.71045107 -1.7778647   0.43008155 ... -0.42037937  1.2255307
    0.47380894]]], shape=(64, 5, 512), dtype=float32)
```

## **进一步阅读**

本节提供了更多有关该主题的资源，如果您希望深入了解。

### **图书**

+   [Python 深度学习进阶](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X), 2019

+   [自然语言处理的 Transformer](https://www.amazon.com/Transformers-Natural-Language-Processing-architectures/dp/1800565798), 2021

### **论文**

+   [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017

## **总结**

在本教程中，您学习了如何在 TensorFlow 和 Keras 中从头开始实现 Transformer 解码器。

具体而言，您学习了：

+   组成 Transformer 解码器的层

+   如何从头开始实现 Transformer 解码器

您有什么问题吗？

在下面的评论中提出您的问题，我将尽力回答。
