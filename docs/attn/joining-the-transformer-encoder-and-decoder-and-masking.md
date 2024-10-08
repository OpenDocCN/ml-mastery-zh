# 结合 Transformer 编码器和解码器及掩码

> 原文：[`machinelearningmastery.com/joining-the-transformer-encoder-and-decoder-and-masking/`](https://machinelearningmastery.com/joining-the-transformer-encoder-and-decoder-and-masking/)

我们已经分别实现并测试了 Transformer [编码器](https://machinelearningmastery.com/implementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras) 和 [解码器](https://machinelearningmastery.com/implementing-the-transformer-decoder-from-scratch-in-tensorflow-and-keras)，现在可以将它们结合成一个完整的模型。我们还将了解如何创建填充和前瞻掩码，以抑制在编码器或解码器计算中不考虑的输入值。我们的最终目标是将完整模型应用于自然语言处理（NLP）。

在本教程中，你将发现如何实现完整的 Transformer 模型并创建填充和前瞻掩码。

完成本教程后，你将了解到：

+   如何为编码器和解码器创建填充掩码

+   如何为解码器创建前瞻掩码

+   如何将 Transformer 编码器和解码器结合成一个模型

+   如何打印出编码器和解码器层的总结

开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2022/04/model_cover-scaled.jpg)

结合 Transformer 编码器和解码器及掩码

照片由 [John O’Nolan](https://unsplash.com/photos/ykeLTANUQyE) 提供，部分权利保留。

## **教程概述**

本教程分为四个部分：

+   Transformer 架构回顾

+   掩码

    +   创建填充掩码

    +   创建前瞻掩码

+   结合 Transformer 编码器和解码器

+   创建 Transformer 模型的实例

    +   打印出编码器和解码器层的总结

## **先决条件**

对于本教程，我们假设你已经熟悉：

+   [Transformer 模型](https://machinelearningmastery.com/the-transformer-model/)

+   [Transformer 编码器](https://machinelearningmastery.com/implementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras)

+   [Transformer 解码器](https://machinelearningmastery.com/implementing-the-transformer-decoder-from-scratch-in-tensorflow-and-keras)

## **Transformer 架构回顾**

[回顾](https://machinelearningmastery.com/the-transformer-model/)我们已经看到 Transformer 架构遵循编码器-解码器结构。左侧的编码器负责将输入序列映射到连续表示序列；右侧的解码器接收编码器的输出以及上一个时间步的解码器输出，以生成输出序列。

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png)

Transformer 架构的编码器-解码器结构

取自“[Attention Is All You Need](https://arxiv.org/abs/1706.03762)”

在生成输出序列时，Transformer 不依赖于递归和卷积。

您已经看到如何分别实现 Transformer 编码器和解码器。在本教程中，您将把两者结合起来，形成一个完整的 Transformer 模型，并在输入值上应用填充和前瞻掩码。

让我们首先了解如何应用掩码。

**用我的书 [使用注意力构建 Transformer 模型](https://machinelearningmastery.com/transformer-models-with-attention/) 启动您的项目**。它提供了**自学教程**和**工作代码**，指导您构建一个完全工作的 Transformer 模型

*将句子从一种语言翻译成另一种语言*…

## **掩码**

### **创建填充掩码**

您应该已经了解在将其馈送到编码器和解码器之前对输入值进行掩码的重要性。

当您继续[训练 Transformer 模型](https://machinelearningmastery.com/training-the-transformer-model)时，将把输入序列馈送到编码器和解码器之前，首先将其零填充到特定的序列长度。填充掩码的重要性在于确保这些零值不会与编码器和解码器同时处理的实际输入值混合在一起。

让我们创建以下函数为编码器和解码器生成填充掩码：

Python

```py
from tensorflow import math, cast, float32

def padding_mask(input):
    # Create mask which marks the zero padding values in the input by a 1
    mask = math.equal(input, 0)
    mask = cast(mask, float32)

    return mask
```

收到输入后，此函数将生成一个张量，标记输入包含零值处的地方为*一*。

因此，如果您输入以下数组：

Python

```py
from numpy import array

input = array([1, 2, 3, 4, 0, 0, 0])
print(padding_mask(input))
```

那么 `padding_mask` 函数的输出将如下所示：

Python

```py
tf.Tensor([0\. 0\. 0\. 0\. 1\. 1\. 1.], shape=(7,), dtype=float32)
```

### **创建前瞻掩码**

需要前瞻掩码以防止解码器关注后续的单词，这样特定单词的预测仅能依赖于其之前的已知输出。

为此，让我们创建以下函数以为解码器生成前瞻掩码：

Python

```py
from tensorflow import linalg, ones

def lookahead_mask(shape):
    # Mask out future entries by marking them with a 1.0
    mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0)

    return mask
```

您将向其传递解码器输入的长度。让我们以 5 为例：

Python

```py
print(lookahead_mask(5))
```

那么 `lookahead_mask` 函数返回的输出如下：

Python

```py
tf.Tensor(
[[0\. 1\. 1\. 1\. 1.]
 [0\. 0\. 1\. 1\. 1.]
 [0\. 0\. 0\. 1\. 1.]
 [0\. 0\. 0\. 0\. 1.]
 [0\. 0\. 0\. 0\. 0.]], shape=(5, 5), dtype=float32)
```

再次，*一* 值掩盖了不应使用的条目。因此，每个单词的预测仅依赖于其之前的单词。

### 想要开始构建使用注意力的 Transformer 模型吗？

现在就注册我的免费 12 天电子邮件速成课程（包含示例代码）。

点击注册，还可免费获取课程的 PDF 电子书版本。

## **连接 Transformer 编码器和解码器**

让我们首先创建`TransformerModel`类，它继承自 Keras 中的`Model`基类：

Python

```py
class TransformerModel(Model):
    def __init__(self, enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate, **kwargs):
        super(TransformerModel, self).__init__(**kwargs)

        # Set up the encoder
        self.encoder = Encoder(enc_vocab_size, enc_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate)

        # Set up the decoder
        self.decoder = Decoder(dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate)

        # Define the final dense layer
        self.model_last_layer = Dense(dec_vocab_size)
        ...
```

创建`TransformerModel`类的第一步是初始化先前实现的`Encoder`和`Decoder`类的实例，并将它们的输出分别分配给变量`encoder`和`decoder`。如果你将这些类保存到单独的 Python 脚本中，不要忘记导入它们。我将代码保存在 Python 脚本*encoder.py*和*decoder.py*中，所以我需要相应地导入它们。

你还将包括一个最终的全连接层，生成最终的输出，类似于[Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)中的 Transformer 架构。

接下来，你将创建类方法`call()`，以将相关输入送入编码器和解码器。

首先生成一个填充掩码，以掩盖编码器输入和编码器输出，当这些被送入解码器的第二个自注意力块时：

Python

```py
...
def call(self, encoder_input, decoder_input, training):

    # Create padding mask to mask the encoder inputs and the encoder outputs in the decoder
    enc_padding_mask = self.padding_mask(encoder_input)
...
```

然后生成一个填充掩码和一个前瞻掩码，以掩盖解码器输入。通过逐元素`maximum`操作将它们结合在一起：

Python

```py
...
# Create and combine padding and look-ahead masks to be fed into the decoder
dec_in_padding_mask = self.padding_mask(decoder_input)
dec_in_lookahead_mask = self.lookahead_mask(decoder_input.shape[1])
dec_in_lookahead_mask = maximum(dec_in_padding_mask, dec_in_lookahead_mask)
...
```

接下来，将相关输入送入编码器和解码器，并通过将解码器输出送入一个最终的全连接层来生成 Transformer 模型输出：

Python

```py
...
# Feed the input into the encoder
encoder_output = self.encoder(encoder_input, enc_padding_mask, training)

# Feed the encoder output into the decoder
decoder_output = self.decoder(decoder_input, encoder_output, dec_in_lookahead_mask, enc_padding_mask, training)

# Pass the decoder output through a final dense layer
model_output = self.model_last_layer(decoder_output)

return model_output
```

将所有步骤结合起来，得到以下完整的代码清单：

Python

```py
from encoder import Encoder
from decoder import Decoder
from tensorflow import math, cast, float32, linalg, ones, maximum, newaxis
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class TransformerModel(Model):
    def __init__(self, enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate, **kwargs):
        super(TransformerModel, self).__init__(**kwargs)

        # Set up the encoder
        self.encoder = Encoder(enc_vocab_size, enc_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate)

        # Set up the decoder
        self.decoder = Decoder(dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate)

        # Define the final dense layer
        self.model_last_layer = Dense(dec_vocab_size)

    def padding_mask(self, input):
        # Create mask which marks the zero padding values in the input by a 1.0
        mask = math.equal(input, 0)
        mask = cast(mask, float32)

        # The shape of the mask should be broadcastable to the shape
        # of the attention weights that it will be masking later on
        return mask[:, newaxis, newaxis, :]

    def lookahead_mask(self, shape):
        # Mask out future entries by marking them with a 1.0
        mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0)

        return mask

    def call(self, encoder_input, decoder_input, training):

        # Create padding mask to mask the encoder inputs and the encoder outputs in the decoder
        enc_padding_mask = self.padding_mask(encoder_input)

        # Create and combine padding and look-ahead masks to be fed into the decoder
        dec_in_padding_mask = self.padding_mask(decoder_input)
        dec_in_lookahead_mask = self.lookahead_mask(decoder_input.shape[1])
        dec_in_lookahead_mask = maximum(dec_in_padding_mask, dec_in_lookahead_mask)

        # Feed the input into the encoder
        encoder_output = self.encoder(encoder_input, enc_padding_mask, training)

        # Feed the encoder output into the decoder
        decoder_output = self.decoder(decoder_input, encoder_output, dec_in_lookahead_mask, enc_padding_mask, training)

        # Pass the decoder output through a final dense layer
        model_output = self.model_last_layer(decoder_output)

        return model_output
```

请注意，你对`padding_mask`函数返回的输出进行了小的更改。它的形状被调整为可广播到它在训练 Transformer 模型时将要掩盖的注意力权重张量的形状。

## **创建 Transformer 模型的实例**

你将使用[Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)论文中指定的参数值：

Python

```py
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_ff = 2048  # Dimensionality of the inner fully connected layer
d_model = 512  # Dimensionality of the model sub-layers' outputs
n = 6  # Number of layers in the encoder stack

dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers
...
```

至于输入相关的参数，你暂时将使用虚拟值，直到你达到[训练完整的 Transformer 模型](https://machinelearningmastery.com/training-the-transformer-model)的阶段。到那时，你将使用实际的句子：

Python

```py
...
enc_vocab_size = 20 # Vocabulary size for the encoder
dec_vocab_size = 20 # Vocabulary size for the decoder

enc_seq_length = 5  # Maximum length of the input sequence
dec_seq_length = 5  # Maximum length of the target sequence
...
```

你现在可以按如下方式创建`TransformerModel`类的实例：

Python

```py
from model import TransformerModel

# Create model
training_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
```

完整的代码清单如下：

Python

```py
enc_vocab_size = 20 # Vocabulary size for the encoder
dec_vocab_size = 20 # Vocabulary size for the decoder

enc_seq_length = 5  # Maximum length of the input sequence
dec_seq_length = 5  # Maximum length of the target sequence

h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_ff = 2048  # Dimensionality of the inner fully connected layer
d_model = 512  # Dimensionality of the model sub-layers' outputs
n = 6  # Number of layers in the encoder stack

dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers

# Create model
training_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
```

### **打印编码器和解码器层的摘要**

你还可以打印出 Transformer 模型的编码器和解码器块的摘要。选择单独打印它们将使你能够查看各个子层的详细信息。为此，将以下代码行添加到`EncoderLayer`和`DecoderLayer`类的`__init__()`方法中：

Python

```py
self.build(input_shape=[None, sequence_length, d_model])
```

然后你需要将以下方法添加到`EncoderLayer`类中：

Python

```py
def build_graph(self):
    input_layer = Input(shape=(self.sequence_length, self.d_model))
    return Model(inputs=[input_layer], outputs=self.call(input_layer, None, True))
```

以及以下方法到`DecoderLayer`类：

Python

```py
def build_graph(self):
    input_layer = Input(shape=(self.sequence_length, self.d_model))
    return Model(inputs=[input_layer], outputs=self.call(input_layer, input_layer, None, None, True))
```

这导致`EncoderLayer`类被修改如下（`call()`方法下的三个点表示与[这里](https://machinelearningmastery.com/implementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras)实现的内容相同）：

Python

```py
from tensorflow.keras.layers import Input
from tensorflow.keras import Model

class EncoderLayer(Layer):
    def __init__(self, sequence_length, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.build(input_shape=[None, sequence_length, d_model])
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()

    def build_graph(self):
        input_layer = Input(shape=(self.sequence_length, self.d_model))
        return Model(inputs=[input_layer], outputs=self.call(input_layer, None, True))

    def call(self, x, padding_mask, training):
        ...
```

类似的更改也可以应用于 `DecoderLayer` 类。

一旦你完成了必要的更改，你可以继续创建 `EncoderLayer` 和 `DecoderLayer` 类的实例，并按如下方式打印它们的总结：

Python

```py
from encoder import EncoderLayer
from decoder import DecoderLayer

encoder = EncoderLayer(enc_seq_length, h, d_k, d_v, d_model, d_ff, dropout_rate)
encoder.build_graph().summary()

decoder = DecoderLayer(dec_seq_length, h, d_k, d_v, d_model, d_ff, dropout_rate)
decoder.build_graph().summary()
```

对编码器的结果总结如下：

Python

```py
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 5, 512)]     0           []                               

 multi_head_attention_18 (Multi  (None, 5, 512)      131776      ['input_1[0][0]',                
 HeadAttention)                                                   'input_1[0][0]',                
                                                                  'input_1[0][0]']                

 dropout_32 (Dropout)           (None, 5, 512)       0           ['multi_head_attention_18[0][0]']

 add_normalization_30 (AddNorma  (None, 5, 512)      1024        ['input_1[0][0]',                
 lization)                                                        'dropout_32[0][0]']             

 feed_forward_12 (FeedForward)  (None, 5, 512)       2099712     ['add_normalization_30[0][0]']   

 dropout_33 (Dropout)           (None, 5, 512)       0           ['feed_forward_12[0][0]']        

 add_normalization_31 (AddNorma  (None, 5, 512)      1024        ['add_normalization_30[0][0]',   
 lization)                                                        'dropout_33[0][0]']             

==================================================================================================
Total params: 2,233,536
Trainable params: 2,233,536
Non-trainable params: 0
__________________________________________________________________________________________________
```

而解码器的结果总结如下：

Python

```py
Model: "model_1"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_2 (InputLayer)           [(None, 5, 512)]     0           []                               

 multi_head_attention_19 (Multi  (None, 5, 512)      131776      ['input_2[0][0]',                
 HeadAttention)                                                   'input_2[0][0]',                
                                                                  'input_2[0][0]']                

 dropout_34 (Dropout)           (None, 5, 512)       0           ['multi_head_attention_19[0][0]']

 add_normalization_32 (AddNorma  (None, 5, 512)      1024        ['input_2[0][0]',                
 lization)                                                        'dropout_34[0][0]',             
                                                                  'add_normalization_32[0][0]',   
                                                                  'dropout_35[0][0]']             

 multi_head_attention_20 (Multi  (None, 5, 512)      131776      ['add_normalization_32[0][0]',   
 HeadAttention)                                                   'input_2[0][0]',                
                                                                  'input_2[0][0]']                

 dropout_35 (Dropout)           (None, 5, 512)       0           ['multi_head_attention_20[0][0]']

 feed_forward_13 (FeedForward)  (None, 5, 512)       2099712     ['add_normalization_32[1][0]']   

 dropout_36 (Dropout)           (None, 5, 512)       0           ['feed_forward_13[0][0]']        

 add_normalization_34 (AddNorma  (None, 5, 512)      1024        ['add_normalization_32[1][0]',   
 lization)                                                        'dropout_36[0][0]']             

==================================================================================================
Total params: 2,365,312
Trainable params: 2,365,312
Non-trainable params: 0
__________________________________________________________________________________________________
```

## **进一步阅读**

本节提供了更多关于该主题的资源，如果你希望深入了解。

### **书籍**

+   [Advanced Deep Learning with Python](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X)，2019

+   [Transformers for Natural Language Processing](https://www.amazon.com/Transformers-Natural-Language-Processing-architectures/dp/1800565798)，2021

### **论文**

+   [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，2017

## **总结**

在本教程中，你学习了如何实现完整的 Transformer 模型以及创建填充和前瞻掩码。

具体来说，你学到了：

+   如何为编码器和解码器创建填充掩码

+   如何为解码器创建前瞻掩码

+   如何将 Transformer 编码器和解码器组合成一个单一模型

+   如何打印出编码器和解码器层的总结

你有任何问题吗？

在下面的评论中提出你的问题，我会尽力回答。
