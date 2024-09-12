# 如何在 TensorFlow 和 Keras 中从头实现多头注意力机制

> 原文：[`machinelearningmastery.com/how-to-implement-multi-head-attention-from-scratch-in-tensorflow-and-keras/`](https://machinelearningmastery.com/how-to-implement-multi-head-attention-from-scratch-in-tensorflow-and-keras/)

我们已经熟悉了 [Transformer 模型](https://machinelearningmastery.com/the-transformer-model/) 及其 [注意力机制](https://machinelearningmastery.com/the-transformer-attention-mechanism/) 的理论。我们已经开始了实现完整模型的旅程，学习如何 [实现缩放点积注意力](https://machinelearningmastery.com/how-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras)。现在，我们将进一步将缩放点积注意力封装成多头注意力机制，这是核心组成部分。我们的最终目标是将完整模型应用于自然语言处理（NLP）。

在本教程中，您将了解如何在 TensorFlow 和 Keras 中从头实现多头注意力机制。

完成本教程后，您将了解：

+   形成多头注意力机制的层。

+   如何从头实现多头注意力机制。

**启动您的项目**，使用我的书籍 [使用注意力构建 Transformer 模型](https://machinelearningmastery.com/transformer-models-with-attention/)。它提供了 **自学教程** 和 **工作代码**，指导您构建一个完全工作的 Transformer 模型，可用于

*将句子从一种语言翻译成另一种语言*...

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2022/03/multihead_cover-scaled.jpg)

如何在 TensorFlow 和 Keras 中从头实现多头注意力机制

照片由 [Everaldo Coelho](https://unsplash.com/photos/YfldCpQuKt4) 拍摄，部分权利保留。

## **教程概述**

本教程分为三个部分；它们分别是：

+   Transformer 架构回顾

    +   Transformer 多头注意力

+   从头实现多头注意力

+   测试代码

## **先决条件**

对于本教程，我们假设您已经熟悉：

+   [注意力的概念](https://machinelearningmastery.com/what-is-attention/)

+   [Transformer 注意力机制](https://machinelearningmastery.com/the-transformer-attention-mechanism)

+   [Transformer 模型](https://machinelearningmastery.com/the-transformer-model/)

+   [缩放点积注意力](https://machinelearningmastery.com/how-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras)

## **Transformer 架构回顾**

[回顾](https://machinelearningmastery.com/the-transformer-model/)你已经看到 Transformer 架构遵循编码器-解码器结构。左侧的编码器负责将输入序列映射到连续表示序列；右侧的解码器接收编码器的输出以及前一个时间步骤的解码器输出，以生成输出序列。

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png)

Transformer 架构的编码器-解码器结构

摘自“[Attention Is All You Need](https://arxiv.org/abs/1706.03762)”

在生成输出序列时，Transformer 不依赖递归和卷积。

你已经看到，Transformer 的解码器部分在架构上与编码器有许多相似之处。编码器和解码器共同拥有的核心机制之一是*多头注意力*机制。

### **Transformer 多头注意力**

每个多头注意力块由四个连续的层组成：

+   在第一层，三个线性（稠密）层分别接收查询、键或值。

+   在第二层，一个缩放点积注意力函数。第一层和第二层执行的操作会根据组成多头注意力块的头数重复执行*h*次，并且并行进行。

+   在第三层，一个连接操作将不同头部的输出连接起来。

+   在第四层，一个最终的线性（稠密）层生成输出。

![](https://machinelearningmastery.com/wp-content/uploads/2021/09/tour_4.png)

多头注意力

摘自“[Attention Is All You Need](https://arxiv.org/abs/1706.03762)”

[回顾](https://machinelearningmastery.com/the-transformer-attention-mechanism/)一下将作为多头注意力实现构建块的重要组件：

+   **查询**、**键**和**值**：这些是每个多头注意力块的输入。在编码器阶段，它们携带相同的输入序列，该序列在经过嵌入和位置编码信息增强后，作为输入提供。同样，在解码器端，输入到第一个注意力块的查询、键和值代表了经过嵌入和位置编码信息增强后的相同目标序列。解码器的第二个注意力块接收来自编码器的输出，形式为键和值，并且将第一个解码器注意力块的归一化输出作为查询。查询和键的维度由$d_k$表示，而值的维度由$d_v$表示。

+   **投影矩阵**：当应用于查询、键和值时，这些投影矩阵会生成每个的不同子空间表示。每个注意力*头*然后对这些查询、键和值的投影版本中的一个进行处理。另一个投影矩阵也会应用于多头注意力块的输出，在每个单独的头的输出被连接在一起之后。投影矩阵在训练过程中学习得到。

现在我们来看看如何在 TensorFlow 和 Keras 中从零实现多头注意力。

**从零实现多头注意力**

我们从创建`MultiHeadAttention`类开始，它继承自 Keras 中的`Layer`基类，并初始化一些你将使用的实例属性（属性描述可以在注释中找到）：

Python

```py
class MultiHeadAttention(Layer):
    def __init__(self, h, d_k, d_v, d_model, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.attention = DotProductAttention()  # Scaled dot product attention 
        self.heads = h  # Number of attention heads to use
        self.d_k = d_k  # Dimensionality of the linearly projected queries and keys
        self.d_v = d_v  # Dimensionality of the linearly projected values
        self.W_q = Dense(d_k)  # Learned projection matrix for the queries
        self.W_k = Dense(d_k)  # Learned projection matrix for the keys
        self.W_v = Dense(d_v)  # Learned projection matrix for the values
        self.W_o = Dense(d_model)  # Learned projection matrix for the multi-head output
        ...
```

注意到之前实现的`DotProductAttention`类的一个实例已经被创建，并且它的输出被分配给了变量`attention`。[回顾](https://machinelearningmastery.com/how-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras)你是这样实现`DotProductAttention`类的：

Python

```py
from tensorflow import matmul, math, cast, float32
from tensorflow.keras.layers import Layer
from keras.backend import softmax

# Implementing the Scaled-Dot Product Attention
class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)

    def call(self, queries, keys, values, d_k, mask=None):
        # Scoring the queries against the keys after transposing the latter, and scaling
        scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))

        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask

        # Computing the weights by a softmax operation
        weights = softmax(scores)

        # Computing the attention by a weighted sum of the value vectors
        return matmul(weights, values)
```

接下来，你将重新调整*线性投影*后的查询、键和值，以便能够并行计算注意力头。

查询、键和值将作为输入传入多头注意力块，其形状为（*batch size*，*sequence length*，*model dimensionality*），其中*batch size*是训练过程中的一个超参数，*sequence length*定义了输入/输出短语的最大长度，*model dimensionality*是模型所有子层生成的输出的维度。然后，它们会通过各自的密集层，线性投影到（*batch size*，*sequence length*，*queries*/*keys*/*values dimensionality*）的形状。

线性投影后的查询、键和值将被重新排列为（*batch size*，*number of heads*，*sequence length*，*depth*），首先将它们重塑为（*batch size*，*sequence length*，*number of heads*，*depth*），然后转置第二和第三维。为此，你将创建类方法`reshape_tensor`，如下所示：

Python

```py
def reshape_tensor(self, x, heads, flag):
    if flag:
        # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
        x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
        x = transpose(x, perm=(0, 2, 1, 3))
    else:
        # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_model)
        x = transpose(x, perm=(0, 2, 1, 3))
        x = reshape(x, shape=(shape(x)[0], shape(x)[1], -1))
    return x
```

`reshape_tensor`方法接收线性投影后的查询、键或值作为输入（同时将标志设置为`True`）以进行如前所述的重新排列。一旦生成了多头注意力输出，它也会被传入相同的函数（这次将标志设置为`False`）以执行反向操作，从而有效地将所有头的结果连接在一起。

因此，下一步是将线性投影后的查询、键和值输入到 `reshape_tensor` 方法中进行重排，然后将它们输入到缩放点积注意力函数中。为此，让我们创建另一个类方法 `call`，如下所示：

Python

```py
def call(self, queries, keys, values, mask=None):
    # Rearrange the queries to be able to compute all heads in parallel
    q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
    # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

    # Rearrange the keys to be able to compute all heads in parallel
    k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
    # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

    # Rearrange the values to be able to compute all heads in parallel
    v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
    # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

    # Compute the multi-head attention output using the reshaped queries, keys and values
    o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
    # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
    ...
```

请注意，`reshape_tensor` 方法除了接收查询、键和值作为输入外，还可以接收一个掩码（默认值为`None`）。

[回顾](https://machinelearningmastery.com/the-transformer-model/) Transformer 模型引入了一个 *前瞻掩码* 以防止解码器关注后续单词，从而使得对特定单词的预测只能依赖于其前面的已知输出。此外，由于词嵌入被零填充到特定的序列长度，还需要引入一个 *填充掩码* 以防止零值与输入一起被处理。这些前瞻掩码和填充掩码可以通过 `mask` 参数传递给缩放点积注意力。

一旦你从所有注意力头中生成了多头注意力输出，最后的步骤是将所有输出连接成一个形状为（*批大小*，*序列长度*，*值的维度*）的张量，并通过一个最终的全连接层。为此，你将向 `call` 方法添加以下两行代码。

Python

```py
...
# Rearrange back the output into concatenated form
output = self.reshape_tensor(o_reshaped, self.heads, False)
# Resulting tensor shape: (batch_size, input_seq_length, d_v)

# Apply one final linear projection to the output to generate the multi-head attention
# Resulting tensor shape: (batch_size, input_seq_length, d_model)
return self.W_o(output)
```

将所有内容整合在一起，你会得到以下的多头注意力实现：

Python

```py
from tensorflow import math, matmul, reshape, shape, transpose, cast, float32
from tensorflow.keras.layers import Dense, Layer
from keras.backend import softmax

# Implementing the Scaled-Dot Product Attention
class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)

    def call(self, queries, keys, values, d_k, mask=None):
        # Scoring the queries against the keys after transposing the latter, and scaling
        scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))

        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask

        # Computing the weights by a softmax operation
        weights = softmax(scores)

        # Computing the attention by a weighted sum of the value vectors
        return matmul(weights, values)

# Implementing the Multi-Head Attention
class MultiHeadAttention(Layer):
    def __init__(self, h, d_k, d_v, d_model, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.attention = DotProductAttention()  # Scaled dot product attention
        self.heads = h  # Number of attention heads to use
        self.d_k = d_k  # Dimensionality of the linearly projected queries and keys
        self.d_v = d_v  # Dimensionality of the linearly projected values
        self.d_model = d_model  # Dimensionality of the model
        self.W_q = Dense(d_k)  # Learned projection matrix for the queries
        self.W_k = Dense(d_k)  # Learned projection matrix for the keys
        self.W_v = Dense(d_v)  # Learned projection matrix for the values
        self.W_o = Dense(d_model)  # Learned projection matrix for the multi-head output

    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_k)
            x = transpose(x, perm=(0, 2, 1, 3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.d_k))
        return x

    def call(self, queries, keys, values, mask=None):
        # Rearrange the queries to be able to compute all heads in parallel
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the keys to be able to compute all heads in parallel
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the values to be able to compute all heads in parallel
        v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Compute the multi-head attention output using the reshaped queries, keys and values
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange back the output into concatenated form
        output = self.reshape_tensor(o_reshaped, self.heads, False)
        # Resulting tensor shape: (batch_size, input_seq_length, d_v)

        # Apply one final linear projection to the output to generate the multi-head attention
        # Resulting tensor shape: (batch_size, input_seq_length, d_model)
        return self.W_o(output)
```

### 想要开始构建具有注意力机制的 Transformer 模型吗？

现在就参加我的免费 12 天电子邮件速成课程（包括示例代码）。

点击注册，并获得课程的免费 PDF 电子书版本。

## **测试代码**

你将使用 Vaswani 等人（2017）在论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 中指定的参数值：

Python

```py
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of the model sub-layers' outputs
batch_size = 64  # Batch size from the training process
...
```

至于序列长度以及查询、键和值，你将暂时使用虚拟数据，直到你到达另一个教程中 [训练完整 Transformer 模型](https://machinelearningmastery.com/training-the-transformer-model) 的阶段，到时你将使用实际句子：

Python

```py
...
input_seq_length = 5  # Maximum length of the input sequence

queries = random.random((batch_size, input_seq_length, d_k))
keys = random.random((batch_size, input_seq_length, d_k))
values = random.random((batch_size, input_seq_length, d_v))
...
```

在完整的 Transformer 模型中，序列长度以及查询、键和值的值将通过词标记化和嵌入过程获得。我们将在另一个教程中覆盖这部分内容。

回到测试过程，下一步是创建 `MultiHeadAttention` 类的新实例，并将其输出赋值给 `multihead_attention` 变量：

Python

```py
...
multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
...
```

由于 `MultiHeadAttention` 类继承自 `Layer` 基类，因此前者的 `call()` 方法将由后者的魔法 `__call()__` 方法自动调用。最后一步是传入输入参数并打印结果：

Python

```py
...
print(multihead_attention(queries, keys, values))
```

将所有内容整合在一起，生成以下代码清单：

Python

```py
from numpy import random

input_seq_length = 5  # Maximum length of the input sequence
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of the model sub-layers' outputs
batch_size = 64  # Batch size from the training process

queries = random.random((batch_size, input_seq_length, d_k))
keys = random.random((batch_size, input_seq_length, d_k))
values = random.random((batch_size, input_seq_length, d_v))

multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
print(multihead_attention(queries, keys, values))
```

运行这段代码将会产生形状为（*批量大小*，*序列长度*，*模型维度*）的输出。请注意，由于查询、键和值的随机初始化以及密集层的参数值，可能会看到不同的输出。

Python

```py
tf.Tensor(
[[[-0.02185373  0.32784638  0.15958631 ... -0.0353895   0.6645204
   -0.2588266 ]
  [-0.02272229  0.32292002  0.16208754 ... -0.03644213  0.66478664
   -0.26139447]
  [-0.01876744  0.32900316  0.16190802 ... -0.03548665  0.6645842
   -0.26155376]
  [-0.02193783  0.32687354  0.15801215 ... -0.03232524  0.6642926
   -0.25795174]
  [-0.02224652  0.32437912  0.1596448  ... -0.0340827   0.6617497
   -0.26065096]]
 ...

 [[ 0.05414441  0.27019292  0.1845745  ...  0.0809482   0.63738805
   -0.34231138]
  [ 0.05546578  0.27191412  0.18483458 ...  0.08379208  0.6366671
   -0.34372014]
  [ 0.05190979  0.27185103  0.18378328 ...  0.08341806  0.63851804
   -0.3422392 ]
  [ 0.05437043  0.27318984  0.18792395 ...  0.08043509  0.6391771
   -0.34357914]
  [ 0.05406848  0.27073097  0.18579456 ...  0.08388947  0.6376929
   -0.34230167]]], shape=(64, 5, 512), dtype=float32)
```

## **进一步阅读**

如果你想深入了解这个主题，本节提供了更多资源。

### **书籍**

+   [Python 深度学习进阶](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X)，2019

+   [自然语言处理中的变形金刚](https://www.amazon.com/Transformers-Natural-Language-Processing-architectures/dp/1800565798)，2021

### **论文**

+   [注意力机制就是一切](https://arxiv.org/abs/1706.03762)，2017

## **总结**

在本教程中，你学会了如何在 TensorFlow 和 Keras 中从头实现多头注意力机制。

具体来说，你学到了：

+   构成多头注意力机制的层

+   如何从头实现多头注意力机制

你有任何问题吗？

在下面的评论中提出你的问题，我会尽力回答。
