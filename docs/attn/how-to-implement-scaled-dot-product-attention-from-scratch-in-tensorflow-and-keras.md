# 如何在 TensorFlow 和 Keras 中从零开始实现缩放点积注意力

> 原文：[`machinelearningmastery.com/how-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras/`](https://machinelearningmastery.com/how-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras/)

在熟悉了 [Transformer 模型](https://machinelearningmastery.com/the-transformer-model/) 及其 [注意力机制](https://machinelearningmastery.com/the-transformer-attention-mechanism/) 的理论之后，我们将开始实现一个完整的 Transformer 模型，首先了解如何实现缩放点积注意力。缩放点积注意力是多头注意力的核心部分，而多头注意力又是 Transformer 编码器和解码器的重要组件。我们的最终目标是将完整的 Transformer 模型应用于自然语言处理（NLP）。

在本教程中，你将学习如何在 TensorFlow 和 Keras 中从零开始实现缩放点积注意力。

完成本教程后，你将知道：

+   构成缩放点积注意力机制的一部分操作

+   如何从零开始实现缩放点积注意力机制

**启动你的项目**，请阅读我的书籍 [构建带有注意力的 Transformer 模型](https://machinelearningmastery.com/transformer-models-with-attention/)。它提供了**自学教程**和**实用代码**，指导你构建一个完全工作的 Transformer 模型。

*如何将句子从一种语言翻译成另一种语言*...

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2022/03/dotproduct_cover.jpg)

如何在 TensorFlow 和 Keras 中从零开始实现缩放点积注意力

图片来源：[谢尔盖·施密特](https://unsplash.com/photos/koy6FlCCy5s)，版权所有。

## **教程概述**

本教程分为三个部分；它们是：

+   Transformer 架构回顾

    +   Transformer 缩放点积注意力

+   从零开始实现缩放点积注意力

+   代码测试

## **前提条件**

对于本教程，我们假设你已经熟悉：

+   [注意力的概念](https://machinelearningmastery.com/what-is-attention/)

+   [注意力机制](https://machinelearningmastery.com/the-attention-mechanism-from-scratch/)

+   [Transformer 注意力机制](https://machinelearningmastery.com/the-transformer-attention-mechanism)

+   [Transformer 模型](https://machinelearningmastery.com/the-transformer-model/)

## **Transformer 架构回顾**

[回忆起](https://machinelearningmastery.com/the-transformer-model/) 见过 Transformer 架构遵循编码器-解码器结构。编码器位于左侧，负责将输入序列映射到一系列连续表示；解码器位于右侧，接收编码器的输出以及前一时间步的解码器输出，生成输出序列。

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png)

Transformer 架构的编码器-解码器结构

取自 “[注意力机制是你所需要的](https://arxiv.org/abs/1706.03762)”

在生成输出序列时，Transformer 不依赖于递归和卷积。

您已经看到 Transformer 的解码器部分在其架构中与编码器有许多相似之处。在它们的多头注意力块内，编码器和解码器共享的核心组件之一是*缩放点积注意力*。

### **Transformer 缩放点积注意力机制**

首先，[回想一下](https://machinelearningmastery.com/the-transformer-attention-mechanism/) 查询（queries）、键（keys）和值（values）作为你将要处理的重要组件。

在编码器阶段，它们在嵌入并通过位置信息增强之后携带相同的输入序列。类似地，在解码器侧，进入第一个注意力块的查询、键和值代表同样经过嵌入和通过位置信息增强的目标序列。解码器的第二个注意力块接收编码器输出作为键和值，并接收第一个注意力块的归一化输出作为查询。查询和键的维度由 $d_k$ 表示，而值的维度由 $d_v$ 表示。

缩放点积注意力将这些查询、键和值作为输入，并首先计算查询与键的点积。然后结果被 $d_k$ 的平方根缩放，生成注意力分数。然后将它们输入 softmax 函数，得到一组注意力权重。最后，注意力权重通过加权乘法操作来缩放值。整个过程可以用数学方式解释如下，其中 $\mathbf{Q}$、$\mathbf{K}$ 和 $\mathbf{V}$ 分别表示查询、键和值：

$$\text{attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax} \left( \frac{\mathbf{Q} \mathbf{K}^\mathsf{T}}{\sqrt{d_k}} \right) \mathbf{V}$$

Transformer 模型中的每个多头注意力块实现了如下所示的缩放点积注意力操作：

![](https://machinelearningmastery.com/wp-content/uploads/2022/03/dotproduct_1.png)

缩放点积注意力和多头注意力

取自“[注意力机制是你所需要的一切](https://arxiv.org/abs/1706.03762)”

您可能注意到，在将注意力分数输入到 softmax 函数之前，缩放点积注意力也可以应用一个掩码。

由于单词嵌入被零填充到特定的序列长度，需要引入一个*填充掩码*，以防止零令牌与编码器和解码器阶段的输入一起处理。此外，还需要一个*前瞻掩码*，以防止解码器关注后续单词，从而特定单词的预测只能依赖于其前面已知的单词输出。

这些前瞻和填充掩码应用于缩放点积注意力中，将输入到 softmax 函数中的所有值设置为-$\infty$，这些值不应考虑。对于每个这些大负输入，softmax 函数将产生一个接近零的输出值，有效地屏蔽它们。当你进入单独的教程实现[编码器](https://machinelearningmastery.com/implementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras)和[解码器](https://machinelearningmastery.com/implementing-the-transformer-decoder-from-scratch-in-tensorflow-and-keras)块时，这些掩码的用途将变得更加清晰。

暂时先看看如何在 TensorFlow 和 Keras 中从零开始实现缩放点积注意力。

### 想要开始使用注意力机制构建 Transformer 模型吗？

现在就免费获取我为期 12 天的电子邮件快速课程（带有示例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## **从零开始实现缩放点积注意力**

为此，您将创建一个名为`DotProductAttention`的类，该类继承自 Keras 中的`Layer`基类。

在其中，您将创建类方法`call()`，该方法接受查询、键和值作为输入参数，还有维度$d_k$和一个掩码（默认为`None`）：

Python

```py
class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)

    def call(self, queries, keys, values, d_k, mask=None):
        ...
```

第一步是在查询和键之间执行点积运算，然后转置后者。结果将通过除以$d_k$的平方根进行缩放。您将在`call()`类方法中添加以下代码行：

Python

```py
...
scores = matmul(queries, keys, transpose_b=True) / sqrt(d_k)
...
```

接下来，您将检查`mask`参数是否已设置为非默认值`None`。

掩码将包含`0`值，表示应在计算中考虑输入序列中的相应标记，或者`1`表示相反。掩码将乘以-1e9 以将`1`值设置为大负数（请记住在前一节中提到过这一点），然后应用于注意力分数：

Python

```py
...
if mask is not None:
    scores += -1e9 * mask
...
```

然后，注意力分数将通过 softmax 函数传递以生成注意力权重：

Python

```py
...
weights = softmax(scores)
...
```

最后一步是通过另一个点积操作用计算出的注意力权重加权值：

Python

```py
...
return matmul(weights, values)
```

完整的代码列表如下：

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

## **测试代码**

你将使用论文中指定的参数值，[Attention Is All You Need](https://arxiv.org/abs/1706.03762)，由 Vaswani 等人（2017 年）：

Python

```py
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
batch_size = 64  # Batch size from the training process
...
```

至于序列长度和查询、键、值，你将暂时使用虚拟数据，直到你在另一个教程中进入[训练完整 Transformer 模型](https://machinelearningmastery.com/training-the-transformer-model)的阶段，那时你将使用实际句子。同样，对于掩码，暂时将其保持为默认值：

Python

```py
...
input_seq_length = 5  # Maximum length of the input sequence

queries = random.random((batch_size, input_seq_length, d_k))
keys = random.random((batch_size, input_seq_length, d_k))
values = random.random((batch_size, input_seq_length, d_v))
...
```

在完整的 Transformer 模型中，序列长度及查询、键、值的值将通过词语标记化和嵌入过程获得。你将在另一个教程中覆盖这些内容。

回到测试过程，下一步是创建`DotProductAttention`类的新实例，将其输出分配给`attention`变量：

Python

```py
...
attention = DotProductAttention()
...
```

由于`DotProductAttention`类继承自`Layer`基类，前者的`call()`方法将由后者的魔术`__call()__`方法自动调用。最后一步是输入参数并打印结果：

Python

```py
...
print(attention(queries, keys, values, d_k))
```

将一切结合起来产生以下代码列表：

Python

```py
from numpy import random

input_seq_length = 5  # Maximum length of the input sequence
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
batch_size = 64  # Batch size from the training process

queries = random.random((batch_size, input_seq_length, d_k))
keys = random.random((batch_size, input_seq_length, d_k))
values = random.random((batch_size, input_seq_length, d_v))

attention = DotProductAttention()
print(attention(queries, keys, values, d_k))
```

运行此代码会产生一个形状为 (*batch size*, *sequence length*, *values dimensionality*) 的输出。请注意，由于查询、键和值的随机初始化，你可能会看到不同的输出。

Python

```py
tf.Tensor(
[[[0.60413814 0.52436507 0.46551135 ... 0.5260341  0.33879933 0.43999898]
  [0.60433316 0.52383804 0.465411   ... 0.5262608  0.33915892 0.43782598]
  [0.62321603 0.5349194  0.46824688 ... 0.531323   0.34432083 0.43554053]
  [0.60013235 0.54162943 0.47391182 ... 0.53600514 0.33722004 0.4192218 ]
  [0.6295709  0.53511244 0.46552944 ... 0.5317217  0.3462567  0.43129003]]
 ...

[[0.20291057 0.18463902 0.641182   ... 0.4706118  0.4194418  0.39908117]
  [0.19932748 0.18717204 0.64831126 ... 0.48373622 0.3995132  0.37968236]
  [0.20611541 0.18079443 0.6374859  ... 0.48258874 0.41704425 0.4016996 ]
  [0.19703123 0.18210654 0.6400498  ... 0.47037745 0.4257752  0.3962079 ]
  [0.19237372 0.18474475 0.64944196 ... 0.49497223 0.38804317 0.36352912]]], 
shape=(64, 5, 64), dtype=float32)
```

## **进一步阅读**

本节提供了更多资源，如果你想深入了解这个话题。

### **书籍**

+   [深入学习 Python](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X)，2019 年

+   [自然语言处理中的 Transformer](https://www.amazon.com/Transformers-Natural-Language-Processing-architectures/dp/1800565798)，2021 年

### **论文**

+   [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，2017 年

## **总结**

在本教程中，你学习了如何在 TensorFlow 和 Keras 中从头实现缩放点积注意力机制。

具体来说，你学到了：

+   组成缩放点积注意力机制的一部分操作

+   如何从头实现缩放点积注意力机制

你有任何问题吗？

在下面的评论中提问，我会尽力回答。
