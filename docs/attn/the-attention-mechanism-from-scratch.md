# 从头开始了解注意力机制

> 原文：[`machinelearningmastery.com/the-attention-mechanism-from-scratch/`](https://machinelearningmastery.com/the-attention-mechanism-from-scratch/)

引入注意力机制是为了提高编码器-解码器模型在机器翻译中的性能。注意力机制的想法是允许解码器以灵活的方式利用输入序列中最相关的部分，通过对所有编码的输入向量进行加权组合，其中最相关的向量被赋予最高的权重。

在本教程中，你将发现注意力机制及其实现。

完成本教程后，你将了解：

+   注意力机制如何使用所有编码器隐藏状态的加权和来灵活地将解码器的注意力集中在输入序列中最相关的部分

+   如何将注意力机制推广到信息可能不一定按顺序相关的任务中

+   如何在 Python 中使用 NumPy 和 SciPy 实现通用注意力机制

**启动你的项目**，可以参考我的书籍 [使用注意力构建 Transformer 模型](https://machinelearningmastery.com/transformer-models-with-attention/)。书中提供了**自学教程**和**可运行的代码**，指导你构建一个功能完整的 Transformer 模型

*将句子从一种语言翻译成另一种语言*...

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/09/attention_mechanism_cover-scaled.jpg)

从头开始了解注意力机制

图片由 [Nitish Meena](https://unsplash.com/photos/RbbdzZBKRDY) 提供，部分权利保留。

## **教程概览**

本教程分为三个部分；它们是：

+   注意力机制

+   通用注意力机制

+   使用 NumPy 和 SciPy 的通用注意力机制

## **注意力机制**

注意力机制由 [Bahdanau 等人 (2014)](https://arxiv.org/abs/1409.0473) 引入，以解决使用固定长度编码向量时出现的瓶颈问题，其中解码器对输入提供的信息的访问有限。这在处理长和/或复杂序列时尤为成问题，因为它们的表示维度被强制与较短或较简单序列的维度相同。

[注意事项](https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/) 请注意，Bahdanau 等人的*注意力机制*被分为*对齐分数*、*权重*和*上下文向量*的逐步计算：

1.  **对齐分数**：对齐模型使用编码的隐藏状态$\mathbf{h}_i$和先前的解码器输出$\mathbf{s}_{t-1}$来计算一个分数$e_{t,i}$，该分数表示输入序列的元素与当前位置$t$的当前输出对齐的程度。对齐模型由一个函数$a(.)$表示，该函数可以通过前馈神经网络实现：

$$e_{t,i} = a(\mathbf{s}_{t-1}, \mathbf{h}_i)$$

1.  **权重**：权重，$\alpha_{t,i}$，通过对先前计算的对齐分数应用 softmax 操作来计算：

$$\alpha_{t,i} = \text{softmax}(e_{t,i})$$

1.  **上下文向量**：在每个时间步骤中，唯一的上下文向量$\mathbf{c}_t$被输入到解码器中。它通过对所有$T$个编码器隐藏状态的加权和来计算：

$$\mathbf{c}_t = \sum_{i=1}^T \alpha_{t,i} \mathbf{h}_i$$

Bahdanau 等人实现了一个用于编码器和解码器的 RNN。

然而，注意力机制可以重新公式化为可以应用于任何序列到序列（简称 seq2seq）任务的一般形式，其中信息可能不一定以顺序方式相关。

> *换句话说，数据库不必由不同步骤的隐藏 RNN 状态组成，而可以包含任何类型的信息。*
> 
> – [高级深度学习与 Python](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X)，2019 年。

## **一般注意力机制**

一般注意力机制使用三个主要组件，即*查询*，$\mathbf{Q}$，*键*，$\mathbf{K}$，和*值*，$\mathbf{V}$。

如果你要将这三个组件与 Bahdanau 等人提出的注意力机制进行比较，那么查询将类似于先前的解码器输出，$\mathbf{s}_{t-1}$，而值将类似于编码的输入，$\mathbf{h}_i$。在 Bahdanau 注意力机制中，键和值是相同的向量。

> *在这种情况下，我们可以将向量$\mathbf{s}_{t-1}$视为对键值对数据库执行的查询，其中键是向量，而隐藏状态$\mathbf{h}_i$是值。*
> 
> – [高级深度学习与 Python](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X)，2019 年。

一般注意力机制执行以下计算：

1.  每个查询向量$\mathbf{q} = \mathbf{s}_{t-1}$与键的数据库进行匹配，以计算分数值。此匹配操作计算为特定查询与每个键向量$\mathbf{k}_i$的点积：

$$e_{\mathbf{q},\mathbf{k}_i} = \mathbf{q} \cdot \mathbf{k}_i$$

1.  分数通过 softmax 操作生成权重：

$$\alpha_{\mathbf{q},\mathbf{k}_i} = \text{softmax}(e_{\mathbf{q},\mathbf{k}_i})$$

1.  然后，通过对值向量$\mathbf{v}_{\mathbf{k}_i}$进行加权求和来计算广义注意力，其中每个值向量都与相应的键配对：

$$\text{attention}(\mathbf{q}, \mathbf{K}, \mathbf{V}) = \sum_i \alpha_{\mathbf{q},\mathbf{k}_i} \mathbf{v}_{\mathbf{k}_i}$$

在机器翻译的背景下，输入句子中的每个词都会被分配自己的查询、键和值向量。这些向量是通过将编码器对特定词的表示与训练过程中生成的三种不同权重矩阵相乘而得到的。

实质上，当广义注意力机制接收到一系列词时，它会将序列中某个特定词的查询向量与数据库中的每个键进行评分。通过这样做，它捕捉到所考虑的词与序列中其他词的关系。然后，它根据注意力权重（从评分中计算得出）对值进行缩放，以保持对与查询相关的词的关注。这样，它会为所考虑的词生成注意力输出。

### 想要开始构建带有注意力机制的 Transformer 模型吗？

现在就报名参加我的 12 天免费邮件速成课程（附带示例代码）。

点击报名并获取课程的免费 PDF 电子书版本。

## **使用 NumPy 和 SciPy 的通用注意力机制**

本节将探讨如何使用 Python 中的 NumPy 和 SciPy 库实现通用注意力机制。

为了简单起见，你将首先计算四个词序列中第一个词的注意力。然后，你将对代码进行泛化，以矩阵形式计算所有四个词的注意力输出。

因此，让我们首先定义四个不同词的词嵌入，以计算注意力。在实际操作中，这些词嵌入将由编码器生成；然而，在这个例子中，你将手动定义它们。

```py
# encoder representations of four different words
word_1 = array([1, 0, 0])
word_2 = array([0, 1, 0])
word_3 = array([1, 1, 0])
word_4 = array([0, 0, 1])
```

下一步生成权重矩阵，你最终会将这些矩阵乘以词嵌入，以生成查询、键和值。在这里，你将随机生成这些权重矩阵；然而，在实际操作中，这些权重矩阵将通过训练学习得到。

```py
...
# generating the weight matrices
random.seed(42) # to allow us to reproduce the same attention values
W_Q = random.randint(3, size=(3, 3))
W_K = random.randint(3, size=(3, 3))
W_V = random.randint(3, size=(3, 3))
```

请注意，这些矩阵的行数等于词嵌入的维度（在本例中为三），以便进行矩阵乘法。

随后，通过将每个词嵌入与每个权重矩阵相乘来生成每个词的查询、键和值向量。

```py
...
# generating the queries, keys and values
query_1 = word_1 @ W_Q
key_1 = word_1 @ W_K
value_1 = word_1 @ W_V

query_2 = word_2 @ W_Q
key_2 = word_2 @ W_K
value_2 = word_2 @ W_V

query_3 = word_3 @ W_Q
key_3 = word_3 @ W_K
value_3 = word_3 @ W_V

query_4 = word_4 @ W_Q
key_4 = word_4 @ W_K
value_4 = word_4 @ W_V
```

只考虑第一个词的情况下，下一步是使用点积操作对其查询向量与所有键向量进行评分。

```py
...
# scoring the first query vector against all key vectors
scores = array([dot(query_1, key_1), dot(query_1, key_2), dot(query_1, key_3), dot(query_1, key_4)])
```

评分值随后通过 softmax 操作生成权重。在此之前，通常将评分值除以关键向量维度的平方根（在此案例中为三），以保持梯度稳定。

```py
...
# computing the weights by a softmax operation
weights = softmax(scores / key_1.shape[0] ** 0.5)
```

最后，通过所有四个值向量的加权总和计算注意力输出。

```py
...
# computing the attention by a weighted sum of the value vectors
attention = (weights[0] * value_1) + (weights[1] * value_2) + (weights[2] * value_3) + (weights[3] * value_4)

print(attention)
```

```py
[0.98522025 1.74174051 0.75652026]
```

为了加快处理速度，相同的计算可以以矩阵形式实现，一次生成所有四个词的注意力输出：

```py
from numpy import array
from numpy import random
from numpy import dot
from scipy.special import softmax

# encoder representations of four different words
word_1 = array([1, 0, 0])
word_2 = array([0, 1, 0])
word_3 = array([1, 1, 0])
word_4 = array([0, 0, 1])

# stacking the word embeddings into a single array
words = array([word_1, word_2, word_3, word_4])

# generating the weight matrices
random.seed(42)
W_Q = random.randint(3, size=(3, 3))
W_K = random.randint(3, size=(3, 3))
W_V = random.randint(3, size=(3, 3))

# generating the queries, keys and values
Q = words @ W_Q
K = words @ W_K
V = words @ W_V

# scoring the query vectors against all key vectors
scores = Q @ K.transpose()

# computing the weights by a softmax operation
weights = softmax(scores / K.shape[1] ** 0.5, axis=1)

# computing the attention by a weighted sum of the value vectors
attention = weights @ V

print(attention)
```

```py
[[0.98522025 1.74174051 0.75652026]
 [0.90965265 1.40965265 0.5       ]
 [0.99851226 1.75849334 0.75998108]
 [0.99560386 1.90407309 0.90846923]]
```

## **进一步阅读**

本节提供了更多关于该主题的资源，如果你希望深入了解。

### **书籍**

+   [使用 Python 的高级深度学习](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X)，2019 年。

+   [深度学习要点](https://www.amazon.com/Deep-Learning-Essentials-hands-fundamentals/dp/1785880365)，2018 年。

### **论文**

+   [通过联合学习对齐和翻译的神经机器翻译](https://arxiv.org/abs/1409.0473)，2014 年。

## **总结**

在本教程中，你了解了注意机制及其实现。

具体来说，你学到了：

+   注意机制如何使用所有编码器隐藏状态的加权总和来灵活地将解码器的注意力集中在输入序列中最相关的部分

+   注意机制如何推广到信息不一定按顺序相关的任务

+   如何使用 NumPy 和 SciPy 实现通用注意机制

你有什么问题吗？

在下面的评论中提出你的问题，我会尽力回答。
