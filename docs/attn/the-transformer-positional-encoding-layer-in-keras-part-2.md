# Keras 中的变压器位置编码层，第二部分

> 原文：[`machinelearningmastery.com/the-transformer-positional-encoding-layer-in-keras-part-2/`](https://machinelearningmastery.com/the-transformer-positional-encoding-layer-in-keras-part-2/)

在[第一部分：变压器模型中位置编码的温和介绍](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1)中，我们讨论了变压器模型的位置信息编码层。我们还展示了如何在 Python 中自行实现该层及其功能。在本教程中，你将实现 Keras 和 Tensorflow 中的位置编码层。然后，你可以在完整的变压器模型中使用此层。

完成本教程后，你将了解：

+   Keras 中的文本向量化

+   Keras 中的嵌入层

+   如何子类化嵌入层并编写你自己的位置编码层。

**启动你的项目**，请参阅我的书籍[《构建具有注意力机制的变压器模型》](https://machinelearningmastery.com/transformer-models-with-attention/)。它提供了**自学教程**和**可运行的代码**，帮助你构建一个完全可用的变压器模型。

*将句子从一种语言翻译成另一种语言*...

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2022/02/ijaz-rafi-photo-1551102076-9f8bb5f3f897.jpg)

Keras 中的变压器位置编码层，第二部分

照片由 Ijaz Rafi 提供。保留部分权利

## 教程概述

本教程分为三个部分；它们是：

1.  Keras 中的文本向量化和嵌入层

1.  在 Keras 中编写你自己的位置编码层

    1.  随机初始化和可调的嵌入

    1.  来自[《Attention Is All You Need》](https://arxiv.org/abs/1706.03762)的固定权重嵌入

1.  位置信息编码层输出的图形视图

## 导入部分

首先，我们来写一段代码以导入所有必需的库：

```py
import tensorflow as tf
from tensorflow import convert_to_tensor, string
from tensorflow.keras.layers import TextVectorization, Embedding, Layer
from tensorflow.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
```

## 文本向量化层

让我们从一组已经预处理和清理过的英文短语开始。文本向量化层创建一个单词字典，并用字典中对应的索引替换每个单词。让我们看看如何使用文本向量化层来映射这两个句子：

1.  我是一个机器人

1.  你也是机器人

请注意，文本已经被转换为小写，并且所有标点符号和文本中的噪声都已被移除。接下来，将这两个短语转换为固定长度为 5 的向量。Keras 的`TextVectorization`层需要一个最大词汇量和初始化时所需的输出序列长度。该层的输出是一个形状为：

`(句子数量，输出序列长度)`

以下代码片段使用`adapt`方法生成词汇表。接下来，它创建文本的向量化表示。

```py
output_sequence_length = 5
vocab_size = 10
sentences = [["I am a robot"], ["you too robot"]]
sentence_data = Dataset.from_tensor_slices(sentences)
# Create the TextVectorization layer
vectorize_layer = TextVectorization(
                  output_sequence_length=output_sequence_length,
                  max_tokens=vocab_size)
# Train the layer to create a dictionary
vectorize_layer.adapt(sentence_data)
# Convert all sentences to tensors
word_tensors = convert_to_tensor(sentences, dtype=tf.string)
# Use the word tensors to get vectorized phrases
vectorized_words = vectorize_layer(word_tensors)
print("Vocabulary: ", vectorize_layer.get_vocabulary())
print("Vectorized words: ", vectorized_words)
```

输出

```py
Vocabulary:  ['', '[UNK]', 'robot', 'you', 'too', 'i', 'am', 'a']
Vectorized words:  tf.Tensor(
[[5 6 7 2 0]
 [3 4 2 0 0]], shape=(2, 5), dtype=int64)
```

### 想要开始构建具有注意力机制的变压器模型吗？

现在免费参加我的 12 天电子邮件速成课程（附有示例代码）。

单击注册，还可获得课程的免费 PDF 电子书版本。

## 嵌入层

Keras 的`Embedding`层将整数转换为密集向量。此层将这些整数映射到随机数，后者在训练阶段进行调整。但是，您也可以选择将映射设置为一些预定义的权重值（稍后显示）。要初始化此层，您需要指定要映射的整数的最大值，以及输出序列的长度。

### 词嵌入

看看这一层是如何将`vectorized_text`转换为张量的。

```py
output_length = 6
word_embedding_layer = Embedding(vocab_size, output_length)
embedded_words = word_embedding_layer(vectorized_words)
print(embedded_words)
```

输出已经用一些注释进行了标注，如下所示。请注意，每次运行此代码时都会看到不同的输出，因为权重已随机初始化。

![词嵌入。](https://machinelearningmastery.com/wp-content/uploads/2022/02/PEKeras_a.png)

词嵌入。由于涉及到随机数，每次运行代码时，输出都会有所不同。

### 位置嵌入

您还需要相应位置的嵌入。最大位置对应于`TextVectorization`层的输出序列长度。

```py
position_embedding_layer = Embedding(output_sequence_length, output_length)
position_indices = tf.range(output_sequence_length)
embedded_indices = position_embedding_layer(position_indices)
print(embedded_indices)
```

输出如下：![位置索引嵌入。](https://machinelearningmastery.com/wp-content/uploads/2022/02/PEKeras_b.png)

位置索引嵌入

### 变换器中位置编码层的输出

在变换器模型中，最终输出是词嵌入和位置嵌入的总和。因此，当设置这两个嵌入层时，您需要确保`output_length`对两者都是相同的。

```py
final_output_embedding = embedded_words + embedded_indices
print("Final output: ", final_output_embedding)
```

输出如下，带有注释。同样，由于随机权重初始化的原因，这将与您的代码运行结果不同。

![](https://machinelearningmastery.com/wp-content/uploads/2022/02/PEKeras_c.png)

添加了词嵌入和位置嵌入后的最终输出

## 子类化 Keras Embedding 层

当实现变换器模型时，您将不得不编写自己的位置编码层。这相当简单，因为基本功能已为您提供。这个[Keras 示例](https://keras.io/examples/nlp/neural_machine_translation_with_transformer/)展示了如何子类化`Embedding`层以实现自己的功能。您可以根据需要添加更多的方法。

```py
class PositionEmbeddingLayer(Layer):
    def __init__(self, sequence_length, vocab_size, output_dim, **kwargs):
        super(PositionEmbeddingLayer, self).__init__(**kwargs)
        self.word_embedding_layer = Embedding(
            input_dim=vocab_size, output_dim=output_dim
        )
        self.position_embedding_layer = Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )

    def call(self, inputs):        
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_words + embedded_indices
```

让我们运行这一层。

```py
my_embedding_layer = PositionEmbeddingLayer(output_sequence_length,
                                            vocab_size, output_length)
embedded_layer_output = my_embedding_layer(vectorized_words)
print("Output from my_embedded_layer: ", embedded_layer_output)
```

输出

```py
Output from my_embedded_layer:  tf.Tensor(
[[[ 0.06798736 -0.02821309  0.00571618  0.00314623 -0.03060734
    0.01111387]
  [-0.06097465  0.03966043 -0.05164248  0.06578685  0.03638128
   -0.03397174]
  [ 0.06715029 -0.02453769  0.02205854  0.01110986  0.02345785
    0.05879898]
  [-0.04625867  0.07500569 -0.05690887 -0.07615659  0.01962536
    0.00035865]
  [ 0.01423577 -0.03938593 -0.08625181  0.04841495  0.06951572
    0.08811047]]

 [[ 0.0163899   0.06895607 -0.01131684  0.01810524 -0.05857501
    0.01811318]
  [ 0.01915303 -0.0163289  -0.04133433  0.06810946  0.03736673
    0.04218033]
  [ 0.00795418 -0.00143972 -0.01627307 -0.00300788 -0.02759011
    0.09251165]
  [ 0.0028762   0.04526488 -0.05222676 -0.02007698  0.07879823
    0.00541583]
  [ 0.01423577 -0.03938593 -0.08625181  0.04841495  0.06951572
    0.08811047]]], shape=(2, 5, 6), dtype=float32)
```

## 变换器中的位置编码：注意力机制是您所需的

注意，上述类创建了一个具有可训练权重的嵌入层。因此，权重被随机初始化并在训练阶段进行调整。[Attention Is All You Need](https://arxiv.org/abs/1706.03762)的作者指定了一个位置编码方案，如下所示。你可以在本教程的[第一部分](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1)中阅读详细信息：\begin{eqnarray}

P(k, 2i) &=& \sin\Big(\frac{k}{n^{2i/d}}\Big)\\

P(k, 2i+1) &=& \cos\Big(\frac{k}{n^{2i/d}}\Big)

\end{eqnarray}如果你想使用相同的位置编码方案，你可以指定自己的嵌入矩阵，如[第一部分](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1)中讨论的那样，该部分展示了如何在 NumPy 中创建自己的嵌入。当指定`Embedding`层时，你需要提供位置编码矩阵作为权重，并设置`trainable=False`。让我们创建一个新的位置嵌入类来完成这一操作。```py
class PositionEmbeddingFixedWeights(Layer):
    def __init__(self, sequence_length, vocab_size, output_dim, **kwargs):
        super(PositionEmbeddingFixedWeights, self).__init__(**kwargs)
        word_embedding_matrix = self.get_position_encoding(vocab_size, output_dim)   
        position_embedding_matrix = self.get_position_encoding(sequence_length, output_dim)                                          
        self.word_embedding_layer = Embedding(
            input_dim=vocab_size, output_dim=output_dim,
            weights=[word_embedding_matrix],
            trainable=False
        )
        self.position_embedding_layer = Embedding(
            input_dim=sequence_length, output_dim=output_dim,
            weights=[position_embedding_matrix],
            trainable=False
        )

    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P

    def call(self, inputs):        
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_words + embedded_indices
```

接下来，我们设置一切以运行这一层。

```py
attnisallyouneed_embedding = PositionEmbeddingFixedWeights(output_sequence_length,
                                            vocab_size, output_length)
attnisallyouneed_output = attnisallyouneed_embedding(vectorized_words)
print("Output from my_embedded_layer: ", attnisallyouneed_output)
```

输出

```py
Output from my_embedded_layer:  tf.Tensor(
[[[-0.9589243   1.2836622   0.23000172  1.9731903   0.01077196
    1.9999421 ]
  [ 0.56205547  1.5004725   0.3213085   1.9603932   0.01508068
    1.9999142 ]
  [ 1.566284    0.3377554   0.41192317  1.9433732   0.01938933
    1.999877  ]
  [ 1.0504174  -1.4061394   0.2314966   1.9860148   0.01077211
    1.9999698 ]
  [-0.7568025   0.3463564   0.18459873  1.982814    0.00861763
    1.9999628 ]]

 [[ 0.14112     0.0100075   0.1387981   1.9903207   0.00646326
    1.9999791 ]
  [ 0.08466846 -0.11334133  0.23099795  1.9817369   0.01077207
    1.9999605 ]
  [ 1.8185948  -0.8322937   0.185397    1.9913884   0.00861771
    1.9999814 ]
  [ 0.14112     0.0100075   0.1387981   1.9903207   0.00646326
    1.9999791 ]
  [-0.7568025   0.3463564   0.18459873  1.982814    0.00861763
    1.9999628 ]]], shape=(2, 5, 6), dtype=float32)
```

## 可视化最终嵌入

为了可视化嵌入，我们将选择两个较大的句子：一个技术性的，另一个只是一个引用。我们将设置`TextVectorization`层以及位置编码层，看看最终输出的效果。

```py
technical_phrase = "to understand machine learning algorithms you need" +\
                   " to understand concepts such as gradient of a function "+\
                   "Hessians of a matrix and optimization etc"
wise_phrase = "patrick henry said give me liberty or give me death "+\
              "when he addressed the second virginia convention in march"

total_vocabulary = 200
sequence_length = 20
final_output_len = 50
phrase_vectorization_layer = TextVectorization(
                  output_sequence_length=sequence_length,
                  max_tokens=total_vocabulary)
# Learn the dictionary
phrase_vectorization_layer.adapt([technical_phrase, wise_phrase])
# Convert all sentences to tensors
phrase_tensors = convert_to_tensor([technical_phrase, wise_phrase], 
                                   dtype=tf.string)
# Use the word tensors to get vectorized phrases
vectorized_phrases = phrase_vectorization_layer(phrase_tensors)

random_weights_embedding_layer = PositionEmbeddingLayer(sequence_length, 
                                                        total_vocabulary,
                                                        final_output_len)
fixed_weights_embedding_layer = PositionEmbeddingFixedWeights(sequence_length, 
                                                        total_vocabulary,
                                                        final_output_len)
random_embedding = random_weights_embedding_layer(vectorized_phrases)
fixed_embedding = fixed_weights_embedding_layer(vectorized_phrases)
```

现在让我们看看两个短语的随机嵌入是什么样的。

```py
fig = plt.figure(figsize=(15, 5))    
title = ["Tech Phrase", "Wise Phrase"]
for i in range(2):
    ax = plt.subplot(1, 2, 1+i)
    matrix = tf.reshape(random_embedding[i, :, :], (sequence_length, final_output_len))
    cax = ax.matshow(matrix)
    plt.gcf().colorbar(cax)   
    plt.title(title[i], y=1.2)
fig.suptitle("Random Embedding")
plt.show()
```

![随机嵌入](https://machinelearningmastery.com/wp-content/uploads/2022/02/PEKeras1.png)

随机嵌入

固定权重层的嵌入如下图所示。

```py
fig = plt.figure(figsize=(15, 5))    
title = ["Tech Phrase", "Wise Phrase"]
for i in range(2):
    ax = plt.subplot(1, 2, 1+i)
    matrix = tf.reshape(fixed_embedding[i, :, :], (sequence_length, final_output_len))
    cax = ax.matshow(matrix)
    plt.gcf().colorbar(cax)   
    plt.title(title[i], y=1.2)
fig.suptitle("Fixed Weight Embedding from Attention is All You Need")
plt.show()
```

![使用正弦位置编码的嵌入](https://machinelearningmastery.com/wp-content/uploads/2022/02/PEKeras2.png)

使用正弦位置编码的嵌入

你可以看到，使用默认参数初始化的嵌入层输出随机值。另一方面，使用正弦波生成的固定权重为每个短语创建了一个独特的签名，其中包含了每个单词位置的信息。

你可以根据具体应用尝试可调或固定权重的实现。

## 进一步阅读

本节提供了更多资源，如果你想深入了解这个话题。

### 书籍

+   [自然语言处理中的 Transformers](https://www.amazon.com/Transformers-Natural-Language-Processing-architectures/dp/1800565798) 作者：Denis Rothman

### 论文

+   [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，2017 年

### 文章

+   [Transformer 注意力机制](https://machinelearningmastery.com/the-transformer-attention-mechanism/)

+   [Transformer 模型](https://machinelearningmastery.com/the-transformer-model/)

+   [用于语言理解的 Transformer 模型](https://www.tensorflow.org/text/tutorials/transformer)

+   [在 Keras 模型中使用预训练的词嵌入](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)

+   [使用序列到序列变换器进行英语到西班牙语翻译](https://keras.io/examples/nlp/neural_machine_translation_with_transformer/)

+   [转换器模型中位置编码的简介（第一部分）](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1)

## 总结

在本教程中，您了解了 Keras 中位置编码层的实现。

具体来说，您学到了：

+   Keras 中的文本向量化层

+   Keras 中的位置编码层

+   创建自己的位置编码类

+   为 Keras 中的位置编码层设置自定义权重

在本文中讨论的位置编码有任何问题吗？在下面的评论中提问，我会尽力回答。
