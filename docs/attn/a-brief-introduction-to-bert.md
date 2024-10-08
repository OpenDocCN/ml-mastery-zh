# BERT 简介

> 原文：[`machinelearningmastery.com/a-brief-introduction-to-bert/`](https://machinelearningmastery.com/a-brief-introduction-to-bert/)

正如我们了解了 [变换器是什么](https://machinelearningmastery.com/the-transformer-model/) 和我们可能如何 [训练变换器模型](https://machinelearningmastery.com/training-the-transformer-model/)，我们注意到它是让计算机理解人类语言的一个很好的工具。然而，变换器最初设计为一个将一种语言翻译成另一种语言的模型。如果我们将其重新用于其他任务，我们可能需要从头开始重新训练整个模型。考虑到训练变换器模型所需的时间非常长，我们希望有一个解决方案，可以使我们能够方便地重用训练好的变换器模型进行多种不同的任务。BERT 就是这样一个模型。它是变换器编码器部分的扩展。

在本教程中，你将了解什么是 BERT 并发现它能做什么。

完成本教程后，你将了解：

+   什么是来自变换器的双向编码表示（BERT）

+   BERT 模型如何被重新用于不同的目的

+   如何使用预训练的 BERT 模型

**通过我的书** [《构建具有注意力的变换器模型》](https://machinelearningmastery.com/transformer-models-with-attention/) **启动你的项目**。它提供了**自学教程**和**工作代码**，指导你构建一个完全可用的变换器模型。

*将句子从一种语言翻译成另一种语言*...

让我们开始吧。

![](img/b5fb41fb006b15fe8b995c010e2212e8.png)

BERT 简介

图片来源：[Samet Erköseoğlu](https://unsplash.com/photos/B0nUaoWnr0M)，保留部分权利。

## **教程概述**

本教程分为四个部分；它们是：

+   从变换器模型到 BERT

+   BERT 能做什么？

+   使用预训练的 BERT 模型进行摘要

+   使用预训练的 BERT 模型进行问答

## **先决条件**

在本教程中，我们假设你已经熟悉：

+   [变换器模型背后的理论](https://machinelearningmastery.com/the-transformer-model/)

+   [变换器模型的实现](https://machinelearningmastery.com/joining-the-transformer-encoder-and-decoder-and-masking/)

## **从变换器模型到 BERT**

在变换器模型中，编码器和解码器连接在一起形成一个 seq2seq 模型，以便你可以执行翻译任务，例如从英语到德语，正如你之前所见。回想一下注意力方程式说：

$$\text{attention}(Q,K,V) = \text{softmax}\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)V$$

但是上述的 $Q$、$K$ 和 $V$ 都是通过变换器模型中的权重矩阵转换得到的嵌入向量。训练一个变换器模型意味着找到这些权重矩阵。一旦权重矩阵被学习到，变换器就成为一个**语言模型**，这意味着它代表了一种理解你用来训练它的语言的方式。

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png)

Transformer 架构的编码器-解码器结构

取自“[Attention Is All You Need](https://arxiv.org/abs/1706.03762)”

转换器具有编码器和解码器部分。顾名思义，编码器将句子和段落转换为理解上下文的内部格式（一个数值矩阵），而解码器则执行相反的操作。结合编码器和解码器使得转换器可以执行序列到序列的任务，例如翻译。如果你去掉转换器的编码器部分，它可以告诉你一些关于上下文的信息，这可能会带来一些有趣的东西。

双向编码器表示的转换器（BERT）利用注意力模型来更深入地理解语言上下文。BERT 是由多个编码器块堆叠而成。输入文本像在转换器模型中一样被分隔成标记，每个标记在 BERT 输出时会被转换成一个向量。

## **BERT 能做什么？**

BERT 模型同时使用**掩码语言模型**（MLM）和**下一个句子预测**（NSP）进行训练。

![](img/703fca0a92cbf0ed7bbb94abed7c69dc.png)

BERT 模型

每个 BERT 训练样本是一对来自文档的句子。这两个句子可以是文档中的连续句子，也可以不是。第一个句子前会加上一个`[CLS]`标记（表示**类别**），每个句子后会加上一个`[SEP]`标记（作为**分隔符**）。然后，将两个句子拼接成一个标记序列，作为一个训练样本。训练样本中的一小部分标记会用特殊标记`[MASK]`掩码或替换为随机标记。

在输入到 BERT 模型之前，训练样本中的标记将被转换成嵌入向量，并添加位置编码，特别是 BERT 还会添加**段落嵌入**以标记标记是来自第一句还是第二句。

BERT 模型的每个输入词将产生一个输出向量。在训练良好的 BERT 模型中，我们期望：

+   对于被掩码的词，输出结果可以揭示原始词是什么。

+   对应于`[CLS]`标记的输出可以揭示两个句子在文档中是否是连续的。

然后，BERT 模型中训练得到的权重可以很好地理解语言上下文。

一旦你拥有这样的 BERT 模型，你可以将其用于许多**下游任务**。例如，通过在编码器上添加一个适当的分类层，并仅将一句话输入模型而不是一对句子，你可以将类别标记`[CLS]`作为情感分类的输入。这是因为类别标记的输出经过训练，可以聚合整个输入的注意力。

另一个例子是将一个问题作为第一句话，将文本（例如，一个段落）作为第二句话，然后第二句话中的输出标记可以标记出问题答案所在的位置。它有效的原因是每个标记的输出在整个输入的上下文中揭示了有关该标记的一些信息。

## 使用预训练的 BERT 模型进行摘要

从头开始训练一个 Transformer 模型需要很长时间。BERT 模型则需要更长的时间。但 BERT 的目的是创建一个可以用于多种不同任务的模型。

有一些预训练的 BERT 模型可以直接使用。接下来，你将看到一些使用案例。以下示例使用的文本来自：

+   [`www.project-syndicate.org/commentary/bank-of-england-gilt-purchases-necessary-but-mistakes-made-by-willem-h-buiter-and-anne-c-sibert-2022-10`](https://www.project-syndicate.org/commentary/bank-of-england-gilt-purchases-necessary-but-mistakes-made-by-willem-h-buiter-and-anne-c-sibert-2022-10)

理论上，BERT 模型是一个编码器，将每个输入标记映射到一个输出向量，这可以扩展到无限长度的标记序列。在实践中，其他组件的实现会施加限制，限制输入大小。通常，几百个标记应该是可以的，因为并非所有实现都能一次处理数千个标记。你可以将整篇文章保存为 `article.txt`（一个副本可以在[这里](https://machinelearningmastery.com/wp-content/uploads/2022/10/article.txt)获取）。如果你的模型需要更小的文本，你可以只使用其中的几个段落。

首先，让我们探讨摘要任务。使用 BERT 的想法是从原始文本中 *提取* 几句话，这些句子代表整个文本。你可以看到这个任务类似于下一句预测，其中如果给定一句话和文本，你希望分类它们是否相关。

为此，你需要使用 Python 模块 `bert-extractive-summarizer`

```py
pip install bert-extractive-summarizer
```

这是一些 Hugging Face 模型的包装器，用于提供摘要任务流水线。Hugging Face 是一个允许你发布机器学习模型的平台，主要用于 NLP 任务。

一旦你安装了 `bert-extractive-summarizer`，生成摘要只需要几行代码：

```py
from summarizer import Summarizer
text = open("article.txt").read()
model = Summarizer('distilbert-base-uncased')
result = model(text, num_sentences=3)
print(result)
```

这将产生以下输出：

```py
Amid the political turmoil of outgoing British Prime Minister Liz Truss’s
short-lived government, the Bank of England has found itself in the
fiscal-financial crossfire. Whatever government comes next, it is vital
that the BOE learns the right lessons. According to a statement by the BOE’s Deputy Governor for
Financial Stability, Jon Cunliffe, the MPC was merely “informed of the
issues in the gilt market and briefed in advance of the operation,
including its financial-stability rationale and the temporary and targeted
nature of the purchases.”
```

这就是完整的代码！在幕后，spaCy 被用于一些预处理，而 Hugging Face 被用于启动模型。使用的模型名为 `distilbert-base-uncased`。DistilBERT 是一个简化版的 BERT 模型，可以更快运行并使用更少的内存。该模型是一个“uncased”模型，这意味着输入文本中的大写或小写在转换为嵌入向量后被视为相同。

摘要模型的输出是一个字符串。由于你在调用模型时指定了`num_sentences=3`，因此摘要是从文本中选择的三句话。这种方法称为**提取式摘要**。另一种方法是**抽象式摘要**，其中摘要是生成的，而不是从文本中提取的。这需要不同于 BERT 的模型。

### 想要开始构建带有注意力机制的变换器模型吗？

立即参加我的免费 12 天电子邮件速成课程（附样本代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## 使用预训练 BERT 模型进行问答

使用 BERT 的另一个示例是将问题与答案匹配。你将问题和文本都提供给模型，并从文本中寻找答案的开始 *和* 结束位置的输出。

一个快速的示例就是如下几行代码，重用前面示例中的相同文本：

```py
from transformers import pipeline
text = open("article.txt").read()
question = "What is BOE doing?"

answering = pipeline("question-answering", model='distilbert-base-uncased-distilled-squad')
result = answering(question=question, context=text)
print(result)
```

在这里，直接使用了 Hugging Face。如果你已经安装了前面示例中使用的模块，那么 Hugging Face Python 模块是你已经安装的依赖项。否则，你可能需要用`pip`进行安装：

```py
pip install transformers
```

而且为了实际使用 Hugging Face 模型，你还应该安装**both** PyTorch 和 TensorFlow：

```py
pip install torch tensorflow
```

上述代码的输出是一个 Python 字典，如下所示：

```py
{'score': 0.42369240522384644,
'start': 1261,
'end': 1344,
'answer': 'to maintain or restore market liquidity in systemically important\nfinancial markets'}
```

在这里，你可以找到答案（即输入文本中的一句话），以及这个答案在标记顺序中的起始和结束位置。这个分数可以被视为模型对答案适合问题的置信度分数。

在后台，模型所做的是生成一个概率分数，用于确定文本中回答问题的最佳起始位置，以及最佳结束位置。然后通过查找最高概率的位置来提取答案。

## **进一步阅读**

本节提供了更多关于该主题的资源，如果你想深入了解。

### **论文**

+   [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，2017

+   [BERT：深度双向变换器语言理解的预训练](https://arxiv.org/abs/1810.04805)，2019

+   [DistilBERT，BERT 的精简版本：更小、更快、更便宜、更轻](https://arxiv.org/abs/1910.01108)，2019

## **总结**

在本教程中，你发现了 BERT 是什么以及如何使用预训练的 BERT 模型。

具体来说，你学到了：

+   BERT 如何作为对变换器模型的扩展创建

+   如何使用预训练 BERT 模型进行提取式摘要和问答
