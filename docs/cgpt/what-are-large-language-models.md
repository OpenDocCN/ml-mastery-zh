# 什么是大型语言模型

> 原文：[`machinelearningmastery.com/what-are-large-language-models/`](https://machinelearningmastery.com/what-are-large-language-models/)

大型语言模型（LLMs）是深度学习模型在处理人类语言方面的最新进展。一些出色的 LLM 使用案例已经得到展示。大型语言模型是一个经过训练的深度学习模型，以类似人类的方式理解和生成文本。其背后是一个大型变压器模型，完成所有的魔法。

在这篇文章中，你将了解大型语言模型的结构及其工作原理。特别是，你将了解到：

+   什么是变压器模型

+   变压器模型如何读取文本并生成输出

+   大型语言模型如何以类似人类的方式生成文本。

![](img/d1eca61cfb81041e4b4b4a858fb09227.png)

什么是大型语言模型。

由作者使用稳定扩散生成的图像。保留部分权利。

**开始并应用 ChatGPT**，请参阅我的书籍[用 ChatGPT 最大化生产力](https://machinelearningmastery.com/productivity-with-chatgpt/)。它提供了**真实的使用案例**和**提示示例**，旨在帮助你快速使用 ChatGPT。

让我们开始吧。

## 概述

本文分为三个部分；它们是：

+   从变压器模型到大型语言模型

+   为什么变压器能够预测文本？

+   大型语言模型是如何构建的？

## 从变压器模型到大型语言模型

对我们人类来说，我们将文本视为单词的集合。句子是单词的序列。文档是章节、部分和段落的序列。然而，对计算机而言，文本仅仅是一串字符。为了让机器理解文本，可以构建一个[基于递归神经网络的模型](https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/)。该模型一次处理一个词或字符，并在整个输入文本被处理完后提供输出。这个模型效果相当好，只是当序列的结尾到达时，它有时会“忘记”序列开头发生了什么。

2017 年，Vaswani 等人发表了一篇论文，《Attention is All You Need》，以建立一个[变压器模型](https://machinelearningmastery.com/building-transformer-models-with-attention-crash-course-build-a-neural-machine-translator-in-12-days/)。它基于注意力机制。与递归神经网络相反，注意力机制允许你一次性看到整个句子（甚至是段落），而不是逐词处理。这使得变压器模型能够更好地理解一个词的上下文。许多最先进的语言处理模型都基于变压器。

为了使用变换器模型处理文本输入，你首先需要将其标记化为一系列词语。这些标记随后被编码为数字，并转换为嵌入，这些嵌入是保留其含义的标记的向量空间表示。接下来，变换器中的编码器将所有标记的嵌入转换为一个上下文向量。

下面是一个文本字符串、其标记化及向量嵌入的示例。请注意，标记化可以是子词，例如文本中的“nosegay”被标记化为“nose”和“gay”。

输入文本示例

```py
As she said this, she looked down at her hands, and was surprised to find that she had put on one of the rabbit's little gloves while she was talking. "How can I have done that?" thought she, "I must be growing small again." She got up and went to the table to measure herself by it, and found that, as nearly as she could guess, she was now about two feet high, and was going on shrinking rapidly: soon she found out that the reason of it was the nosegay she held in her hand: she dropped it hastily, just in time to save herself from shrinking away altogether, and found that she was now only three inches high.
```

标记化文本

```py
['As', ' she', ' said', ' this', ',', ' she', ' looked', ' down', ' at', ' her', ' hands', ',', ' and', ' was', ' surprised', ' to', ' find', ' that', ' she', ' had', ' put', ' on', ' one', ' of', ' the', ' rabbit', "'s", ' little', ' gloves', ' while', ' she', ' was', ' talking', '.', ' "', 'How', ' can', ' I', ' have', ' done', ' that', '?"', ' thought', ' she', ',', ' "', 'I', ' must', ' be', ' growing', ' small', ' again', '."', ' She', ' got', ' up', ' and', ' went', ' to', ' the', ' table', ' to', ' measure', ' herself', ' by', ' it', ',', ' and', ' found', ' that', ',', ' as', ' nearly', ' as', ' she', ' could', ' guess', ',', ' she', ' was', ' now', ' about', ' two', ' feet', ' high', ',', ' and', ' was', ' going', ' on', ' shrinking', ' rapidly', ':', ' soon', ' she', ' found', ' out', ' that', ' the', ' reason', ' of', ' it', ' was', ' the', ' nose', 'gay', ' she', ' held', ' in', ' her', ' hand', ':', ' she', ' dropped', ' it', ' hastily', ',', ' just', ' in', ' time', ' to', ' save', ' herself', ' from', ' shrinking', ' away', ' altogether', ',', ' and', ' found', ' that', ' she', ' was', ' now', ' only', ' three', ' inches', ' high', '.']
```

上述文本的嵌入

```py
[ 2.49 0.22 -0.36 -1.55 0.22 -2.45 2.65 -1.6 -0.14 2.26
 -1.26 -0.61 -0.61 -1.89 -1.87 -0.16 3.34 -2.67 0.42 -1.71
 ...
 2.91 -0.77 0.13 -0.24 0.63 -0.26 2.47 -1.22 -1.67 1.63
 1.13 0.03 -0.68 0.8 1.88 3.05 -0.82 0.09 0.48 0.33]
```

上下文向量就像整个输入的精髓。利用这个向量，变换器解码器基于线索生成输出。例如，你可以提供原始输入作为线索，让变换器解码器生成自然跟随的下一个词。然后，你可以重复使用相同的解码器，但这次的线索将是之前生成的下一个词。这个过程可以重复，以从一个引导句开始创建整个段落。

![](img/fc1e4d7b9ab803c6ae6c1c8da35d116b.png)

变换器架构

这个过程被称为自回归生成。这就是大型语言模型的工作方式，除了这种模型是一个变换器模型，可以处理非常长的输入文本，其上下文向量非常大，因此能够处理非常复杂的概念，并且其编码器和解码器具有许多层。

## 为什么变换器可以预测文本？

在他的博客文章“[递归神经网络的非凡有效性](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)”中，Andrej Karpathy 证明了递归神经网络可以相当好地预测文本中的下一个词。这不仅仅是因为人类语言中存在限制词语在句子中不同位置使用的规则（即语法），还因为语言中存在冗余。

根据 Claude Shannon 的影响力论文“《印刷英语的预测与熵》”，尽管英语有 27 个字母（包括空格），但其熵为每个字母 2.1 比特。如果字母是随机使用的，那么熵将为 4.8 比特，这使得在一种人类语言文本中预测下一个词变得更容易。机器学习模型，尤其是变换器模型，擅长做出这样的预测。

通过重复这一过程，变压器模型能够逐字生成整个段落。然而，对于变压器模型来说，什么是语法？本质上，语法表示词汇在语言中的使用方式，将其分类为不同的词性，并要求在句子中按照特定的顺序排列。尽管如此，列举所有语法规则仍然具有挑战性。实际上，变压器模型并不会明确地存储这些规则，而是通过示例隐式地获得它们。模型可能会学习超越语法规则的内容，扩展到这些示例中呈现的思想，但变压器模型必须足够大。

## 大型语言模型是如何构建的？

大型语言模型是大规模的变压器模型。它大到通常无法在单台计算机上运行。因此，它自然是通过 API 或 Web 界面提供的服务。正如你所预期的，这种大型模型在记住语言的模式和结构之前，必须从大量文本中学习。

例如，支撑 ChatGPT 服务的 GPT-3 模型是在来自互联网的大量文本数据上进行训练的。这包括书籍、文章、网站以及各种其他来源。在训练过程中，模型学习词汇、短语和句子之间的统计关系，从而能够在给定提示或查询时生成连贯且上下文相关的回应。

从大量文本中提炼出知识，GPT-3 模型因此能够理解多种语言，并且掌握各种主题的知识。这也是为什么它能够生成不同风格的文本。虽然你可能会惊讶于大型语言模型能够进行翻译、文本摘要和问答，但如果你考虑到这些实际上是匹配主文本的特殊“语法”，即提示（prompts），那么这并不令人惊讶。

## 摘要

已经开发了多个大型语言模型。示例包括来自 OpenAI 的 GPT-3 和 GPT-4、来自 Meta 的 LLaMA 以及来自 Google 的 PaLM2。这些模型能够理解语言并生成文本。在这篇文章中，你了解到：

+   大型语言模型基于变压器架构。

+   注意机制使得 LLMs 能够捕捉词语之间的长程依赖关系，因此模型能够理解上下文。

+   大型语言模型基于先前生成的标记以自回归的方式生成文本。
