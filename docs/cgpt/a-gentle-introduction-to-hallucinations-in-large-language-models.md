# 对大语言模型中幻觉的温和介绍

> 原文：[`machinelearningmastery.com/a-gentle-introduction-to-hallucinations-in-large-language-models/`](https://machinelearningmastery.com/a-gentle-introduction-to-hallucinations-in-large-language-models/)

大语言模型（LLMs）以“幻觉”而闻名。这是一种行为，即模型以准确的方式说出虚假的知识。在这篇文章中，你将了解为什么幻觉是 LLM 的一种特性。具体来说，你将了解：

+   为什么大语言模型会出现幻觉

+   如何让幻觉为你服务

+   如何减少幻觉

**开始使用并应用 ChatGPT**，可以参考我的书籍 [《用 ChatGPT 最大化生产力》](https://machinelearningmastery.com/productivity-with-chatgpt/)。这本书提供了**实际案例**和**提示示例**，旨在让你快速上手使用 ChatGPT。

让我们开始吧！[](../Images/b45994193a49a6c8b003fe66305f1902.png)

对大语言模型中幻觉的温和介绍

图片由作者使用 Stable Diffusion 生成。保留所有权利。

## 概述

本文分为三个部分，它们是

+   大语言模型中的幻觉是什么

+   使用幻觉

+   减少幻觉的影响

## 大语言模型中的幻觉是什么

大语言模型是一个经过训练的机器学习模型，它根据你提供的提示生成文本。模型的训练使其具备了从我们提供的训练数据中获得的一些知识。很难判断模型记住了什么知识或者忘记了什么。实际上，当模型生成文本时，它无法判断生成的内容是否准确。

在 LLM 的上下文中，“幻觉”指的是模型生成的不正确、不合理或不真实的文本现象。由于 LLM 不是数据库或搜索引擎，它们不会引用其响应所基于的来源。这些模型生成的文本是基于你提供的提示进行外推的。外推的结果不一定得到任何训练数据的支持，但却是与提示最相关的内容。

为了理解幻觉，你可以从一些文本中构建一个二字母的马尔科夫模型：提取一段长文本，建立每对相邻字母的表格，并统计次数。例如，“大语言模型中的幻觉”会生成“HA”，“AL”，“LL”，“LU”等，其中“LU”出现了一次，“LA”出现了两次。现在，如果你从提示“L”开始，你产生“LA”的可能性是产生“LL”或“LS”的两倍。然后以提示“LA”开始，你有相等的概率产生“AL”，“AT”，“AR”或“AN”。然后你可以尝试以提示“LAT”继续这个过程。最终，这个模型发明了一个不存在的新词。这是统计模式的结果。你可以说你的马尔科夫模型出现了拼写幻觉。

大型语言模型（LLMs）中的幻觉现象并不比这更复杂，即使模型更为复杂。从高层次看，幻觉是由于有限的上下文理解造成的，因为模型必须将提示和训练数据转化为一种抽象，其中一些信息可能会丢失。此外，训练数据中的噪音也可能提供一种偏差的统计模式，导致模型做出你不期望的回应。

## 使用幻觉

你可以将幻觉视为大型语言模型中的一种特性。如果你希望模型具有创造力，你可能希望看到它们产生幻觉。例如，如果你要求 ChatGPT 或其他大型语言模型为你提供一个奇幻故事的情节，你希望它生成一个全新的角色、场景和情节，而不是从现有的故事中复制。这只有在模型没有查阅它们训练过的数据时才有可能。

你可能会在寻找多样性时希望出现幻觉，例如，要求提供创意。这就像是让模型为你进行头脑风暴。你希望从现有的训练数据中得到一些派生的想法，而不是完全相同的东西。幻觉可以帮助你探索不同的可能性。

许多语言模型都有一个“温度”参数。你可以通过 API 控制 ChatGPT 的温度，而不是通过网页界面。这是一个随机性的参数。更高的温度可以引入更多的幻觉。

## 减少幻觉

语言模型不是搜索引擎或数据库。幻觉是不可避免的。令人恼火的是，模型生成的文本中可能包含难以发现的错误。

如果污染的训练数据导致了幻觉，你可以清理数据并重新训练模型。然而，大多数模型过于庞大，无法在自己的设备上进行训练。即使是微调现有模型也可能在普通硬件上不可行。最佳的解决方法可能是对结果进行人工干预，并在模型出现严重错误时要求其重新生成。

避免幻觉的另一种解决方案是受控生成。这意味着在提示中提供足够的细节和约束，以限制模型的自由度，从而减少幻觉的产生。提示工程的目的是为模型指定角色和场景，以指导生成过程，从而避免无边界的幻觉。

## 总结

在这篇文章中，你了解了大型语言模型如何产生幻觉。特别是，

+   为什么幻觉会有用

+   如何限制幻觉

值得注意的是，虽然幻觉可以得到缓解，但可能无法完全消除。在创造力和准确性之间存在权衡。