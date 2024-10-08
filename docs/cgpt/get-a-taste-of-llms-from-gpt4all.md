# 体验 GPT4All 的 LLM

> 原文：[`machinelearningmastery.com/get-a-taste-of-llms-from-gpt4all/`](https://machinelearningmastery.com/get-a-taste-of-llms-from-gpt4all/)

最近，大型语言模型变得越来越流行。ChatGPT 非常时尚。尝试使用 ChatGPT 了解 LLM 的工作原理很容易，但有时你可能希望有一个可以在计算机上离线运行的替代方案。在这篇文章中，你将了解 GPT4All 作为一个可以在计算机上安装的 LLM。特别是，你将学习

+   什么是 GPT4All

+   如何安装 GPT4All 的桌面客户端

+   如何在 Python 中运行 GPT4All

**开始使用 ChatGPT**，可以参考我的书 [《利用 ChatGPT 最大化生产力》](https://machinelearningmastery.com/productivity-with-chatgpt/)。它提供了 **实际应用案例** 和 **提示示例**，旨在帮助你快速上手 ChatGPT。

让我们开始吧！[](../Images/c623826cbb400d3a3cbc609df251a94c.png)

体验 GPT4All 的 LLM

图片由作者使用 Stable Diffusion 生成。保留所有权利。

**更新：**

+   **2023-10-10**：刷新了 gpt4all 模块 1.0.12 版本的 Python 代码

## 概述

本文分为三部分；它们是：

+   什么是 GPT4All？

+   如何获取 GPT4All

+   如何在 Python 中使用 GPT4All

## 什么是 GPT4All？

“GPT”一词源自于 Radford 等人在 2018 年发表的论文《通过生成预训练提升语言理解》。这篇论文描述了变换器模型如何被证明能够理解人类语言。

自那时以来，许多人尝试使用变换器架构开发语言模型，已经发现一个足够大的模型可以产生优异的结果。然而，许多开发的模型都是专有的。这些模型要么作为付费订阅的服务提供，要么在具有某些限制条款的许可下提供。有些甚至由于体积过大而无法在普通硬件上运行。

GPT4All 项目试图让 LLM 在普通硬件上对公众开放。它允许你训练和部署你的模型。也提供了预训练模型，其体积较小，可以合理地在 CPU 上运行。

## 如何获取 GPT4All

让我们仅关注使用预训练模型。

在撰写本文时，GPT4All 可从 [`gpt4all.io/index.html`](https://gpt4all.io/index.html) 获取，你可以将其作为桌面应用程序运行或使用 Python 库。你可以下载适合你操作系统的安装程序以运行桌面客户端。客户端的体积只有几百 MB。你应该会看到如下的安装界面：![](img/4486a6688a08d8b8b6472c99ea7faaf4.png)

安装客户端后，首次启动时会提示你安装一个模型，模型的大小可以达到几 GB。你可以选择“`gpt4all-j-v1.3-groovy`”（GPT4All-J 模型）。这是一个相对较小但受欢迎的模型。

![](img/381d212b3d682cc1fa359bb0873be2e9.png)

一旦客户端和模型准备好，你可以在输入框中键入你的消息。模型可能期望特定形式的输入，例如特定的语言或风格。这个模型期望一种对话风格（如 ChatGPT），并且通常处理英语效果很好。例如，下面是它对输入“给我一个包含 10 种颜色及其 RGB 代码的列表”的回应：

![](img/7aac7ad319c2da868217154cc3202ace.png)

## 如何在 Python 中使用 GPT4All

GPT4All 的关键组件是模型。桌面客户端只是一个接口。除了客户端之外，你还可以通过 Python 库调用模型。

这个库名为“`gpt4all`”，你可以使用 `pip` 命令安装：

```py
pip install gpt4all
```

**注意：** 这是一个快速发展的库，功能可能会发生变化。以下代码已在版本 1.0.12 上测试，但在未来版本中可能无法使用。

之后，你可以用 Python 只需几行代码使用它：

```py
import pprint

import gpt4all

model = gpt4all.GPT4All("orca-mini-7b.ggmlv3.q4_0.bin")
with model.chat_session():
    response = model.generate("Give me a list of 10 colors and their RGB code")
    print(response)
    pprint.pprint(model.current_chat_session)
```

运行上述代码将下载模型文件（如果你还没有下载）。之后，模型会被加载，输入被提供，响应将作为字符串返回。打印的输出可能是：

```py
 Sure, here's a list of 10 colors along with their RGB codes:

1\. Red (255, 0, 0)
2\. Blue (0, 0, 255)
3\. Green (0, 255, 0)
4\. Yellow (255, 255, 0)
5\. Orange (255, 165, 0)
6\. Purple (192, 118, 192)
7\. Pink (255, 192, 203)
8\. Maroon (153, 42, 102)
9\. Teal (0, 128, 128)
10\. Lavender (238, 102, 147)
```

会话的聊天记录存储在模型的属性 `current_chat_session` 中，格式为 Python 列表。示例如下：

```py
[{'content': '### System:\n'
             'You are an AI assistant that follows instruction extremely well. '
             'Help as much as you can.',
  'role': 'system'},
 {'content': 'Give me a list of 10 colors and their RGB code', 'role': 'user'},
 {'content': " Sure, here's a list of 10 colors along with their RGB codes:\n"
             '\n'
             '1\. Red (255, 0, 0)\n'
             '2\. Blue (0, 0, 255)\n'
             '3\. Green (0, 255, 0)\n'
             '4\. Yellow (255, 255, 0)\n'
             '5\. Orange (255, 165, 0)\n'
             '6\. Purple (192, 118, 192)\n'
             '7\. Pink (255, 192, 203)\n'
             '8\. Maroon (153, 42, 102)\n'
             '9\. Teal (0, 128, 128)\n'
             '10\. Lavender (238, 102, 147)',
  'role': 'assistant'}]
```

聊天记录是一个格式为 Python 字典的对话序列，字典包含 `role` 和 `content` 两个键。`role` 可以是 `"system"`、`"assistant"` 或 `"user"`，而 `content` 是文本字符串。如果你像示例那样与模型聊天，你的角色是 `"user"`，计算机的响应是 `"assistant"`。你可以继续使用 `generate()` 调用来继续对话。下面是一个示例：

```py
import pprint

import gpt4all

model = gpt4all.GPT4All("orca-mini-7b.ggmlv3.q4_0.bin")
with model.chat_session():
    questions = [
        "Can you explain what is a large language model?",
        "Can you give some examples applications?",
        "Are there any limitations?",
        "Summarize the above in two sentences.",
    ]
    for question in questions:
        answer = model.generate(question)
        print("Q:", question)
        print("A:", answer)

    pprint.pprint(model.current_chat_session)
```

请注意，你在 for 循环中多次调用了模型。每次响应时，模型会将输出追加到聊天消息列表中，因此你积累了上下文。然后你添加一个新的对话，再次调用模型。这就是模型如何记住聊天记录的方式。下面是上述代码如何回应你问题的示例：

```py
Q: Can you explain what is a large language model?
A:  A large language model (LLM) is a type of artificial intelligence model that uses deep learning techniques to analyze and generate human-like language. These models are trained on massive amounts of text data, which allows them to learn complex patterns and relationships in the language. LLMs can be used for a variety of natural language processing tasks such as language translation, sentiment analysis, speech recognition, and more. They are typically implemented using deep learning frameworks like TensorFlow or PyTorch and require large amounts of computational resources to train and run effectively.
Q: Can you give some examples applications?
A:  Yes, there are many applications for LLMs, including:

1\. Chatbots: LLMs can be used to power chatbots that can understand natural language and respond appropriately to user queries.

2\. Virtual assistants: LLMs can help power virtual assistants like Siri or Alexa, allowing them to understand and respond to user requests in a more human-like way.

3\. Language translation: LLMs can be used to translate text from one language to another with high accuracy.

4\. Sentiment analysis: LLMs can analyze social media posts or customer feedback to determine the sentiment behind the content, which can help businesses understand their customers better.

5\. Speech recognition: LLMs can be used to power speech recognition systems that allow users to speak commands and receive responses in natural language.
Q: Are there any limitations?
A:  Yes, there are some limitations to LLMs, including:

1\. Data requirements: LLMs require large amounts of data to train effectively. This means that the model may not perform well on tasks with limited training data.

2\. Overfitting: LLMs can sometimes overfit the training data, meaning they memorize the data rather than learning from it. This can lead to poor generalization and accuracy issues when tested on new data.

3\. Hardware requirements: LLMs require powerful hardware to run effectively, which may not be available or affordable for all users.

4\. Interpretability: LLMs can be difficult to interpret and understand how they are making decisions, which may limit their usefulness in some applications.
Q: Summarize the above in two sentences.
A:  There are limitations to LLMs such as data requirements, overfitting, hardware requirements, and interpretability.
```

因此，上述代码结束时积累的聊天记录将如下所示：

```py
[{'content': '### System:\n'
             'You are an AI assistant that follows instruction extremely well. '
             'Help as much as you can.',
  'role': 'system'},
 {'content': 'Can you explain what is a large language model?', 'role': 'user'},
 {'content': ' A large language model (LLM) is a type of artificial '
             'intelligence model that uses deep learning techniques to analyze '
             'and generate human-like language. These models are trained on '
             'massive amounts of text data, which allows them to learn complex '
             'patterns and relationships in the language. LLMs can be used for '
             'a variety of natural language processing tasks such as language '
             'translation, sentiment analysis, speech recognition, and more. '
             'They are typically implemented using deep learning frameworks '
             'like TensorFlow or PyTorch and require large amounts of '
             'computational resources to train and run effectively.',
  'role': 'assistant'},
 {'content': 'Can you give some examples applications?', 'role': 'user'},
 {'content': ' Yes, there are many applications for LLMs, including:\n'
             '\n'
             '1\. Chatbots: LLMs can be used to power chatbots that can '
             'understand natural language and respond appropriately to user '
             'queries.\n'
             '\n'
             '2\. Virtual assistants: LLMs can help power virtual assistants '
             'like Siri or Alexa, allowing them to understand and respond to '
             'user requests in a more human-like way.\n'
             '\n'
             '3\. Language translation: LLMs can be used to translate text from '
             'one language to another with high accuracy.\n'
             '\n'
             '4\. Sentiment analysis: LLMs can analyze social media posts or '
             'customer feedback to determine the sentiment behind the content, '
             'which can help businesses understand their customers better.\n'
             '\n'
             '5\. Speech recognition: LLMs can be used to power speech '
             'recognition systems that allow users to speak commands and '
             'receive responses in natural language.',
  'role': 'assistant'},
 {'content': 'Are there any limitations?', 'role': 'user'},
 {'content': ' Yes, there are some limitations to LLMs, including:\n'
             '\n'
             '1\. Data requirements: LLMs require large amounts of data to '
             'train effectively. This means that the model may not perform '
             'well on tasks with limited training data.\n'
             '\n'
             '2\. Overfitting: LLMs can sometimes overfit the training data, '
             'meaning they memorize the data rather than learning from it. '
             'This can lead to poor generalization and accuracy issues when '
             'tested on new data.\n'
             '\n'
             '3\. Hardware requirements: LLMs require powerful hardware to run '
             'effectively, which may not be available or affordable for all '
             'users.\n'
             '\n'
             '4\. Interpretability: LLMs can be difficult to interpret and '
             'understand how they are making decisions, which may limit their '
             'usefulness in some applications.',
  'role': 'assistant'},
 {'content': 'Summarize the above in two sentences.', 'role': 'user'},
 {'content': ' There are limitations to LLMs such as data requirements, '
             'overfitting, hardware requirements, and interpretability.',
  'role': 'assistant'}]
```

你可能会从其他模型中获得更好的结果。由于模型的随机性，你也可能得到不同的结果。

## 总结

GPT4All 是一个你可以在计算机上使用的不错工具。它允许你探索与大型语言模型的互动，并帮助你更好地理解模型的能力和限制。在这篇文章中，你了解到：

+   GPT4All 有一个桌面客户端，你可以将其安装到你的计算机上。

+   GPT4All 提供了一个 Python 接口，允许你在代码中与语言模型进行交互。

+   有多个语言模型可供选择。
