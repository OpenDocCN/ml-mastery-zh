# 本地使用 Gemma 2 的三种方式

> 原文：[`machinelearningmastery.com/3-ways-of-using-gemma-2-locally/`](https://machinelearningmastery.com/3-ways-of-using-gemma-2-locally/)

![本地使用 Gemma 2 的三种方式](img/dc201ce24fbf2415fcd6b8efaa23d8b9.png)

作者提供的图片

在 Gemma 1 成功推出后，Google 团队推出了一个更先进的模型系列，称为 Gemma 2。这个新的大型语言模型（LLMs）系列包括 90 亿（9B）和 270 亿（27B）参数的模型。Gemma 2 提供比其前身更高的性能和更大的推理效率，同时具有显著的安全性提升。这两款模型都优于 Llama 3 和 Gork 1 模型。

在本教程中，我们将了解三种应用程序，这些应用程序将帮助你更快地在本地运行 Gemma 2 模型。要在本地体验最先进的模型，你只需安装应用程序，下载模型并开始使用。就是这么简单。

## 1\. Jan

从官方网站下载并安装 [Jan](https://jan.ai/)。Jan 是我最喜欢的运行和测试各种开源及专有 LLM 的应用程序。它非常容易设置，并且在导入和使用本地模型方面具有很高的灵活性。

启动 Jan 应用程序并进入 Model Hub 菜单。然后，将 Hugging Face 仓库的以下链接粘贴到搜索栏中并按回车：**bartowski/gemma-2-9b-it-GGUF**。

![本地使用 Gemma 2 的三种方式](img/c045fd6057727f77d34b830d311e49d4.png)

作者提供的图片

你将被重定向到一个新窗口，在那里你可以选择不同的量化版本。我们将下载“Q4-K-M”版本。

![本地使用 Gemma 2 的三种方式](img/feecad737c904007ebe0ce768d31faab.png)

作者提供的图片

从右侧面板的模型菜单中选择已下载的模型并开始使用。

这个量化模型的当前版本每秒提供 37 个令牌，但如果使用不同的版本，你的速度可能会更快。

![本地使用 Gemma 2 的三种方式](img/381427dd20956e2f431dcf699c811325.png)

作者提供的图片

## 2\. Ollama

访问官方网站下载并安装 [Ollama](https://ollama.com/download)。它在开发者和熟悉终端及 CLI 工具的人中非常受欢迎。即使是新用户，也很容易设置。

安装完成后，请启动 Ollama 应用程序并在你喜欢的终端中输入以下命令。我在 Windows 11 上使用 Powershell。

```py
$ ollama run gemma2
```

根据你的互联网速度，下载模型大约需要半小时。

![本地使用 Gemma 2 的三种方式](img/90d90e4cc42e6e641be1ee8f86ef102d.png)

作者提供的图片

下载完成后，你可以开始提示并在终端中使用它。

![本地使用 Gemma 2 的三种方式](img/cb577dd2885bfbb57d95a15b4ce7a945.png)

作者提供的图片

### 通过从 GGUF 模型文件导入来使用 Gemma2

如果你已经有了一个 GGUF 模型文件并想与 Ollama 一起使用，那么你首先需要创建一个名为“Modelfile”的新文件，并输入以下命令：

```py
FROM ./gemma-2-9b-it-Q4_K_M.gguf
```

之后，使用 Modelfile 创建模型，该文件指向你目录中的 GGUF 文件。

```py
$ ollama create gemma2 -f Modelfile
```

当模型转移成功完成后，请输入以下命令开始使用它。

```py
$ ollama run gemma2
```

![本地使用 Gemma 2 的 3 种方法](img/09f32729029bc5a6271fb85a2e1a366c.png)

作者提供的图片

## 3\. Msty

从官方网站下载并安装 [Msty](https://msty.app/)。Msty 是一个新兴的竞争者，正成为我最喜欢的应用程序。它提供了大量功能和模型。你甚至可以连接到专有模型或 Ollama 服务器。它是一个简单而强大的应用程序，你应该试一试。

安装应用程序成功后，请启动程序并通过点击左侧面板上的按钮导航到“本地 AI 模型”。

![本地使用 Gemma 2 的 3 种方法](img/65b590dc4d6f53a22d6d6ed3309221f4.png)

作者提供的图片

点击“下载更多模型”按钮，并在搜索栏中输入以下链接：**bartowski/gemma-2-9b-it-GGUF**。确保你已选择 Hugging Face 作为模型中心。

![本地使用 Gemma 2 的 3 种方法](img/a3916fdae1506bc3d0579bd6f49f3fc5.png)

作者提供的图片

下载完成后，开始使用它。

![本地使用 Gemma 2 的 3 种方法](img/366b19acdb4f31ea717fac23260f2b66.png)

作者提供的图片

### 使用 Msty 与 Ollama

如果你想在聊天机器人应用程序中使用 Ollama 模型而不是终端，可以使用 Msty 的与 Ollama 连接选项。这很简单。

1.  首先，去终端并启动 Ollama 服务器。

```py
$ ollama serve
```

复制服务器链接。

```py
>>> </b>listen tcp 127.0.0.1:11434
```

1.  导航到“本地 AI 模型”菜单，并点击右上角的设置按钮。

1.  然后，选择“远程模型提供者”并点击“添加新提供者”按钮。

1.  接下来，将模型提供者选择为“Ollama 远程”，并输入 Ollama 服务器的服务端点链接。

1.  点击“重新获取模型”按钮，选择“gemma2:latest”，然后点击“添加”按钮。

![本地使用 Gemma 2 的 3 种方法](img/3de2e35d87ed8a7b7c1a1ef031eaac78.png)

作者提供的图片

1.  在聊天菜单中，选择新的模型并开始使用它。

![本地使用 Gemma 2 的 3 种方法](img/23a14432ea4ae3189717e92abefc426f.png)

作者提供的图片

## 结论

我们评测的这三款应用程序功能强大，具有大量功能，将提升你使用 AI 模型的本地体验。你需要做的就是下载应用程序和模型，其余的都很简单。

我使用 Jan 应用程序测试开源 LLM 的性能，并生成代码和内容。它速度快且私密，我的数据从未离开过我的笔记本电脑。

在本教程中，我们学习了如何使用 Jan、Ollama 和 Msty 本地运行 Gemma 2 模型。这些应用程序具有重要的功能，将提升你使用本地 LLM 的体验。

我希望你喜欢我的简短教程。 我喜欢分享我热衷并且经常使用的产品和应用。
