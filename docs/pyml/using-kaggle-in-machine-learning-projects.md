# 在机器学习项目中使用 Kaggle

> 原文：[`machinelearningmastery.com/using-kaggle-in-machine-learning-projects/`](https://machinelearningmastery.com/using-kaggle-in-machine-learning-projects/)

你可能听说过 Kaggle 数据科学竞赛，但你知道 Kaggle 还有许多其他功能可以帮助你完成下一个机器学习项目吗？对于寻找数据集以进行下一个机器学习项目的人，Kaggle 允许你访问他人公开的数据集并分享你自己的数据集。对于希望构建和训练自己机器学习模型的人，Kaggle 还提供了一个浏览器内的笔记本环境和一些免费的 GPU 小时。你还可以查看其他人的公开笔记本！

除了网站之外，Kaggle 还提供了一个命令行界面（CLI），你可以在命令行中使用它来访问和下载数据集。

让我们立即深入探索 Kaggle 所提供的内容！

完成本教程后，你将学到：

+   什么是 Kaggle？

+   如何将 Kaggle 作为你机器学习工作流的一部分

+   使用 Kaggle API 的命令行界面（CLI）

**通过我的新书** [《Python 机器学习》](https://machinelearningmastery.com/python-for-machine-learning/)，**快速启动你的项目**，包括*逐步教程*和*所有示例的 Python 源代码*文件。

让我们开始吧！！[](../Images/9fa78f4ce40291046ce558c63d368f74.png)

在机器学习项目中使用 Kaggle

图片由[Stefan Widua](https://unsplash.com/photos/kOuaZs7jDZE)提供。保留部分权利。

## 概述

本教程分为五个部分；它们是：

+   什么是 Kaggle？

+   设置 Kaggle 笔记本

+   使用带有 GPU/TPU 的 Kaggle 笔记本

+   在 Kaggle 笔记本中使用 Kaggle 数据集

+   使用 Kaggle CLI 工具中的 Kaggle 数据集

## 什么是 Kaggle？

Kaggle 可能以其举办的数据科学竞赛而最为人知，其中一些竞赛提供了五位数的奖池，并有数百支队伍参赛。除了这些竞赛，Kaggle 还允许用户发布和搜索数据集，这些数据集可以用于他们的机器学习项目。要使用这些数据集，你可以在浏览器中使用 Kaggle 笔记本或 Kaggle 的公共 API 来下载数据集，然后在你的机器学习项目中使用这些数据集。

![](https://machinelearningmastery.com/wp-content/uploads/2022/04/kaggle_featured_competitions.png)

Kaggle 竞赛

此外，Kaggle 还提供一些课程和讨论页面，供你学习更多关于机器学习的知识，并与其他机器学习从业者交流！

在本文的其余部分，我们将重点介绍如何利用 Kaggle 的数据集和笔记本来帮助我们在自己的机器学习项目中工作或寻找新的项目。

## 设置 Kaggle 笔记本

要开始使用 Kaggle 笔记本，你需要创建一个 Kaggle 账户，可以使用现有的 Google 账户或使用你的电子邮件创建一个。

然后，前往“代码”页面。

![](https://machinelearningmastery.com/wp-content/uploads/2022/04/kaggle_sidebar_notebook.png)

Kaggle 首页的左侧边栏，代码标签

然后你将能够看到你自己的笔记本以及其他人发布的公共笔记本。要创建自己的笔记本，点击“新建笔记本”。

![](https://machinelearningmastery.com/wp-content/uploads/2022/04/kaggle_code_page.png)

Kaggle 代码页面

这将创建你的新笔记本，它看起来像一个 Jupyter 笔记本，具有许多类似的命令和快捷键。

![](https://machinelearningmastery.com/wp-content/uploads/2022/04/kaggle_empty_notebook.png)

Kaggle 笔记本

你还可以通过前往“文件 -> 编辑器类型”在笔记本编辑器和脚本编辑器之间切换。

![](https://machinelearningmastery.com/wp-content/uploads/2022/04/kaggle_toggle_script.png)

更改 Kaggle 笔记本中的编辑器类型

将编辑器类型更改为脚本会显示如下内容：

![](https://machinelearningmastery.com/wp-content/uploads/2022/04/kaggle_empty_script.png)

Kaggle 笔记本脚本编辑器类型

### 想要开始学习 Python 用于机器学习吗？

现在就获取我的免费 7 天电子邮件速成课程（包含示例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## 使用 GPUs/TPUs 的 Kaggle

谁不喜欢用于机器学习项目的免费 GPU 时间呢？ GPUs 可以大幅加速机器学习模型的训练和推断，尤其是深度学习模型。

Kaggle 提供了一些免费的 GPUs 和 TPUs 配额，你可以用来进行你的项目。在撰写本文时，验证手机号码后每周 GPU 的可用时间为 30 小时，TPU 的可用时间为 20 小时。

要将加速器附加到你的笔记本，请前往“设置 ▷ 环境 ▷ 偏好设置”。

![](https://machinelearningmastery.com/wp-content/uploads/2022/04/kaggle_environment_preferences.png)

更改 Kaggle 笔记本环境偏好设置

你将被要求通过手机号码验证你的账户。

![](https://machinelearningmastery.com/wp-content/uploads/2022/04/verify_phone.png)

验证手机号码

然后会出现一个页面，列出你剩余的使用量，并提到开启 GPUs 会减少可用的 CPUs 数量，因此在进行神经网络训练/推断时才可能是一个好主意。

![](https://machinelearningmastery.com/wp-content/uploads/2022/04/use_accelerator.png)

向 Kaggle 笔记本添加 GPU 加速器

## 使用 Kaggle 数据集与 Kaggle 笔记本

机器学习项目是数据饥饿的怪物，找到当前项目的数据集或寻找新的项目数据集总是一项繁琐的工作。幸运的是，Kaggle 拥有由用户和比赛提供的丰富数据集。这些数据集对寻找当前机器学习项目的数据或寻找新项目创意的人来说都是宝贵的财富。

让我们探索如何将这些数据集添加到我们的 Kaggle 笔记本中。

首先，点击右侧边栏中的“添加数据”。

![](https://machinelearningmastery.com/wp-content/uploads/2022/04/add_data.png)

将数据集添加到 Kaggle 笔记本环境中

应该会出现一个窗口，显示一些公开可用的数据集，并提供将自己的数据集上传以供 Kaggle 笔记本使用的选项。

![](https://machinelearningmastery.com/wp-content/uploads/2022/04/kaggle_datasets.png)

在 Kaggle 数据集中进行搜索

我将使用经典的泰坦尼克数据集作为本教程的示例，您可以通过在窗口右上角的搜索框中输入搜索词来找到它。

![](https://machinelearningmastery.com/wp-content/uploads/2022/04/search_titanic_dataset.png)

使用“Titanic”关键词过滤的 Kaggle 数据集

之后，数据集可以在笔记本中使用。要访问文件，请查看文件的路径并在其前面加上`../input/{path}`。例如，泰坦尼克数据集的文件路径是：

```py
../input/titanic/train_and_test2.csv
```

在笔记本中，我们可以使用以下命令读取数据：

```py
import pandas

pandas.read_csv("../input/titanic/train_and_test2.csv")
```

这将从文件中获取数据：

![](https://machinelearningmastery.com/wp-content/uploads/2022/04/kaggle_notebook_read_dataset.png)

在 Kaggle 笔记本中使用泰坦尼克数据集

## 使用 Kaggle CLI 工具操作 Kaggle 数据集

Kaggle 还拥有一个公共 API 及 CLI 工具，我们可以用来下载数据集、参与比赛等。我们将探讨如何使用 CLI 工具设置和下载 Kaggle 数据集。

要开始，请使用以下命令安装 CLI 工具：

```py
pip install kaggle
```

对于 Mac/Linux 用户，您可能需要：

```py
pip install --user kaggle
```

然后，您需要创建一个 API 令牌进行身份验证。请访问 Kaggle 网页，点击右上角的个人资料图标，然后进入帐户。

![](https://machinelearningmastery.com/wp-content/uploads/2022/04/kaggle_account.png)

进入 Kaggle 帐户设置

从那里，向下滚动到创建新的 API 令牌：

![](https://machinelearningmastery.com/wp-content/uploads/2022/04/create_api_token.png)

为 Kaggle 公共 API 生成新的 API 令牌

这将下载一个 `kaggle.json` 文件，您将用它来通过 Kaggle CLI 工具进行身份验证。您必须将其放置在正确的位置以使其正常工作。对于 Linux/Mac/Unix 系统，应放置在 `~/.kaggle/kaggle.json`，对于 Windows 用户，应放置在 `C:\Users\<Windows 用户名>\.kaggle\kaggle.json`。如果放错位置并在命令行中调用 `kaggle`，将会出现错误：

```py
OSError: Could not find kaggle.json. Make sure it’s location in … Or use the environment method
```

现在，让我们开始下载这些数据集吧！

要使用搜索词（如 titanic）搜索数据集，我们可以使用：

```py
kaggle datasets list -s titanic
```

搜索 titanic，我们得到：

```py
$ kaggle datasets list -s titanic
ref                                                          title                                           size  lastUpdated          downloadCount  voteCount  usabilityRating
-----------------------------------------------------------  ---------------------------------------------  -----  -------------------  -------------  ---------  ---------------
datasets/heptapod/titanic                                    Titanic                                         11KB  2017-05-16 08:14:22          37681        739  0.7058824
datasets/azeembootwala/titanic                               Titanic                                         12KB  2017-06-05 12:14:37          13104        145  0.8235294
datasets/brendan45774/test-file                              Titanic dataset                                 11KB  2021-12-02 16:11:42          19348        251  1.0
datasets/rahulsah06/titanic                                  Titanic                                         34KB  2019-09-16 14:43:23           3619         43  0.6764706
datasets/prkukunoor/TitanicDataset                           Titanic                                        135KB  2017-01-03 22:01:13           4719         24  0.5882353
datasets/hesh97/titanicdataset-traincsv                      Titanic-Dataset (train.csv)                     22KB  2018-02-02 04:51:06          54111        377  0.4117647
datasets/fossouodonald/titaniccsv                            Titanic csv                                      1KB  2016-11-07 09:44:58           8615         50  0.5882353
datasets/broaniki/titanic                                    titanic                                        717KB  2018-01-30 04:08:45           8004        128  0.1764706
datasets/pavlofesenko/titanic-extended                       Titanic extended dataset (Kaggle + Wikipedia)  134KB  2019-03-06 09:53:24           8779        130  0.9411765
datasets/jamesleslie/titanic-cleaned-data                    Titanic: cleaned data                           36KB  2018-11-21 11:50:18           4846         53  0.7647059
datasets/kittisaks/testtitanic                               test titanic                                    22KB  2017-03-13 15:13:12           1658         32  0.64705884
datasets/yasserh/titanic-dataset                             Titanic Dataset                                 22KB  2021-12-24 14:53:06           1011         25  1.0
datasets/abhinavralhan/titanic                               titanic                                         22KB  2017-07-30 11:07:55            628         11  0.8235294
datasets/cities/titanic123                                   Titanic Dataset Analysis                        22KB  2017-02-07 23:15:54           1585         29  0.5294118
datasets/brendan45774/gender-submisson                       Titanic: all ones csv file                      942B  2021-02-12 19:18:32            459         34  0.9411765
datasets/harunshimanto/titanic-solution-for-beginners-guide  Titanic Solution for Beginner's Guide           34KB  2018-03-12 17:47:06           1444         21  0.7058824
datasets/ibrahimelsayed182/titanic-dataset                   Titanic dataset                                  6KB  2022-01-27 07:41:54            334          8  1.0
datasets/sureshbhusare/titanic-dataset-from-kaggle           Titanic DataSet from Kaggle                     33KB  2017-10-12 04:49:39           2688         27  0.4117647
datasets/shuofxz/titanic-machine-learning-from-disaster      Titanic: Machine Learning from Disaster         33KB  2017-10-15 10:05:34           3867         55  0.29411766
datasets/vinicius150987/titanic3                             The Complete Titanic Dataset                   277KB  2020-01-04 18:24:11           1459         23  0.64705884
```

要下载列表中的第一个数据集，我们可以使用：

```py
kaggle datasets download -d heptapod/titanic --unzip
```

使用 Jupyter 笔记本来读取文件，类似于 Kaggle 笔记本示例，给我们提供了：

![](https://machinelearningmastery.com/wp-content/uploads/2022/04/jupyter_titanic.png)

在 Jupyter 笔记本中使用 Titanic 数据集

当然，某些数据集的大小非常大，您可能不希望将它们保留在自己的磁盘上。尽管如此，这是 Kaggle 提供的免费资源之一，供您的机器学习项目使用！

## 进一步阅读

此部分提供了更多资源，如果您对深入研究此主题感兴趣。

+   Kaggle: [`www.kaggle.com`](https://www.kaggle.com)

+   Kaggle API 文档：[`www.kaggle.com/docs/api`](https://www.kaggle.com/docs/api)

## 摘要

在本教程中，您学到了 Kaggle 是什么，我们如何使用 Kaggle 获取数据集，甚至在 Kaggle 笔记本中使用一些免费的 GPU/TPU 实例。您还看到了我们如何使用 Kaggle API 的 CLI 工具下载数据集，以便在本地环境中使用。

具体来说，您学到了：

+   什么是 Kaggle

+   如何在 Kaggle 笔记本中使用 GPU/TPU 加速器

+   如何在 Kaggle 笔记本中使用 Kaggle 数据集或使用 Kaggle 的 CLI 工具下载它们
