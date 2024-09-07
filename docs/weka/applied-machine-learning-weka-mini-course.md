# Weka 机器学习迷你课程

> 原文：<https://machinelearningmastery.com/applied-machine-learning-weka-mini-course/>

最后更新于 2021 年 2 月 2 日

### *14 天成为机器学习实践者*

机器学习是一项引人入胜的研究，但是你实际上是如何在自己的问题上使用它的呢？

你可能会对如何最好地为机器学习准备数据、使用哪些算法或者如何选择一个模型而不是另一个模型感到困惑。

在这篇文章中，你将发现一个由 14 部分组成的速成课程，使用 Weka 平台进行应用机器学习，没有一个数学方程或一行编程代码。

完成本迷你课程后:

*   您将知道如何端到端地处理数据集，并交付一组预测或高表现模型。
*   您将了解 Weka 机器学习工作台，包括如何探索算法和设计受控实验。
*   您将知道如何创建问题的多个视图，评估多种算法，并使用统计数据为您自己的预测建模问题选择最佳表现模型。

**用我的新书[用 Weka](https://machinelearningmastery.com/machine-learning-mastery-weka/) 启动你的项目**，包括*的分步教程*和清晰的*截图*所有示例。

我们开始吧。

(**提示** : Y *您可能想要打印或标记此页面，以便稍后可以参考它*)

![Applied Machine Learning With Weka Mini-Course](img/89415d9f17dafc6827aa020501510543.png)

Weka 迷你课程应用机器学习
图片由[利昂·亚科夫](https://www.flickr.com/photos/106447493@N05/15549957014/)提供，保留部分权利。

## 这个迷你课程是给谁的？

在我们开始之前，让我们确保你在正确的地方。下面的列表提供了一些关于本课程是为谁设计的一般指南。

不要惊慌，如果你没有完全匹配这些点，你可能只需要在一个或另一个领域刷起来就能跟上。

你是一个懂一点机器学习的开发者。

这意味着你知道一些机器学习的基础知识，比如交叉验证，一些算法和偏差方差的权衡。这并不意味着你是一个机器学习博士，只是你知道地标或知道在哪里可以找到它们。

这门迷你课程不是机器学习的教科书。

它将把你从一个知道一点机器学习的开发人员带到一个可以使用 Weka 平台从头到尾处理数据集并交付一组预测或高表现模型的开发人员。

## 迷你课程概述(期待什么)

这门迷你课程分为 14 个部分。

每节课都设计为用 30 分钟左右。你可能会更快地完成一些任务，而对于其他任务，你可能会选择做得更深入，花更多的时间。

你可以随心所欲地快速或缓慢地完成每一部分。一个舒适的时间表可能是在两周内每天完成一节课。强烈推荐。

在接下来的 14 课中，您将涉及以下主题:

*   **第 01 课**:下载并安装 Weka。
*   **第 02 课**:加载标准机器学习数据集。
*   **第 03 课**:描述性统计和可视化。
*   **第 04 课**:重新缩放数据。
*   **第 05 课**:对数据进行特征选择。
*   **第 06 课**:Weka 中的机器学习算法。
*   **第 07 课**:评估模型表现。
*   **第 08 课**:数据的基线表现。
*   **第 09 课**:分类算法。
*   **第十课**:回归算法。
*   **第 11 课**:集成算法。
*   **第 12 课**:比较算法的表现。
*   **第 13 课**:调整算法参数。
*   **第十四课**:保存你的模型。

这会很有趣的。

但是你必须做一些工作，一点阅读，一点在 Weka 的修补。你想开始应用机器学习，对吗？

(**提示** : *这些课的答案都可以在这个博客上找到，使用搜索功能*)

如有任何问题，请在下面的评论中发表。

在评论中分享你的结果。

坚持住，不要放弃！

## 第 01 课:下载并安装 Weka

首先要做的是在你的工作站上安装 Weka 软件。

Weka 是免费的开源软件。它是用 Java 编写的，可以在任何支持 Java 的平台上运行，包括:

*   窗户。
*   Mac OS X。
*   Linux。

您可以将 Weka 作为独立软件或与 Java 捆绑的版本下载。

如果您的系统上还没有安装 Java，我建议下载并安装一个与 Java 捆绑的版本。

1.  本课您的任务是访问 [Weka 下载页面](https://waikato.github.io/weka-wiki/downloading_weka/)，在您的工作站上下载并安装 Weka。

## 第 2 课:加载标准机器学习数据集

现在您已经安装了 Weka，您需要加载数据。

Weka 旨在以称为 ARFF 的本地格式加载数据。这是一种经过修改的 CSV 格式，其中包含有关每个属性(列)类型的附加信息。

您的 Weka 安装包括一个子目录，其中包含许多 ARFF 格式的标准机器学习数据集，您可以随时加载。

Weka 还支持从原始 CSV 文件和数据库加载数据，并根据需要将数据转换为 ARFF。

在本课中，您将在 Weka 浏览器中加载一个标准数据集。

1.  启动 Weka(点击鸟图标)，这将启动 Weka 图形用户界面选择器。
2.  点击“浏览器”按钮，这将打开 Weka 浏览器界面。
3.  单击“打开文件...”按钮，导航到您的 Weka 安装中的数据/目录，并加载 diabetes.arff 数据集。

注意，如果您的 Weka 安装中没有数据/目录，或者您找不到它，请下载。zip 版本的 Weka 从 [Weka 下载网页](https://waikato.github.io/weka-wiki/downloading_weka/)中，解压并访问数据/目录。

您刚刚在 Weka 中加载了第一个数据集。

尝试加载数据/目录中的一些其他数据集。

尝试从 [UCI 机器学习资源库](https://archive.ics.uci.edu/ml/)下载一个原始 CSV 文件，并将其加载到 Weka 中。

## 第 03 课:描述性统计和可视化

一旦您可以在 Weka 中加载数据，看一看它是很重要的。

Weka 允许您查看根据数据计算的描述性统计数据。它还提供可视化工具。

在本课中，您将使用 Weka 了解更多关于您的数据的信息。

1.  打开 Weka 图形用户界面选择器。
2.  打开 Weka 浏览器。
3.  加载 data/diabetes.arff 数据集。
4.  单击“属性”列表中的不同属性，并查看“选定属性”窗格中的详细信息。
5.  单击“全部可视化”按钮查看所有属性分布。
6.  单击“可视化”选项卡，查看所有属性的散点图矩阵。

轻松查看“预处理”选项卡中不同属性的详细信息，并在“可视化”选项卡中调整散点图矩阵。

## 第 04 课:重新缩放数据

原始数据通常不适合建模。

通常，您可以通过重新调整属性来提高机器学习模型的表现。

在本课中，您将学习如何在 Weka 中使用数据过滤器来重新缩放数据。您将规范化数据集的所有属性，将它们重新缩放到 0 到 1 的一致范围。

1.  打开 Weka 图形用户界面选择器，然后打开 Weka 浏览器。
2.  加载 data/diabetes.arff 数据集。
3.  点按“过滤器”面板中的“选择”按钮，然后选择“无监督”
4.  单击“应用”按钮。

查看“选定属性”窗格中每个属性的详细信息，并注意刻度的变化。

探索使用其他数据过滤器，如标准化过滤器。

通过单击加载的过滤器的名称并更改其参数，探索如何配置过滤器。

通过单击“预处理”选项卡上的“保存...”按钮，测试保存修改后的数据集供以后使用。

## 第 5 课:对数据执行要素选择

并非数据集中的所有属性都与您想要预测的属性相关。

您可以使用特征选择来识别那些与您的输出变量最相关的属性。

在本课中，您将熟悉使用不同的要素选择方法。

1.  打开 Weka 图形用户界面选择器，然后打开 Weka 浏览器。
2.  加载 data/diabetes.arff 数据集。
3.  单击“选择属性”选项卡。
4.  单击“属性评估器”窗格中的“选择”按钮，并选择“相关属性评估”。
    1.  您将看到一个对话框，要求您在使用此功能选择方法时根据需要更改为“Ranker”搜索方法。单击“是”按钮。
5.  单击“开始”按钮运行功能选择方法。

查看“属性选择输出”窗格中的输出，并记下每个属性的相关分数，数字越大表示相关特征越多。

探索其他特征选择方法，如使用信息增益(熵)。

探索在“预处理”选项卡和“移除”按钮中选择要从数据集中移除的要素。

## 第六课:Weka 的机器学习算法

Weka 工作台的一个主要好处是它提供了大量的机器学习算法。

你需要了解机器学习算法。

在本课中，您将详细了解 Weka 中的机器学习算法。

1.  打开 Weka 图形用户界面选择器，然后打开 Weka 浏览器。
2.  加载 data/diabetes.arff 数据集。
3.  单击“分类”选项卡。
4.  单击“选择”按钮，注意算法的不同分组。
5.  单击所选算法的名称进行配置。
6.  单击配置窗口上的“更多”按钮，了解有关实现的更多信息。
7.  单击配置窗口上的“功能”按钮，了解如何使用它的更多信息。
8.  请注意窗口上的“打开”和“保存”按钮，在这里可以保存和加载不同的配置。
9.  将鼠标悬停在配置参数上，并注意工具提示帮助。
10.  单击“开始”按钮运行算法。

浏览可用的算法。请注意，无论数据集是分类(预测类别)还是回归(预测实值)类型的问题，有些算法都不可用。

探索并了解更多关于 Weka 中可用的各种算法。

选择和配置算法获得信心。

## 第 07 课:评估模型表现

既然你知道如何选择和配置不同的算法，你就需要知道如何评估一个算法的表现。

在本课中，您将学习在 Weka 中评估计法表现的不同方法。

1.  打开 Weka 图形用户界面选择器，然后打开 Weka 浏览器。
2.  加载 data/diabetes.arff 数据集。
3.  单击“分类”选项卡。

“测试选项”窗格列出了可用于评估计法表现的各种不同技术。

*   黄金标准是 10 倍的“交叉验证”。默认情况下，这是选中的。对于小数据集，折叠次数可以从 10 次调整到 5 次，甚至 3 次。
*   如果数据集非常大，并且希望快速评估计法，可以使用“百分比分割”选项。默认情况下，该选项将在 66%的数据集上进行训练，并使用剩余的 34%来评估模型的表现。
*   或者，如果您有一个包含验证数据集的单独文件，您可以通过选择“提供的测试集”选项来评估您的模型。您的模型将在整个训练数据集上进行训练，并在单独的数据集上进行评估。
*   最后，您可以在整个训练数据集上评估模型的表现。如果您对描述性模型比对预测性模型更感兴趣，这将非常有用。

单击“开始”按钮，使用您选择的测试选项运行给定的算法。

尝试不同的测试选项。

通过单击“更多选项...”按钮，进一步细化配置中的测试选项。

## 第 08 课:数据的基准表现

当您开始评估数据集上的多种机器学习算法时，您需要一个比较基准。

基线结果为您提供了一个参考点，让您知道给定算法的结果是好是坏，以及好到什么程度。

在本课中，您将了解可以用作分类和回归算法基线的 ZeroR 算法。

1.  打开 Weka 图形用户界面选择器，然后打开 Weka 浏览器。
2.  加载 data/diabetes.arff 数据集。
3.  单击“分类”选项卡。默认情况下，选择零算法。
4.  单击“开始”按钮。

这将使用数据集上的 10 倍交叉验证来运行 ZeroR 算法。

零规则算法也称为零规则，是一种可以用来计算数据集上所有算法的表现基线的算法。这是“最差”的结果，任何表现更好的算法对你的问题都有一定的技巧。

在分类算法上，ZeroR 算法将总是预测最丰富的类别。如果数据集具有相同数量的类，它将预测第一个类别值。

在糖尿病数据集上，这导致 65%的分类准确率。

对于回归问题，ZeroR 算法将总是预测平均输出值。

在一系列不同的数据集上实验 ZeroR 算法。这是一种算法，你应该总是先运行，然后再开发基线。

## 第九课:分类算法之旅

Weka 提供了大量的分类算法。

在本课中，您将发现 5 种最常用的分类算法，可用于您的分类问题。

1.  打开 Weka 图形用户界面选择器，然后打开 Weka 浏览器。
2.  加载 data/diabetes.arff 数据集。
3.  单击“分类”选项卡。
4.  单击“选择”按钮。

可用于分类的 5 个顶级算法包括:

*   逻辑回归(函数。后勤)。
*   朴素贝叶斯。天真贝叶斯)。
*   k-最近邻居(懒惰。IBk)。
*   分类和回归树(树。REPTree)。
*   支持向量机(函数。SMO)。

尝试这些顶级算法。

在不同的类别数据集上尝试它们，比如有两个类的数据集和有更多类的数据集。

## 第 10 课:回归算法之旅

分类算法是 Weka 的专长，但这些算法中有很多都可以用于回归。

回归是对真实价值结果的预测(比如一美元金额)，不同于预测类别的分类(比如“狗”或“猫”)。

在本课中，您将发现 5 种最常用的回归算法，可用于您的回归问题。

可以从[Weka 数据集下载网页](https://sourceforge.net/projects/weka/files/datasets/)下载一套标准回归机器学习数据集。下载数据集-回归问题的 numeric.jar 档案，标题为:

*   一个包含 37 个回归问题的 jar 文件，从各种来源获得

用你最喜欢的解压程序解压。jar 文件，您将有一个名为 numeric/的新目录，其中包含 37 个可以处理的回归问题。

1.  打开 Weka 图形用户界面选择器，然后打开 Weka 浏览器。
2.  加载数据/housing.arff 数据集。
3.  单击“分类”选项卡。
4.  单击“选择”按钮。

可以用于衰退的 5 个顶级算法包括:

*   线性回归(函数。线性回归)。
*   支持向量回归(函数。SMOReg)。
*   k-最近邻居(懒惰。IBk)。
*   分类和回归树(树。REPTree)。
*   人工神经网络(函数。多边感知器)。

尝试这些顶级算法。

在不同的回归数据集上尝试它们。

## 第 11 课:集成算法之旅

Weka 非常容易使用，这可能是它相对于其他平台的最大优势。

除此之外，Weka 还提供了一大套集成机器学习算法，这可能是 Weka 相对于其他平台的第二大优势。

值得花时间去擅长使用 Weka 的集成算法。在这节课中，你将发现 5 个你可以使用的顶级集成机器学习算法。

1.  打开 Weka 图形用户界面选择器，然后打开 Weka 浏览器。
2.  加载 data/diabetes.arff 数据集。
3.  单击“分类”选项卡。
4.  单击“选择”按钮。

您可以使用的 5 种顶级集成算法包括:

*   装袋(meta。装袋)。
*   随机森林(树木。RandomForest)。
*   AdaBoost (meta。AdaBoost)。
*   投票(元。投票)。
*   堆叠(元。堆叠)。

尝试这些顶级算法。

大多数集成方法都允许您选择子模型。使用子模型的不同组合进行实验。以非常不同的方式工作并产生不同预测的技术组合通常会产生更好的表现。

在不同的分类和回归数据集上尝试它们。

## 第 12 课:比较算法的表现

Weka 提供了一种不同的工具，专门用于比较算法，称为 Weka 实验环境。

Weka 实验环境允许您使用机器学习算法设计和执行受控实验，然后分析结果。

在本课中，您将在 Weka 中设计您的第一个实验，并发现如何使用 Weka 实验环境来比较机器学习算法的表现。

1.  打开“Weka 选择器图形用户界面”。
2.  点击“实验者”按钮，打开“Weka 实验环境”。
3.  单击“新建”按钮。
4.  单击“数据集”窗格中的“添加新...”按钮，并选择“数据/糖尿病. arff”。
5.  单击“算法”窗格中的“添加新...”按钮，并添加“零”和“IBk”。
6.  单击“运行”选项卡，然后单击“开始”按钮。
7.  单击“分析”选项卡，单击“实验”按钮，然后单击“执行测试”按钮。

你刚刚设计、执行并分析了你在 Weka 的第一个受控实验的结果。

您将 ZeroR 算法与糖尿病数据集上默认配置的 IBk 算法进行了比较。

结果表明，IBK 的分类准确率高于零，并且这种差异具有统计学意义(结果旁边的小“v”字符)。

扩展实验，添加更多算法，重新运行实验。

更改“分析”选项卡上的“测试基准”以更改哪组结果被用作与其他结果进行比较的参考。

## 第 13 课:调整算法参数

为了充分利用机器学习算法，你必须根据你的问题调整方法的参数。

你不能事先知道如何最好地做到这一点，因此你必须尝试许多不同的参数。

Weka 实验环境允许您设计受控实验来比较不同算法参数的结果，以及差异是否具有统计学意义。

在本课中，您将设计一个实验来比较 k 近邻算法的参数。

1.  打开“Weka 选择器图形用户界面”。
2.  点击“实验者”按钮，打开“Weka 实验环境”
3.  单击“新建”按钮。
4.  单击“数据集”窗格中的“添加新...”按钮，并选择“数据/糖尿病. arff”。
5.  单击“算法”窗格中的“添加新……”按钮，添加 3 个“IBk”算法副本。
6.  单击列表中的每个 IBk 算法，然后单击“编辑所选...”按钮，将 3 种不同算法的“KNN”分别更改为 1、3、5。
7.  单击“运行”选项卡，然后单击“开始”按钮。
8.  单击“分析”选项卡，单击“实验”按钮，然后单击“执行测试”按钮。

您刚刚设计、执行并分析了一个受控实验的结果，以比较算法参数。

我们可以看到，大 K 值的结果优于默认值 1，差异显著。

探索改变 KNN 的其他配置属性，并在开发调整机器学习算法的实验中建立信心。

## 第 14 课:保存您的模型

一旦你在你的问题上找到了一个表现最好的模型，你需要最终确定它以备后用。

在最后一课中，您将了解如何训练最终模型，并将其保存到文件中以备后用。

1.  打开 Weka 图形用户界面选择器，然后打开 Weka 浏览器。
2.  加载 data/diabetes.arff 数据集。
3.  单击“分类”选项卡。
4.  将“测试选项”更改为“使用训练集”，然后单击“开始”按钮。
5.  右键单击“结果列表”中的结果，然后单击“保存模型”，并输入类似“糖尿病-最终”的文件名。

您刚刚在整个训练数据集上训练了一个最终模型，并将结果模型保存到一个文件中。

你可以将这个模型加载回 Weka，并使用它对新数据进行预测。

1.  右键单击“结果列表”，单击“加载模型”，并选择您的模型文件(“糖尿病-最终模型”)。
2.  将“测试选项”更改为“提供的测试集”，然后选择数据/糖尿病。arff(这可能是您没有预测的新文件)
3.  单击“测试选项”中的“更多选项”，并将“输出预测”更改为“纯文本”
4.  右键单击加载的模型，然后选择“在当前测试集上重新评估模型”。

新的预测现在将列在“分类器输出”窗格中。

尝试保存不同的模型，并对全新的数据集进行预测。

## Weka 迷你课程复习机器学习

恭喜你，你成功了。干得好！

花点时间回顾一下你已经走了多远:

*   你发现了如何启动和使用 Weka 探索者和 Weka 实验环境，也许是第一次。
*   您加载数据，分析数据，并使用数据过滤器和特征选择为建模准备数据。
*   你发现了一套机器学习算法，以及如何设计受控实验来评估它们的表现。

不要轻视这一点，你在短时间内已经取得了很大的进步。这只是您与 Weka 一起进行应用机器学习之旅的开始。不断练习和发展你的技能。

你喜欢这个迷你课程吗？你有什么问题或症结吗？
留言让我知道。