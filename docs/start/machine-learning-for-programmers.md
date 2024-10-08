# 面向程序员的机器学习

> 原文： [`machinelearningmastery.com/machine-learning-for-programmers/`](https://machinelearningmastery.com/machine-learning-for-programmers/)

## 从开发人员跳到
机器学习从业者

或者，我对这个问题的回答：

### _ 如何开始机器学习？_

> 我是开发人员。我读过一本关于机器学习的书或一些帖子。我看过一些 Coursera 机器学习课程。我还是不知道怎么开始......

这听起来很熟悉吗？

![Machine Learning Frustration](img/13ece4eae4e3eed65ff08dc886a5d217.jpg)

对机器学习书籍和课程感到沮丧？
你如何开始机器学习？
摄影： [Peter Alfred Hess](https://www.flickr.com/photos/peterhess/2976755407/) ，保留一些权利

开发人员在我的时事通讯中询问的最常见问题是：

> _ 我如何开始机器学习？_

老实说我记不清已经回答了多少次。

在这篇文章中，我列出了关于这个主题的所有最好的想法。

*   您将了解为什么传统的机器学习教学方法不适合您。
*   您将了解如何翻转整个模型。
*   你会发现我可以用来开始的简单但非常有效的解毒剂。

我们进入吧......

## 对机器学习感兴趣的开发人员

您是一名开发人员，您对进入机器学习很感兴趣。那么为何不？这是目前的单热门话题，它是一个迷人而快速发展的领域。

你看过一些博文。你试图深入，但书籍很糟糕。数学专注。理论重点。算法集中。

![Machine Learning for Programmers - How Do I Get Started](img/38706a9d5ba22355859d83c7bf2a2fa3.jpg)

听起来有点熟？您是否尝试过书籍，MOOC，博客文章，还不知道如何开始机器学习？

你尝试一些视频课程。您注册并诚实地尝试了经常被引用的 [Coursera Stanford 机器学习 MOOC](https://www.coursera.org/learn/machine-learning) 。它并不比书籍和详细的博客文章好多少。你无法看到所有的大惊小怪，为什么建议初学者。

您甚至可能尝试过一些小型数据集，可能是入门级的 Kaggle 竞赛。

问题是你无法将书籍和课程中的理论，算法和数学与问题联系起来。这是一个巨大的差距。海湾。 **你应该如何开始机器学习？**

## 机器学习工程师

当你想到未来时，一旦你掌握了对机器学习这种难以理解的理解，你的工作会是什么样子？你是如何在日常生活中使用新发现的机器学习技能的？

我想我能看到它。你是一名机器学习工程师。你是一个知道如何“做”机器学习的开发人员。

![Machine Learning for Programmers - Dream](img/1716f15e7c733f6be44142d8ee955a26.jpg)

您想从开发人员转变为可以进行机器学习的开发人员吗？

### 场景 1：一次性模型

你的老板走过去说：

> 嘿，你知道机器学习，对吗？您能否使用去年的客户数据来预测销售渠道中的哪些客户可能转换？我想在下周的董事会演讲中使用它......

我称之为一次性模型。

你的老板很好地定义了这个问题。她为您提供了数据，如果必须的话，这些数据足够小，可以在 MS Excel 中查看和理解。她想要准确可靠的预测。

你可以交付。更重要的是，您可以解释结果的所有相关警告。

### 场景 2：嵌入式模型

您和您的团队正在收集利益相关方对软件项目的要求。要求用户能够在软件中徒手绘制形状，并且软件要确定其形状，并将其转换为清晰明确的版本并对其进行适当标记。

您很快就会发现解决此问题的最佳（也是唯一可行的？）方法是设计和训练预测模型并将其嵌入到您的软件产品中。

我称之为嵌入式模型。有变化（例如模型是静态的还是更新的，以及它是本地的还是通过 API 远程调用），但这只是细节。

在这种情况下，关键在于您有经验可以注意到最好用预测模型解决的问题以及设计，训练和部署它的技能。

### 场景 3：深层模型

您已经开始了新工作，而您正在处理的系统至少由一个预测模型组成。维护和添加功能需要了解模型，输入和输出。模型的准确率是软件产品的一个特征，您的部分工作将是改进它。

例如，作为常规预发布系统测试的一部分，您必须证明模型的准确率（在历史数据上验证时）具有与先前版本相同或更好的技能。

我称之为深层模型。您将需要深入了解一个特定的预测模型，并使用您的经验和技能来改进和验证其准确率，作为日常职责的一部分。

### 开发人员“_ 做 _”机器学习

这些场景让您了解成为一名知道如何进行机器学习的开发人员的感受。它们是现实的，因为它们都是我所处过的场景或我必须完成的任务的变化。

所有这三种情况都使一件事非常明确。虽然机器学习是一个迷人的领域，但对于开发人员来说，机器学习算法只是另一种技巧，如多线程或三维图形编程。然而，它们是一组强大的方法，对于特定类别的问题是绝对必需的。

## 传统答案：“_ 我该如何开始？_ “

那么你如何开始机器学习呢？

如果你破解一本关于机器学习的书来寻求这个问题的答案，你会感到震惊。他们从定义开始，继续进行不断增加的复杂性的概念和算法的数学描述。

![Machine Learning for Programmers - The Traditional Approach](img/3d4804faf74416912920d2718801153e.jpg)

“如何开始机器学习”这一问题的传统答案是自下而上的。

定义和数学描述清晰，简洁且通常是明确的。问题是，它们干燥，乏味，需要必要的数学背景来解析和解释。

机器学习通常被教授为大学的研究生课程，这是有原因的。这是因为这种“第一原则”教学方式需要多年的理解前提条件。

例如，建议你有一个良好的基础：

*   统计
*   可能性
*   线性代数
*   多变量统计
*   结石

如果你稍微偏离一些更奇特和有趣的算法，这会变得更糟。

这种自下而上和算法固定的机器学习方法是普遍存在的。

在线课程，MOOC 和 YouTube 视频模仿大学教学机器学习的方法。再次，如果你有背景，或者你已经完成了半年到十年的学习以获得更高的学位，那么这很好。它对普通开发人员没有帮助。

如果你躲到像 [Quora](http://www.quora.com/How-do-I-learn-machine-learning-1) ， [StackExchange](http://programmers.stackexchange.com/questions/79476/what-skills-are-needed-for-machine-learning-jobs) 或 [Reddit](https://www.reddit.com/r/learnprogramming/comments/3d5ajk/where_to_start_with_machine_learning/) 这样的问题和答案论坛，并温顺地问如何开始，你会被同样的回应打耳光。通常这种反应来自同样失败的开发人员。这是一个同样糟糕建议的大打印室。

难怪诚实和勤奋的开发人员想要做正确的事情，认为他们必须回到学校并获得硕士或博士学位。在他们觉得有资格“做”机器学习之前。

## 传统方法是错误的！

想想这种自下而上的方法来教授机器学习一秒钟。它严谨而系统，听起来就像是表面上的正确理念。怎么会错？

### 自下而上编程（或者，如何杀死新手程序员）

想象一下，你是一个年轻的开发者。您已经学习了一些这种语言，并开始学习如何创建独立软件。

你告诉朋友和家人你想进入一个每天都要开始编程的职业。他们告诉你，你需要先获得计算机科学学位才能获得程序员的工作。

您注册并开始计算机科学学位。学期开学后，你会接触到越来越多的深奥代数，微积分和离散数学。您使用过时的编程语言。您对编程和构建软件波动的热情。

![Machine Learning for Programmers - Gap in Bottom Up](img/bde9d62e07471463b18a651591ec25c6.jpg)

开始进行机器学习的传统方法在从业者的道路上存在差距。

也许你以某种方式把它带到另一边。回顾过去，您意识到您没有学过现代软件开发实践，语言，工具或您在创建和交付软件时可以使用的任何东西。

看到机器学习教学的相似之处？

值得庆幸的是，编程已经存在了很长时间，足够受欢迎并且对于经济而言非常重要，我们已经找到了其他方法来为初露头角的年轻（或旧）程序员提供他们实际做他们想要做的事情所需的技能 - 例如创建软件。

用[可计算性](https://en.wikipedia.org/wiki/Computability_theory)或[计算复杂度](https://en.wikipedia.org/wiki/Computational_complexity_theory)的理论来加载一个萌芽程序员的头脑是没有意义的，甚至是算法和数据结构的深层细节。一些有用的材料（后者在算法复杂性和数据结构上）可以在以后出现。也许是有重点的材料 - 但重要的是在已经编程和交付软件的工程师的背景下，而不是孤立的。

谢天谢地，我们专注于软件工程学位。我们也有像 [codecademy](https://www.codecademy.com/) 这样的资源，你可以学习编程......是的，实际编程。

如果一个开发人员想“做”机器学习，他们是否真的不得不花费大量时间和数十或数十万美元来获得必要的数学和更高学位？

答案当然不是！有一个更好的办法。

## 更好的方法

与计算机科学一样，您不能只是颠倒模型并自上而下地教授相同的材料。

原因在于，就像计算机科学课程从未涉及到涵盖开发和交付软件的实际问题的科目，机器学习课程和书籍都很短。他们停在算法上。

您需要一种自上而下的机器学习方法。一种方法，您可以专注于您想要的实际结果：使用现代和“_ 最佳 _”工具和平台从端到端工作真正的机器学习问题。

![Machine Learning for Programmers - A Better Approach](img/3cf413c8473531f22e5e8c6fc486f4a7.jpg)

学习机器学习的更好方法，从端到端的工作机器学习问题开始。

这就是我认为你的黄砖路看起来像。

### 1.系统过程的可重复结果

一旦你知道了一些工具，用机器学习算法来解决问题并称之为“_ 完成 _”相对容易。

这可能很危险。

你怎么知道你做完了？你怎么知道结果有什么好处？您如何知道数据集上的结果是否可靠？

在处理机器学习问题时，您需要系统化。这是一个项目，就像一个软件项目，良好的流程可以实现从项目到项目的可重复的高质量结果。

考虑这样一个过程，您可以考虑一些明确的要求，例如：

*   指导您从端到端，从问题规范到结果的呈现或部署的过程。就像一个软件项目，你可以认为你已经完成了，但你可能不是。从一开始就考虑到最终可交付成果设定了明确的项目停止条件并集中精力。
*   一步一步的过程，以便您始终知道下一步该做什么。不知道下一步该做什么是一个项目杀手。
*   保证“_ 良好 _”结果的过程，例如优于平均水平或足以满足项目需求。项目需要以已知的置信水平可靠地提供良好的结果是非常常见的，不一定是可能的最佳精度。
*   一个对特定工具，编程语言和算法时尚不变的过程。工具来去匆匆，过程必须是自适应的。考虑到该领域的算法痴迷，学术界总会出现新的强大算法。

![Machine Learning for Programmers - Select a Systematic Process](img/013ddf5e4f3158bbd6eaf42bc199a76d.jpg)

选择一个系统且可重复的流程，您可以使用该流程始终如一地提供结果。

有很多很棒的流程，包括一些可以适应您需求的旧流程。

例如：

*   [数据库中的知识发现](http://machinelearningmastery.com/what-is-data-mining-and-kdd/)（KDD），我在这里改编的[](http://machinelearningmastery.com/process-for-working-through-machine-learning-problems/)
*   [CRISP-DM](https://en.wikipedia.org/wiki/Cross_Industry_Standard_Process_for_Data_Mining)
*   [OSEMN](http://machinelearningmastery.com/how-to-work-through-a-problem-like-a-data-scientist/)
*   其他…

选择或调整最适合您的流程并满足上述要求。

### 2.将“_ 最佳品种 _”工具映射到您的过程中

机器学习工具和库来来往往，但在任何单一时间点，您都必须使用最佳映射到您选择的交付结果的过程。

您不想评估和选择任何旧算法或库，您希望所谓的“_ 最佳 _”能够为您提供快速，可靠和高质量的结果并自动化您可以负担得起的流程

同样，您将不得不自己做出这些选择。如果你问任何人，你会听到他们的偏见，通常是他们正在使用的最新工具。

我有自己的偏见，我喜欢使用不同的工具和平台来完成不同类型的工作。

例如，在上面列出的场景中，我会建议以下最好的工具：

*   **一次性预测模型**： [Weka 平台](http://machinelearningmastery.com/how-to-run-your-first-classifier-in-weka/)，因为我可以加载 CSV，设计实验并在没有编程的情况下立即获得最佳模型（ []看到我对过程的映射](http://machinelearningmastery.com/template-for-working-through-machine-learning-problems-in-weka/)）。
*   **嵌入式预测模型**：Python 与 [scikit-learn](http://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/) ，因为我可以用它所部署的同一语言开发模型。 IPython 是向更广泛的团队展示您的管道和结果的好方法。 MLaaS 也是更大数据的选择。
*   **深潜模型**： [R](http://machinelearningmastery.com/how-to-get-started-with-machine-learning-algorithms-in-r/) 与[插入符号](http://machinelearningmastery.com/caret-r-package-for-applied-predictive-modeling/)，因为我可以快速自动地尝试很多最先进的模型并设计更多使用整个 R 平台进行更精细的特征选择，特征工程和算法调整实验。

实际上，根据情况的具体情况，这三种工具会在三种情况下流失。

![Machine Learning for Programmers - Select Tools](img/e8ee3b384634f311cc805035a204e77b.jpg)

将您首选的机器学习工具映射到您选择的系统过程，以解决问题。

与开发一样，您需要研究工具以从中获得最大收益。您还需要将耳朵贴近地面，并在可用时跳转到更新的更好的工具，永远适应您的可重复过程。

### 3.针对半正式工作产品的实践

通过练习 - 通过开发大量软件，您可以擅长开发。使用这种熟悉的方法来获得机器学习。您在每个项目中练习的过程越多越好（理想情况下，端到端的工作问题）。

#### 仔细选择您的练习数据集

您想要选择现实而非人为的数据集。有数百个免费数据集，其复杂性不断增加。

*   我建议从 [UCI 机器学习库](http://machinelearningmastery.com/practice-machine-learning-with-small-in-memory-datasets-from-the-uci-machine-learning-repository/)等地方的小型内存数据集开始。它们众所周知，相对干净，是您感受新工艺和工具的好地方。
*   从那里，我会推荐更大的内存数据集，比如来自某些 [Kaggle](https://www.kaggle.com/) 和 [KDD 杯](http://www.sigkdd.org/kddcup/index.php)比赛的数据集。它们更脏一些，需要你在更多不同的技能上进行工作。

坚持表格数据，这是我建议所有学生。

处理图像和文本数据本身就是新的和不同的领域（分别是计算机视觉和自然语言处理），需要您学习这些领域的专门方法和工具。如果这些是你想要或需要工作的问题类型，那么最好从那里开始，并且有很多可用的资源。

我将详细介绍如何在“[练习机器学习与 UCI 机器学习库](http://machinelearningmastery.com/practice-machine-learning-with-small-in-memory-datasets-from-the-uci-machine-learning-repository/)中的小内存数据集”中进行有针对性的练习。

#### 记录您的结果并建立公共工作组合

从每个项目中创建并保留半正式结果（我将结果称为“工作产品”）。我的意思是，将您所做的和您学到的内容写入某种独立文档，以便您可以回顾并利用未来和后续项目的结果。

这类似于为每个编程项目保留一个目录作为开发人员，并重用以前项目中的代码和想法。它加速了很多旅程，我强烈推荐它。

保留任何脚本，代码和生成的图像，但写出您的发现也很重要。可以认为它类似于代码中的注释。独立的报道可以是简单的 PPT 或文本文件，也可以是在 YouTube 上的聚会或视频中精心制作的演示文稿。

![Machine Learning for Programmers - Targeted Practice](img/91c15cda983c7981bb761821363eacf9.jpg)

完成并完成离散项目，编写结果并构建项目组合。

将每个项目保存在公共版本控制存储库（如 [GitHub](https://github.com/) ）中，以便其他初学者可以向您学习并扩展您的工作。从您的博客，LinkedIn 或任何地方链接到项目，并使用公共投资组合来展示您不断增长的技能和能力。

在我的帖子“[构建机器学习组合：完成小型项目并展示您的技能](http://machinelearningmastery.com/build-a-machine-learning-portfolio/)”中查看有关这一重要想法的更多信息。

公共 GitHub 存储库组合正迅速成为实际关注技能和交付成果的公司招聘过程中的简历。

## 是的，这种方法是为开发人员量身定制的

我们上面列出的是一种可以作为开发人员学习，开始并在机器学习方面取得进步的方法。

你很可能对这种方法是否真的适合你有所怀疑。让我谈谈你的一些担忧。

### 您不需要编写代码

您可能是一名 Web 开发人员或类似的人，您不会编写大量代码。您可以使用此方法开始并应用机器学习。像 [Weka](http://machinelearningmastery.com/how-to-run-your-first-classifier-in-weka/) 这样的工具可以轻松地设计机器学习实验并构建模型而无需任何编程。

编写代码可以解锁更多不同的工具和功能，但这不是必需的，并且不需要先行。

### 你不需要擅长数学

就像开发中你不需要了解可计算性或 [Big O 符号](https://en.wikipedia.org/wiki/Big_O_notation)编写代码并发布有用且可靠的软件一样，你可以在没有背景的情况下端到端地解决机器学习问题在统计学，概率和线性代数中。

重要的是要注意，我们不是从理论开始，但我们不会忽视它。在需要时潜入并提取您需要的方法或算法。事实上，你将无法阻挡自己。原因是，工作机器学习问题令人上瘾和消费。为了获得更好的结果和更准确的预测，您将从您可以找到的任何资源中汲取经验，学习足够的知识，为您提供适用于您的问题的智慧。

如果您的目标是掌握理论，那么这种方法就会变慢，效率也会降低。这就是为什么当透过那个镜头看时它如此不舒服。从作为开发人员进行机器学习的目标来看，它很有意义。

### 你不需要更高的学位

这种知识没有看门人。这一切都可用，你现在可以自己研究它。在开始处理机器学习问题之前，您不需要花费大量时间和金钱来获得学位。

如果你的心得到了更高的学位，为什么不首先开始研究机器学习问题，并在你完成一小部分已完成项目后的几周或几个月内查看一个学位。您将更清楚地了解该领域的范围和您喜欢的部分。

我确实回过头来获得更高的学位。我喜欢做研究，但我喜欢处理真正的问题并提供客户真正关心的结果。另外，在我开始学位之前，我正在研究机器学习问题，我只是没有意识到我已经拥有了资源和一条路在我面前。

这是我热衷于说服像你这样的开发人员你有你现在需要的东西的原因之一。

![Machine Learning for Programmers - Limiting Beliefs2](img/810e529baae63a22dbe63ed5f4d355ff.jpg)

很容易找到借口不开始机器学习。

### 您不需要大数据

开发了机器学习算法，并且可以最好地理解小数据。数据足够小，您可以在 MS Excel 中查看，加载到内存中并在桌面工作站上完成工作。

大数据！=机器学习。您可以使用大数据构建预测模型，但将此视为您对域的技能集的专业化。我通常建议我的学生在开始机器学习时从小内存数据集开始。

如果大数据机器学习是您想要工作的领域，那么从那里开始。

### 您不需要桌面超级计算机

确实，像深度学习这样的一些最先进的算法需要非常强大的万亿核 GPU。它们是强大且令人兴奋的算法。它们也是可以解决您可以使用桌面 CPU 计算的较小问题的算法。

在您访问大型计算机之前，您不需要推迟开始机器学习。

在您购买台式超级计算机或租用非常大的 [EC2 实例](https://aws.amazon.com/)之前，花一些时间学习如何在更小的更好理解的数据集上充分利用这些算法可能是值得的。

### 你不需要很多时间

我们都有忙碌的生活，但如果你真的想要一些你需要投入的时间。

我以前说过，工作机器学习问题让人上瘾。如果你陷入机器学习竞赛，你会很乐意牺牲一个月的晚间电视来从你的算法中挤出几个百分点。

话虽这么说，如果你从一个清晰的过程和一流的工具开始，你可以在一两个小时内完成一个端到端的数据集，可能会分散一两个晚上。其中一些和您在已完成的机器学习项目组合中有一个滩头阵地，您可以开始利用更大和更有趣的问题。

将其分解为 Kanban 板上的零食大小任务，并留出时间开始。

## 开发人员犯下的最大错误以及如何避免它们

自从我推出机器学习掌握以来，我已经提供了近两年的建议。在那段时间里，我看到了五个我希望你避免的常见陷阱。

1.  **没有采取行动**：这一切都已经布局，但我看到很多开发者没有采取行动。观看电视或阅读新闻要比在迷人的研究领域建立新的有价值的技能容易得多。你可以带马到水......
2.  **挑选太大的问题**：我经常看到开发人员选择处理太难的第一个或第二个数据集。它太大，太复杂或太脏，而且还没有为迎接挑战做好准备。可怕的是，“失败”会扼杀动机，而开发人员也会离开这个过程。选择你可以完成并在 60 分钟内写完的小问题。在你采取更大的事情之前做一段时间。
3.  **从零开始实现算法**：我们有算法实现。完成。至少做得足以让你在接下来的几年做有趣的事情。如果您的目标是学习如何开发和提供可靠和准确的预测模型，那么不要花时间从零开始实现算法，使用库。另一方面，如果你想专注于实现算法，那么明确地将它作为你的目标并专注于它。
4.  **不坚持流程**：与敏捷软件开发一样，如果您偏离流程，轮子可能会很快脱离您的项目，结果往往是一团糟。从头到尾坚持从头到尾系统地解决问题的过程是关键。您可以重温“_ 您发现的那个有趣的东西......_ ”作为后续迷你项目（后续工作的 _ 想法 _“在你的写作中），但是完成流程并交付。
5.  **不使用资源**：有许多关于机器学习的优秀论文，书籍和博客文章。您可以利用这些资源来改进流程，工具的使用和结果的准确率。使用第三方资源从算法和数据集中获取更多信息。获取有关问题的算法和框架的想法。智慧的金块可以改变你的项目进程。请记住，如果采用自上而下的过程，理论必须在后端进行。花点时间了解您的最终模型。

不要让任何这些事发生在你身上！

## 你的下一步

我们已经涵盖了很多方面，我希望我开始说服你，你可以开始并在机器学习方面取得进步。您是一名可以进行机器学习的开发人员的未来是非常真实且非常可获得的。

您接下来的步骤是：

1.  选择一个过程（[或只使用此过程](http://machinelearningmastery.com/process-for-working-through-machine-learning-problems/)）。
2.  选择一个工具或平台（[或只使用这个](http://machinelearningmastery.com/how-to-run-your-first-classifier-in-weka/)）。
3.  选择你的第一个数据集（[或只使用这个](https://archive.ics.uci.edu/ml/datasets/Iris)）。
4.  在下面的评论中报告并执行！

嘿，你觉得这篇文章有用吗？发表评论！

**更新**：查看这个方便的思维导图，总结了这篇文章中的重要概念（感谢 Simeon 的建议！）。

![Machine Learning For Programmers Mind Map](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2015/08/Machine-Learning-For-Programmers-Mind-Map.png)

手绘思维导图，总结了本文中的重要概念。
[点击图片放大]