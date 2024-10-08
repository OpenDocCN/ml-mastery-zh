# 对抗机器学习数据集中不平衡类别的 8 种策略

> 原文： [`machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/`](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)

这件事发生在你身上吗？

您正在处理数据集。您可以创建分类模型并立即获得 90％的准确度。你觉得“很棒”。你深入一点，发现 90％的数据属于一个类。该死的！

这是一个不平衡数据集的例子，它可能导致令人沮丧的结果。

在这篇文章中，您将发现可用于在具有不平衡数据的机器学习数据集上提供出色结果的策略。

![Class Imbalance](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2015/08/Class-Imbalance.jpg)

在机器学习中找到一些平衡点。
摄影：MichaEli，保留一些权利。

## 掌握不平衡的数据

我一直收到有关课程不平衡的电子邮件，例如：

> 我有一个二分类问题，在我的训练集中有一个类与 60：1 的比例。我使用了逻辑回归，结果似乎只是忽略了一个类。

还有这个：

> 我正在研究分类模型。在我的数据集中，我有三个不同的标签要分类，让它们分别为 A，B 和 C.但是在训练数据集中，我有一个 70％体积的数据集，B 为 25％，C 为 5％。大部分时间我的结果都适合 A.你能否建议我如何解决这个问题？

我写了很长的技术列表来尝试思考解决这个问题的最佳方法。我终于接受了我的一个学生的建议：

> 也许您即将发布的一篇博文可以解决训练模型以解决高度不平衡数据的问题，并概述一些技术和期望。

## 挫折！

不平衡的数据可能会让您感到很沮丧。

当你发现你的数据有不平衡的类并且你认为你得到的所有好结果都变成了谎言时，你会感到非常沮丧。

当书籍，文章和博客文章似乎没有为您提供有关处理数据不平衡的良好建议时，下一波挫折就会出现。

放松，有很多选择，我们将全面介绍它们。您可以为不平衡数据构建预测模型。

## 什么是不平衡数据？

不平衡数据通常是指分类问题的问题，其中类没有被平等地表示。

例如，您可能有一个包含 100 个实例（行）的 2 类（二进制）分类问题。总共 80 个实例标记为 Class-1，其余 20 个实例标记为 Class-2。

这是一个不平衡的数据集，Class-1 与 Class-2 实例的比例为 80:20 或更简洁，为 4：1。

对于两类分类问题以及多分类问题，您可能会遇到类不平衡问题。大多数技术都可以使用。

剩下的讨论将假设一个两类分类问题，因为它更容易思考和描述。

### 不平衡是常见的

大多数分类数据集在每个类中没有完全相同数量的实例，但是小的差异通常无关紧要。

有些问题是阶级不平衡不仅仅是常见的，而是预期的。例如，在那些表征欺诈性交易的数据集中，这些数据集是不平衡的。绝大多数交易将在“非欺诈”类中进行，而极少数交易将在“欺诈”类中。

另一个例子是客户流失数据集，绝大多数客户都使用该服务（“No-Churn”类），少数客户取消订阅（“Churn”类）。

如果上面的示例中存在类似 4：1 的适度类不平衡，则可能会导致问题。

### 准确率悖论

[准确率悖论](https://en.wikipedia.org/wiki/Accuracy_paradox)是本文简介中确切情况的名称。

在这种情况下，您的准确度测量可以说明您具有出色的准确率（例如 90％），但准确率仅反映了潜在的阶级分布。

这是非常常见的，因为分类准确率通常是我们在评估分类问题模型时使用的第一个衡量标准。

### 把它全部放在红色！

当我们在不平衡的数据集上训练时，我们的模型中发生了什么？

正如您可能已经猜到的那样，我们在不平衡数据上获得 90％准确度的原因（在 Class-1 中有 90％的实例）是因为我们的模型会查看数据并巧妙地决定最好的做法是始终预测“1 级”并实现高精度。

使用简单的基于规则的算法时最好看到这一点。如果在最终模型中打印出规则，您将看到它很可能预测一个类，而不管它被要求预测的数据。

## 打击不平衡训练数据的 8 种策略

我们现在了解什么是类不平衡，以及它为什么提供误导性的分类准确率。

那么我们有什么选择呢？

### 1）你能收集更多数据吗？

您可能认为这很愚蠢，但收集更多数据几乎总是被忽视。

你能收集更多数据吗？花点时间考虑一下您是否能够收集有关问题的更多数据。

较大的数据集可能会在类上显示不同且可能更平衡的视角。

稍后当我们查看重采样数据集时，更多次要类的示例可能会有用。

### 2）尝试更改表现指标

准确率不是使用不平衡数据集时使用的指标。我们已经看到它具有误导性。

在使用不平衡类别时，有一些指标可以告诉您更真实的故事。

在我的帖子“[分类准确率不够：可以使用的更多表现指标](http://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/)”中，我提供了更多关于选择不同表现指标的建议。

在那篇文章中，我看了一个不平衡的数据集，它描述了患者乳腺癌复发的特征。

从那篇文章中，我建议查看以下表现测量，这些测量可以比传统的分类准确度更深入地了解模型的准确率：

*   **混淆矩阵**：将预测分解为表格，显示正确的预测（对角线）和不正确的预测类型（分配了哪些类别的错误预测）。
*   **精度**：分类器精确度的度量。
*   **召回**：分类器完整性的度量
*   **F1 分数（或 F 分数）**：精确度和召回率的加权平均值。

我还建议你看一下以下内容：

*   **Kappa（或 [Cohen 的 kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa) ）**：分类准确率由数据中类别的不平衡归一化。
*   **ROC 曲线**：与精确度和召回率一样，精度分为灵敏度和特异性，可以根据这些值的平衡阈值选择模型。

您可以学习更多关于使用 ROC 曲线来比较我们的帖子“[评估和比较分类器表现与 ROC 曲线](http://machinelearningmastery.com/assessing-comparing-classifier-performance-roc-curves-2/)”的分类准确率。

还不确定吗？从 kappa 开始，它将让您更好地了解正在发生的事情，而不是分类准确率。

### 3）尝试重采样数据集

您可以更改用于构建预测模型的数据集，以获得更平衡的数据。

此更改称为对数据集进行采样，您可以使用两种主要方法来均衡类：

1.  您可以从代表性不足的类中添加实例的副本，称为过采样（或更正式的替换采样），或
2.  您可以从过度表示的类中删除实例，称为欠采样。

这些方法通常很容易实现并且运行速度快。他们是一个很好的起点。

事实上，我建议你总是在所有不平衡数据集上尝试这两种方法，只是为了看看它是否能提高你的首选精确度量。

您可以在 Wikipedia 文章“[过采样和数据分析中的欠采样](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis)”中学到更多内容。

#### 一些经验法则

*   当您拥有大量数据（数十或数十万个实例或更多）时，请考虑对欠采样进行测试
*   当您没有大量数据（数万条记录或更少）时，请考虑测试过采样
*   考虑测试随机和非随机（例如分层）采样方案。
*   考虑测试不同的重采样比率（例如，您不必在二分类问题中以 1：1 的比例为目标，尝试其他比率）

### 4）尝试生成合成样品

生成合成样本的一种简单方法是从少数类中的实例中随机采样属性。

您可以在数据集中凭经验对它们进行采样，或者您可以使用像 Naive Bayes 这样的方法，可以在反向运行时独立地对每个属性进行采样。您将拥有更多不同的数据，但可能无法保留属性之间的非线性关系。

您可以使用系统算法生成合成样本。最流行的此类算法称为 SMOTE 或合成少数过采样技术。

顾名思义，SMOTE 是一种过采样方法。它的工作原理是从次要类创建合成样本，而不是创建副本。该算法选择两个或更多个类似的实例（使用距离测量）并且通过差异内的相邻实例的随机量一次扰动实例一个属性。

了解有关 SMOTE 的更多信息，请参阅 2002 年原创题为“ [SMOTE：合成少数过采样技术](http://www.jair.org/papers/paper953.html)”的论文。

SMOTE 算法有许多实现，例如：

*   在 Python 中，看一下“ [UnbalancedDataset](https://github.com/fmfn/UnbalancedDataset) ”模块。它提供了许多 SMOTE 实现以及您可以尝试的各种其他重采样技术。
*   在 R 中， [DMwR 包](https://cran.r-project.org/web/packages/DMwR/index.html)提供了 SMOTE 的实现。
*   在 Weka 中，您可以使用 [SMOTE 监督滤波器](http://weka.sourceforge.net/doc.packages/SMOTE/weka/filters/supervised/instance/SMOTE.html)。

### 5）尝试不同的算法

与往常一样，我强烈建议您不要在每个问题上使用您喜欢的算法。您应该至少在给定问题上对各种不同类型的算法进行采样检查。

有关点检查算法的更多信息，请参阅我的文章“为什么你应该在机器学习问题上进行现场检查算法”。

话虽这么说，决策树通常在不平衡的数据集上表现良好。查看用于创建树的类变量的拆分规则可以强制解决这两个类。

如果有疑问，请尝试一些流行的决策树算法，如 C4.5，C5.0，CART 和随机森林。

对于使用决策树的一些 R 代码示例，请参阅我的帖子“带有决策树的 R 中的[非线性分类”。](http://machinelearningmastery.com/non-linear-classification-in-r-with-decision-trees/)

有关在 Python 和 scikit-learn 中使用 CART 的示例，请参阅我的帖子标题为“[让你的手弄乱 Scikit-Learn Now](http://machinelearningmastery.com/get-your-hands-dirty-with-scikit-learn-now/) ”。

### 6）尝试惩罚模型

您可以使用相同的算法，但为他们提供不同的视角。

惩罚分类会对模型造成额外成本，以便在训练期间对少数群体犯下分类错误。这些处罚可能会使模型偏向于更多地关注少数民族。

通常，类惩罚或权重的处理专用于学习算法。存在惩罚版本的算法，例如惩罚的 SVM 和惩罚的 LDA。

也可以为惩罚模型提供通用框架。例如，Weka 有一个 [CostSensitiveClassifier](http://weka.sourceforge.net/doc.dev/weka/classifiers/meta/CostSensitiveClassifier.html) ，可以包装任何分类器并应用自定义惩罚矩阵进行未命中分类。

如果您被锁定在特定算法中并且无法重新取样或者您的结果不佳，则需要使用惩罚。它提供了另一种“平衡”类的方法。设置惩罚矩阵可能很复杂。您很可能必须尝试各种惩罚方案，看看什么最适合您的问题。

### 7）尝试不同的观点

有专门针对不平衡数据集的研究领域。他们有自己的算法，措施和术语。

从这些角度看一看并思考你的问题有时会让一些想法失去理智。

您可能要考虑的两个是**异常检测**和**变化检测**。

[异常检测](https://en.wikipedia.org/wiki/Anomaly_detection)是罕见事件的检测。这可能是由于其振动或由系统调用序列指示的程序的恶意活动而指示的机器故障。与正常操作相比，这些事件很少见。

思维的这种转变将次要类别视为异常类，这可能有助于您考虑分离和分类样本的新方法。

[变化检测](https://en.wikipedia.org/wiki/Change_detection)类似于异常检测，除了寻找异常之外，它正在寻找变化或差异。这可能是使用模式或银行交易所观察到的用户行为的变化。

这两种转变都会对分类问题采取更实时的立场，这可能会为您提供一些思考问题的新方法，也许还有一些尝试的技巧。

### 8）尝试获得创意

真正爬进你的问题并思考如何将其分解为更容易处理的小问题。

为了获得灵感，请回顾 Quora 上非常有创意的答案，回答问题“[在分类中，您如何处理不平衡的训练集？](http://www.quora.com/In-classification-how-do-you-handle-an-unbalanced-training-set) “

例如：

> 将较大的类分解为较少数量的其他类...
> 
> ...使用一类分类器......（例如对待异常检测）
> 
> ......将不平衡的训练集重新取样到不是一个平衡集，而是几个。在这些集合上运行分类集合可以产生比单独一个分类器更好的结果

这些只是您可以尝试的一些有趣且富有创意的想法。

有关更多想法，请查看 reddit 帖子“[分类时的这些评论，当我 80％的训练集属于一个类](https://www.reddit.com/r/MachineLearning/comments/12evgi/classification_when_80_of_my_training_set_is_of/)”时。

## 选择一种方法并采取行动

您不需要是算法向导或统计学家来从不平衡数据集构建准确可靠的模型。

我们已经介绍了许多可用于建模不平衡数据集的技术。

希望您可以从架子上立即应用一两个，例如更改精度指标和重采样数据集。两者都很快，并会立即产生影响。

_**你打算尝试哪种方法？**_

## 最后一句话，从小开始

请记住，我们无法知道哪种方法最适合您以及您正在处理的数据集。

您可以使用一些专家启发式方法来选择这种方法，但最后，我能给您的最佳建议是“成为科学家”并根据经​​验测试每种方法，并选择能够为您提供最佳结果的方法。

从小处着手，以你学到的东西为基础。

## 想要更多？进一步阅读...

如果你知道在哪里看，就会有关于阶级不平衡的资源，但它们很少而且很少。

我看起来和以下是我认为是作物的奶油。如果您想更深入地了解一些关于处理课堂不平衡的学术文献，请查看下面的一些链接。

### 图书

*   [不平衡学习：基础，算法和应用](http://www.amazon.com/dp/1118074629?tag=inspiredalgor-20)

### 文件

*   [不平衡数据集的数据挖掘：概述](http://link.springer.com/chapter/10.1007/978-0-387-09823-4_45)
*   [从不平衡数据中学习](http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=5128907)
*   [解决不平衡训练集的诅咒：单方面选择](http://sci2s.ugr.es/keel/pdf/algorithm/congreso/kubat97addressing.pdf)（PDF）
*   [几种平衡机器学习训练数据方法的行为研究](http://dl.acm.org/citation.cfm?id=1007735)

你觉得这篇文章有用吗？还有问题吗？

发表评论并告诉我您的问题以及您在处理不平衡课程时仍然遇到的任何问题。