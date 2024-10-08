# 不平衡分类的成本敏感学习

> 原文：<https://machinelearningmastery.com/cost-sensitive-learning-for-imbalanced-classification/>

大多数机器学习算法都假设模型产生的所有错误分类错误都是相等的。

这通常不是不平衡分类问题的情况，在不平衡分类问题中，遗漏一个正类或少数类比错误地将一个例子从负类或多数类中分类出来更糟糕。有许多真实的例子，如检测垃圾邮件、诊断医疗状况或识别欺诈。在所有这些情况下，假阴性(错过一个病例)比假阳性更糟糕或更昂贵。

**成本敏感学习**是机器学习的一个子领域，在训练机器学习模型时会考虑预测误差的成本(以及潜在的其他成本)。它是与不平衡学习领域密切相关的一个研究领域，涉及在具有偏斜类分布的数据集上的分类。因此，为成本敏感学习开发和使用的许多概念化和技术可以用于不平衡的分类问题。

在本教程中，您将发现对不平衡分类的成本敏感学习的温和介绍。

完成本教程后，您将知道:

*   不平衡分类问题通常对假阳性分类错误的重视程度不同于假阴性。
*   成本敏感学习是机器学习的一个子领域，它涉及在训练机器学习算法时明确定义和使用成本。
*   成本敏感技术可以分为三类，包括数据重采样、算法修改和集成方法。

**用我的新书[Python 不平衡分类](https://machinelearningmastery.com/imbalanced-classification-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![Cost-Sensitive Learning for Imbalanced Classification](img/b11c006b3ac9fec47230f3f558575f35.png)

不平衡分类的成本敏感学习
图片由 [bvi4092](https://flickr.com/photos/bvi4092/33035819163/) 提供，保留部分权利。

## 教程概述

本教程分为四个部分；它们是:

1.  并非所有分类错误都相等
2.  对成本敏感的学习
3.  成本敏感的不平衡分类
4.  对成本敏感的方法

## 并非所有分类错误都相等

分类是一个预测建模问题，包括预测观察的类别标签。

可能有许多类标签，即所谓的多类分类问题，尽管最简单也可能是最常见的分类问题类型有两类，被称为二分类。

大多数为分类而设计的机器学习算法都假设每个观察到的类都有相同数量的例子。

实际上并不总是这样，具有偏斜类分布的数据集被称为不平衡分类问题。

> 在对成本敏感的学习中，不是每个实例被正确或不正确地分类，而是每个类(或实例)被给予错误分类成本。因此，问题不是试图优化准确率，而是最小化总的错误分类成本。

—第 50 页，[不平衡学习:基础、算法和应用](https://amzn.to/32K9K6d)，2013。

除了假设类分布是平衡的，大多数机器学习算法还假设分类器产生的预测误差是相同的，即所谓的误分类。

对于二进制分类问题，尤其是那些类别分布不平衡的问题，通常不是这样。

> 大多数分类器假设错误分类成本(假阴性和假阳性成本)是相同的。在大多数现实应用中，这个假设是不成立的。

——[不平衡数据的成本敏感学习方法](https://ieeexplore.ieee.org/document/5596486)，2010。

对于不平衡分类问题，来自多数类的例子被称为负类，并被赋予类标签 0。那些来自少数民族的例子被称为正类，并被指定为类标签 1。

这种否定与肯定命名约定的原因是，多数类的示例通常代表正常或无事件情况，而少数类的示例代表例外或事件情况。

*   **多数类**:否定或无事件分配类标签 0。
*   **少数民族类**:正面或事件被赋予类标签 1。

现实世界中不平衡的二进制分类问题通常对每一个可能出现的分类错误都有不同的解释。例如，将一个负面案例归类为正面案例通常比将一个正面案例归类为负面案例少得多。

如果我们考虑不平衡二进制分类问题的分类器的目标是正确地检测阳性病例，并且阳性病例代表我们最感兴趣的异常或事件，这是有意义的。

我们可以用一些例子来说明这一点。

**银行贷款问题**:考虑一个银行想确定是否给客户贷款的问题。拒绝一个好客户的贷款并不像给一个可能永远不会还款的坏客户贷款那么糟糕。

**癌症诊断问题**:考虑一个医生想要确定患者是否患有癌症的问题。诊断一个健康的癌症患者，并进行更多的医学检查，比让一个癌症患者出院更好。

> ……在某一癌症的医学诊断中，如果将癌症视为阳性类别，将非癌症(健康)视为阴性，则遗漏了一个癌症(患者实际上是阳性，但被归类为阴性；因此它也被称为“假阴性”)比假阳性错误严重得多(因此昂贵)。

——[成本敏感学习，《机器学习百科全书》](https://amzn.to/2PamKhX)，2010。

**欺诈检测问题**:考虑一个保险公司想要确定某个理赔是否存在欺诈的问题。将好的索赔确定为欺诈性索赔，并对客户进行跟进，这比接受欺诈性保险索赔要好。

从这些例子中我们可以看出，错误分类错误通常是不可取的，但一种错误分类比另一种更糟糕。具体来说，无论我们以何种方式来衡量目标领域的背景，将阳性病例预测为阴性病例都是更有害、更昂贵或更糟糕的。

> ……将一个实际的正面例子错误地归类为反面，往往比将一个实际的反面例子归类为正面更为昂贵。

——[成本敏感学习，《机器学习百科全书》](https://amzn.to/2PamKhX)，2010。

将每种错误分类错误视为相同的机器学习算法无法满足这些类型问题的需要。

因此，少数类在训练数据中的代表性不足，以及正确识别少数类中的例子的重要性增加，使得不平衡分类成为应用机器学习中最具挑战性的问题之一。

> 类不平衡是机器学习算法面临的挑战性问题之一。

——[不平衡数据的成本敏感学习方法](https://ieeexplore.ieee.org/document/5596486)，2010。

## 对成本敏感的学习

机器学习有一个子领域，专注于学习和使用数据模型，这些数据在进行预测等时具有不均衡的惩罚或成本。

这个领域通常被称为成本敏感机器学习，或者更简单地称为成本敏感学习。

> ……机器学习算法需要对它所处理的成本敏感，更好的情况是在模型拟合过程中考虑成本。这导致了机器学习中相对较新的研究课题——成本敏感的机器学习。

—第十三页，[成本敏感型机器学习](https://amzn.to/2qFgswK)，2011 年。

传统上，机器学习算法在数据集上训练，并寻求最小化误差。在数据上拟合模型解决了一个优化问题，在这个问题中，我们明确地寻求最小化误差。一系列函数可以用来计算模型对训练数据的误差，更一般的术语称为损失。我们寻求最小化模型在训练数据上的损失，这与谈论误差最小化是一样的。

*   **误差最小化**:训练机器学习算法时的常规目标是最小化模型在训练数据集上的误差。

在对成本敏感的学习中，与不正确预测相关联的一种惩罚，被称为“*成本*”我们也可以把惩罚的反面称为“T2”利益，尽管这种框架很少使用。

*   **成本**:与不正确预测相关的惩罚。

成本敏感学习的目标是最小化训练数据集上模型的成本，其中假设不同类型的预测误差具有不同且已知的相关成本。

*   **成本最小化**:成本敏感学习的目标是最小化训练数据集上模型的成本。

> 成本敏感学习是一种将错误分类成本(可能还有其他类型的成本)考虑在内的学习。这种学习的目标是使总成本最小化。

——[成本敏感学习，《机器学习百科全书》](https://amzn.to/2PamKhX)，2010。

不平衡分类和成本敏感学习之间存在紧密耦合。具体来说，不平衡的学习问题可以使用成本敏感的学习来解决。

然而，对成本敏感的学习是一个单独的子领域的研究，成本的定义可能比预测误差或分类误差更广泛。这意味着，尽管成本敏感学习中的一些方法可以有助于不平衡分类，但并非所有成本敏感学习技术都是不平衡学习技术，相反，并非所有用于解决不平衡学习的方法都适用于成本敏感学习。

为了使这一点具体化，我们可以考虑在数据集上训练模型时可能希望考虑或测量成本的各种其他方法。例如，[彼得·特尼](https://www.apperceptual.com)在他 2000 年发表的题为“[归纳概念学习中的成本类型](https://arxiv.org/abs/cs/0212034)的论文中列出了机器学习中可能考虑的九种成本类型

概括来说，它们是:

*   错误分类错误(或更一般的预测错误)的成本。
*   测试或评估的成本。
*   教师或标签费用。
*   干预或改变观察系统的成本。
*   干预带来的不必要的成就或结果的代价。
*   计算成本或计算复杂性。
*   案例或数据收集的成本。
*   人机交互或构建问题并使用软件来适应和使用模型的成本。
*   不稳定或变化的代价被称为概念漂移。

尽管成本和成本敏感学习对许多现实问题至关重要，但直到最近，这一概念还是一个很大程度上被忽视的新话题。

> 在概念学习的实际应用中，涉及到许多不同类型的成本。大多数机器学习文献忽略了所有类型的成本…

——[归纳概念学习中的成本类型](https://arxiv.org/abs/cs/0212034)，2000。

上面的列表强调了我们对不平衡分类感兴趣的成本只是更广泛的成本敏感学习领域可能考虑的成本范围的一种类型。

在下一节中，我们将仔细研究如何利用错误分类成本敏感学习的思想来帮助不平衡的分类。

## 成本敏感的不平衡分类

不平衡分类的成本敏感学习的重点是首先为可能出现的错误分类错误类型分配不同的成本，然后使用专门的方法将这些成本考虑在内。

使用成本矩阵的思想可以最好地理解不同的误分类成本。

让我们从回顾混淆矩阵开始。

一个[混淆矩阵](https://machinelearningmastery.com/ufaqs/what-is-a-confusion-matrix/)是一个模型对分类任务所做预测的总结。这是一个表，总结了每个类的预测数量，用每个示例所属的实际类来分隔。

最好使用具有负类和正类的二分类问题来理解，通常分别分配 0 和 1 类标签。表中的列代表示例所属的实际类，行代表预测类(尽管行和列的含义可以而且经常互换，不会失去意义)。表格中的单元格是满足行和列条件的示例数的计数，每个单元格都有一个特定的通用名称。

下面列出了二进制分类任务的混淆矩阵示例，显示了表中四个单元格中的值的通用名称。

```py
                   | Actual Negative | Actual Positive
Predicted Negative | True Negative   | False Negative
Predicted Positive | False Positive  | True Positive
```

我们可以看到，我们最感兴趣的是错误，即所谓的假阳性和假阴性，在许多不平衡的分类任务中，假阴性可能是我们最感兴趣的。

现在，我们可以考虑具有相同行和列的同一个表，并为每个单元格分配一个成本。这叫做成本矩阵。

*   **成本矩阵**:为混淆矩阵中的每个单元格分配成本的矩阵。

下面的例子是一个成本矩阵，我们使用符号 *C()* 来表示成本，第一个值表示为预测类，第二个值表示实际类。混淆矩阵中每个单元格的名称也作为首字母缩略词列出，例如，假阳性是 FP。

```py
                   | Actual Negative | Actual Positive
Predicted Negative | C(0,0), TN      | C(0,1), FN
Predicted Positive | C(1,0), FP      | C(1,1), TP
```

我们可以看到，假阳性的成本是 C(1，0)，假阴性的成本是 C(0，1)。

成本矩阵的这个公式和符号来自查尔斯·埃尔坎 2001 年的开创性论文，题目是“T2 对成本敏感学习的基础”

这个矩阵的一个直觉是，错误分类的成本总是高于正确分类，否则，可以通过预测一个类来最小化成本。

> 从概念上讲，错误标注一个例子的成本应该总是大于正确标注的成本。

——[成本敏感学习的基础](https://dl.acm.org/citation.cfm?id=1642224)，2001。

例如，我们可能在每个类中没有分配正确预测的成本，假阳性的成本为 5，假阴性的成本为 88。

```py
                   | Actual Negative | Actual Positive
Predicted Negative | 0               | 88
Predicted Positive | 5               | 0
```

我们可以使用这个框架将分类器的总成本定义为假阴性和假阳性的成本加权和。

*   总成本= C(0，1) *假阴性+ C(1，0) *假阳性

这是我们在对成本敏感的学习中寻求最小化的价值，至少在概念上是这样。

> CSL 的目的是建立一个错误分类成本(总成本)最小的模型

——[不平衡数据的成本敏感学习方法](https://ieeexplore.ieee.org/document/5596486)，2010。

必须仔细定义成本矩阵的值。与传统机器学习模型中误差函数的选择一样，成本或成本函数的选择将决定模型的质量和实用性，该模型适合于训练数据。

> 成本敏感学习的有效性强烈依赖于所提供的成本矩阵。提供的参数对训练和预测步骤都至关重要。

—第 66 页，[从不平衡数据集](https://amzn.to/307Xlva)中学习，2018。

在某些问题领域，定义成本矩阵可能是显而易见的。在保险索赔的例子中，假阳性的成本可能是客户跟进公司的货币成本，假阴性的成本可能是保险索赔的成本。

在其他领域，定义成本矩阵可能具有挑战性。例如，在癌症诊断测试示例中，假阳性的成本可能是执行后续测试的货币成本，而让生病的患者回家并病情加重的等价美元成本是多少？在这种情况下，成本矩阵可以由领域专家或经济学家定义，也可以不定义。

此外，成本可能是一个复杂的多维函数，包括货币成本、声誉成本等等。

不平衡分类任务的一个很好的起点是基于逆类分布分配成本。

> 在许多情况下，我们无法获得领域专家的帮助，并且在分类器训练期间没有关于成本矩阵的先验信息可用。当我们想要将成本敏感学习作为解决不平衡问题的方法时，这是一个常见的场景…

—第 67 页，[从不平衡数据集](https://amzn.to/307Xlva)中学习，2018。

例如，我们可能有一个数据集，少数类中的示例与多数类中的示例的比例为 1 比 100 (1:100)。这个比率可以反过来用作错误分类错误的成本，其中假阴性的成本是 100，假阳性的成本是 1。

```py
                   | Actual Negative | Actual Positive
Predicted Negative | 0               | 100
Predicted Positive | 1               | 0
```

一般来说，这是一种设置成本的有效启发式方法，尽管它假设在训练数据中观察到的类分布代表更广泛的问题，并且适用于所选择的对成本敏感的方法。

因此，使用这种启发式方法作为起点是一个好主意，然后测试一系列类似的相关成本或比率，以确认它是合理的。

## 对成本敏感的方法

对成本敏感的机器学习方法是那些明确使用成本矩阵的方法。

鉴于我们对不平衡分类的关注，我们对那些以某种方式使用不同错误分类成本的成本敏感技术特别感兴趣。

> 对成本敏感的学习方法通过使用不同的成本矩阵来解决不平衡学习的问题，这些成本矩阵描述了对任何特定数据示例进行错误分类的成本。

—第 3-4 页，[不平衡学习:基础、算法和应用](https://amzn.to/32K9K6d)，2013。

可能有三组主要的成本敏感方法与不平衡学习最相关；它们是:

1.  成本敏感重采样
2.  成本敏感算法
3.  对成本敏感的集成

让我们依次仔细看看每一个。

### 成本敏感重采样

在不平衡分类中，数据重采样指的是变换训练数据集以更好地平衡类分布的技术。

这可能涉及选择性地从多数类中删除示例，称为欠采样。更常见的是，它指的是复制或合成少数类中的新示例，称为过采样，或欠采样和过采样的组合。

数据重采样是一种可以直接用于成本敏感学习的技术。重采样的重点不是平衡倾斜的类分布，而是改变训练数据集的组成，以满足成本矩阵的期望。

这可能涉及直接对数据分布进行重采样，或者使用一种方法对数据集中的示例进行加权。这种方法可以称为训练数据集的成本比例加权或成本比例重采样。

> 我们提出并评估了一系列基于训练示例的成本比例加权的方法……,这可以通过将权重馈送到分类算法(在 boosting 中经常这样做)或(以黑盒方式)通过仔细的二次采样来实现。

——[按成本比例示例加权的成本敏感学习](https://dl.acm.org/citation.cfm?id=952181)，2003 年。

对于使用类别分布定义成本矩阵的不平衡分类，数据重采样技术没有区别。

### 成本敏感算法

机器学习算法很少专门为成本敏感的学习而开发。

相反，现有机器学习算法的财富可以被修改以利用成本矩阵。

这可能涉及对每种算法的独特修改，并且开发和测试可能相当耗时。已经为流行的算法提出了许多这样的算法特定的扩展，如决策树和支持向量机。

> 在所有的分类器中，代价敏感决策树的归纳无疑获得了最多的关注。

—第 69 页，[从不平衡数据集](https://amzn.to/307Xlva)中学习，2018。

Sklearn Python 机器学习库通过以下分类器上的 *class_weight* 参数提供了这些成本敏感扩展的示例:

*   [静止无功补偿器](https://Sklearn.org/stable/modules/generated/sklearn.svm.SVC.html)
*   [决定相反分类器](https://Sklearn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

修改现有算法的另一种更通用的方法是，在训练算法时，使用成本作为错误分类的惩罚。考虑到大多数机器学习算法都是为了最小化误差而训练的，因此在训练过程中，错误分类的成本会被添加到误差中，或者用于对误差进行加权。

这种方法可以用于迭代训练的算法，如逻辑回归和人工神经网络。

Sklearn 库通过以下分类器上的 *class_weight* 参数提供了这些成本敏感扩展的示例:

*   物流损耗
*   [重制](https://Sklearn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)

[Keras Python 深度学习库](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)还提供了在训练模型时通过 [fit()函数](https://keras.io/models/sequential/)上的 *class_weight* 参数对神经网络的成本敏感增强的使用进行访问。

同样，当逆类分布被用作成本矩阵时，算法的成本敏感扩展和算法的不平衡分类扩展之间的界限是模糊的。

在对成本敏感的机器学习领域，这些算法用“*对成本敏感的*前缀来表示，例如对成本敏感的逻辑回归，而在不平衡学习领域，这些算法用“*类加权的*前缀来表示，例如类加权的逻辑回归或简单的加权逻辑回归。

### 对成本敏感的集成

第二组不同的方法是设计用于过滤或组合来自传统机器学习模型的预测的技术，以便将误分类成本考虑在内。

这些方法被称为“包装器方法”，因为它们包装了一个标准的机器学习分类器。他们也被称为“T2”元学习器“T3”或“T4”集合“T5”，因为他们学习如何使用或组合来自其他模型的预测。

> 对成本敏感的元学习将现有的对成本不敏感的分类器转换为对成本敏感的分类器，而无需修改它们。因此，它可以被视为一个中间件组件，用于预处理训练数据，或者对成本不敏感的学习算法的输出进行后处理。

——[成本敏感学习，《机器学习百科全书》](https://amzn.to/2PamKhX)，2010。

也许最简单的方法是使用机器学习模型来预测类别成员的概率，然后在阈值上使用线性搜索，在该阈值下，示例被分配给每个清晰的类别标签，从而最小化错误分类的成本。

这通常被称为“阈值化”*或阈值优化，并且更一般地用于二进制分类任务，尽管它可以被容易地修改以最小化成本，而不是特定类型的分类误差度量。*

 *MetaCost 是一种数据预处理技术，它在训练数据集中重新标记示例，以便最小化成本。

> ……我们提出了一种有原则的方法，通过在任意分类器周围包装一个成本最小化过程，使其对成本敏感。

——[元成本:一种使分类器对成本敏感的通用方法](https://dl.acm.org/citation.cfm?id=312220)，1999。

在元成本中，首先将分类器的袋装集成拟合到训练数据集中，以识别那些需要重新标记的示例，创建具有重新标记示例的数据集的变换版本，然后丢弃该集成，并使用变换后的数据集来训练分类器模型。

另一个重要领域是考虑成本矩阵的决策树集成的修改，例如装袋和提升算法，最显著的是 AdaBoost 的成本敏感版本，如 AdaCost。

> AdaCost 是 AdaBoost 的变体，是一种对成本敏感的误分类 boosting 方法。它使用错误分类的成本来更新连续提升轮的训练分布。

——[AdaCost:误分类成本敏感提升](https://dl.acm.org/citation.cfm?id=657651)，1999。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 报纸

*   [阶级不平衡问题:系统研究](https://dl.acm.org/citation.cfm?id=1293954)，2002。
*   [归纳概念学习中的成本类型](https://arxiv.org/abs/cs/0212034)，2000。
*   [成本敏感学习的基础](https://dl.acm.org/citation.cfm?id=1642224)，2001。
*   [按成本比例示例加权的成本敏感学习](https://dl.acm.org/citation.cfm?id=952181)，2003。
*   [不平衡数据的成本敏感学习方法](https://ieeexplore.ieee.org/document/5596486)，2010。
*   [元成本:一种使分类器对成本敏感的通用方法](https://dl.acm.org/citation.cfm?id=312220)，1999。
*   [AdaCost:误分类成本敏感提升](https://dl.acm.org/citation.cfm?id=657651)，1999。

### 书

*   [成本敏感学习，机器学习百科](https://amzn.to/2PamKhX)，2010。
*   [成本敏感型机器学习](https://amzn.to/2qFgswK)，2011。
*   [第四章成本敏感学习，从不平衡数据集学习](https://amzn.to/307Xlva)，2018。
*   [不平衡学习:基础、算法和应用](https://amzn.to/32K9K6d)，2013。

### 文章

*   [混淆矩阵，维基百科](https://en.wikipedia.org/wiki/Confusion_matrix)。

## 摘要

在本教程中，您发现了不平衡分类的成本敏感学习。

具体来说，您了解到:

*   不平衡分类问题通常对假阳性分类错误的重视程度不同于假阴性。
*   成本敏感学习是机器学习的一个子领域，它涉及在训练机器学习算法时明确定义和使用成本。
*   成本敏感技术可以分为三类，包括数据重采样、算法修改和集成方法。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。*