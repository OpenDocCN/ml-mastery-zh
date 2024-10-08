# 用于比较机器学习算法的统计显着性检验

> 原文： [`machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/`](https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/)

比较机器学习方法和选择最终模型是应用机器学习中的常见操作。

模型通常使用重采样方法进行评估，例如 k 折叠交叉验证，从中计算并直接比较平均技能分数。虽然简单，但这种方法可能会产生误导，因为很难知道平均技能得分之间的差异是真实的还是统计侥幸的结果。

设计统计显着性检验以解决该问题，并且假设它们是从相同分布中抽取而量化观察技能分数样本的可能性。如果拒绝这种假设或零假设，则表明技能分数的差异具有统计学意义。

虽然不是万无一失，但统计假设检验可以提高您对解释的信心以及在模型选择过程中对结果的呈现。

在本教程中，您将发现选择统计假设检验来比较机器学习模型的重要性和挑战。

完成本教程后，您将了解：

*   统计假设检验有助于比较机器学习模型和选择最终模型。
*   统计假设检验的幼稚应用可能导致误导性结果。
*   正确使用统计检验具有挑战性，对于使用 McNemar 检验或 5×2 交叉验证与修改后的配对 Student t 检验存在一些共识。

让我们开始吧。

*   **更新 Oct / 2018** ：添加了使用 [McNemar 测试](https://machinelearningmastery.com/mcnemars-test-for-machine-learning/)的示例的链接。

![Statistical Significance Tests for Comparing Machine Learning Algorithms](img/5416b82764f164702c545e65ebdd8329.jpg)

用于比较机器学习算法的统计显着性检验
照片由[FotografíasdeJavier](https://www.flickr.com/photos/69487670@N04/6351929452/) 拍摄，保留一些权利。

## 教程概述

本教程分为 5 个部分;他们是：

1.  模型选择问题
2.  统计假设检验
3.  选择假设检验的问题
4.  一些调查结果摘要
5.  建议

## 模型选择问题

应用机器学习的一个重要部分是模型选择。

我们可以用最简单的形式来描述：

> 鉴于对数据集的两种机器学习方法的评估，您选择哪种模型？

您选择具有最佳技能的模型。

也就是说，在对看不见的数据做出预测时估计技能的模型是最佳的。在分类和回归问题的情况下，这可能分别是最大准确度或最小误差。

选择具有最佳技能的模型的挑战在于确定您可以信任每种模型的估计技能的程度。更普遍：

> 两种机器学习模型之间的技能差异是真实的，还是由于统计机会？

我们可以使用统计假设检验来解决这个问题。

## 统计假设检验

通常，用于比较样本的统计假设检验量化了在假设样本具有相同分布的情况下观察两个数据样本的可能性。

统计检验的假设称为零假设，我们可以计算统计指标并对其进行解释，以决定是否接受或拒绝原假设。

在基于其估计技能选择模型的情况下，我们有兴趣知道两个模型之间是否存在实际或统计上显着的差异。

*   如果测试结果表明没有足够的证据拒绝零假设，那么任何观察到的模型技能差异都可能是由于统计机会。
*   如果测试结果表明有足够的证据拒绝零假设，则模型技能的任何观察到的差异可能是由于模型的差异。

测试的结果是概率性的，这意味着，可以正确地解释结果，并且结果对于类型 I 或类型 II 错误是错误的。简而言之，假阳性或假阴性发现。

通过统计显着性检验比较机器学习模型会产生一些预期，反过来会影响可以使用的统计检验类型;例如：

*   **技能评估**。必须选择特定的模型技能度量。这可能是分类准确度（一个比例）或平均绝对误差（汇总统计量），这将限制可以使用的测试类型。
*   **重复估计**。需要一个技能分数样本来计算统计量。对相同或不同数据重复训练和测试给定模型将影响可以使用的测试类型。
*   **估算分布**。技能分数估计的样本将具有分布，可能是高斯分布或者可能不是。这将确定是否可以使用参数测试或非参数测试。
*   **中心趋势**。通常使用诸如平均值或中值的汇总统计来描述和比较模型技能，这取决于技能分数的分布。测试可能会也可能不会直接考虑到这一点。

统计检验的结果通常是检验统计量和 p 值，两者都可以在结果的呈现中被解释和使用，以便量化模型之间差异的置信水平或显着性。这允许更强的声明作为模型选择的一部分而不是使用统计假设检验。

鉴于使用统计假设检验似乎是模型选择的一部分，您如何选择适合您特定用例的测试？

## 选择假设检验的问题

让我们看一个用于评估和比较均衡二分类问题的分类器的常见示例。

通常的做法是使用分类准确度评估分类方法，使用 10 倍交叉验证评估每个模型，假设 10 个模型技能估计样本的高斯分布，并使用样本的平均值作为总结模特的技巧。

我们可以要求使用此过程评估的每个分类器通过 10 倍交叉验证在完全相同的数据集分割上进行评估。这将给出两个分类器之间匹配的配对测量的样本，因为每个分类器在相同的 10 个测试集上进行评估。

然后我们可以选择并使用配对的[T 检验](https://en.wikipedia.org/wiki/Student%27s_t-test)来检查两个模型之间平均准确度的差异是否具有统计学意义，例如拒绝假设两个样本具有相同分布的零假设。

事实上，这是使用这种方法将分类器与可能数百篇已发表论文进行比较的常用方法。

问题是，配对 T 检验的关键假设已被违反。

即，每个样本中的观察结果不是独立的。作为 k 折交叉验证程序的一部分，将在训练数据集（k-1）中使用给定的观察。这意味着估计的技能分数是依赖的，而不是独立的，反过来，测试中的 t 统计量的计算将与误差统计和 p 值的任何解释一起误导。

这种观察需要仔细了解所使用的重采样方法，在这种情况下是 k 次交叉验证，以及所选假设检验的期望，在这种情况下是配对 T 检验。没有这个背景，测试看起来是合适的，结果将被计算和解释，一切都会好看。

不幸的是，在应用机器学习中为模型选择选择适当的统计假设检验比首次出现时更具挑战性。幸运的是，越来越多的研究有助于指出朴素方法的缺陷，并提出修正和替代方法。

## 一些调查结果摘要

在本节中，我们来看一些关于机器学习中模型选择的适当统计显着性检验选择的研究。

### 使用 McNemar 的测试或 5×2 交叉验证

也许关于这一主题的开创性工作是 1998 年题为“[用于比较监督分类学习算法](http://ieeexplore.ieee.org/document/6790639/)的近似统计检验”的论文，作者是 Thomas Dietterich。

这是关于该主题的优秀论文和推荐阅读。它首先介绍了一个很好的框架，用于在机器学习项目中考虑可能需要进行统计假设检验的点，讨论与比较分类器机器学习方法相关的常见违反统计检验的期望，以及对方法的经验评估的结束确认调查结果。

> 本文回顾了五种近似的统计检验，用于确定一种学习算法是否优于另一种学习算法。

本文中统计假设检验的选择和经验评估的重点是 I 型误差或误报的校准。也就是说，选择最小化在没有这种差异时建议显着差异的情况的测试。

本文中有许多重要发现。

第一个发现是，对于通过训练数据集的随机重采样估计的技能结果，使用配对 T 检验绝对不应该进行。

> ......我们可以自信地得出结论，不应该采用重采样 t 检验。

在随机重采样的情况下和在 k 折交叉验证的情况下（如上所述），违反了配对 t 检验的假设。然而，在 k 折交叉验证的情况下，t 检验将是乐观的，导致 I 类错误更高，但只有适度的 II 类错误。这意味着这种组合可用于避免类型 II 错误比屈服于类型 I 错误更重要的情况。

> 10 倍交叉验证 t 检验具有高 I 型误差。然而，它也具有高功率，因此，可以推荐在 II 类错误（未能检测到算法之间的真正差异）更重要的情况下。

Dietterich 建议 [McNemar 的统计假设检验](https://machinelearningmastery.com/mcnemars-test-for-machine-learning/)用于数据量有限且每种算法只能评估一次的情况。

McNemar 的测试类似于卡方测试，在这种情况下用于确定算法的列联表中观察到的比例的差异是否与预期的比例显着不同。对于需要数天或数周训练的大型深度学习神经网络，这是一个有用的发现。

> 我们的实验引导我们推荐 McNemar 的测试，适用于学习算法只能运行一次的情况。

Dietterich 还推荐了他自己设计的重新取样方法，称为 5×2 交叉验证，涉及 5 次重复的 2 倍[交叉验证](https://en.wikipedia.org/wiki/Cross-validation_(statistics))。

选择两个折叠以确保每个观察仅出现在训练或测试数据集中以用于模型技能的单个估计。在结果上使用配对 T 检验，更新以更好地反映有限的自由度，因为估计的技能分数之间存在依赖关系。

> 我们的实验引导我们推荐 5 x 2cv t 测试，适用于学习算法足够高效运行十次的情况

### 5×2 交叉验证的改进

自该论文发表以来，使用 McNemar 测试或 5×2 交叉验证已成为 20 年来大部分时间的主要建议。

尽管如此，已经进行了进一步的改进，以更好地纠正配对 T 检验，以避免重复 k-交叉验证违反独立性假设。

其中两篇重要论文包括：

Claude Nadeau 和 Yoshua Bengio 在 2003 年题为“[推断误差](https://link.springer.com/article/10.1023/A:1024068626366)”的论文中提出了进一步的修正。这是一张浓密的纸，不适合胆小的人。

> 该分析允许我们构建两个方差估计，其考虑了由于选择训练集而导致的可变性以及测试示例的选择。其中一个提出的估计看起来类似于 cv 方法（Dietterich，1998），并且专门设计用于高估方差以产生保守推理。

Remco Bouckaert 和 Eibe Frank 在其 2004 年题为“[评估比较学习算法的重要性测试的可复制性](https://link.springer.com/chapter/10.1007/978-3-540-24775-3_3)”的论文中采取了不同的观点，并认为复制结果的能力比 I 型或 II 型错误更重要。

> 在本文中，我们认为测试的可复制性也很重要。我们说如果测试结果强烈依赖于用于执行它的数据的特定随机分区，那么测试具有低可复制性

令人惊讶的是，他们建议使用 100 次随机重采样或 10×10 倍交叉验证与 Nadeau 和 Bengio 校正配对 Student-t 测试，以实现良好的可复制性。

后一种方法在 Ian Witten 和 Eibe Frank 的书以及他们的开源数据挖掘平台 Weka 中被推荐，将 Nadeau 和 Bengio 校正称为“_ 校正的重采样 t 检验 _”。

> 已经提出了对标准 t 检验的各种修改来规避这个问题，所有这些修改都是启发式的并且缺乏合理的理论依据。在实践中似乎运行良好的是校正的重采样 t 检验。 [...]相同的修改统计量可用于重复交叉验证，这只是重复保持的特殊情况，其中一个交叉验证的各个测试集不重叠。

- 第 159 页，第五章，可信度：评估已经学到的东西，[数据挖掘：实用机器学习工具和技术](https://amzn.to/2GgeHch)，第三版，2011。

## 建议

在应用机器学习中选择模型选择的统计显着性检验时，没有银子弹。

让我们看看您可以在机器学习项目中使用的五种方法来比较分类器。

### 1.独立数据样本

如果您有接近无限的数据，请收集 k 个单独的训练和测试数据集，以计算每种方法的 10 个真正独立的技能分数。

然后，您可以正确应用配对学生的 t 检验。这是最不可能的，因为我们经常处理小数据样本。

> ...假设存在基本上无限的数据，以便可以使用正确大小的几个独立数据集。实际上，通常只有一个有限大小的数据集。可以做些什么？

- 第 158 页，第五章，可信度：评估已经学到的东西，[数据挖掘：实用机器学习工具和技术](https://amzn.to/2GgeHch)，第三版，2011。

### 2.接受 10 倍 CV 的问题

可以使用朴素的 10 倍交叉验证与未修改的配对 T 检验。

它具有相对于其他方法的良好可重复性和适度的 II 型错误，但已知具有高 I 型错误。

> 实验还建议在解释 10 倍交叉验证 t 检验的结果时要谨慎。该测试具有升高的 I 型错误概率（高达目标水平的两倍），尽管它不像重采样 t 检验的问题那么严重。

- [用于比较监督分类学习算法的近似统计检验](http://ieeexplore.ieee.org/document/6790639/)，1998。

这是一个选项，但推荐非常弱。

### 3.使用 McNemar 的测试或 5×2 CV

McNemar 测试单次分类准确率结果的长达 20 年的建议和 5×2 倍交叉验证，并在一般情况下使用修改后的配对 Student t 检验。

此外，Nadeau 和 Bengio 对测试统计的进一步校正可以与 Weka 的开发者推荐的 5×2 倍交叉验证或 10×10 倍交叉验证一起使用。

使用修改的 t 统计量的一个挑战是没有现成的实现（例如在 SciPy 中），需要使用第三方代码以及这带来的风险。您可能必须自己实现。

选择统计方法的可用性和复杂性是一个重要的考虑因素，Gitte Vanwinckelen 和 Hendrik Blockeel 在其 2012 年题为“[估计模型准确率与重复交叉验证](https://lirias.kuleuven.be/handle/123456789/346385)”的论文中表示：

> 虽然这些方法经过精心设计，并且可以通过多种方式改进以前的方法，但它们与以前的方法具有相同的风险，即方法越复杂，研究人员使用它的风险就越高。 ，或错误地解释结果。

我有一个在这里使用 McNemar 测试的例子：

*   [如何计算 McNemar 的比较两台机器学习分类器的测试](https://machinelearningmastery.com/mcnemars-test-for-machine-learning/)

### 4.使用非参数配对测试

我们可以使用非参数测试来做出更少的假设，例如不假设技能分数的分布是高斯分布。

一个例子是 [Wilcoxon 符号秩检验](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)，它是配对 T 检验的非参数版本。该测试具有比配对 t 检验更少的统计功效，尽管在违反 t 检验的期望时更多功率，例如独立性。

这种统计假设检验被推荐用于比较 Janez Demsar 在其 2006 年论文“多重数据集中的分类器的统计比较”中的不同数据集的算法。

> 因此，我们建议使用 Wilcoxon 测试，除非满足 t 检验假设，因为我们有许多数据集，或者因为我们有理由相信跨数据集的表现测量值是正常分布的。

尽管该测试是非参数的，但它仍假设每个样本内的观察是独立的（例如 iid），并且使用 k 折交叉验证将产生依赖样本并违反该假设。

### 5.使用估算统计

可以计算估计统计量而不是统计假设检验，例如置信区间。考虑到评估模型的重采样方法，这些将遭受类似的问题，其中违反了独立性的假设。

Tom Mitchell 在其 1997 年的书中提出了类似的建议，建议将统计假设检验的结果作为启发式估计，并寻求关于模型技能估计的置信区间：

> 总而言之，没有一种基于有限数据的比较学习方法的程序满足我们想要的所有约束。明智的是要记住，当可用数据有限时，统计模型很少完全符合测试学习算法的实际约束。然而，它们确实提供了近似置信区间，这对于解释学习方法的实验比较有很大帮助。

- 第 150 页，第五章，评估假设，[机器学习](https://amzn.to/2pE6l83)，1997。

诸如自举之类的统计方法可用于计算可防御的非参数置信区间，其可用于呈现结果和比较分类器。这是一种简单而有效的方法，您可以随时使用，我建议一般。

> 实际上，置信区间已经获得了对自助区域中任何主题的最理论研究。

- 第 321 页， [Bootstrap 简介](https://amzn.to/2ISXPKe)，1994。

## 扩展

本节列出了一些扩展您可能希望探索的教程的想法。

*   查找并列出三篇错误使用未经修改的配对 T 检验的研究论文，以比较和选择机器学习模型。
*   总结在 Thomas Dietterich 1998 年论文中提出的机器学习项目中使用统计假设检验的框架。
*   查找并列出三篇正确使用 McNemar 测试或 5×2 交叉验证进行比较并选择机器学习模型的研究论文。

如果你探索任何这些扩展，我很想知道。

## 进一步阅读

如果您希望深入了解，本节将提供有关该主题的更多资源。

### 文件

*   [用于比较监督分类学习算法的近似统计检验](http://ieeexplore.ieee.org/document/6790639/)，1998。
*   [推广错误的推断](https://link.springer.com/article/10.1023/A:1024068626366)，2003。
*   [评估用于比较学习算法的重要性测试的可复制性](https://link.springer.com/chapter/10.1007/978-3-540-24775-3_3)，2004。
*   [关于通过重复交叉验证估算模型准确率](https://lirias.kuleuven.be/handle/123456789/346385)，2012。
*   [多数据集上分类器的统计比较](http://www.jmlr.org/papers/v7/demsar06a.html)，2006。

### 图书

*   第五章，评估假设，[机器学习](https://amzn.to/2pE6l83)，1997。
*   第五章，可信度：评估已经学到的东西，[数据挖掘：实用机器学习工具和技术](https://amzn.to/2GgeHch)，第三版，2011。
*   [引导程序简介](https://amzn.to/2ISXPKe)，1994。

### 用品

*   [维基百科上的 T 检验](https://en.wikipedia.org/wiki/Student%27s_t-test)
*   [维基百科](https://en.wikipedia.org/wiki/Cross-validation_(statistics))的交叉验证（统计）
*   [McNemar 对维基百科的测试](https://en.wikipedia.org/wiki/McNemar%27s_test)
*   [Wilcoxon 对维基百科的签名等级测试](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)

### 讨论

*   [对于模型选择/比较，我应该使用哪种测试？](https://stats.stackexchange.com/questions/217466/for-model-selection-comparison-what-kind-of-test-should-i-use)
*   [如何进行假设检验以比较不同的分类器](https://stats.stackexchange.com/questions/89064/how-to-perform-hypothesis-testing-for-comparing-different-classifiers)
*   [Wilcoxon 秩和检验方法](https://stats.stackexchange.com/questions/93369/wilcoxon-rank-sum-test-methodology)
*   [如何选择 t 检验或非参数检验，例如小样本中的 Wilcoxon](https://stats.stackexchange.com/questions/121852/how-to-choose-between-t-test-or-non-parametric-test-e-g-wilcoxon-in-small-sampl)

## 摘要

在本教程中，您发现了选择统计假设检验来比较机器学习模型的重要性和挑战。

具体来说，你学到了：

*   统计假设检验有助于比较机器学习模型和选择最终模型。
*   统计假设检验的幼稚应用可能导致误导性结果。
*   正确使用统计检验具有挑战性，对于使用 McNemar 检验或 5×2 交叉验证与修改后的配对 Student t 检验存在一些共识。

你有任何问题吗？
在下面的评论中提出您的问题，我会尽力回答。