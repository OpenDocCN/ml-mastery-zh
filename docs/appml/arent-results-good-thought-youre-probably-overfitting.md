# 为什么我的结果不如我想的那么好？你可能过拟合了

> 原文： [`machinelearningmastery.com/arent-results-good-thought-youre-probably-overfitting/`](https://machinelearningmastery.com/arent-results-good-thought-youre-probably-overfitting/)

我们都知道运行分析的满意度，并看到结果以我们希望的方式回归：80％的准确度; 85％; 90％？

只是转向我们正在编写的报告的结果部分，并将数字放入其中，这种诱惑很强烈。但是等待：一如既往，这并不是那么简单。

屈服于这种特殊的诱惑可能会破坏其他完全有效的分析的影响。

对于大多数机器学习算法，考虑如何生成这些结果非常重要：不仅仅是算法，而是数据集及其使用方式会对获得的结果产生重大影响。应用于太小数据集的复杂算法可能导致过拟合，从而导致误导性良好的结果。

![Light at the end of the tunnel](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/11/Light-at-the-end-of-the-tunnel.jpg)

在隧道尽头的光
照片由 [darkday](http://www.flickr.com/photos/drainrat/14928494590) ，保留一些权利

## 什么是过拟合？

当机器学习算法（例如分类器）不仅识别数据集中的信号而且识别噪声时，发生过拟合。所有数据集都很吵。实验中记录的值可能会受到许多问题的影响：

*   机械问题，例如热量或湿度改变记录装置的特性;
*   身体问题：有些老鼠比其他老鼠大;
*   或者只是被调查系统中的固有噪音。例如，来自 DNA 的蛋白质的产生本质上是有噪声的，不是如通常可视化的那样在稳定的流中发生，而是在一系列步骤中，每个步骤是随机的，取决于在适当时间存在合适的分子。
*   从人类受试者收集的数据同样受到诸如一天中的时间，受试者的健康状况，甚至他们的情感等因素的影响。

随着数据集中参数数量的增加，情况会恶化。例如，具有 100 个记录的数据集（每个具有 500 个观测值）非常容易过拟合，而具有 5 个观测值的 1000 个记录每个将远不是问题。

> 当您的模型相对于数据点数量的参数太多时，您很容易高估模型的效用。

- Jessica Su in“[过拟合的直观解释是什么？](http://www.quora.com/What-is-an-intuitive-explanation-of-overfitting) “

## 为什么过拟合问题？

大多数机器学习算法的目的是找到从数据中的信号，重要值到输出的映射。噪声干扰了这种映射的建立。

过拟合的实际结果是，似乎在其训练数据上表现良好的分类器可能对来自同一问题的新数据表现不佳，可能非常糟糕。从数据集到数据集，数据中的信号几乎相同，但噪声可能非常不同。

如果分类器适合噪声和信号，它将无法将信号与新数据上的噪声分开。开发大多数分类器的目的是让它们以可预测的方式推广到新数据。

> 过拟合的模型通常具有较差的预测表现，因为它可能夸大数据中的微小波动

- 过拟合，[维基百科](http://en.wikipedia.org/wiki/Overfitting)。

## 克服过拟合

克服过拟合有两种主要方法：三组验证和交叉验证。

### 三套验证

道德分析师如何克服过拟合的问题？最简单，最难实现的解决方案就是拥有大量的数据。有了足够的数据，分析师就可以在一组数据（训练集）上开发和算法，然后在一个全新的，看不见的数据集上测试其表现，这些数据集由相同的方法（测试集）生成。

仅使用两个数据集的问题是，只要您使用测试集，它就会被污染。对集合 1 进行训练，对集合 2 进行测试，然后使用这些测试的结果来修改算法的过程意味着集合 2 实际上是训练数据的一部分。

为了完全客观，需要第三个数据集（验证集）。验证集应保持光荣隔离，直到所有训练完成。验证集上训练的分类器的结果是应该报告的结果。

一种流行的方法是在训练集上训练分类器，并且每几次迭代，在测试集上测试其表现。最初，当数据集中的信号拟合时，训练集和测试集的误差将下降。

然而，最终，分类器将开始适应噪声，尽管训练集上的错误率仍然降低，但测试集上的错误率将开始增加。此时应停止训练，并将训练好的分类器应用于验证集以估计实际表现。

这个过程因此变成：

1.  开发算法;
2.  第 1 组训练（训练集）;
3.  测试第 2 组（测试装置）;
4.  使用步骤 3 的结果修改算法或停止训练;
5.  迭代步骤 1 到 4 直到满意算法的结果;
6.  在第 3 组（验证集）上运行算法;
7.  报告步骤 6 的结果。

可悲的是，很少有项目产生足够的数据，使分析师能够沉迷于三数据集方法的奢侈品。必须找到一种替代方案，其中每个结果由分类器生成，该分类器在其训练中未使用该数据项。

### 交叉验证

通过交叉验证，整个可用数据集被分成大小相等或更小的子集。假设我们有 100 个观测数据集。我们可以将它分成 33 个，33 个和 34 个观察值的三个子集。我们将这三个子集称为 set1，set2 和 set3。

为了开发我们的分类器，我们使用了三分之二的数据;说 set1 和 set2，训练算法。然后，我们在 set3 上运行分类器，到目前为止看不见，并记录这些结果。

然后使用另外的三分之二，例如 set1 和 set3 重复该过程，并记录 set2 上的结果。类似地，在 set2 和 set3 上训练的分类器产生 set1 的结果。

然后组合三个结果集，并成为整个数据集的结果。

上述过程称为三重交叉验证，因为使用了三个数据集。可以使用任意数量的子集;十倍交叉验证被广泛使用。当然，最终的交叉验证方案是在除了一个案例之外的所有数据上训练每个分类器，然后在左边的情况下运行它。这种做法被称为留一法验证。

> 交叉验证对于防止数据建议的测试假设（称为“III 型错误”）非常重要，特别是在进一步的样品有害，昂贵或无法收集的情况下。

— Overfitting, [Wikipedia](http://en.wikipedia.org/wiki/Overfitting).

#### 交叉验证的优点，以避免过拟合

任何形式的交叉验证的主要优点是每个结果都是使用未经过该结果训练的分类器生成的。

此外，因为每个分类器的训练集由大多数数据组成，所以分类器虽然可能略有不同，但应该大致相同。在留一法的情况下尤其如此，其中每个分类器在几乎相同的数据集上训练。

#### 交叉验证的缺点

使用交叉验证有两个主要缺点：

1.  用于生成结果的分类器不是单个分类器，而是一组密切相关的分类器。如前所述，这些分类器应该非常相似，并且这个缺点通常不被认为是主要的。
2.  测试集不能再用于修改分类算法。因为算法是针对大多数数据进行训练，然后在较小的子集上进行测试，所以这些结果不再被视为“看不见”。无论结果如何，都应该报告。从理论上讲，这是一个重大缺陷，但在实践中很少出现这种情况。

总之，如果数据充足，则应使用三组验证方法。但是，当数据集有限时，交叉验证会以原则方式最佳地利用数据。

## 统计方法

由于过拟合是一个普遍存在的问题，因此已经有大量研究使用统计方法来避免这个问题。一些标准教科书对这些方法有很好的报道，包括：

*   Duda，R。O.，Hart，P。E.，＆amp; Stork，D.G。（2012）。 [模式分类](http://www.amazon.com/dp/0471056693?tag=inspiredalgor-20)：John Wiley＆amp;儿子。
*   Bishop，C。M.（2006）。 [模式识别和机器学习](http://www.amazon.com/dp/0387310738?tag=inspiredalgor-20)（第 1 卷）：施普林格纽约。

## 避免过拟合的教程

例如，使用 R 统计语言，使用 R 统计语言，请参阅“[评估模型表现 - 过拟合和数据大小对预测影响的实际示例](http://www.r-bloggers.com/evaluating-model-performance-a-practical-example-of-the-effects-of-overfitting-and-data-size-on-prediction/)”。

有关使用 SPSS 的详细教程，请参阅幻灯片“ [逻辑回归 - 完整问题](http://www.utexas.edu/courses/schwab/sw388r7/SolvingProblems/LogisticRegression_CompleteProblems.ppt)”（PPT）。

有关 SAS 用户指南的介绍，请参阅“ [GLMSELECT 程序](http://support.sas.com/documentation/cdl/en/statug/65328/HTML/default/viewer.htm#statug_glmselect_details25.htm)”。

## 进一步阅读

有关过拟合的实际效果的有趣概述可以在麻省理工学院技术评论中找到，题为“[大数据临近预报的新兴陷阱](http://www.technologyreview.com/view/530131/the-emerging-pitfalls-of-nowcasting-with-big-data/)”。

来自 CalTech 的优秀入门讲座在 YouTube 上提供，名为“ [Overfitting](https://www.youtube.com/watch?v=EQWr3GGCdzw) ”：

&lt;iframe allowfullscreen="" frameborder="0" height="281" src="https://www.youtube.com/embed/EQWr3GGCdzw?feature=oembed" width="500"&gt;&lt;/iframe&gt;

来自阿姆斯特丹自由大学（Vrije Universiteit Amsterdam）的一篇更详细的文章，题为“[你看到的可能不是你所得到的：回归型模型过拟合的非技术性介绍](http://www.cs.vu.nl/~eliens/sg/local/theory/overfitting.pdf)”（PDF）。