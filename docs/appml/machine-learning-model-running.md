# 机器学习模型运行期间要做什么

> 原文： [`machinelearningmastery.com/machine-learning-model-running/`](https://machinelearningmastery.com/machine-learning-model-running/)

最近有一个问题是“[如何在 ml 脚本运行时不浪费时间/拖延？](http://www.reddit.com/r/MachineLearning/comments/2uznyb/how_to_not_wastetimeprocrastinate_while_ml/) “。

我认为这是一个重要的问题。我认为这个问题的答案显示了你的工作方法的组织或成熟程度。

我在这个问题上留下了[小评论](http://www.reddit.com/r/MachineLearning/comments/2uznyb/how_to_not_wastetimeprocrastinate_while_ml/cof6co0)，但是在这篇文章中，我详细阐述了我的答案并给出了一些关于如何考虑这个问题，最小化它甚至完全避免它的观点。

![What to do during machine learning model runs](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2015/02/What-to-do-during-machine-learning-model-runs.jpg)

在机器学习模型运行期间做什么
照片由 [Mark Fischer](https://www.flickr.com/photos/fischerfotos/7519614160) ，保留一些权利

## 少运行实验

考虑一下为什么要执行模型运行。您几乎可以肯定地执行[探索性数据分析](http://machinelearningmastery.com/understand-problem-get-better-results-using-exploratory-data-analysis/ "Understand Your Problem and Get Better Results Using Exploratory Data Analysis")的形式。

您试图了解您的问题，以达到特定准确度的结果。您可能需要报告结果，或者您可能希望模型可以运行。

您的实验旨在教您一些有关该问题的信息。因此，您需要清楚地了解要从您执行的每个实验中学习的内容。

如果您没有明确的明确问题，实验结果会启发，请考虑是否需要运行实验。

当您获得问题的经验答案时，请尊重这些结果。尽力将新知识整合到您对问题的理解中。这可能是半正式的工作产品，如日报或技术报告。

## 运行更快的实验

现代编程的编译运行修复循环非常有效。立竿见影的回报可以让您不断测试想法和课程更正。

这个过程并不总是那么有效。作为工程师，您曾经使用笔和纸手工设计模块和桌面检查他们的逻辑。如果你在编程中做任何数学，你很可能仍然使用这个过程。

一个有用的现代工具是单元测试，它使桌面检查过程自动化，使它们可重复。良好的测试设计的准则是速度。反馈越直接，您就可以更快地纠正错误并修复错误。

**这里的教训是你想要速度。**

您希望快速获得问题的经验答案，以便您可以提出后续问题。这并不意味着设计糟糕的实验。这意味着使实验变得很大或足够详细以回答一个问题。

实现更快速实验的最简单方法是使用减少的数据样本。它是如此简单的技术，它经常被忽视。

通常情况下，您正在寻找的效果可以与数据进行可预测的比例，无论是数据本身的属性，如异常值还是数据模型的准确率。

## 运行调整作为实验

一些实验本身就很慢，就像调整超参数一样。事实上，当您的追求是优化精确度时，调整可能会让人上瘾。

完全避免手动调整任何参数，这是一个陷阱！我的建议是使用随机或网格搜索等搜索方法设计有条理的调整实验。

收集结果并使用实验建议的最佳参数。

如果您想获得更好的结果，请在参数空间中减少超立方体设计后续实验，并将搜​​索算法更改为使用基于梯度（或准梯度）的方法。

## 在停机时间运行实验

避免在最有效的时间内进行实验。如果您在白天工作时间完成工作，请不要占用您的机器，并在此时关注模型运行等阻塞任务。

安排您的实验在您不工作时运行。在晚上，午餐时间和周末进行实验。

要在停机时间运行实验，意味着您需要安排它们。如果您能够批量实验，这将变得更加容易。

你可以花时间在一批中设计 5-10 个实验，准备模型运行并在你的关闭时间顺序或并行地运行实验。

这可能需要纪律来解决问题和实验所服务的答案。这样做的好处将是您获得的有关问题的深度知识以及获得问题的速度。

## 在场外进行实验

有些实验可能需要数天或数周，这意味着在工作站上运行它们实际上是不可行的。

对于长时间运行的实验，您可以利用云中的计算服务器（如 EC2 和朋友）或本地计算服务器。无论是哪种语言环境，都不能实时使用计算服务器。您提出问题并收到答案。

计算服务器的最有效使用是拥有一个问题队列和一个过程，用于消费并将答案集成到您不断增长的问题知识库中。

例如，您可以设置每天（或夜晚）运行一个实验的目标，无论如何。我经常尝试在新项目上坚持这种模式。这有助于保持高势头。

当想法减弱时，您可以通过轻率的优化实验填充队列，以调整表现良好的模型的参数，这是一个可以随时备份的持续后台任务。

## 在实验运行时进行计划

有时您必须在工作站上实时运行实验。模型运行时，工作站必须阻塞。原因将是一些迫切的实时要求，你不能拖延。

发生这种情况时，请记住您的项目并没有阻止您的想法，只有您的工作站。

拉出文本编辑器或笔和纸（首选，这样您就不会从实验运行中偷走任何循环）。利用这段时间深入思考您的项目。制作如下列表：

*   列出并优先考虑您想要执行的实验
*   列出问题，预期答案，所需的设置并影响每个实验的结果。
*   列出并优先考虑您可以做出的假设和实验以对其提出异议。
*   列出并优先考虑您要编写单元测试的代码区域。
*   列出您的问题的替代视角和框架。

要有创意，并考虑测试有关该项目的长期信念。

我喜欢在一天结束时做我的创造性工作，让我的潜意识在我睡觉时解决问题。我也喜欢在我的工作站上进行实验过夜，让它与我的潜意识一起思考。

## 摘要

在这篇文章中，您已经发现了一些方法来解决机器学习模型运行期间的高效问题。

以下是您可以使用的关键策略的摘要：

*   考虑每个实验是否需要使用它将为您理解问题作为评估标准提供的贡献。
*   设计实验运行速度更快，并使用数据样本来实现加速。
*   永远不要手动调整超参数，总是设计自动化实验来回答模型校准的问题。
*   在您的停工期间进行实验，例如过夜，午休和周末。
*   批量设计实验，以便您可以排队并安排执行。
*   委派实验运行来计算工作站以外的服务器以提高效率。
*   如果您必须运行阻止实时实验，请利用该时间深入思考您的问题，设计未来的实验并挑战基本假设。