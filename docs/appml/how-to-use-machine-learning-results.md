# 如何使用机器学习结果

> 原文： [`machinelearningmastery.com/how-to-use-machine-learning-results/`](https://machinelearningmastery.com/how-to-use-machine-learning-results/)

一旦[发现](http://machinelearningmastery.com/how-to-evaluate-machine-learning-algorithms/ "How to Evaluate Machine Learning Algorithms")和[调整](http://machinelearningmastery.com/how-to-improve-machine-learning-results/ "How to Improve Machine Learning Results")一个可行的问题模型，就可以利用该模型了。你可能需要重新审视你的[为什么](http://machinelearningmastery.com/how-to-define-your-machine-learning-problem/ "How to Define Your Machine Learning Problem")，并提醒自己，你需要一个解决你正在解决的问题的形式。

在对结果执行某些操作之前，不会解决此问题。在这篇文章中，您将学习在将原型模型转换为生产系统时回答问题和考虑因素的结果。

![Presentation of Results](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/01/presentation-of-results.jpg)

结果展示
照片归属于 [Phil Sexton](http://www.flickr.com/photos/john_hall_associates/3175199088/sizes/l/) ，保留一些权利

根据您要解决的问题类型，结果的呈现方式会有很大差异。利用机器学习成果的结果有两个主要方面：

*   报告结果
*   操作系统

## 报告结果

一旦你发现了一个好的模型和一个足够好的结果（或者不是，视情况而定），你将需要总结所学的内容并将其呈现给利益相关者。这可能是您自己，客户或您工作的公司。

使用 powerpoint 模板并解决下面列出的部分。您可能希望编写一个分页器并将部分部分用作部分标题。即使在您为自己做的小型实验项目（例如教程和比赛）上也要尝试遵循此过程。很容易在项目上花费过多的时间，并且您希望确保捕获您沿途学到的所有伟大的事物。

以下是报告项目结果时可以完成的部分。

*   **上下文**（为什么）：定义问题所在的环境，并为研究问题设置动机。
*   **问题**（问题）：简要描述问题是你出去解答的问题。
*   **解决方案**（答案）：简要描述解决方案，作为您在上一节中提出的问题的答案。请明确点。
*   **调查结果**：您在观众感兴趣的路上发现的项目符号列表。它们可能是数据中的发现，已完成或未起作用的方法，或者您在旅程中获得的模型表现优势。
*   **限制**：考虑模型不起作用的地方或模型未回答的问题。不要回避这些问题，如果你可以定义它不擅长的地方，那么定义模型擅长的地方会更加可信。
*   **结论**（为什么+问题+答案）：重新审视为什么，研究问题和你在一个简单的小包装中发现的答案，这些包装易于记忆并为自己和他人重复。

您要呈现的受众类型将定义您要详细说明的详细信息量。通过结果报告来完成项目的纪律，即使是小型项目，也会加速您在现场的学习。在这些小型项目中，我强烈建议在博客或社区上分享辅助项目的结果，并获得您可以捕获的反馈，并将其带入您下一个项目的开始。

## 操作化

您已经找到了一个足以解决您希望将其投入生产的问题的模型。对于有趣的项目，这可能是您工作站上的操作安装，一直到将模型集成到现有企业应用程序中。范围很广。在本节中，您将学习操作模型的三个关键方面，您可以在将系统投入生产之前仔细考虑。

您应该仔细考虑的三个方面是算法实现，模型的自动测试以及随时间跟踪模型的表现。这三个问题很可能会影响您选择的模型类型。

### 算法实现

您可能正在使用研究库来发现表现最佳的方法。研究库中的算法实现可以很好，但它们也可以针对问题的一般情况而不是您正在使用的特定情况编写。

通过将这样的实现直接投入生产，可以非常认真地考虑您可能创建的依赖关系和技术债务。请考虑查找支持您要使用的方法的生产级库。如果此时切换到生产级别库，则可能必须重复算法调整过程。

您也可以考虑自己实现算法。此选项可能会引入风险，具体取决于您选择的算法的复杂程度以及它使用的实现技巧。即使使用开源代码，也可能存在许多复杂的操作，这些操作可能很难内化并且可靠地重现。

### 模型测试

编写自动化测试，验证可以构建模型并重复实现最低级别的表现。还要为任何数据准备步骤编写测试。您可能希望控制每个单元测试运行的算法使用的随机性（随机数种子），以便测试 100％可重复。

### 跟踪

添加基础设施以监控模型的表现，并在精度降至最低水平以下时发出警报。跟踪可以在单独的环境中实时发生，也可以在重新创建的模型上使用实时数据样本。引发警报可以指示数据中的模型所学习的结构已经改变（概念漂移）并且可能需要更新或调整模型。

有些模型类型可以执行在线学习并自行更新。仔细考虑允许模型在生产环境中自我更新。在某些情况下，管理模型更新过程并切换模型（其内部配置）可能更明智，因为它们经过验证更具表现。

## 摘要

在这篇文章中，您了解到在交付结果之前，项目不会被视为完成。结果可能会呈现给您自己或您的客户，并且在呈现结果时需要遵循最低结构。

在生产环境中使用学习模型时，您还学习了三个问题，特别是算法实现的性质，模型测试和正在进行的跟踪。