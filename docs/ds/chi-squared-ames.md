# 车库与否？通过卡方检验获取爱荷华州艾姆斯市的住房见解

> 原文：[`machinelearningmastery.com/chi-squared-ames/`](https://machinelearningmastery.com/chi-squared-ames/)

卡方独立性检验是一种统计程序，用于评估两个分类变量之间的关系——确定它们是相关的还是独立的。探索房产的视觉吸引力及其对估值的影响是很有趣的。但是你多久将房子的外观与车库等功能特征联系在一起？通过卡方检验，你可以确定这些特征之间是否存在统计学上显著的关联。

让我们开始吧。

![](img/a9f80be1f13b40c791938e24656cb22f.png)

车库与否？通过卡方检验获取爱荷华州艾姆斯市的住房见解

图片由[Damir Kopezhanov](https://unsplash.com/photos/gray-sedan-w-bRrLmXODg)提供。一些权利保留。

## 概述

本文分为四部分，它们是：

+   理解卡方检验

+   卡方检验的工作原理

+   揭示外部质量与车库存在之间的关系

+   重要注意事项

## 理解卡方检验

卡方（$\chi²$）检验之所以有用，是因为它能够测试**分类变量**之间的关系。它在处理名义或顺序数据时特别有价值，因为这些数据被划分为类别或组。卡方检验的主要目的是确定两个分类变量之间是否存在统计学上显著的关联。换句话说，它有助于回答以下问题：

+   **两个分类变量是否相互独立？**

    +   如果变量是独立的，一个变量的变化与另一个变量的变化无关。它们之间没有关联。

+   **两个分类变量之间是否存在显著关联？**

    +   如果变量之间存在关联，一个变量的变化与另一个变量的变化相关。卡方检验有助于量化这种关联是否具有统计学意义。

在你的研究中，你关注房子的外部质量（分为“优秀”或“一般”）及其与车库有无的关系。为了使卡方检验的结果有效，必须满足以下条件：

+   **独立性：** 观察值必须是独立的，意味着一个结果的发生不应影响另一个结果。我们的数据集符合这一点，因为每个条目代表一个独立的房子。

+   **样本大小：** 数据集不仅要随机抽样，而且要足够大以具代表性。我们的数据来自爱荷华州艾姆斯市，符合这一标准。

+   **期望频率：** 每个列联表中的单元格应该有至少 5 的期望频率。这对于检验的可靠性至关重要，因为卡方检验依赖于大样本近似。你将通过创建和可视化期望频率来展示这一条件。

**启动你的项目**，参考我的书籍[数据科学入门指南](https://machinelearning.samcart.com/products/beginners-guide-data-science/)。它提供了**自学教程**和**可运行的代码**。

## 卡方检验的工作原理

卡方检验将数据中的观察频率与假设中的期望频率进行比较。

卡方检验通过将列联表中类别的观察频率与在独立假设下预期的频率进行比较来工作。列联表是两个分类变量的交叉表，显示每个类别组合中有多少观察值。

+   **零假设 ($H_0$)：** 卡方检验中的零假设假定两个变量之间独立，即观察到的频率（有或没有车库）应该匹配。

+   **备择假设 ($H_1$)：** 备择假设表明两个变量之间存在显著关联，即观察到的频率（有或没有车库）应该根据另一个变量的值（房屋质量）有所不同。

卡方检验中的检验统计量通过比较列联表中每个单元格的观察频率和期望频率来计算。观察频率与期望频率之间的差异越大，卡方统计量越大。卡方检验产生一个 p 值，表示在独立假设下观察到观察到的关联（或更极端的关联）的概率。如果 p 值低于选择的显著性水平 $\alpha$（通常为 0.05），则拒绝独立性零假设，表明存在显著的关联。

## **揭示外部质量与车库存在之间的关联**

使用[Ames 房屋数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)，你着手确定房屋的外部质量与车库存在或不存在之间是否存在关联。让我们深入分析的细节，并辅以相应的 Python 代码。

```py
# Importing the essential libraries
import pandas as pd
from scipy.stats import chi2_contingency

# Load the dataset
Ames = pd.read_csv('Ames.csv')

# Extracting the relevant columns
exterqual_garagefinish_data = Ames[['ExterQual', 'GarageFinish']].copy()

# Filling missing values in the 'GarageFinish' column with 'No Garage'
exterqual_garagefinish_data['GarageFinish'].fillna('No Garage', inplace=True)

# Grouping 'GarageFinish' into 'With Garage' and 'No Garage'
exterqual_garagefinish_data['Garage Group'] \
    = exterqual_garagefinish_data['GarageFinish'] \
      .apply(lambda x: 'With Garage' if x != 'No Garage' else 'No Garage')

# Grouping 'ExterQual' into 'Great' and 'Average'
exterqual_garagefinish_data['Quality Group'] \
    = exterqual_garagefinish_data['ExterQual'] \
      .apply(lambda x: 'Great' if x in ['Ex', 'Gd'] else 'Average')

# Constructing the simplified contingency table
simplified_contingency_table \
    = pd.crosstab(exterqual_garagefinish_data['Quality Group'],
                  exterqual_garagefinish_data['Garage Group'])

#Printing the Observed Frequency
print("Observed Frequencies:")
observed_df = pd.DataFrame(simplified_contingency_table,
                           index=["Average", "Great"],
                           columns=["No Garage", "With Garage"])
print(observed_df)
print()

# Performing the Chi-squared test
chi2_stat, p_value, _, expected_freq = chi2_contingency(simplified_contingency_table)

# Printing the Expected Frequencies
print("Expected Frequencies:")
print(pd.DataFrame(expected_freq,
                   index=["Average", "Great"],
                   columns=["No Garage", "With Garage"]).round(1))
print()

# Printing the results of the test
print(f"Chi-squared Statistic: {chi2_stat:.4f}")
print(f"p-value: {p_value:.4e}")
```

输出应为：

```py
Observed Frequencies:
         No Garage  With Garage
Average        121         1544
Great            8          906

Expected Frequencies:
         No Garage  With Garage
Average       83.3       1581.7
Great         45.7        868.3

Chi-squared Statistic: 49.4012
p-value: 2.0862e-12
```

上述代码执行了三个步骤：

**数据加载与准备：**

+   你首先加载了数据集，并提取了相关列：`ExterQual`（外部质量）和`GarageFinish`（车库完成情况）。

+   识别到`GarageFinish`中的缺失值后，你明智地用标签`"No Garage"`填补这些缺失值，表示没有车库的房屋。

**简化的数据分组：**

+   你进一步将`GarageFinish`数据分类为两组：“有车库”（对于任何类型的有车库的房屋）和“无车库”。

+   同样，你将`ExterQual`数据分为“优秀”（具有优秀或良好外部质量的房屋）和“一般”（具有平均或一般外部质量的房屋）两组。

**卡方检验:**

+   将数据适当准备后，你构建了一个列联表来展示新形成类别之间的观察频率。它们是输出中打印的两个表格。

+   然后，你使用 SciPy 对这个列联表执行了卡方检验。p 值被打印出来，发现远小于$\alpha$。从测试中获得的极低 p 值表明，在这个数据集中，房屋外部质量与车库存在之间存在显著的统计关联。

+   仔细观察期望频率可以满足卡方检验的第三个条件，即每个单元格至少需要 5 次发生。

通过这项分析，你不仅使数据更加精炼和简化，使其更易解释，还提供了关于两个感兴趣的分类变量之间关联的统计证据。

### 想要开始学习数据科学初学者指南吗？

现在就参加我的免费电子邮件快速课程（带有示例代码）。

点击注册，并且还可以获得课程的免费 PDF 电子书版本。

## 重要注意事项

尽管卡方检验非常实用，但也有其局限性：

+   **非因果关系:** 虽然测试可以确定关联，但不能推断因果关系。因此，即使房屋外部质量与其车库存在之间存在显著联系，也不能得出一个导致另一个的结论。

+   **方向性:** 这项测试表明了一种关联，但并未指明其方向。然而，我们的数据表明，在外部质量“优秀”的房屋中更有可能有车库，而在“一般”标记的房屋中则可能性较小。

+   **幅度:** 该测试不提供关系强度的见解。其他度量指标，如克拉默 V，对此更具信息性。

+   **外部效度:** 我们的结论仅适用于 Ames 数据集。在将这些发现推广到其他地区时需谨慎。

## **进一步阅读**

#### 线上

+   [独立性的卡方检验 | Stat Trek](https://stattrek.com/chi-square-test/independence)

+   [scipy.stats.chi2_contingency](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html) API

+   [卡方检验](https://zh.wikipedia.org/wiki/%E5%8D%A1%E6%96%B9%E6%A3%80%E9%AA%8C) 在维基百科

#### **资源**

+   [Ames 数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)

+   [Ames 数据字典](https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt)

## **总结**

在这篇文章中，你深入探讨了卡方检验及其在 Ames 住房数据集上的应用。你发现了房屋外部质量与车库存在之间的显著关联。

具体来说，你学到了：

+   卡方检验的基本原理和实际应用。

+   卡方检验揭示了 Ames 数据集中房屋外部质量与车库存在之间的显著关联。与“普通”评级的房屋相比，具有“优秀”外部质量评级的房屋更有可能拥有车库，这一趋势具有统计学意义。

+   卡方检验的重要警示和局限性。

你有任何问题吗？请在下面的评论中提问，我会尽力回答。
