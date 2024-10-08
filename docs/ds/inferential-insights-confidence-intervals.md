# 推断性见解：置信区间如何揭示艾姆斯房地产市场的情况

> 原文：[`machinelearningmastery.com/inferential-insights-confidence-intervals/`](https://machinelearningmastery.com/inferential-insights-confidence-intervals/)

在数据的广阔宇宙中，有时重要的不是你能看到什么，而是你能推断出什么。置信区间，作为推断统计学的基石，使你能够根据样本数据对更大的人群做出有根据的猜测。利用艾姆斯住房数据集，让我们揭开置信区间的概念，并看看它们如何为房地产市场提供可操作的见解。

让我们开始吧。

![](img/9bb0c1d442e18c72634e116d8129967a.png)

推断性见解：置信区间如何揭示艾姆斯房地产市场的情况。

图片来源：[Jonathan Klok](https://unsplash.com/photos/gray-and-black-wooden-bridge-across-mountains-covered-by-trees-JS8RhWVk74Q)。部分权利保留。

## 概述

本文分为以下几个部分：

+   推断统计学的核心

+   什么是置信区间？

+   使用置信区间估算销售价格

+   理解置信区间背后的假设

## 推断统计学的核心

推断统计学使用数据样本对所抽样的总体进行推断。主要组成部分包括：

+   **置信区间：** 人口参数可能落在的范围。

+   **假设检验：** 对总体参数进行推断的过程。

当研究整个总体不切实际，需要从代表性样本中得出见解时，推断统计学是不可或缺的，就像在[Ames 属性数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)的情况一样。

**通过我的书籍** [《数据科学入门指南》](https://machinelearning.samcart.com/products/beginners-guide-data-science/) **来启动你的项目**。它提供了**自学教程**和**可运行的代码**。

## 什么是置信区间？

想象一下你从一个城市随机抽取了一些房屋，并计算了平均销售价格。虽然这为你提供了一个单一的估算值，但如果有一个范围可以表示整个城市的真实平均销售价格可能落在何处，那不是更有信息量吗？这个范围估计就是置信区间提供的。实际上，置信区间给了我们一个范围，在这个范围内，我们可以合理地确定（例如，95% 确定）真实的人口参数（如均值或比例）位于其中。

## 使用置信区间估算销售价格

虽然像均值和中位数这样的点估计能让我们了解中心趋势，但它们无法告知我们真实的总体参数可能落在什么范围内。置信区间填补了这一空白。例如，如果你想估计所有房屋的平均销售价格，你可以使用数据集计算 95%的置信区间。这一区间将给我们一个范围，我们可以 95%确信所有房屋的真实平均销售价格落在这个范围内。

你将使用 t 分布来找到置信区间：

```py
# Import the necessary libraries and load the data
import scipy.stats as stats
import pandas as pd
Ames = pd.read_csv('Ames.csv')

#Define the confidence level and degrees of freedom
confidence_level = 0.95
degrees_freedom = Ames['SalePrice'].count() - 1

#Calculate the confidence interval for 'SalePrice'
confidence_interval = stats.t.interval(confidence_level, degrees_freedom,
                                       loc=Ames['SalePrice'].mean(),
                                       scale=Ames['SalePrice'].sem())

# Print out the sentence with the confidence interval figures
print(f"The 95% confidence interval for the "
      f"true mean sales price of all houses in Ames is "
      f"between \${confidence_interval[0]:.2f} and \${confidence_interval[1]:.2f}.")
```

```py
The 95% confidence interval for the true mean sales price of all houses in Ames 
is between $175155.78 and $180951.11.
```

置信区间提供了一个范围，依据一定的置信水平，我们相信这个范围涵盖了真实的总体参数。解释这个范围可以让我们理解估计的变异性和精确度。如果均值‘SalePrice’的 95%置信区间是(\$175,156, \$180,951)，我们可以 95%确信所有 Ames 物业的真实平均销售价格介于\$175,156 和\$180,951 之间。

```py
# Import additional libraries
import matplotlib.pyplot as plt

# Plot the main histogram
plt.figure(figsize=(10, 7))
plt.hist(Ames['SalePrice'], bins=30, color='lightblue', edgecolor='black', alpha=0.5, label='Sales Prices Distribution')

# Vertical lines for sample mean and confidence interval with adjusted styles
plt.axvline(Ames['SalePrice'].mean(), color='blue', linestyle='-', label=f'Mean: ${Ames["SalePrice"].mean():,.2f}')
plt.axvline(confidence_interval[0], color='red', linestyle='--', label=f'Lower 95% CI: ${confidence_interval[0]:,.2f}')
plt.axvline(confidence_interval[1], color='green', linestyle='--', label=f'Upper 95% CI: ${confidence_interval[1]:,.2f}')

# Annotations and labels
plt.title('Distribution of Sales Prices with Confidence Interval', fontsize=20)
plt.xlabel('Sales Price', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.xlim([min(Ames['SalePrice']) - 5000, max(Ames['SalePrice']) + 5000])
plt.legend()
plt.grid(axis='y')
plt.show()
```

![](img/be6d577c4c53c3c3d902a916ac8d0d96.png)

销售价格和均值的分布

在上面的可视化中，直方图表示了 Ames 住房数据集中的销售价格分布。蓝色垂直线对应样本均值，提供了平均销售价格的点估计。虚线红色和绿色线分别表示 95%的下限和上限置信区间。

让我们*深入了解*在\$150,000 和\$200,000 之间的价格区间。

```py
# Creating a second plot focused on the mean and confidence intervals
plt.figure(figsize=(10, 7))
plt.hist(Ames['SalePrice'], bins=30, color='lightblue', edgecolor='black', alpha=0.5, label='Sales Prices')

# Zooming in around the mean and confidence intervals
plt.xlim([150000, 200000])

# Vertical lines for sample mean and confidence interval with adjusted styles
plt.axvline(Ames['SalePrice'].mean(), color='blue', linestyle='-', label=f'Mean: ${Ames["SalePrice"].mean():,.2f}')
plt.axvline(confidence_interval[0], color='red', linestyle='--', label=f'Lower 95% CI: ${confidence_interval[0]:,.2f}')
plt.axvline(confidence_interval[1], color='green', linestyle='--', label=f'Upper 95% CI: ${confidence_interval[1]:,.2f}')

# Annotations and labels for the zoomed-in plot
plt.title('Zoomed-in View of Mean and Confidence Intervals', fontsize=20)
plt.xlabel('Sales Price', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.legend()
plt.grid(axis='y')
plt.show()
```

![](img/72bb43c6c325005626a4a201c6af6fab.png)

置信区间的解释如下：我们 95%确信所有房屋的真实平均销售价格介于\$175,156 和\$180,951 之间。这个范围考虑了从样本估计总体参数时固有的变异性。从收集的样本中计算出的均值是\$178,053，但整个总体的实际值可能有所不同。换句话说，这个区间较窄，因为它是基于大量样本计算得出的。

### 想要开始数据科学初学者指南吗？

立即参加我的免费邮件速成课程（包括示例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## 理解置信区间背后的假设

在 Ames 房地产市场这个动态环境中熟练应用置信区间，关键在于理解支撑我们分析的基础假设。

**假设 1：随机抽样。**我们的分析假设数据是通过随机抽样过程收集的，确保 Ames 中的每个物业都有相等的被包含机会。这种随机性增强了我们对整个房地产市场发现的推广性。

**假设 2：中心极限定理 (CLT) 和大样本。**我们分析中的一个关键假设是中心极限定理 (CLT)，它使得在计算置信区间时可以使用 t 分布。中心极限定理认为，对于大样本，样本均值的抽样分布近似于正态分布，无论总体的分布如何。在我们的案例中，有 2,579 个观察值，中心极限定理得到了可靠满足。

**假设 3：独立性。**我们假设各个房屋的销售价格彼此独立。这个假设至关重要，确保一个房屋的销售价格不会影响另一个房屋的销售价格。这在 Ames 多样化的房地产市场中特别相关。

**假设 4：已知或估计的总体标准差（用于 Z-区间）。**虽然我们的主要方法涉及使用 t 分布，但值得注意的是，置信区间也可以通过 Z 分数计算，这要求知道或可靠估计总体标准差。然而，我们的分析偏向于 t 分布，当样本量较小或总体标准差未知时，t 分布更为稳健。

**假设 5：** **连续数据。** 置信区间应用于连续数据。在我们的背景中，Ames 的房屋销售价格是连续变量，因此置信区间适合用于估计总体参数。

这些假设构成了我们分析的基础，认识它们的作用并评估其有效性对可靠且有洞察力的房地产市场分析至关重要。违反这些假设可能会危及我们结论的可靠性。总之，我们的方法论基于 t 分布，利用这些假设为 Ames 的市场趋势和物业价值提供了细致的洞察。

## 进一步阅读

#### 教程

+   [置信区间教程](https://www.khanacademy.org/math/statistics-probability/confidence-intervals-one-sample)

+   [scipy.stats.t](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html) API

#### 资源

+   [Ames 数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)

+   [Ames 数据字典](https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt)

## **总结**

在这次探索中，我们通过 Ames 房地产数据集介绍了置信区间的概念。通过理解 Ames 房屋真实平均销售价格可能落在的范围，利益相关者可以在房地产市场中做出更明智的决策。

具体来说，你学到了：

+   推断统计中的置信区间的基础概念。

+   如何估计和解释 Ames 房地产市场中平均销售价格的 95% 置信区间。

+   置信区间计算的关键假设。

你有任何问题吗？请在下面的评论中提问，我将尽力回答。
