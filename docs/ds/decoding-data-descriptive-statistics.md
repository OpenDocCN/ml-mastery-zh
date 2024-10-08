# 解读数据：Ames 房价数据集的描述性统计入门

> 原文：[`machinelearningmastery.com/decoding-data-descriptive-statistics/`](https://machinelearningmastery.com/decoding-data-descriptive-statistics/)

你在 Ames 数据集上开始了你的数据科学之旅，通过描述性统计来分析数据。Ames 房价数据集的丰富性使得描述性统计能够将数据提炼成有意义的总结。这是分析中的初步步骤，提供了数据集主要方面的简洁总结。它们的意义在于简化复杂性，帮助数据探索，促进比较分析，并启用数据驱动的叙述。

当你深入研究 Ames 属性数据集时，你将探索描述性统计的变革力量，将大量数据提炼成有意义的总结。在这个过程中，你将发现关键指标及其解释的细微差别，比如均值大于中位数的偏度含义。

让我们开始吧。

![](img/0f66c31898eeb2f47fd4736e48c27aae.png)

解读数据：Ames 房价数据集的描述性统计入门

图片来源：[lilartsy](https://unsplash.com/photos/person-holding-on-red-pen-while-writing-on-book-333oj7zFsdg)。保留部分权利。

## 概述

本文分为三个部分；它们是：

+   描述性统计的基本概念

+   Ames 数据集的数据深入分析

+   视觉叙事

## 描述性统计的基本概念

本文将展示如何利用描述性统计来理解数据。让我们回顾一下统计学如何帮助描述数据。

### 中心趋势：数据的核心

中心趋势捕捉数据集的核心或典型值。最常见的衡量指标包括：

+   **均值（平均值）：** 所有值之和除以值的数量。

+   **中位数：** 当数据有序时的中间值。

+   **众数：** 出现频率最高的值。

### 离散度：分布和变异性

离散度揭示了数据集内的分布和变异性。主要的衡量指标包括：

+   **范围：** 最大值和最小值之间的差异。

+   **方差：** 均值的平方差的平均值。

+   **标准差：** 方差的平方根。

+   **四分位距（IQR）：** 第 25 百分位数和第 75 百分位数之间的范围。

### 形状和位置：数据的轮廓和标志

形状和位置揭示了数据集的分布形式和关键标记，其特征由以下指标描述：

+   **偏度：** 分布的不对称性。如果中位数大于均值，我们说数据是左偏的（大值更常见）。相反，它是右偏的。

+   **峰度：** 分布的“尾部程度”。换句话说，就是你看到异常值的频率。如果你比正常分布更频繁地看到极端的大值或小值，你可以说数据是**尖峰分布**的。

+   **百分位数：** 某个百分比的观测值低于的值。第 25、第 50 和第 75 百分位数也称为**四分位数**。

描述性统计使数据能够清晰而简明地讲述其故事。

**通过我的书** [《数据科学入门指南》](https://machinelearning.samcart.com/products/beginners-guide-data-science/) **启动你的项目**。它提供了**自学教程**和**有效的代码**。

## 使用艾姆斯数据集进行数据分析

要深入了解艾姆斯数据集，我们的重点是“SalePrice”属性。

```py
# Importing libraries and loading the dataset
import pandas as pd
Ames = pd.read_csv('Ames.csv')

# Descriptive Statistics of Sales Price
sales_price_description = Ames['SalePrice'].describe()
print(sales_price_description)
```

Python

```py
count 2579.000000
mean 178053.442420
std 75044.983207
min 12789.000000
25% 129950.000000
50% 159900.000000
75% 209750.000000
max 755000.000000
Name: SalePrice, dtype: float64
```

这总结了“SalePrice”，展示了计数、均值、标准差和百分位数。

```py
median_saleprice = Ames['SalePrice'].median()
print("Median Sale Price:", median_saleprice)

mode_saleprice = Ames['SalePrice'].mode().values[0]
print("Mode Sale Price:", mode_saleprice)
```

```py
Median Sale Price: 159900.0
Mode Sale Price: 135000
```

在艾姆斯的房屋平均“SalePrice”（或均值）大约为\$178,053.44，而中位数价格为\$159,900，这表明一半的房屋售价低于这个值。这些测量值之间的差异暗示了高价值房屋对平均值的影响，而众数则提供了最频繁的售价洞察。

```py
range_saleprice = Ames['SalePrice'].max() - Ames['SalePrice'].min()
print("Range of Sale Price:", range_saleprice)

variance_saleprice = Ames['SalePrice'].var()
print("Variance of Sale Price:", variance_saleprice)

std_dev_saleprice = Ames['SalePrice'].std()
print("Standard Deviation of Sale Price:", std_dev_saleprice)

iqr_saleprice = Ames['SalePrice'].quantile(0.75) - Ames['SalePrice'].quantile(0.25)
print("IQR of Sale Price:", iqr_saleprice)
```

```py
Range of Sale Price: 742211
Variance of Sale Price: 5631749504.563301
Standard Deviation of Sale Price: 75044.9832071625
IQR of Sale Price: 79800.0
```

“SalePrice”的范围从\$12,789 到\$755,000，展示了艾姆斯房地产价值的巨大多样性。方差大约为\$5.63 亿，突显了价格的显著波动，标准差约为\$75,044.98 进一步强调了这一点。四分位间距（IQR），表示数据中间的 50%，为\$79,800，反映了房价的集中范围。

```py
skewness_saleprice = Ames['SalePrice'].skew()
print("Skewness of Sale Price:", skewness_saleprice)

kurtosis_saleprice = Ames['SalePrice'].kurt()
print("Kurtosis of Sale Price:", kurtosis_saleprice)

tenth_percentile = Ames['SalePrice'].quantile(0.10)
ninetieth_percentile = Ames['SalePrice'].quantile(0.90)
print("10th Percentile:", tenth_percentile)
print("90th Percentile:", ninetieth_percentile)

q1_saleprice = Ames['SalePrice'].quantile(0.25)
q2_saleprice = Ames['SalePrice'].quantile(0.50)
q3_saleprice = Ames['SalePrice'].quantile(0.75)
print("Q1 (25th Percentile):", q1_saleprice)
print("Q2 (Median/50th Percentile):", q2_saleprice)
print("Q3 (75th Percentile):", q3_saleprice)
```

```py
Skewness of Sale Price: 1.7607507033716905
Kurtosis of Sale Price: 5.430410648673599
10th Percentile: 107500.0
90th Percentile: 272100.0000000001
Q1 (25th Percentile): 129950.0
Q2 (Median/50th Percentile): 159900.0
Q3 (75th Percentile): 209750.0
```

艾姆斯的“SalePrice”显示出约 1.76 的正偏度，表明分布右侧有较长或较胖的尾巴。这种偏度突显了平均售价受到一部分高价房产的影响，而大多数房屋的成交价格低于这个平均值。这种偏度量化了分布的不对称性或偏离对称性，突出了高价房产在提升平均值方面的不成比例影响。当平均值（均值）超过中位数时，微妙地表示存在高价房产，导致右偏分布，尾部显著向右延伸。大约 5.43 的峰度值进一步强调了这些洞察，表明可能存在增加分布尾部重量的极端值或异常值。

深入探讨，四分位值提供了数据中央趋势的见解。Q1 为\$129,950，Q3 为\$209,750，这些四分位数涵盖了四分位数间距，代表数据中间的 50%。这种划分强调了价格的中央分布，提供了定价范围的细致描绘。此外，10 百分位和 90 百分位分别位于\$107,500 和\$272,100，作为关键界限。这些百分位划定了 80%房价所在的范围，突显了物业估值的广泛范围，并强调了 Ames 住房市场的多面性。

## 视觉叙事

视觉化为数据注入了生命，讲述了其故事。让我们深入探讨 Ames 数据集中“SalePrice”特征的视觉叙事。

```py
# Importing visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Setting up the style
sns.set_style("whitegrid")

# Calculate Mean, Median, Mode for SalePrice
mean_saleprice = Ames['SalePrice'].mean()
median_saleprice = Ames['SalePrice'].median()
mode_saleprice = Ames['SalePrice'].mode().values[0]

# Plotting the histogram
plt.figure(figsize=(14, 7))
sns.histplot(x=Ames['SalePrice'], bins=30, kde=True, color="skyblue")
plt.axvline(mean_saleprice, color='r', linestyle='--', label=f"Mean: ${mean_saleprice:.2f}")
plt.axvline(median_saleprice, color='g', linestyle='-', label=f"Median: ${median_saleprice:.2f}")
plt.axvline(mode_saleprice, color='b', linestyle='-.', label=f"Mode: ${mode_saleprice:.2f}")

# Calculating skewness and kurtosis for SalePrice
skewness_saleprice = Ames['SalePrice'].skew()
kurtosis_saleprice = Ames['SalePrice'].kurt()

# Annotations for skewness and kurtosis
plt.annotate('Skewness: {:.2f}\nKurtosis: {:.2f}'.format(Ames['SalePrice'].skew(), Ames['SalePrice'].kurt()),
             xy=(500000, 100), fontsize=14, bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="aliceblue"))

plt.title('Histogram of Ames\' Housing Prices with KDE and Reference Lines')
plt.xlabel('Housing Prices')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

![](https://machinelearningmastery.com/wp-content/uploads/2024/01/Figure_3-1.png)

上述直方图提供了 Ames 房价的引人注目的视觉表示。接近\$150,000 的显著峰值强调了在这一特定价格范围内房屋的集中度。与直方图相辅的是核密度估计（KDE）曲线，它提供了数据分布的平滑表示。KDE 本质上是直方图的估计，但具有**无限窄的区间**优势，提供了数据的更连续视图。它作为直方图的“极限”或精细版本，捕捉了离散分箱方法中可能遗漏的细微差别。

值得注意的是，KDE 曲线的右尾与我们之前计算出的正偏斜度一致，强调了**低于均值的房屋浓度**较高。颜色线条——红色表示均值，绿色表示中位数，蓝色表示众数——作为关键标记，允许快速比较和理解分布的中央趋势与更广泛数据景观的关系。这些视觉元素一起提供了对 Ames 房价分布和特征的全面洞察。

Python

```py
from matplotlib.lines import Line2D

# Horizontal box plot with annotations
plt.figure(figsize=(12, 8))

# Plotting the box plot with specified color and style
sns.boxplot(x=Ames['SalePrice'], color='skyblue', showmeans=True, meanprops={"marker": "D", "markerfacecolor": "red",
                                                                             "markeredgecolor": "red", "markersize":10})

# Plotting arrows for Q1, Median and Q3
plt.annotate('Q1', xy=(q1_saleprice, 0.30), xytext=(q1_saleprice - 70000, 0.45),
             arrowprops=dict(edgecolor='black', arrowstyle='->'), fontsize=14)
plt.annotate('Q3', xy=(q3_saleprice, 0.30), xytext=(q3_saleprice + 20000, 0.45),
             arrowprops=dict(edgecolor='black', arrowstyle='->'), fontsize=14)
plt.annotate('Median', xy=(q2_saleprice, 0.20), xytext=(q2_saleprice - 90000, 0.05),
             arrowprops=dict(edgecolor='black', arrowstyle='->'), fontsize=14)

# Titles, labels, and legends
plt.title('Box Plot Ames\' Housing Prices', fontsize=16)
plt.xlabel('Housing Prices', fontsize=14)
plt.yticks([])  # Hide y-axis tick labels
plt.legend(handles=[Line2D([0], [0], marker='D', color='w', markerfacecolor='red', markersize=10, label='Mean')],
           loc='upper left', fontsize=14)

plt.tight_layout()
plt.show()
```

![](https://machinelearningmastery.com/wp-content/uploads/2024/01/Figure_3-2.png)

箱线图提供了中央趋势、范围和异常值的简洁表示，提供了 KDE 曲线或直方图难以清晰展示的见解。跨越 Q1 到 Q3 的四分位数间距（IQR）捕捉了数据中间的 50%，提供了价格中央范围的清晰视图。此外，代表均值的红色钻石位于中位数的右侧，突显了高价值物业对平均值的影响。

解读箱形图的关键在于其“胡须”。左胡须从箱体的左边缘延伸到下边界内的最小数据点，表示低于 Q1 的 1.5 倍 IQR 的价格。相对地，右胡须从箱体的右边缘延伸到上边界内的最大数据点，涵盖了高于 Q3 的 1.5 倍 IQR 的价格。这些胡须作为界限，划分了数据在中央 50%之外的扩展，超出这些范围的点通常被标记为潜在异常值。

异常值以单独的点表示，突出了价格特别高的房屋，可能是奢侈物业或具有独特特征的房屋。箱形图中的异常值是低于 Q1 的 1.5 倍 IQR 或高于 Q3 的 1.5 倍 IQR。在上图中，低端没有异常值，但高端有很多。识别和理解这些异常值至关重要，因为它们可以揭示 Ames 房地产市场中的独特市场动态或异常情况。

像这样的可视化将原始数据赋予生命，编织引人入胜的叙事，揭示可能在简单数字中隐藏的见解。随着我们前进，认识到并接受可视化在数据分析中的深远影响至关重要——它具有独特的能力，能够传达仅凭文字或数字无法捕捉的细微差别和复杂性。

### 想要开始学习数据科学初学者指南吗？

立即参加我的免费电子邮件速成课程（附带示例代码）。

点击注册，同时获得课程的免费 PDF 电子书版本。

## **进一步阅读**

本节提供了更多资源，以便您深入了解该主题。

#### **资源**

+   [Ames 数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)

+   [Ames 数据字典](https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt)

## **总结**

在本教程中，我们通过描述性统计深入探讨了 Ames Housing 数据集，以揭示关于房产销售的关键见解。我们计算并可视化了基本的统计测量，强调了集中趋势、离散度和形态的价值。通过利用可视化叙事和数据分析，我们将原始数据转化为引人入胜的故事，揭示了 Ames 房价的复杂性和模式。

具体来说，您学习了：

+   如何利用描述性统计从 Ames Housing 数据集中提取有意义的见解，重点关注‘SalePrice’属性。

+   均值、中位数、众数、范围和 IQR 等测量指标的意义，以及它们如何讲述 Ames 房价的故事。

+   可视化叙事的力量，特别是直方图和箱形图，在直观呈现和解释数据的分布和变异性方面。

有任何问题吗？请在下面的评论中提问，我会尽力回答。
