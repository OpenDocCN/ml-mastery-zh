# 利用 ANOVA 和 Kruskal-Wallis 检验分析大萧条对房价的影响

> 原文：[`machinelearningmastery.com/leveraging-anova-and-kruskal-wallis-tests-to-analyze-the-impact-of-the-great-recession-on-housing-prices/`](https://machinelearningmastery.com/leveraging-anova-and-kruskal-wallis-tests-to-analyze-the-impact-of-the-great-recession-on-housing-prices/)

在房地产领域，许多因素会影响房产价格。经济状况、市场需求、地理位置甚至房产销售年份都可能发挥重要作用。2007 年至 2009 年是美国房地产市场动荡的时期。这段时间，通常被称为“大萧条”，经历了房价的大幅下跌、止赎案件激增和金融市场的广泛动荡。大萧条对房价的影响深远，许多房主发现自己所拥有的房产价值低于贷款金额。这一下滑的连锁反应在全国范围内产生了影响，一些地区的房价下跌更为严重，恢复也更为缓慢。

鉴于此背景，分析来自爱荷华州艾姆斯的房屋数据尤为引人注目，因为数据集覆盖了 2006 年至 2010 年，这一时期囊括了大萧条的高峰和余波。在如此经济波动的环境下，销售年份是否会影响艾姆斯的销售价格？在本文中，你将深入探讨艾姆斯房屋数据集，使用探索性数据分析（EDA）和两种统计检验：ANOVA 和 Kruskal-Wallis 检验来探索这一问题。

让我们开始吧。

![](img/ecd9c55889ddd98de74ed6aeb02bb03b.png)

利用 ANOVA 和 Kruskal-Wallis 检验分析大萧条对房价的影响

照片由[Sharissa Johnson](https://unsplash.com/photos/brown-rock-formation-on-sea-during-daytime-t0uKpnS2SIg)提供。部分权利保留。

## 概述

本文分为三个部分，它们是：

+   EDA：可视化洞察

+   使用 ANOVA 评估不同年份的销售价格变异性

+   Kruskal-Wallis 检验：ANOVA 的非参数替代方法

## EDA：可视化洞察

首先，让我们加载[Ames Housing 数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)并将不同年份的销售数据与依赖变量：销售价格进行比较。

```py
# Importing the essential libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
Ames = pd.read_csv('Ames.csv')

# Convert 'YrSold' to a categorical variable
Ames['YrSold'] = Ames['YrSold'].astype('category')

plt.figure(figsize=(10, 6))
sns.boxplot(x=Ames['YrSold'], y=Ames['SalePrice'], hue=Ames['YrSold'])
plt.title('Boxplot of Sales Prices by Year', fontsize=18)
plt.xlabel('Year Sold', fontsize=15)
plt.ylabel('Sales Price (US$)', fontsize=15)
plt.legend('')
plt.show()
```

![](img/014eb9c949387baae02790f1a09af710.png)

比较销售价格的趋势

从箱线图中可以观察到，销售价格在不同年份之间相当一致，因为每年的情况都相似。让我们使用 pandas 中的`groupby`函数来进一步分析。

```py
# Calculating mean and median sales price by year
summary_table = Ames.groupby('YrSold')['SalePrice'].agg(['mean', 'median'])

# Rounding the values for better presentation
summary_table = summary_table.round(2)
print(summary_table)
```

输出结果是：

```py
             mean    median
YrSold                     
2006    176615.62  157000.0
2007    179045.08  159000.0
2008    178170.02  162700.0
2009    180387.64  162000.0
2010    173971.67  157900.0
```

从表格中可以得出以下观察结果：

1.  **均值**销售价格在 2009 年最高，约为\$180,388，而 2010 年最低，约为\$173,972。

1.  **中位数**销售价格在 2008 年最高，为\$162,700，在 2006 年最低，为\$157,000。

1.  即使每年的均值和中位数销售价格接近，也存在轻微的变化。这表明，虽然可能有一些离群值影响了均值，但它们并没有极端偏斜。

1.  在这五年里，销售价格似乎没有保持一致的上升或下降趋势，考虑到这一时期的更大经济背景（大萧条），这一点非常有趣。

结合箱线图，这张表提供了销售价格在不同年份间分布和中心倾向的全面视角。这为进一步的统计分析奠定了基础，以确定观察到的差异（或其缺失）是否具有统计学显著性。

**启动你的项目**，使用我的书籍[《数据科学初学者指南》](https://machinelearning.samcart.com/products/beginners-guide-data-science/)。它提供了**自学教程**和**工作代码**。

## 使用 ANOVA 评估销售价格在不同年份之间的变异性

ANOVA（方差分析）帮助我们测试三个或更多独立组之间是否存在任何统计显著差异。其零假设是所有组的平均值相等。这可以视为支持多组比较的 t 检验版本。它利用 F 检验统计量来检查每组内方差（$\sigma²$）与所有组间方差的差异。

假设设置为：

+   $H_0$: 所有年份的销售价格均值相等。

+   $H_1$: 至少有一个年份的平均销售价格不同。

你可以使用`scipy.stats`库运行你的测试，如下所示：

```py
# Import an additional library
import scipy.stats as stats

# Perform the ANOVA
f_value, p_value = stats.f_oneway(*[Ames['SalePrice'][Ames['YrSold'] == year]
                                    for year in Ames['YrSold'].unique()])

print(f_value, p_value)
```

这两个数值是：

```py
0.4478735462379817 0.774024927554816
```

ANOVA 测试的结果为：

+   **F 值:** 0.4479

+   **p 值:** 0.7740

鉴于高的*p 值*（大于常见显著水平 0.05），你无法拒绝零假设（$H_0$）。这表明，在数据集中，销售价格的平均值之间不存在统计显著差异。

当你的 ANOVA 结果提供了跨不同年份平均值平等的见解时，确保检验假设条件的符合性至关重要。让我们深入验证 ANOVA 测试的三个假设，以验证你的发现。

**假设 1：观察值的独立性。** 由于每个观察（房屋销售）是相互独立的，这一假设得到满足。

**假设 2：残差的正态性。** 对于 ANOVA 的有效性，模型的残差应当近似服从**正态分布**，因为这是 F 检验背后的模型。你可以通过视觉和统计方法来检验这一点。

可以使用 QQ 图进行视觉评估：

```py
# Import an additional library
import statsmodels.api as sm

# Fit an ordinary least squares model and get residuals
model = sm.OLS(Ames['SalePrice'], Ames['YrSold'].astype('int')).fit()
residuals = model.resid

# Plot QQ plot
sm.qqplot(residuals, line='s')
plt.title('Normality Assessment of Residuals via QQ Plot', fontsize=18)
plt.xlabel('Theoretical Quantiles', fontsize=15)
plt.ylabel('Sample Residual Quantiles', fontsize=15)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
```

![](https://machinelearningmastery.com/wp-content/uploads/2024/01/Figure_2-3.png)

上述 QQ 图作为评估数据集残差正态性的重要视觉工具，提供了观察数据与正态分布理论期望的匹配程度的见解。在此图中，每个点代表一对分位数：一个来自数据的残差，另一个来自标准正态分布。如果你的数据完全符合正态分布，那么 QQ 图上的所有点应恰好落在红色的 45 度参考线上。图示出相对于 45 度参考线的偏离，暗示了可能的正态性偏离。

统计评估可以使用 Shapiro-Wilk 检验，它提供了一种正式的正态性检验方法。该检验的原假设是数据服从正态分布。这个检验也可以在 SciPy 中使用：

```py
#Import shapiro from scipy.stats package
from scipy.stats import shapiro

# Shapiro-Wilk Test
shapiro_stat, shapiro_p = shapiro(residuals)
print(f"Shapiro-Wilk Test Statistic: {shapiro_stat}\nP-value: {shapiro_p}")
```

输出为：

```py
Shapiro-Wilk Test Statistic: 0.8774482011795044
P-value: 4.273399796804962e-41
```

低 p 值（通常是*p* < 0.05）表明拒绝原假设，说明残差不符合正态分布。这表明 ANOVA 的第二个假设违反了，该假设要求残差服从正态分布。QQ 图和 Shapiro-Wilk 检验都得出了相同的结论：残差不严格遵循正态分布。因此，ANOVA 的结果可能无效。

**假设 3：方差齐性。** 组（年份）的方差应大致相等。这恰好是 Levene 检验的原假设。因此，你可以使用它来验证：

```py
# Check for equal variances using Levene's test
levene_stat, levene_p = stats.levene(*[Ames['SalePrice'][Ames['YrSold'] == year]
                                      for year in Ames['YrSold'].unique()])

print(f"Levene's Test Statistic: {levene_stat}\nP-value: {levene_p}")
```

输出为：

```py
Levene's Test Statistic: 0.2514412478357097
P-value: 0.9088910499612235
```

由于 Levene 检验的 p 值为 0.909，你不能拒绝原假设，表明不同年份的销售价格方差在统计上是齐性的，满足 ANOVA 的第三个关键假设。

综合来看，以下代码运行 ANOVA 检验并验证三个假设：

```py
# Importing the essential libraries
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro

# Load the dataset
Ames = pd.read_csv('Ames.csv')

# Perform the ANOVA
f_value, p_value = stats.f_oneway(*[Ames['SalePrice'][Ames['YrSold'] == year]
                                    for year in Ames['YrSold'].unique()])
print("F-value:", f_value)
print("p-value:", p_value)

# Fit an ordinary least squares model and get residuals
model = sm.OLS(Ames['SalePrice'], Ames['YrSold'].astype('int')).fit()
residuals = model.resid

# Plot QQ plot
sm.qqplot(residuals, line='s')
plt.title('Normality Assessment of Residuals via QQ Plot', fontsize=18)
plt.xlabel('Theoretical Quantiles', fontsize=15)
plt.ylabel('Sample Residual Quantiles', fontsize=15)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# Shapiro-Wilk Test
shapiro_stat, shapiro_p = shapiro(residuals)
print(f"Shapiro-Wilk Test Statistic: {shapiro_stat}\nP-value: {shapiro_p}")

# Check for equal variances using Levene's test
levene_stat, levene_p = stats.levene(*[Ames['SalePrice'][Ames['YrSold'] == year]
                                      for year in Ames['YrSold'].unique()])

print(f"Levene's Test Statistic: {levene_stat}\nP-value: {levene_p}")
```

## Kruskal-Wallis 检验：ANOVA 的非参数替代方法

Kruskal-Wallis 检验是一种非参数方法，用于比较三个或更多独立组的中位值，是单因素 ANOVA 的合适替代方法（特别是当 ANOVA 假设不成立时）。

非参数统计是一类不对数据的潜在分布做明确假设的统计方法。与假设特定分布的 *参数* 检验不同（例如上述假设 2 中的正态分布），*非参数* 检验更加灵活，适用于可能不符合参数方法严格假设的数据。非参数检验特别适用于处理有序或名义数据，以及可能表现出偏斜或重尾的数据。这些检验关注值的顺序或等级，而非具体值本身。非参数检验，包括 Kruskal-Wallis 检验，提供了一种灵活且不依赖分布的统计分析方法，使其适用于各种数据类型和情况。

Kruskal-Wallis 检验下的假设设置为：

+   $H_0$: 所有年份的销售价格分布相同。

+   $H_1$: 至少有一年销售价格的分布不同。

你可以使用 SciPy 运行 Kruskal-Wallis 检验，步骤如下：

```py
# Perform the Kruskal-Wallis H-test
H_statistic, kruskal_p_value = stats.kruskal(*[Ames['SalePrice'][Ames['YrSold'] == year]
                                               for year in Ames['YrSold'].unique()])

print(H_statistic, kruskal_p_value)
```

输出结果为：

```py
2.1330989438609236 0.7112941815590765
```

Kruskal-Wallis 检验的结果为：

+   **H 统计量：** 2.133

+   **p 值：** 0.7113

**注意**：Kruskal-Wallis 检验并不特别测试均值的差异（如 ANOVA），而是测试分布的差异。这可以包括中位数、形状和范围的差异。

鉴于较高的 *p 值*（大于常见的显著性水平 0.05），你不能拒绝原假设。这表明在数据集中不同年份的中位销售价格之间没有统计学上显著的差异。让我们深入验证 Kruskal-Wallis 检验的 3 个假设，以验证你的发现。

**假设 1：观察值的独立性。** 这一点与 ANOVA 相同；每个观察值相互独立。

**假设 2：响应变量应为有序、区间或比例变量。** 销售价格是比例变量，因此满足此假设。

**假设 3：响应变量的分布应在所有组中相同。** 可以使用视觉和数值方法进行验证。

```py
# Plot histograms of Sales Price for each year
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 8), sharex=True)

for idx, year in enumerate(sorted(Ames['YrSold'].unique())):
    sns.histplot(Ames[Ames['YrSold'] == year]['SalePrice'], kde=True, ax=axes[idx], color='skyblue')
    axes[idx].set_title(f'Distribution of Sales Prices for Year {year}', fontsize=16)
    axes[idx].set_ylabel('Frequency', fontsize=14)
    if idx == 4:
        axes[idx].set_xlabel('Sales Price', fontsize=15)
    else:
        axes[idx].set_xlabel('')

plt.tight_layout()
plt.show()
```

![](img/7f43551a4866b31650c7046cf6766098.png)

不同年份的销售价格分布

堆叠直方图表明各年份销售价格分布一致，每一年显示出类似的范围和峰值，尽管频率上有轻微的变化。

此外，你还可以进行配对 Kolmogorov-Smirnov 检验，这是一种非参数检验，用于比较两个概率分布的相似性。它在 SciPy 中可用。你可以使用原假设为两个分布相等，备择假设为不相等的版本：

```py
# Run KS Test from scipy.stats
from scipy.stats import ks_2samp
results = {}
for i, year1 in enumerate(sorted(Ames['YrSold'].unique())):
    for j, year2 in enumerate(sorted(Ames['YrSold'].unique())):
        if i < j:
            ks_stat, ks_p = ks_2samp(Ames[Ames['YrSold'] == year1]['SalePrice'], 
                                     Ames[Ames['YrSold'] == year2]['SalePrice'])
            results[f"{year1} vs {year2}"] = (ks_stat, ks_p)

# Convert the results into a DataFrame for tabular representation
ks_df = pd.DataFrame(results).transpose()
ks_df.columns = ['KS Statistic', 'P-value']
ks_df.reset_index(inplace=True)
ks_df.rename(columns={'index': 'Years Compared'}, inplace=True)

print(ks_df)
```

这表明：

```py
  Years Compared  KS Statistic   P-value
0   2006 vs 2007      0.038042  0.798028
1   2006 vs 2008      0.052802  0.421325
2   2006 vs 2009      0.062235  0.226623
3   2006 vs 2010      0.040006  0.896946
4   2007 vs 2008      0.039539  0.732841
5   2007 vs 2009      0.044231  0.586558
6   2007 vs 2010      0.051508  0.620135
7   2008 vs 2009      0.032488  0.908322
8   2008 vs 2010      0.052752  0.603031
9   2009 vs 2010      0.053236  0.586128
```

尽管我们仅满足了 ANOVA 的 3 个假设中的 2 个，但我们已经满足了 Kruskal-Wallis 检验的所有必要条件。成对的 Kolmogorov-Smirnov 检验表明，不同年份的销售价格分布非常一致。具体而言，高 p 值（均大于常见的显著性水平 0.05）意味着没有足够的证据来拒绝每年销售价格来自同一分布的假设。这些发现满足了 Kruskal-Wallis 检验的假设，即响应变量的分布在所有组中应相同。这强调了尽管面临更广泛的经济背景，从 2006 年到 2010 年埃姆斯的销售价格分布的稳定性。

### 想要开始数据科学的初学者指南吗？

现在就来参加我的免费电子邮件速成课程（包含示例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## **进一步阅读**

#### 在线

+   [ANOVA | 统计解决方案](https://www.statisticssolutions.com/anova-analysis-of-variance/)

+   [Python 中的 Kruskal-Wallis H 检验](https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/)

+   [方差分析](https://en.wikipedia.org/wiki/Analysis_of_variance) 维基百科

+   [scipy.stats.f_oneway](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html) API

+   [scipy.stats.shapiro](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html) API

+   [scipy.stats.levene](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html) API

+   [scipy.stats.kruskal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html) API

+   [scipy.stats.ks_2samp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html) API

#### **资源**

+   [埃姆斯数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)

+   [埃姆斯数据字典](https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt)

## 总结

在房地产的多维世界中，包括销售年份在内的多个因素可能会影响房产价格。美国住房市场在 2007 年至 2009 年大萧条期间经历了相当大的动荡。本研究关注的是 2006 年至 2010 年间来自爱荷华州埃姆斯的住房数据，旨在确定销售年份是否会影响销售价格，尤其是在这一动荡时期。

本分析使用了 ANOVA 和 Kruskal-Wallis 检验来评估不同年份销售价格的变异性。虽然 ANOVA 的结果具有指导性，但并非所有的基本假设都得到了满足，特别是残差的正态性。相反，Kruskal-Wallis 检验满足了所有的标准，提供了更可靠的见解。因此，单独依赖 ANOVA 可能会产生误导，而没有 Kruskal-Wallis 检验的佐证。

单因素 ANOVA 和 Kruskal-Wallis 检验都得出了相一致的结果，表明不同年份之间的销售价格没有统计学上的显著差异。考虑到 2006 到 2010 年间经济的动荡，这一结果尤为引人注目。研究结果表明，艾姆斯的房地产价格非常稳定，主要受地方因素影响。

具体来说，你学到了：

+   验证统计检验假设的重要性，例如 ANOVA 的残差正态性挑战。

+   参数检验（ANOVA）和非参数检验（Kruskal-Wallis）在比较数据分布中的重要性和应用。

+   如何利用地方因素来保护房地产市场，如爱荷华州的艾姆斯，免受更广泛经济下滑的影响，强调房地产定价的微妙性质。

你有任何问题吗？请在下面的评论中提出你的问题，我将尽力回答。
