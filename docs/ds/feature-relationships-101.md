# 特征关系 101：来自 Ames Housing 数据的教训

> 原文：[`machinelearningmastery.com/feature-relationships-101/`](https://machinelearningmastery.com/feature-relationships-101/)

在房地产领域，了解物业特征及其对销售价格的影响至关重要。在此探索中，我们将深入分析 Ames Housing 数据集，揭示各种特征之间的关系以及它们与销售价格的相关性。通过数据可视化的力量，我们将揭示模式、趋势和见解，以指导从房主到房地产开发商的利益相关者。

让我们开始吧。

![](img/9ff9df9b842e10d8c54dacf73e85ef53.png)

特征关系 101：来自 Ames Housing 数据的教训

图片由 [Andraz Lazic](https://unsplash.com/photos/white-feather-on-body-of-water-in-shallow-focus-64sgR8HV_68) 提供。部分权利保留。

## 概述

本文分为三部分；它们是：

+   揭示相关性

+   通过热图进行可视化

+   通过散点图分析特征关系

## 揭示相关性

相关性是一个统计度量，显示两个变量共同变化的程度。正相关表示一个变量增加时，另一个变量也倾向于增加，反之亦然。相反，负相关表示一个变量增加时，另一个变量倾向于减少。

```py
# Load the Dataset
import pandas as pd
Ames = pd.read_csv('Ames.csv')

# Calculate the correlation of all features with 'SalePrice'
correlations = Ames.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False)

# Display the top 10 features most correlated with 'SalePrice'
top_correlations = correlations[1:11]
print(top_correlations)
```

这将打印：

```py
OverallQual     0.790661
GrLivArea       0.719980
TotalBsmtSF     0.652268
1stFlrSF        0.642623
GarageCars      0.639017
GarageArea      0.635029
YearBuilt       0.544569
FullBath        0.535175
GarageYrBlt     0.521105
YearRemodAdd    0.514720
Name: SalePrice, dtype: float64
```

从 Ames Housing 数据集中，与房价最相关的主要特征有：

+   **OverallQual:** 房屋的总体质量，评分范围从 1（非常差）到 10（非常优秀）。

+   **GrLivArea:** 地上生活面积，以平方英尺为单位。包括不在地下室中的生活区域。

+   **TotalBsmtSF:** 总地下室面积，以平方英尺表示。这包括了地下室的完成和未完成区域。

+   **1stFlrSF:** 一楼平方英尺，表示房屋第一层的大小。

+   **GarageCars:** 车库容量。表示车库可以容纳的汽车数量。

+   **GarageArea:** 车库面积，以平方英尺为单位。这显示了车库所覆盖的总面积。

+   **YearBuilt:** 原始建设日期，指示房屋主要施工完成的年份。

+   **FullBath:** 地上完整浴室数量。计算不在地下室中的完整浴室数量（即有洗手池、马桶和浴缸或淋浴）。

+   **GarageYrBlt:** 车库建造年份。指定车库建造的年份。对于没有车库的房屋，此特征可能为空。

+   **YearRemodAdd:** 改造日期。指示改造或增加的年份，如果没有改造或增加，则与建设年份相同。

最相关的特征是具有最佳预测能力的特征。如果你建立一个预测房价的模型，这些特征是输入特征中成功可能性较高的子集。相关特征也可能是由某些共同因素引起的，这本身是数据科学中的一个主题，你可能会想要调查和详细阐述。

上述代码打印了`correlations[1:11]`，因为`correlations[0]`是 SalesPrice，按照定义其值为 1.0。从特征选择的角度来看，你还应该检查`correlations[-10:]`，以了解最负相关的特征，这些特征可能在解释价格方面也很强大。然而，在这个特定的数据集中并非如此。

**通过我的书籍** [《数据科学入门指南》](https://machinelearning.samcart.com/products/beginners-guide-data-science/) **开启你的项目**。它提供了**自学教程**和**工作代码**。

## 使用热图进行可视化

热图提供了一种强大的可视化工具，用于在二维空间中表示数据，颜色指示了大小或频率。在相关性分析的背景下，热图可以生动地展示多个特征之间的关系强度和方向。让我们深入了解一张展示与*SalePrice*最相关的顶级特征的热图。

```py
# Import required libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Select the top correlated features including SalePrice
selected_features = list(top_correlations.index) + ['SalePrice']

# Compute the correlations for the selected features
correlation_matrix = Ames[selected_features].corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Generate a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=.5, fmt=".2f", vmin=-1, vmax=1)

# Title
plt.title("Heatmap of Correlations among Top Features with SalePrice", fontsize=16)

# Show the heatmap
plt.show()
```

## ![](https://machinelearningmastery.com/wp-content/uploads/2023/12/Figure_1.png)

热图是同时可视化多个变量之间关系强度和方向的绝佳方式。热图中每个单元格的颜色强度对应于相关性的大小，暖色表示正相关，冷色表示负相关。由于上述热图仅涉及 10 个正相关性最高的列，因此没有蓝色。

在上面的热图中，我们可以观察到以下内容：

+   **OverallQual**，表示房屋的整体质量，与**SalePrice**的相关性最强，相关系数约为 0.79。这意味着随着房屋质量的提高，销售价格也有上升的趋势。

+   **GrLivArea**和**TotalBsmtSF**，分别表示地上生活面积和总地下室面积，也与销售价格显示出强烈的正相关。

+   大多数特征与**SalePrice**呈正相关，这表明当这些特征增加或改善时，房屋的销售价格也有上升的趋势。

+   值得注意的是，有些特征彼此相关。例如，**GarageCars**和**GarageArea**之间的相关性很强，这很有意义，因为较大的车库可以容纳更多的汽车。

这些见解对房地产领域的各类利益相关者来说非常宝贵。例如，房地产开发商可以集中精力提升房屋的特定特征，以提高其市场价值。

以下是完整的代码：

```py
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the Dataset
Ames = pd.read_csv('Ames.csv')

# Calculate the top 10 features most correlated with 'SalePrice'
correlations = Ames.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False)
top_correlations = correlations[1:11]

# Select the top correlated features including SalePrice
selected_features = list(top_correlations.index) + ['SalePrice']

# Compute the correlations for the selected features
correlation_matrix = Ames[selected_features].corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Generate a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=.5, fmt=".2f", vmin=-1, vmax=1)
plt.title("Heatmap of Correlations among Top Features with SalePrice", fontsize=16)
plt.show()
```

## 通过散点图解剖特征关系

虽然相关性提供了对关系的初步理解，但进一步可视化这些关系至关重要。例如，散点图可以更清晰地描绘两个特征如何相互作用。此外，区分相关性和因果关系也很重要。高相关性并不一定意味着一个变量导致另一个变量的变化，它仅仅表示存在关系。

```py
# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Ames = pd.read_csv('Ames.csv')

# Setting up the figure and axes
fig, ax = plt.subplots(2, 2, figsize=(15, 12))

# Scatter plot for SalePrice vs. OverallQual
sns.scatterplot(x=Ames['OverallQual'], y=Ames['SalePrice'], ax=ax[0, 0], color='blue', alpha=0.6)
ax[0, 0].set_title('House Prices vs. Overall Quality')
ax[0, 0].set_ylabel('House Prices')
ax[0, 0].set_xlabel('Overall Quality')

# Scatter plot for SalePrice vs. GrLivArea
sns.scatterplot(x=Ames['GrLivArea'], y=Ames['SalePrice'], ax=ax[0, 1], color='red', alpha=0.6)
ax[0, 1].set_title('House Prices vs. Ground Living Area')
ax[0, 1].set_ylabel('House Prices')
ax[0, 1].set_xlabel('Above Ground Living Area (sq. ft.)')

# Scatter plot for SalePrice vs. TotalBsmtSF
sns.scatterplot(x=Ames['TotalBsmtSF'], y=Ames['SalePrice'], ax=ax[1, 0], color='green', alpha=0.6)
ax[1, 0].set_title('House Prices vs. Total Basement Area')
ax[1, 0].set_ylabel('House Prices')
ax[1, 0].set_xlabel('Total Basement Area (sq. ft.)')

# Scatter plot for SalePrice vs. 1stFlrSF
sns.scatterplot(x=Ames['1stFlrSF'], y=Ames['SalePrice'], ax=ax[1, 1], color='purple', alpha=0.6)
ax[1, 1].set_title('House Prices vs. First Floor Area')
ax[1, 1].set_ylabel('House Prices')
ax[1, 1].set_xlabel('First Floor Area (sq. ft.)')

# Adjust layout
plt.tight_layout(pad=3.0)
plt.show()
```

![](https://machinelearningmastery.com/wp-content/uploads/2023/12/Figure_2.png)

散点图强调了销售价格与关键特征之间的强正相关关系。随着整体质量、地面生活面积、地下室面积和一楼面积的增加，房屋通常会获得更高的价格。然而，一些例外和离群点表明其他因素也影响最终销售价格。一个特别的例子是上面的“房价与地面生活面积”散点图：在 2500 平方英尺及以上，点的分布较散，表明在这个面积范围内，房价的变化不与面积有强相关或无法有效解释。

### 想开始数据科学初学者指南吗？

立即参加我的免费电子邮件速成课程（包括示例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## **进一步阅读**

本节提供了更多关于该主题的资源，供你深入了解。

#### **资源**

+   [Ames 数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)

+   [Ames 数据字典](https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt)

## **总结**

在探索 Ames Housing 数据集时，我们踏上了了解各种属性特征与销售价格之间关系的旅程。通过热力图和散点图，我们揭示了对房地产利益相关者有重大影响的模式和见解。

具体而言，你学到了：

+   相关性的重要性及其在理解属性特征与销售价格之间关系中的意义。

+   热力图在直观表示多个特征之间的相关性中的实用性。

+   散点图所增加的深度，强调了超越简单相关系数的个体特征动态分析的重要性。

你有任何问题吗？请在下面的评论中提问，我会尽力回答。
