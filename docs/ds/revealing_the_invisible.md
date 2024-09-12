# 揭示隐形：可视化 Ames 住房中的缺失值

> 原文：[`machinelearningmastery.com/revealing_the_invisible/`](https://machinelearningmastery.com/revealing_the_invisible/)

数字时代带来了一个数据驱动决策至关重要的时代，房地产就是一个典型的例子。像 Ames 房产这样的综合数据集，为数据爱好者提供了丰富的宝藏。通过细致的探索和分析这些数据集，人们可以发现模式，获得洞见，并做出明智的决策。

从这篇文章开始，你将踏上一次引人入胜的旅程，深入探索 Ames 房产的复杂领域，主要集中在数据科学技术上。

让我们开始吧。

![](img/62b3f5a77321631871e0ca726f549d6e.png)

揭示隐形：可视化 Ames 住房中的缺失值

照片由[Joakim Honkasalo](https://unsplash.com/photos/beige-and-black-lighthouse-on-hill-with-starry-sky-xNRWtb6mkao)提供，版权所有

## 概述

本文分为三个部分，分别是：

+   **Ames 房产数据集**

***   加载与评估数据集*   揭示与可视化缺失值**

**## **Ames**房产**数据集**

每个数据集都有一个故事，理解其背景可以提供宝贵的背景信息。虽然 Ames Housing Dataset 在学术界广为人知，但我们今天分析的数据集，[`Ames.csv`](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)，是一个更全面的 Ames 房产细节集合。

学者 Dr. Dean De Cock 认识到房地产领域需要一个新的、强大的数据集。他细致地编制了[Ames Housing Dataset](https://raw.githubusercontent.com/Padre-Media/dataset/main/decock.pdf)，该数据集自此成为了新兴数据科学家和研究人员的基石。由于其详尽的细节，这个数据集在捕捉房地产属性的诸多方面方面表现出色。它已成为许多预测建模练习的基础，并为探索性数据分析提供了丰富的领域。

Ames Housing Dataset 被设想为旧有 Boston Housing Dataset 的现代替代品。覆盖了 2006 年至 2010 年间在爱荷华州 Ames 的住宅销售，它呈现了多种变量，为高级回归技术奠定了基础。

这段时间在美国历史上具有特别重要的意义。2007-2008 年之前的时期见证了住房价格的急剧上涨，受到投机狂潮和次贷危机的推动。这一切在 2007 年末以住房泡沫的毁灭性崩溃告终，这一事件在《大空头》等叙事中被生动地记录了下来。这次崩溃的余波席卷全国，导致了大萧条。住房价格暴跌，止赎数量激增，许多美国人发现自己在抵押贷款中陷入困境。Ames 数据集提供了这一动荡时期的一个快照，记录了在国家经济动荡中进行的房地产销售。

**通过我的书籍** [《数据科学初学者指南》](https://machinelearning.samcart.com/products/beginners-guide-data-science/) **快速启动你的项目**。它提供了**自学教程**和**实用代码**。

## **加载****&****调整****数据****集** 

对于那些进入数据科学领域的人来说，拥有合适的工具是至关重要的。如果你需要一些帮助来设置你的 Python 环境，这个 [全面指南](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/) 是一个极好的资源。

**数据集****维度**：在深入复杂分析之前，了解数据集的基本结构和数据类型是至关重要的。这个步骤为后续探索提供了路线图，并确保你能够根据数据的性质调整你的分析。在环境设置好之后，让我们加载数据集，并评估其行数（代表各个物业）和列数（代表这些物业的属性）。

```py
# Load the Ames dataset
import pandas as pd
Ames = pd.read_csv('Ames.csv')

# Dataset shape
print(Ames.shape)

rows, columns = Ames.shape
print(f"The dataset comprises {rows} properties described across {columns} attributes.")
```

Python

```py
(2579, 85)
The dataset comprises 2579 properties described across 85 attributes.
```

**数据****类型**：识别每个属性的数据类型有助于塑造我们的分析方法。数值属性可以通过均值或中位数等措施来总结，而类别属性则适合用众数（最频繁的值）。

```py
# Determine the data type for each feature
data_types = Ames.dtypes

# Tally the total by data type
type_counts = data_types.value_counts()

print(type_counts)
```

```py
object     44
int64      27
float64    14
dtype: int64
```

**数据字典**：数据字典通常伴随全面的数据集，是一个非常实用的资源。它提供了每个特征的详细描述，说明其含义、可能的值，有时甚至包括其收集逻辑。对于像 Ames properties 这样包含广泛特征的数据集，数据字典可以成为清晰的指引。通过参考附带的 [数据字典](https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt)，分析师、数据科学家，甚至领域专家都可以更深入地理解每个特征的细微差别。无论是解读不熟悉特征的含义还是辨别特定值的重要性，数据字典都作为一个全面的指南。它架起了原始数据与可操作洞察之间的桥梁，确保分析和决策基于充分的信息。

```py
# Determine the data type for each feature
data_types = Ames.dtypes

# View a few datatypes from the dataset (first and last 5 features)
print(data_types)
```

```py
PID                int64
GrLivArea          int64
SalePrice          int64
MSSubClass         int64
MSZoning          object
                  ...   
SaleCondition     object
GeoRefNo         float64
Prop_Addr         object
Latitude         float64
Longitude        float64
Length: 85, dtype: object
```

Ground Living Area 和 Sale Price 是数值（int64）数据类型，而 Sale Condition（在本例中为字符串类型的对象）是类别数据类型。

## **揭示****和****可视化****缺失****值**

现实世界的数据集很少是完美整理的，通常会给分析师带来缺失值的挑战。这些数据的空白可能由于各种原因产生，如数据收集错误、系统限制或信息缺失。解决缺失值不仅仅是技术上的必要性，而是一个关键步骤，对后续分析的完整性和可靠性有显著影响。

理解缺失值的模式对于知情的数据分析至关重要。这些见解指导了适当的填补方法的选择，这些方法基于可用信息填补缺失数据，从而影响结果的准确性和可解释性。此外，评估缺失值模式有助于决策特征选择；大量缺失数据的特征可能会被排除，以提高模型性能并集中在更可靠的信息上。总之，掌握缺失值的模式确保了稳健和可靠的数据分析，指导填补策略并优化特征选择，以获得更准确的见解。

**NaN 或 None？** 在 pandas 中，`isnull()`函数用于检测 DataFrame 或 Series 中的缺失值。具体来说，它识别以下类型的缺失数据：

+   `np.nan`（Not a Number），通常用于表示缺失的数值数据

+   `None`，这是 Python 内置的对象，用于表示值的缺失或空值

`nan`和`NaN`只是不同的方式来指代 NumPy 的`np.nan`，`isnull()`会将它们识别为缺失值。这里是一个快速示例。

```py
# Import NumPy
import numpy as np

# Create a DataFrame with various types of missing values
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': ['a', 'b', None, 'd', 'e'],
    'C': [np.nan, np.nan, np.nan, np.nan, np.nan],
    'D': [1, 2, 3, 4, 5]
})

# Use isnull() to identify missing values
missing_data = df.isnull().sum()

print(df)
print()
print(missing_data)
```

```py
     A     B   C  D
0  1.0     a NaN  1
1  2.0     b NaN  2
2  NaN  None NaN  3
3  4.0     d NaN  4
4  5.0     e NaN  5
```

```py
A    1
B    1
C    5
D    0
dtype: int64
```

**可视化****缺失****值****：** 在可视化缺失数据时，工具如**DataFrames**、**missingno**、**matplotlib**和**seaborn**非常有用。通过根据缺失值的百分比对特征进行排序并将其放入 DataFrame 中，你可以轻松地对受缺失数据影响最大的特征进行排名。

```py
# Calculating the percentage of missing values for each column
missing_data = Ames.isnull().sum()
missing_percentage = (missing_data / len(Ames)) * 100

# Combining the counts and percentages into a DataFrame for better visualization
missing_info = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_percentage})

# Sorting the DataFrame by the percentage of missing values in descending order
missing_info = missing_info.sort_values(by='Percentage', ascending=False)

# Display columns with missing values
print(missing_info[missing_info['Missing Values'] > 0])
```

```py
              Missing Values  Percentage
PoolQC                  2570   99.651028
MiscFeature             2482   96.238852
Alley                   2411   93.485847
Fence                   2054   79.643273
FireplaceQu             1241   48.119426
LotFrontage              462   17.913920
GarageCond               129    5.001939
GarageQual               129    5.001939
GarageFinish             129    5.001939
GarageYrBlt              129    5.001939
GarageType               127    4.924389
Longitude                 97    3.761148
Latitude                  97    3.761148
BsmtExposure              71    2.753005
BsmtFinType2              70    2.714230
BsmtFinType1              69    2.675456
BsmtQual                  69    2.675456
BsmtCond                  69    2.675456
GeoRefNo                  20    0.775494
Prop_Addr                 20    0.775494
MasVnrArea                14    0.542846
MasVnrType                14    0.542846
BsmtFullBath               2    0.077549
BsmtHalfBath               2    0.077549
GarageArea                 1    0.038775
BsmtFinSF1                 1    0.038775
Electrical                 1    0.038775
TotalBsmtSF                1    0.038775
BsmtUnfSF                  1    0.038775
BsmtFinSF2                 1    0.038775
GarageCars                 1    0.038775
```

**missingno**包提供了缺失数据的快速图形表示。可视化中的白线或空隙表示缺失值。然而，它仅能容纳最多 50 个标签变量。超过此范围，标签开始重叠或变得不可读，默认情况下，大型显示会省略它们。

```py
import missingno as msno
import matplotlib.pyplot as plt
msno.matrix(Ames, sparkline=False, fontsize=20)
plt.show()
```

![](https://machinelearningmastery.com/wp-content/uploads/2024/01/Figure_1.png)

使用`missingno.matrix()`对缺失值进行可视化表示。

使用`msno.bar()`视觉展示在提取前 15 个缺失值最多的特征后，通过列提供了清晰的图示。

```py
# Calculating the percentage of missing values for each column
missing_data = Ames.isnull().sum()
missing_percentage = (missing_data / len(Ames)) * 100

# Combining the counts and percentages into a DataFrame for better visualization
missing_info = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_percentage})

# Sort the DataFrame columns by the percentage of missing values
sorted_df = Ames[missing_info.sort_values(by='Percentage', ascending=False).index]

# Select the top 15 columns with the most missing values
top_15_missing = sorted_df.iloc[:, :15]

#Visual with missingno
msno.bar(top_15_missing)
plt.show()
```

![](https://machinelearningmastery.com/wp-content/uploads/2024/01/Screenshot-2024-01-08-at-19.50.04.png)

使用 `missingno.bar()` 来可视化缺失数据的特征。

上图表示 Pool Quality、Miscellaneous Feature 和通往房产的 Alley 访问类型是缺失值最多的三个特征。

```py
import seaborn as sns
import matplotlib.pyplot as plt

# Filter to show only the top 15 columns with the most missing values
top_15_missing_info = missing_info.nlargest(15, 'Percentage')

# Create the horizontal bar plot using seaborn
plt.figure(figsize=(12, 8))
sns.barplot(x='Percentage', y=top_15_missing_info.index, data=top_15_missing_info, orient='h')
plt.title('Top 15 Features with Missing Percentages', fontsize=20)
plt.xlabel('Percentage of Missing Values', fontsize=16)
plt.ylabel('Features', fontsize=16)
plt.yticks(fontsize=11)
plt.show()
```

![](https://machinelearningmastery.com/wp-content/uploads/2024/01/Figure_2-1.png)

使用 seaborn 横向条形图来可视化缺失数据。

使用 seaborn 制作的横向条形图可以让你以垂直格式列出缺失值最多的特征，增加了可读性和美观性。

处理缺失值不仅仅是技术要求；这是一项重要的步骤，可能会影响你的机器学习模型的质量。理解和可视化这些缺失值是这场复杂舞蹈的第一步。

### 想要开始数据科学初学者指南吗？

现在就报名参加我的免费电子邮件速成课程（附示例代码）。

点击注册，并获取课程的免费 PDF 电子书版本。

## **进一步阅读**

如果你想深入了解这个话题，本节提供了更多资源。

#### 教程

+   [Anaconda 设置教程](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

#### **论文**

+   [1. Ames, Iowa: 作为学期结束回归项目的波士顿住房数据替代方案 by Dr. Dean De Cock](https://raw.githubusercontent.com/Padre-Media/dataset/main/decock.pdf)

#### **资源**

+   [Ames 房产数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)

+   [Ames 数据字典](https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt)

## **总结**

在本教程中，你开始探索 Ames Properties 数据集，这是一个针对数据科学应用的全面房产数据集。

具体来说，你学到了：

+   **关于 Ames 数据集的背景，包括其先驱和学术重要性。**

***   **如何提取数据集的维度、数据类型和缺失值。*****   **如何使用像 `missingno`、Matplotlib 和 Seaborn 等包来快速可视化缺失数据。******

******有任何问题吗？请在下面的评论中提问，我会尽力回答。********
