# 数据协调：分段、连接、数据透视和合并的交响曲

> 原文：[`machinelearningmastery.com/harmonizing-data-a-symphony-of-segmenting-concatenating-pivoting-and-merging/`](https://machinelearningmastery.com/harmonizing-data-a-symphony-of-segmenting-concatenating-pivoting-and-merging/)

在数据科学项目中，你收集的数据通常并不是你想要的形状。你通常需要创建派生特征，将数据子集汇总为总结形式，或根据一些复杂的逻辑选择数据的部分。这不是一种假设情况。在大大小小的项目中，你在第一步获得的数据很可能远非理想。

作为数据科学家，你必须熟练地将数据格式化为正确的形状，以使后续步骤更加轻松。在接下来的内容中，你将学习如何在 pandas 中切片和切块数据集，并将其重新组合成不同的形式，以使有用的数据更加突出，从而使分析更加容易。

让我们开始吧。

![](img/b696b299f843eaf29bccadb880187738.png)

数据协调：分段、连接、数据透视和合并的交响曲

图片由 [Samuel Sianipar](https://unsplash.com/photos/group-of-person-playing-violin-4TNd3hsW3PM) 提供。保留部分权利。

## 概述

本文分为两个部分；它们是：

+   分段与连接：与 Pandas 的编舞

+   数据透视与合并：与 Pandas 的舞蹈

## 分段与连接：与 Pandas 的编舞

你可能会提出一个有趣的问题：房产建造年份如何影响其价格？为了研究这个问题，你可以将数据集按照‘SalePrice’分成四个四分位数——低、中、高和优质，并分析这些分段中的建造年份。这种系统化的数据集划分不仅为集中分析铺平了道路，还揭示了可能在整体回顾中被隐藏的趋势。

### **分段策略：‘SalePrice’ 的四分位数**

首先，创建一个新的列，将房产的‘SalePrice’整齐地分类到你定义的价格类别中：

```py
# Import the Pandas Library
import pandas as pd

# Load the dataset
Ames = pd.read_csv('Ames.csv')

# Define the quartiles
quantiles = Ames['SalePrice'].quantile([0.25, 0.5, 0.75])

# Function to categorize each row
def categorize_by_price(row):
    if row['SalePrice'] <= quantiles.iloc[0]:
        return 'Low'
    elif row['SalePrice'] <= quantiles.iloc[1]:
        return 'Medium'
    elif row['SalePrice'] <= quantiles.iloc[2]:
        return 'High'
    else:
        return 'Premium'

# Apply the function to create a new column
Ames['Price_Category'] = Ames.apply(categorize_by_price, axis=1)
print(Ames[['SalePrice','Price_Category']])
```

通过执行上述代码，你已将数据集丰富了一个新的名为‘Price_Category’的列。以下是你获得的输出的一个快照：

```py
      SalePrice Price_Category
0        126000            Low
1        139500         Medium
2        124900            Low
3        114000            Low
4        227000        Premium
...         ...            ...
2574     121000            Low
2575     139600         Medium
2576     145000         Medium
2577     217500        Premium
2578     215000        Premium

[2579 rows x 2 columns]
```

### **使用经验累积分布函数 (ECDF) 可视化趋势**

现在你可以将原始数据集分成四个 DataFrame，并继续可视化每个价格类别中建造年份的累积分布。这一视觉效果将帮助你一目了然地了解与定价相关的房产建造历史趋势。

```py
# Import Matplotlib & Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Split original dataset into 4 DataFrames by Price Category
low_priced_homes = Ames.query('Price_Category == "Low"')
medium_priced_homes = Ames.query('Price_Category == "Medium"')
high_priced_homes = Ames.query('Price_Category == "High"')
premium_priced_homes = Ames.query('Price_Category == "Premium"')

# Setting the style for aesthetic looks
sns.set_style("whitegrid")

# Create a figure
plt.figure(figsize=(10, 6))

# Plot each ECDF on the same figure
sns.ecdfplot(data=low_priced_homes, x='YearBuilt', color='skyblue', label='Low')
sns.ecdfplot(data=medium_priced_homes, x='YearBuilt', color='orange', label='Medium')
sns.ecdfplot(data=high_priced_homes, x='YearBuilt', color='green', label='High')
sns.ecdfplot(data=premium_priced_homes, x='YearBuilt', color='red', label='Premium')

# Adding labels and title for clarity
plt.title('ECDF of Year Built by Price Category', fontsize=16)
plt.xlabel('Year Built', fontsize=14)
plt.ylabel('ECDF', fontsize=14)
plt.legend(title='Price Category', title_fontsize=14, fontsize=14)

# Show the plot
plt.show()
```

下面是 ECDF 图，它提供了你所分类数据的视觉总结。ECDF，即经验累积分布函数，是一种用于描述数据集数据点分布的统计工具。它表示低于或等于某个值的数据点比例或百分比。从本质上讲，它为你提供了一种可视化不同值的数据点分布的方式，提供有关数据的形状、扩展和集中趋势的见解。ECDF 图尤其有用，因为它们便于不同数据集之间的比较。注意每个价格类别的曲线如何呈现出房屋趋势的叙述。

![](https://machinelearningmastery.com/wp-content/uploads/2024/02/Figure_1-2.png) 从图中可以明显看出，较低和中等价格的房屋在较早的年份建造的频率更高，而高价和奢侈价格的房屋则倾向于较新的建筑。了解到房产年龄在价格段中有显著差异后，你找到了使用 `Pandas.concat()` 的充分理由。

### **使用 `Pandas.concat()` 叠加数据集**

作为数据科学家，你们经常需要叠加数据集或其部分，以获取更深入的见解。`Pandas.concat()` 函数是你完成这些任务的瑞士军刀，使你能够精确而灵活地合并 DataFrame。这一强大函数类似于 SQL 的`UNION`操作，但`Pandas.concat()` 提供了更大的灵活性——它允许对 DataFrame 进行垂直和水平的拼接。当你处理具有不匹配列的数据集或需要按公共列对齐时，这一特性变得不可或缺，显著拓宽了你的分析范围。以下是如何将分段 DataFrame 合并以比较“经济型”和“豪华型”房屋的市场类别：

```py
# Stacking Low and Medium categories into an "affordable_homes" DataFrame
affordable_homes = pd.concat([low_priced_homes, medium_priced_homes])

# Stacking High and Premium categories into a "luxury_homes" DataFrame
luxury_homes = pd.concat([high_priced_homes, premium_priced_homes])
```

通过这些，你可以对比和分析那些使得经济型房屋与昂贵房屋不同的特征。

通过我的书籍 [数据科学初学者指南](https://machinelearning.samcart.com/products/beginners-guide-data-science/)，**快速启动你的项目**。它提供了**自学教程**和**可运行的代码**。

## 数据透视与合并：与 Pandas 共舞

在将数据集划分为“经济型”和“豪华型”房屋，并探讨其建设年份的分布后，你现在将关注于影响房产价值的另一个维度：设施，特别是壁炉的数量。在你深入了解数据集的合并——一个`Pandas.merge()`作为与 SQL 的`JOIN`相媲美的强大工具——之前，你必须首先通过更精细的视角检查你的数据。

透视表是总结和分析特定数据点的绝佳工具。它们让你能够汇总数据并揭示模式，从而指导后续的合并操作。通过创建透视表，你可以编制一个清晰且组织良好的概述，显示按壁炉数量分类的平均居住面积和房屋数量。这一初步分析不仅丰富了你对两个市场细分的理解，还为你展示的复杂合并技术奠定了坚实的基础。

**使用透视表创建有洞察力的总结**

我们首先构建‘经济型’和‘豪华型’住宅类别的透视表。这些表将总结每种壁炉类别的平均居住面积（GrLivArea）并提供每个类别的房屋数量。这种分析至关重要，因为它展示了房屋吸引力和价值的一个关键方面——壁炉的存在和数量，以及这些特征在不同的市场细分中如何变化。

```py
# Creating pivot tables with both mean living area and home count
pivot_affordable = affordable_homes.pivot_table(index='Fireplaces', 
                                                aggfunc={'GrLivArea': 'mean', 'Fireplaces': 'count'})
pivot_luxury = luxury_homes.pivot_table(index='Fireplaces', 
                                        aggfunc={'GrLivArea': 'mean', 'Fireplaces': 'count'})

# Renaming columns and index labels separately
pivot_affordable.rename(columns={'GrLivArea': 'AvLivArea', 'Fireplaces': 'HmCount'}, inplace=True)
pivot_affordable.index.name = 'Fire'

pivot_luxury.rename(columns={'GrLivArea': 'AvLivArea', 'Fireplaces': 'HmCount'}, inplace=True)
pivot_luxury.index.name = 'Fire'  

# View the pivot tables
print(pivot_affordable)
print(pivot_luxury)
```

使用这些透视表，你现在可以轻松地可视化和比较壁炉等特征如何与居住面积相关，并且这些特征在每个细分中的出现频率。第一个透视表是从‘经济型’住宅数据框中创建的，显示了这个分组中的大多数房产没有壁炉。

```py
      HmCount    AvLivArea
Fire                      
0         931  1159.050483
1         323  1296.808050
2          38  1379.947368
```

第二个透视表来源于‘豪华型’住宅数据框，展示了这个子集中的房产壁炉数量范围从零到四个，其中一个壁炉最为常见。

```py
      HmCount    AvLivArea
Fire                      
0         310  1560.987097
1         808  1805.243812
2         157  1998.248408
3          11  2088.090909
4           1  2646.000000
```

通过创建透视表，你将数据提炼成一个适合下一步分析的形式——使用`Pandas.merge()`将这些洞察合并，以查看这些特征在更广泛市场中的相互作用。

上面的透视表是最简单的版本。更高级的版本允许你不仅指定索引，还指定列作为参数。思想类似：你选择两列，一列指定为`index`，另一列指定为`columns`参数，这两列的值会被聚合并形成一个矩阵。矩阵中的值就是`aggfunc`参数所指定的结果。

你可以考虑以下示例，它产生了与上述类似的结果：

```py
pivot = Ames.pivot_table(index="Fireplaces",
                         columns="Price_Category",
                         aggfunc={'GrLivArea':'mean', 'Fireplaces':'count'})
print(pivot)
```

这将打印：

```py
               Fireplaces                          GrLivArea
Price_Category       High    Low Medium Premium         High          Low       Medium      Premium
Fireplaces
0                   228.0  520.0  411.0    82.0  1511.912281  1081.496154  1257.172749  1697.439024
1                   357.0  116.0  207.0   451.0  1580.644258  1184.112069  1359.961353  1983.031042
2                    52.0    9.0   29.0   105.0  1627.384615  1184.888889  1440.482759  2181.914286
3                     5.0    NaN    NaN     6.0  1834.600000          NaN          NaN  2299.333333
4                     NaN    NaN    NaN     1.0          NaN          NaN          NaN  2646.000000
```

通过比较，例如，零壁炉的低档和中档住宅数量分别为 520 和 411，你可以看到结果是相同的，即 931 = 520+411，这与之前获得的结果一致。你会发现二级列标记为低、中、高和优质，因为你在`pivot_table()`中指定了“Price_Category”作为`columns`参数。`aggfunc`参数中的字典给出了顶级列。

### **迈向更深的洞察：利用`Pandas.merge()`进行比较分析**

在揭示了壁炉、房屋数量和生活区域在细分数据集中的关系后，你已经做好了进一步分析的准备。使用`Pandas.merge()`，你可以叠加这些见解，就像 SQL 的`JOIN`操作根据相关列合并两个或多个表中的记录一样。这种技术将允许你在共同属性上合并细分数据，进行超越分类的比较分析。

我们的第一次操作使用**外连接**来合并经济型和豪华型房屋数据集，确保没有任何类别的数据丢失。这种方法特别具有启发性，因为它揭示了所有房屋的完整范围，无论它们是否共享相同数量的壁炉。

```py
pivot_outer_join = pd.merge(pivot_affordable, pivot_luxury, on='Fire', how='outer', suffixes=('_aff', '_lux')).fillna(0)
print(pivot_outer_join)
```

```py
      HmCount_aff  AvLivArea_aff  HmCount_lux  AvLivArea_lux
Fire                                                        
0           931.0    1159.050483          310    1560.987097
1           323.0    1296.808050          808    1805.243812
2            38.0    1379.947368          157    1998.248408
3             0.0       0.000000           11    2088.090909
4             0.0       0.000000            1    2646.000000
```

在这种情况下，外连接的功能类似于右连接，捕捉到两个市场细分中存在的每种壁炉类别。值得注意的是，在经济型价格范围内没有具有 3 个或 4 个壁炉的房产。你需要为`suffixes`参数指定两个字符串，因为“`HmCount`”和“`AvLivArea`”列在`pivot_affordable`和`pivot_luxury`两个数据框中都存在。你会看到“`HmCount_aff`”在 3 和 4 个壁炉的情况下为零，因为你需要它们作为外连接的占位符，以匹配`pivot_luxury`中的行。

接下来，你可以使用**内连接**，专注于经济型和豪华型房屋共享相同数量壁炉的交集。这种方法突出了两个细分市场之间的核心相似性。

```py
pivot_inner_join = pd.merge(pivot_affordable, pivot_luxury, on='Fire', how='inner', suffixes=('_aff', '_lux'))
print(pivot_inner_join)
```

```py
      HmCount_aff  AvLivArea_aff  HmCount_lux  AvLivArea_lux
Fire                                                        
0             931    1159.050483          310    1560.987097
1             323    1296.808050          808    1805.243812
2              38    1379.947368          157    1998.248408
```

有趣的是，在这种情况下，内连接的功能类似于左连接，展示了两个数据集中的类别。你看不到 3 和 4 个壁炉对应的行，因为这是内连接的结果，并且在数据框`pivot_affordable`中没有这样的行。

最后，**交叉连接**允许你检查经济型和豪华型房屋属性的每一种可能组合，提供了一个全面的视图，展示不同特征在整个数据集中的交互。结果有时被称为两个数据框的**笛卡尔积**。

```py
# Resetting index to display cross join
pivot_affordable.reset_index(inplace=True)
pivot_luxury.reset_index(inplace=True)

pivot_cross_join = pd.merge(pivot_affordable, pivot_luxury, how='cross', suffixes=('_aff', '_lux')).round(2)
print(pivot_cross_join)
```

结果如下，这展示了交叉连接的结果，但在这个数据集中并没有提供任何特殊的见解。

```py
    Fire_aff  HmCount_aff  AvLivArea_aff  Fire_lux  HmCount_lux  AvLivArea_lux
0          0          931        1159.05         0          310        1560.99
1          0          931        1159.05         1          808        1805.24
2          0          931        1159.05         2          157        1998.25
3          0          931        1159.05         3           11        2088.09
4          0          931        1159.05         4            1        2646.00
5          1          323        1296.81         0          310        1560.99
6          1          323        1296.81         1          808        1805.24
7          1          323        1296.81         2          157        1998.25
8          1          323        1296.81         3           11        2088.09
9          1          323        1296.81         4            1        2646.00
10         2           38        1379.95         0          310        1560.99
11         2           38        1379.95         1          808        1805.24
12         2           38        1379.95         2          157        1998.25
13         2           38        1379.95         3           11        2088.09
14         2           38        1379.95         4            1        2646.00
```

### **从合并数据中提取见解**

完成这些合并操作后，你可以深入探究它们揭示的见解。每种连接类型都为房屋市场的不同方面提供了洞察：

+   **外连接**揭示了最广泛的属性范围，强调了各个价格点上壁炉等设施的多样性。

+   **内连接**细化了你的视角，专注于在壁炉数量上经济型和豪华型住宅的直接比较，提供了更清晰的标准市场情况。

+   **交叉连接**提供了一个详尽的特征组合，非常适合假设分析或理解潜在市场扩展。

在进行这些合并后，你会观察到**在可负担得起的住房中**：

+   没有壁炉的房屋平均总生活面积约为 1159 平方英尺，占最大份额。

+   当壁炉数量增加到一个时，平均生活面积扩展到约 1296 平方英尺，突显了生活空间的明显增加。

+   尽管拥有两个壁炉的房屋数量较少，但其平均生活面积更大，约为 1379 平方英尺，突显了附加设施与更宽敞生活空间的关联趋势。

相比之下，你会观察到**在奢华住房中**：

+   奢华市场中的房屋起点是没有壁炉的房屋，平均面积为 1560 平方英尺，比可负担得起的房屋要大得多。

+   随着壁炉数量的增加，平均生活面积的跃升更加明显，单壁炉房屋的平均面积约为 1805 平方英尺。

+   拥有两个壁炉的房屋进一步放大了这一趋势，提供了接近 1998 平方英尺的平均生活面积。少见的三或四个壁炉的房屋标志着生活空间的显著增加，四个壁炉的房屋的生活空间高达 2646 平方英尺。

这些观察提供了一个迷人的视角，展示了壁炉等设施不仅增加了房屋的吸引力，还似乎是更大生活空间的标志，尤其是从可负担得起的市场细分到奢华市场细分。

### 想要开始学习《数据科学初学者指南》吗？

现在就参加我的免费电子邮件速成课程（包含示例代码）。

点击注册，并获得课程的免费 PDF 电子书版本。

## **进一步阅读**

#### 教程

+   [`Pandas.concat()` 方法](https://pandas.pydata.org/docs/reference/api/pandas.concat.html)

+   [Pandas 的 `DataFrame.pivot_table()` 方法](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot_table.html)

+   [`Pandas.merge()` 方法](https://pandas.pydata.org/docs/reference/api/pandas.merge.html#pandas.merge)

#### **资源**

+   [Ames 数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)

+   [Ames 数据字典](https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt)

## **总结**

在这次关于使用 Python 和 Pandas 进行数据协调技术的全面探索中，你已深入了解了数据集的分段、连接、透视和合并的复杂性。从基于价格类别将数据集划分为有意义的部分，到可视化建筑年份的趋势，再到使用`Pandas.concat()`堆叠数据集以分析更广泛的市场类别，以及利用数据透视表在细分中总结和分析数据点，你覆盖了一系列必备的数据操作和分析技巧。此外，通过利用`Pandas.merge()`比较分段数据集，并从不同类型的合并操作（外连接、内连接、交叉连接）中获取洞察，你解锁了数据集成和探索的强大能力。掌握这些技术，数据科学家和分析师可以自信地导航复杂的数据领域，发现隐藏的模式，并提取有价值的洞察，从而推动明智的决策制定。

具体来说，你学到了：

+   如何基于价格类别将数据集划分为有意义的部分，并可视化建筑年份的趋势。

+   使用`Pandas.concat()`堆叠数据集并分析更广泛的市场类别。

+   数据透视表在总结和分析数据点中的作用。

+   如何利用`Pandas.merge()`比较分段数据集，并从不同类型的合并操作（外连接、内连接、交叉连接）中获取洞察。

有任何问题吗？请在下面的评论中提问，我会尽力回答。
