# 超越 SQL：使用 Pandas 将房地产数据转化为可操作的见解

> 原文：[`machinelearningmastery.com/beyond-sql-transforming-real-estate-data-into-actionable-insights-with-pandas/`](https://machinelearningmastery.com/beyond-sql-transforming-real-estate-data-into-actionable-insights-with-pandas/)

在数据分析中，SQL 作为一个强大的工具，以其在管理和查询数据库方面的强大功能而闻名。Python 中的 pandas 库为数据科学家带来了类似 SQL 的功能，使其能够进行复杂的数据操作和分析，而无需传统的 SQL 数据库。在接下来的内容中，你将应用 Python 中的类似 SQL 的函数来剖析和理解数据。

![](img/3564c6b63ef18ee9b4c8ba58d3b66c41.png)

超越 SQL：使用 Pandas 将房地产数据转化为可操作的见解

照片由[Lukas W.](https://unsplash.com/photos/white-and-black-panda-on-brown-wooden-fence-during-daytime-e3mu-MTj7Xk)提供，部分权利保留。

让我们开始吧。

## 概述

本文分为三个部分，它们是：

+   使用 Pandas 的`DataFrame.query()`方法探索数据

+   数据聚合与分组

+   精通 Pandas 中的行和列选择

+   利用数据透视表进行深入的房市分析

## 使用 Pandas 的`DataFrame.query()`方法探索数据

pandas 中的`DataFrame.query()`方法允许根据指定条件选择行，类似于 SQL 的`SELECT`语句。从基础开始，你可以基于单个或多个条件过滤数据，从而为更复杂的数据查询奠定基础。

```py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
Ames = pd.read_csv('Ames.csv')

# Simple querying: Select houses priced above $600,000
high_value_houses = Ames.query('SalePrice > 600000')
print(high_value_houses)
```

在上述代码中，你利用 pandas 中的`DataFrame.query()`方法筛选出价格高于\$600,000 的房屋，将结果存储在一个名为`high_value_houses`的新 DataFrame 中。此方法允许基于作为字符串指定的条件对数据进行简洁易读的查询。在这种情况下，条件为`'SalePrice > 600000'`。

下面的结果 DataFrame 展示了所选的高价值房产。查询有效地将数据集缩小到售价超过\$600,000 的房屋，仅展示了 5 个符合该标准的房产。过滤后的视图提供了对 Ames 数据集中高价值房产的集中观察，提供了有关这些高价值房产的特征和位置的见解。

```py
            PID  GrLivArea  ...   Latitude  Longitude
65    528164060       2470  ...  42.058475 -93.656810
584   528150070       2364  ...  42.060462 -93.655516
1007  528351010       4316  ...  42.051982 -93.657450
1325  528320060       3627  ...  42.053228 -93.657649
1639  528110020       2674  ...  42.063049 -93.655918

[5 rows x 85 columns]
```

在下面的下一个示例中，让我们进一步探索`DataFrame.query()`方法的功能，以根据更具体的标准过滤 Ames Housing 数据集。查询选择了卧室数量超过 3 间（`BedroomAbvGr > 3`）且价格低于\$300,000（`SalePrice < 300000`）的房屋。此条件组合通过逻辑与操作符（`&`）实现，使你能够同时应用多个过滤器到数据集中。

```py
# Advanced querying: Select houses with more than 3 bedrooms and priced below $300,000
specific_houses = Ames.query('BedroomAbvGr > 3 & SalePrice < 300000')
print(specific_houses)
```

该查询的结果存储在一个新的 DataFrame 中，名为`specific_houses`，其中包含满足两个条件的所有属性。通过打印`specific_houses`，你可以检查既相对较大（就卧室而言）又负担得起的房屋的详细信息，目标是特定的住房市场细分，这可能会吸引那些寻找在特定预算范围内有宽敞居住选择的家庭。

```py
            PID  GrLivArea  ...   Latitude  Longitude
5     908128060       1922  ...  42.018988 -93.671572
23    902326030       2640  ...  42.029358 -93.612289
33    903400180       1848  ...  42.029544 -93.627377
38    527327050       2030  ...  42.054506 -93.631560
40    528326110       2172  ...  42.055785 -93.651102
...         ...        ...  ...        ...        ...
2539  905101310       1768  ...  42.033393 -93.671295
2557  905107250       1440  ...  42.031349 -93.673578
2562  535101110       1584  ...  42.048256 -93.619860
2575  905402060       1733  ...  42.027669 -93.666138
2576  909275030       2002  ...        NaN        NaN

[352 rows x 85 columns]
```

高级查询成功识别出 Ames Housing 数据集中总计 352 套符合指定条件的房屋：有超过 3 个卧室且售价低于\$300,000。这个属性子集突显了市场上的一个重要部分，提供宽敞的居住选择而不会超出预算，适合寻找经济实惠但宽敞住房的家庭或个人。为了进一步探讨这个子集的动态，让我们可视化销售价格与地面居住面积之间的关系，并增加一个额外的层次来指示卧室数量。这个图形表示将帮助你理解居住空间和卧室数量如何影响这些房屋在指定条件下的可负担性和吸引力。

```py
# Visualizing the advanced query results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', hue='BedroomAbvGr', data=specific_houses, palette='viridis')
plt.title('Sales Price vs. Ground Living Area')
plt.xlabel('Ground Living Area (sqft)')
plt.ylabel('Sales Price ($)')
plt.legend(title='Bedrooms Above Ground')
plt.show()
```

![](img/d1de7e8953b049490ac746cc4db78258.png)

散点图显示了销售价格与卧室数量和居住面积之间的分布

上面的散点图生动地展示了销售价格、居住面积和卧室数量之间的微妙关系，突出了 Ames 住房市场中这一细分市场的多样选择。它突显了较大的居住空间和额外卧室如何影响房产的价值，为关注宽敞且经济实惠住房的潜在买家和投资者提供了宝贵的见解。这个视觉分析不仅使数据更易于理解，还强调了 Pandas 在揭示关键市场趋势中的实际效用。

用我的书[《数据科学入门指南》](https://machinelearning.samcart.com/products/beginners-guide-data-science/)**启动你的项目**。它提供了**自学教程**和**实用代码**。

## **数据聚合与分组**

聚合和分组在总结数据洞察中至关重要。在你探索的第一部分中建立在基础查询技术的基础上，让我们深入研究 Python 中数据聚合和分组的力量。类似于 SQL 的`GROUP BY`子句，pandas 提供了一个强大的`groupby()`方法，使你能够将数据分成子集以进行详细分析。在你旅程的下一个阶段，专注于利用这些功能揭示 Ames Housing 数据集中隐藏的模式和洞察。具体来说，你将检查在不同邻里中，超过三个卧室且售价低于\$300,000 的房屋的平均售价。通过聚合这些数据，你旨在突出 Ames, Iowa 空间画布上住房可负担性和库存的变化。

```py
# Advanced querying: Select houses with more than 3 bedrooms and priced below $300,000
specific_houses = Ames.query('BedroomAbvGr > 3 & SalePrice < 300000')

# Group by neighborhood, then calculate the average and total sale price, and count the houses
grouped_data = specific_houses.groupby('Neighborhood').agg({
    'SalePrice': ['mean', 'count']
})

# Renaming the columns for clarity
grouped_data.columns = ['Average Sales Price', 'House Count']

# Round the average sale price to 2 decimal places
grouped_data['Averages Sales Price'] = grouped_data['Average Sales Price'].round(2)

print(grouped_data)
```

```py
             Average Sales Price  House Count
Neighborhood                                 
BrDale                 113700.00            1
BrkSide                154840.00           10
ClearCr                206756.31           13
CollgCr                233504.17           12
Crawfor                199946.68           19
Edwards                142372.41           29
Gilbert                222554.74           19
IDOTRR                 146953.85           13
MeadowV                135966.67            3
Mitchel                152030.77           13
NAmes                  158835.59           59
NPkVill                143000.00            1
NWAmes                 203846.28           39
NoRidge                272222.22           18
NridgHt                275000.00            3
OldTown                142586.72           43
SWISU                  147493.33           15
Sawyer                 148375.00           16
SawyerW                217952.06           16
Somerst                247333.33            3
StoneBr                270000.00            1
Timber                 247652.17            6
```

通过使用 Seaborn 进行可视化，我们的目标是创建一个直观且易于访问的数据表示。你可以创建一个条形图，展示按社区划分的平均销售价格，并附上房屋数量的注释，以在一个连贯的图表中展示价格和数量。

```py
# Ensure 'Neighborhood' is a column (reset index if it was the index)
grouped_data_reset = grouped_data.reset_index().sort_values(by='Average Sales Price')

# Set the aesthetic style of the plots
sns.set_theme(style="whitegrid")

# Create the bar plot
plt.figure(figsize=(12, 8))
barplot = sns.barplot(
    x='Neighborhood',
    y='Average Sales Price',
    data=grouped_data_reset,
    palette="coolwarm",
    hue='Neighborhood',
    legend=False,
    errorbar=None  # Removes the confidence interval bars
)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Annotate each bar with the house count, using enumerate to access the index for positioning
for index, value in enumerate(grouped_data_reset['Average Sales Price']):
    house_count = grouped_data_reset.loc[index, 'House Count']
    plt.text(index, value, f'{house_count}', ha='center', va='bottom')

plt.title('Average Sales Price by Neighborhood', fontsize=18)
plt.xlabel('Neighborhood')
plt.ylabel('Average Sales Price ($)')

plt.tight_layout()  # Adjust the layout
plt.show()
```

![](img/8acde4372862c4994564c610b58a0a32.png)

按升序比较社区的平均销售价格

分析和随后的可视化强调了在艾姆斯（Ames），爱荷华州，符合特定标准（超过三间卧室且价格低于 \$300,000）的房屋在可负担性和可获得性上的显著差异。这不仅展示了 SQL 类函数在 Python 中用于实际数据分析的实际应用，还提供了有关地方房地产市场动态的宝贵见解。

## 精通 Pandas 中的行和列选择

从 DataFrame 中选择特定的数据子集是常见的需求。你可以使用两个强大的方法：`DataFrame.loc[]` 和 `DataFrame.iloc[]`。这两者的目的相似——选择数据——但它们在引用行和列时有所不同。

### 理解 `DataFrame.loc[]` 方法

`DataFrame.loc[]` 是一种基于标签的数据选择方法，这意味着你使用行和列的标签来选择数据。对于基于列名和行索引选择数据，它非常直观，尤其是当你知道感兴趣的具体标签时。

**语法**：`DataFrame.loc[row_label, column_label]`

**目标**：让我们选择所有超过 3 间卧室、价格低于 \$300,000 的房屋，位于以高平均销售价格著称的特定社区（基于你之前的发现），并显示它们的 ‘Neighborhood’，‘SalePrice’ 和 ‘GrLivArea’。

```py
# Assuming 'high_value_neighborhoods' is a list of neighborhoods with higher average sale prices
high_value_neighborhoods = ['NridgHt', 'NoRidge', 'StoneBr']

# Use df.loc[] to select houses based on your conditions and only in high-value neighborhoods
high_value_houses_specific = Ames.loc[(Ames['BedroomAbvGr'] > 3) &
                                      (Ames['SalePrice'] < 300000) &
                                      (Ames['Neighborhood'].isin(high_value_neighborhoods)),
                                      ['Neighborhood', 'SalePrice', 'GrLivArea']]

print(high_value_houses_specific.head())
```

```py
    Neighborhood  SalePrice  GrLivArea
40       NoRidge     291000       2172
162      NoRidge     285000       2225
460      NridgHt     250000       2088
468      NoRidge     268000       2295
490      NoRidge     260000       2417
```

#### 理解 `DataFrame.iloc[]` 方法

相对而言，`DataFrame.iloc[]` 是一种基于整数位置的索引方法。这意味着你使用整数来指定要选择的行和列。它特别有用，以按位置访问 DataFrame 中的数据。

**语法**：`DataFrame.iloc[row_position, column_position]`

**目标**：接下来的重点是揭示艾姆斯数据集中不妥协于空间的经济实惠的住房选项，特别是寻找至少有 3 间以上的卧室且价格低于 \$300,000 的房屋，排除高价值社区。

```py
# Filter for houses not in the 'high_value_neighborhoods', 
# with at least 3 bedrooms above grade, and priced below $300,000
low_value_spacious_houses = Ames.loc[(~Ames['Neighborhood'].isin(high_value_neighborhoods)) & 
                                     (Ames['BedroomAbvGr'] >= 3) & 
                                     (Ames['SalePrice'] < 300000)]

# Sort these houses by 'SalePrice' to highlight the lower end explicitly
low_value_spacious_houses_sorted = low_value_spacious_houses.sort_values(by='SalePrice').reset_index(drop=True)

# Using df.iloc to select and print the first 5 observations of such low-value houses
low_value_spacious_first_5 = low_value_spacious_houses_sorted.iloc[:5, :]

# Print only relevant columns to match the earlier high-value example: 'Neighborhood', 'SalePrice', 'GrLivArea'
print(low_value_spacious_first_5[['Neighborhood', 'SalePrice', 'GrLivArea']])
```

```py
  Neighborhood  SalePrice  GrLivArea
0       IDOTRR      40000       1317
1       IDOTRR      50000       1484
2       IDOTRR      55000       1092
3       Sawyer      62383        864
4      Edwards      63000       1112
```

在你探索`DataFrame.loc[]`和`DataFrame.iloc[]`时，你发现了 pandas 在行和列选择方面的能力，展示了这些方法在数据分析中的灵活性和强大功能。通过来自 Ames Housing 数据集的实际示例，你看到`DataFrame.loc[]`允许基于标签进行直观的选择，适用于根据已知标签定位特定数据。相对而言，`DataFrame.iloc[]`提供了一种通过整数位置精确访问数据的方法，提供了一种用于位置选择的基本工具，特别适用于需要关注数据片段或样本的场景。无论是在特定社区中筛选高价值物业，还是在更广泛的市场中识别入门级住房，掌握这些选择技巧都丰富了你的数据科学工具包，使数据探索更加有针对性和富有洞察力。

### 想要开始学习《数据科学初学者指南》吗？

现在就参加我的免费邮件速成课程吧（附带示例代码）。

点击注册，还可以免费获得课程的 PDF 电子书版本。

## **利用数据透视表进行深入的住房市场分析**

当你深入探讨 Ames Housing 数据集时，你的分析之旅将引入 pandas 中数据透视表的强大功能。数据透视表作为一个宝贵的工具，可以总结、分析和呈现复杂数据，以易于消化的格式。这种技术允许你交叉汇总和分段数据，以揭示可能否则会被隐藏的模式和见解。在本节中，你将利用数据透视表更深入地剖析住房市场，重点关注社区特征、卧室数量和销售价格之间的相互作用。

为了为你的数据透视表分析做好准备，你将数据集筛选为价格低于 \$300,000 且至少有一个高于地面的卧室。这个标准关注更实惠的住房选项，确保你的分析对更广泛的受众仍然相关。然后，你将构建一个数据透视表，按社区和卧室数量分段平均销售价格，旨在揭示影响 Ames 住房可负担性和偏好的模式。

```py
# Import an additional library
import numpy as np

# Filter for houses priced below $300,000 and with at least 1 bedroom above grade
affordable_houses = Ames.query('SalePrice < 300000 & BedroomAbvGr > 0')

# Create a pivot table to analyze average sale price by neighborhood and number of bedrooms
pivot_table = affordable_houses.pivot_table(values='SalePrice',
                                            index='Neighborhood',
                                            columns='BedroomAbvGr',
                                            aggfunc='mean').round(2)

# Fill missing values with 0 for better readability and to indicate no data for that segment
pivot_table = pivot_table.fillna(0)

# Adjust pandas display options to ensure all columns are shown
pd.set_option('display.max_columns', None)

print(pivot_table)
```

在我们讨论一些见解之前，先快速查看一下数据透视表。

```py
BedroomAbvGr          1          2          3          4          5          6
Neighborhood                                                                  
Blmngtn       178450.00  197931.19       0.00       0.00       0.00       0.00
Blueste       192500.00  128557.14  151000.00       0.00       0.00       0.00
BrDale             0.00   99700.00  111946.43  113700.00       0.00       0.00
BrkSide        77583.33  108007.89  140058.67  148211.11  214500.00       0.00
ClearCr       212250.00  220237.50  190136.36  209883.20  196333.33       0.00
CollgCr       154890.00  181650.00  196650.98  233504.17       0.00       0.00
Crawfor       289000.00  166345.00  193433.75  198763.94  210000.00       0.00
Edwards        59500.00  117286.27  134660.65  137332.00  191866.67  119900.00
Gilbert            0.00  172000.00  182178.30  223585.56  204000.00       0.00
Greens        193531.25       0.00       0.00       0.00       0.00       0.00
GrnHill            0.00  230000.00       0.00       0.00       0.00       0.00
IDOTRR         67378.00   93503.57  111681.13  144081.82  162750.00       0.00
Landmrk            0.00       0.00  137000.00       0.00       0.00       0.00
MeadowV        82128.57  105500.00   94382.00  128250.00  151400.00       0.00
Mitchel       176750.00  150366.67  168759.09  149581.82  165500.00       0.00
NAmes         139500.00  133098.93  146260.96  159065.22  180360.00  144062.50
NPkVill            0.00  134555.00  146163.64  143000.00       0.00       0.00
NWAmes             0.00  177765.00  183317.12  201165.00  253450.00       0.00
NoRidge            0.00  262000.00  259436.67  272222.22       0.00       0.00
NridgHt       211700.00  215458.55  264852.71  275000.00       0.00       0.00
OldTown        83333.33  105564.32  136843.57  136350.91  167050.00   97500.00
SWISU          60000.00  121044.44  132257.88  143444.44  158500.00  148633.33
Sawyer        185000.00  124694.23  138583.77  148884.62       0.00  146166.67
SawyerW       216000.00  156147.41  185192.14  211315.00       0.00  237863.25
Somerst       205216.67  191070.18  225570.39  247333.33       0.00       0.00
StoneBr       223966.67  211468.75  233750.00  270000.00       0.00       0.00
Timber             0.00  217263.64  200536.04  241202.60  279900.00       0.00
Veenker       247566.67  245150.00  214090.91       0.00       0.00       0.00
```

上述数据透视表提供了一个全面的快照，展示了不同卧室数量在各社区中如何影响平均销售价格。此分析揭示了几个关键见解：

+   **各社区的可负担性**：你可以一目了然地看到哪些社区提供了具有特定卧室数量的最实惠的住房选项，有助于有针对性地寻找住房。

+   **卧室数量对价格的影响**：该表格突出显示了卧室数量如何影响每个社区的销售价格，提供了对大房子的溢价的评估。

+   **市场缺口和机会**：零值区域表明缺乏符合特定标准的住房，这意味着市场上可能存在缺口或开发者和投资者的机会。

通过利用数据透视表进行分析，你成功地将 Ames 房地产市场中的复杂关系提炼为一种既易于访问又信息丰富的格式。这一过程不仅展示了 pandas 和类似 SQL 的分析技术之间的强大协同作用，还强调了先进数据处理工具在揭示房地产市场中可操作见解的重要性。尽管数据透视表非常有见地，但其真正的潜力是在与可视化分析相结合时发挥的。

为了进一步阐明你的发现并使其更具直观性，你将从数值分析过渡到可视化表示。热图是实现这一目的的优秀工具，特别是在处理像这样的多维数据时。然而，为了提高热图的清晰度并将注意力集中在可操作的数据上，你将使用自定义的颜色方案，以明确突出不存在的社区和卧室数量组合。

```py
# Import an additional library
import matplotlib.colors

# Create a custom color map
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])

# Mask for "zero" values to be colored with a different shade
mask = pivot_table == 0

# Set the size of the plot
plt.figure(figsize=(14, 10))

# Create a heatmap with the mask
sns.heatmap(pivot_table,
            cmap=cmap,
            annot=True,
            fmt=".0f",
            linewidths=.5,
            mask=mask,
            cbar_kws={'label': 'Average Sales Price ($)'})

# Adding title and labels for clarity
plt.title('Average Sales Price by Neighborhood and Number of Bedrooms', fontsize=16)
plt.xlabel('Number of Bedrooms Above Grade', fontsize=12)
plt.ylabel('Neighborhood', fontsize=12)

# Display the heatmap
plt.show()
```

![](img/f4316886e11be4175e0a26fad3ac5191.png)

显示按社区划分的平均销售价格的热图

热图生动地展示了不同社区中按卧室数量划分的平均销售价格的分布。这种颜色编码的视觉辅助工具使得哪些 Ames 地区为各种规模的家庭提供了最实惠的住房选择一目了然。此外，针对零值的独特阴影——表示不存在的社区和卧室数量组合——是市场分析中的一个关键工具。它突出了市场中可能存在需求但供应不足的缺口，为开发者和投资者提供了宝贵的见解。值得注意的是，你的分析还揭示了“旧城”社区的 6 间卧室的房屋价格低于\$100,000。这一发现为大型家庭或寻找高卧室数量且价格合理的投资物业的投资者指明了优异的价值。

通过这一视觉探索，你不仅加深了对房地产市场动态的理解，还展示了高级数据可视化在房地产分析中的不可或缺的作用。数据透视表配合热图，展示了复杂的数据处理和可视化技术如何揭示房地产领域的有价值见解。

## **进一步阅读**

本节提供了更多相关资源，如果你想深入了解，请参考。

#### Python 文档

+   [Pandas 的`DataFrame.query()`方法](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html#pandas.DataFrame.query)

+   [Pandas 的`DataFrame.groupby()`方法](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html)

+   [Pandas 的 `DataFrame.loc[]` 方法](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html)

+   [Pandas 的 `DataFrame.iloc[]` 方法](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html)

+   [Pandas 的 `DataFrame.pivot_table()` 方法](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot_table.html)

#### **资源**

+   [Ames 数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)

+   [Ames 数据字典](https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt)

## **总结**

通过对 Ames 住房数据集的全面探索，突显了 pandas 在进行复杂数据分析中的多功能性和强大能力，常常在不依赖传统数据库的环境中实现或超越 SQL 的可能性。从精准确定详细的住房市场趋势到识别独特的投资机会，你展示了一系列技术，使分析师具备了深度数据探索所需的工具。具体来说，你学会了如何：

+   利用 `DataFrame.query()` 进行数据选择，类似于 SQL 的 `SELECT` 语句。

+   使用 `DataFrame.groupby()` 进行数据汇总和总结，类似于 SQL 的 `GROUP BY`。

+   应用高级数据处理技术，如 `DataFrame.loc[]`、`DataFrame.iloc[]` 和 `DataFrame.pivot_table()` 以进行更深入的分析。

有任何问题吗？请在下方评论中提出问题，我会尽力回答。
