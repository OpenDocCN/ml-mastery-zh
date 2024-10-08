# 探索字典、分类变量和填补 Ames 数据集中的数据

> 原文：[`machinelearningmastery.com/classifying_variables/`](https://machinelearningmastery.com/classifying_variables/)

房地产市场是一个复杂的生态系统，由诸如位置、物业特征、市场趋势和经济指标等多个变量驱动。一个深入探讨这一复杂性的 数据集是 Ames Housing 数据集。该数据集来自 Iowa 的 Ames，包括各种物业及其特征，从小巷通行方式到物业的整体状况。

在这篇文章中，你的目标是通过数据科学技术更详细地了解这个数据集。具体而言，你将关注如何识别分类变量和数值变量，因为理解这些变量对于任何数据驱动的决策过程至关重要。

开始吧。

![](img/901cfca1dbcdd2b628de1d387de9db36.png)

探索字典、分类变量和填补 Ames 数据集中的数据

图片由[Brigitte Tohm](https://unsplash.com/photos/pink-petaled-flowers-bouquet-1i4P2B148FQ)提供。保留所有权利。

## 概述

本文分为三部分：

+   数据字典的重要性

+   识别分类变量和数值变量

+   缺失数据填补

## **数据字典的重要性**

分析 Ames Housing 数据集的一个关键第一步是利用其[数据字典](https://jse.amstat.org/v19n3/decock/DataDocumentation.txt)。这个版本不仅列出了特征和定义，还将其分类为**名义型**、**顺序型**、**离散型**和**连续型**，以指导我们的分析方法。

+   **名义型变量**是没有顺序的类别，如“邻里”。它们有助于识别用于分组分析的细分领域。

+   **顺序型变量**具有明确的顺序（例如“厨房质量”）。它们允许进行排序和基于顺序的分析，但不意味着类别之间的间距相等。

+   **离散型变量**是可计数的数字，如“卧室”。它们在汇总或比较数量的分析中至关重要。

+   **连续型变量**在连续尺度上进行测量，例如“地块面积”。它们支持广泛的统计分析，依赖于详细的细节。

理解这些变量类型也有助于选择适当的可视化技术。**名义型和顺序型变量**适合使用条形图，这可以有效地突出类别差异和排名。相反，**离散型和连续型变量**则最好通过直方图、散点图和折线图来表示，这些图表能够展示数据的分布、关系和趋势。

**启动你的项目**，请阅读我的书籍[《数据科学初学者指南》](https://machinelearning.samcart.com/products/beginners-guide-data-science/)。它提供了**自学教程**和**工作代码**。

## **识别分类变量和数值变量**

基于我们对数据字典的理解，让我们深入探讨如何使用 Python 的 pandas 库在 Ames 数据集中实际区分分类和数值变量。这一步骤对指导我们后续的数据处理和分析策略至关重要。

```py
# Load and obtain the data types from the Ames dataset
import pandas as pd
Ames = pd.read_csv('Ames.csv')

print(Ames.dtypes)
print(Ames.dtypes.value_counts())
```

执行上述代码将产生以下输出，通过数据类型对每个特征进行分类：

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

object     44
int64      27
float64    14
dtype: int64
```

该输出显示数据集包含`object`（44 个变量）、`int64`（27 个变量）和`float64`（14 个变量）数据类型。在这里，`object`通常表示名义变量，即没有固有顺序的分类数据。与此同时，`int64`和`float64`则表示数值数据，这些数据可以是离散的（`int64`用于可计数的数字）或连续的（`float64`用于在连续范围内可测量的量）。

现在我们可以利用 pandas 的`select_dtypes()`方法明确区分 Ames 数据集中的数值特征和分类特征。

```py
# Build on the above block of code
# Separating numerical and categorical features
numerical_features = Ames.select_dtypes(include=['int64', 'float64']).columns
categorical_features = Ames.select_dtypes(include=['object']).columns

# Displaying the separated lists
print("Numerical Features:", numerical_features)
print("Categorical Features:", categorical_features)
```

`numerical_features`捕获存储为`int64`和`float64`的变量，分别指示可计数和可测量的量。相对而言，`categorical_features`包括类型为`object`的变量，通常表示没有量化值的名义或顺序数据：

```py
Numerical Features: Index(['PID', 'GrLivArea', 'SalePrice', 'MSSubClass', 'LotFrontage', 'LotArea',
       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
       'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
       '2ndFlrSF', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold', 'GeoRefNo', 'Latitude', 'Longitude'],
      dtype='object')
Categorical Features: Index(['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition', 'Prop_Addr'],
      dtype='object')
```

值得注意的是，某些变量，例如‘MSSubClass’，尽管被编码为数值，但实际上作为分类数据使用，这突显了参考数据字典以确保准确分类的重要性。同样，像‘MoSold’（售出月份）和‘YrSold’（售出年份）这样的特征虽然在本质上是数值的，但在没有进行数学运算的情况下，它们通常可以被视为分类变量。我们可以使用 pandas 中的`astype()`方法将这些转换为分类特征。

```py
# Building on the above 2 blocks of code
Ames['MSSubClass'] = Ames['MSSubClass'].astype('object')
Ames['YrSold'] = Ames['YrSold'].astype('object')
Ames['MoSold'] = Ames['MoSold'].astype('object')
print(Ames.dtypes.value_counts())
```

在执行此转换后，`object`数据类型的列数已增加到 47（之前为 44），而`int64`已降至 24（之前为 27）。

```py
object     47
int64      24
float64    14
dtype: int64
```

对数据字典、数据集的性质和领域专业知识的仔细评估可以有助于正确地重新分类数据类型。

## **缺失数据填补**

处理缺失数据是每个数据科学家面临的挑战。忽略缺失值或处理不当可能导致分析偏差和错误结论。填补技术的选择通常取决于数据的性质——分类数据或数值数据。此外，数据字典中的信息将会有用（例如 Pool Quality），在这些情况下，缺失值（“NA”）有其意义，即特定属性的缺失。

**带缺失值的分类特征的数据填补**

你可以识别分类数据类型并按缺失数据对它们的影响程度进行排序。

```py
# Calculating the percentage of missing values for each column
missing_data = Ames.isnull().sum()
missing_percentage = (missing_data / len(Ames)) * 100
data_type = Ames.dtypes

# Combining the counts and percentages into a DataFrame for better visualization
missing_info = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_percentage,
                             'Data Type':data_type})

# Sorting the DataFrame by the percentage of missing values in descending order
missing_info = missing_info.sort_values(by='Percentage', ascending=False)

# Display columns with missing values of 'object' data type
print(missing_info[(missing_info['Missing Values'] > 0) & (missing_info['Data Type'] == 'object')])
```

```py
              Missing Values  Percentage Data Type
PoolQC                  2570   99.651028    object
MiscFeature             2482   96.238852    object
Alley                   2411   93.485847    object
Fence                   2054   79.643273    object
FireplaceQu             1241   48.119426    object
GarageCond               129    5.001939    object
GarageQual               129    5.001939    object
GarageFinish             129    5.001939    object
GarageType               127    4.924389    object
BsmtExposure              71    2.753005    object
BsmtFinType2              70    2.714230    object
BsmtFinType1              69    2.675456    object
BsmtQual                  69    2.675456    object
BsmtCond                  69    2.675456    object
Prop_Addr                 20    0.775494    object
MasVnrType                14    0.542846    object
Electrical                 1    0.038775    object
```

数据字典指示，类别特征缺失值的整个列表表示该特征在给定属性中缺失，除了“Electrical”外。基于这一见解，我们可以用“mode”来插补电气系统的 1 个缺失数据点，并用`"None"`（带引号以使其成为 Python 字符串）插补其他所有缺失值。

```py
# Building on the above block of code
# Imputing Missing Categorical Data

mode_value = Ames['Electrical'].mode()[0]
Ames['Electrical'].fillna(mode_value, inplace=True)

missing_categorical = missing_info[(missing_info['Missing Values'] > 0)
                           & (missing_info['Data Type'] == 'object')]

for item in missing_categorical.index.tolist():
    Ames[item].fillna("None", inplace=True)

print(Ames[missing_categorical.index].isnull().sum())
```

这确认现在类别特征的缺失值已不再存在：

```py
PoolQC          0
MiscFeature     0
Alley           0
Fence           0
FireplaceQu     0
GarageCond      0
GarageQual      0
GarageFinish    0
GarageType      0
BsmtExposure    0
BsmtFinType2    0
BsmtFinType1    0
BsmtQual        0
BsmtCond        0
Prop_Addr       0
MasVnrType      0
Electrical      0
```

**缺失值的数值特征插补**

我们可以应用上述演示的相同技术来识别数值数据类型，并按其受到缺失数据影响的程度进行排名。

```py
# Build on the above blocks of code
# Import Numpy
import numpy as np

# Calculating the percentage of missing values for each column
missing_data = Ames.isnull().sum()
missing_percentage = (missing_data / len(Ames)) * 100
data_type = Ames.dtypes

# Combining the counts and percentages into a DataFrame for better visualization
missing_info = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_percentage,
                             'Data Type':data_type})

# Sorting the DataFrame by the percentage of missing values in descending order
missing_info = missing_info.sort_values(by='Percentage', ascending=False)

# Display columns with missing values of numeric data type
print(missing_info[(missing_info['Missing Values'] > 0)
                   & (missing_info['Data Type'] == np.number)])
```

```py
              Missing Values  Percentage Data Type
LotFrontage              462   17.913920   float64
GarageYrBlt              129    5.001939   float64
Longitude                 97    3.761148   float64
Latitude                  97    3.761148   float64
GeoRefNo                  20    0.775494   float64
MasVnrArea                14    0.542846   float64
BsmtFullBath               2    0.077549   float64
BsmtHalfBath               2    0.077549   float64
BsmtFinSF2                 1    0.038775   float64
GarageArea                 1    0.038775   float64
BsmtFinSF1                 1    0.038775   float64
BsmtUnfSF                  1    0.038775   float64
TotalBsmtSF                1    0.038775   float64
GarageCars                 1    0.038775   float64
```

上述说明了与缺失类别数据相比，缺失数值数据的实例较少。然而，数据字典对于直接插补并不十分有用。在数据科学中是否插补缺失数据在很大程度上取决于分析的目标。通常，数据科学家可能会生成多个插补值，以考虑插补过程中的不确定性。常见的多重插补方法包括（但不限于）均值、 медиан和回归插补。作为基准，我们将在这里演示如何使用均值插补，但根据任务的不同可能会参考其他技术。

```py
# Build on the above blocks of code
# Initialize a DataFrame to store the concise information
concise_info = pd.DataFrame(columns=['Feature', 'Missing Values After Imputation', 
                                     'Mean Value Used to Impute'])

# Identify and impute missing numerical values, and store the related concise information
missing_numeric_df = missing_info[(missing_info['Missing Values'] > 0)
                           & (missing_info['Data Type'] == np.number)]

for item in missing_numeric_df.index.tolist():
    mean_value = Ames[item].mean(skipna=True)
    Ames[item].fillna(mean_value, inplace=True)

    # Append the concise information to the concise_info DataFrame
    concise_info.loc[len(concise_info)] = pd.Series({
        'Feature': item,
        'Missing Values After Imputation': Ames[item].isnull().sum(),
        # This should be 0 as we are imputing all missing values
        'Mean Value Used to Impute': mean_value
    })

# Display the concise_info DataFrame
print(concise_info)
```

这将打印：

```py
         Feature Missing Values After Imputation  Mean Value Used to Impute
0    LotFrontage                               0               6.851063e+01
1    GarageYrBlt                               0               1.976997e+03
2      Longitude                               0              -9.364254e+01
3       Latitude                               0               4.203456e+01
4       GeoRefNo                               0               7.136762e+08
5     MasVnrArea                               0               9.934698e+01
6   BsmtFullBath                               0               4.353900e-01
7   BsmtHalfBath                               0               6.208770e-02
8     BsmtFinSF2                               0               5.325950e+01
9     GarageArea                               0               4.668646e+02
10    BsmtFinSF1                               0               4.442851e+02
11     BsmtUnfSF                               0               5.391947e+02
12   TotalBsmtSF                               0               1.036739e+03
13    GarageCars                               0               1.747867e+00
```

有时，我们也可以选择不对缺失值进行任何插补，以保留原始数据集的真实性，并在必要时删除那些没有完整和准确数据的观察值。或者，你也可以尝试建立一个机器学习模型来**猜测**基于同一行中的其他数据的缺失值，这就是回归插补的原理。作为上述基准插补的最终步骤，让我们交叉检查是否还有缺失值。

```py
# Build on the above blocks of code
missing_values_count = Ames.isnull().sum().sum()
print(f'The DataFrame has a total of {missing_values_count} missing values.')
```

你应该看到：

```py
The DataFrame has a total of 0 missing values.
```

恭喜！我们已经成功地通过基准操作插补了 Ames 数据集中的所有缺失值。值得注意的是，还有许多其他技术可以用来插补缺失数据。作为数据科学家，探索各种选项并确定最适合给定背景的方法对生成可靠和有意义的结果至关重要。

### 想开始学习数据科学初学者指南吗？

立即参加我的免费电子邮件速成课程（附样例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## **进一步阅读**

#### **资源**

+   [Ames 数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)

+   [Ames 数据字典（扩展版）](https://jse.amstat.org/v19n3/decock/DataDocumentation.txt)

## **总结**

在本教程中，我们通过数据科学技术的视角探讨了 Ames 住房数据集。我们讨论了数据字典在理解数据集变量中的重要性，并深入研究了帮助有效识别和处理这些变量的 Python 代码片段。

了解你所处理的变量的本质对于任何数据驱动的决策过程至关重要。正如我们所见，Ames 数据字典在这方面作为一个宝贵的指南。结合 Python 强大的数据处理库，处理像 Ames Housing 数据集这样的复杂数据集变得更加可控。

具体来说，你学到了：

+   **在评估数据类型和填补策略时数据字典的重要性。**

***数值特征和类别特征的识别与重新分类方法。*****   **如何使用 pandas 库填补缺失的类别特征和数值特征。******

******你有任何问题吗？请在下方评论中提出你的问题，我会尽力回答。******
