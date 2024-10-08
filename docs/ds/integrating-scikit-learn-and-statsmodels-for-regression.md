# 结合 Scikit-Learn 和 Statsmodels 进行回归

> 原文：[`machinelearningmastery.com/integrating-scikit-learn-and-statsmodels-for-regression/`](https://machinelearningmastery.com/integrating-scikit-learn-and-statsmodels-for-regression/)

统计学和机器学习都旨在从数据中提取洞察，但它们的方法大相径庭。传统统计学主要关注推断，使用整个数据集来检验假设和估计关于更大人群的概率。相对而言，机器学习强调预测和决策，通常采用训练-测试分割的方法，其中模型从数据的一个部分（训练集）学习，并在未见过的数据（测试集）上验证其预测。

在这篇文章中，我们将展示一个看似简单的线性回归技术如何从这两个角度来看待。我们将通过使用 Scikit-Learn 进行机器学习和 Statsmodels 进行统计推断，探讨它们的独特贡献。

让我们开始吧。

![](img/ebca9f6b8212bb3833601155dab5008c.png)

结合 Scikit-Learn 和 Statsmodels 进行回归。

图片由[Stephen Dawson](https://unsplash.com/photos/turned-on-monitoring-screen-qwtCeJ5cLYs)提供。版权所有。

## 概述

本文分为三个部分：

+   监督学习：分类与回归

+   从机器学习角度深入回归

+   通过统计洞察提升理解

## 监督学习：分类与回归

监督学习是机器学习的一个分支，其中模型在标记的数据集上进行训练。这意味着训练数据集中的每个示例都与正确的输出配对。训练完成后，模型可以将其学到的知识应用于新的、未见过的数据。

在监督学习中，我们遇到两个主要任务：分类和回归。这些任务取决于我们要预测的输出类型。如果目标是预测类别，例如确定一封邮件是否为垃圾邮件，那么我们处理的是分类任务。相反，如果我们估计一个值，例如根据汽车的特征计算每加仑多少英里（MPG），这属于回归。输出的性质——一个类别还是一个数字——引导我们选择合适的方法。

在这一系列中，我们将使用[Ames 住房数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)。该数据集提供了与房屋相关的全面特征，包括建筑细节、条件和位置，旨在预测每栋房屋的“SalePrice”（销售价格）。

Python

```py
# Load the Ames dataset
import pandas as pd
Ames = pd.read_csv('Ames.csv')

# Display the first few rows of the dataset and the data type of 'SalePrice'
print(Ames.head())

sale_price_dtype = Ames['SalePrice'].dtype
print(f"The data type of 'SalePrice' is {sale_price_dtype}.")
```

这应该输出：

```py
         PID  GrLivArea  SalePrice  ...          Prop_Addr   Latitude  Longitude
0  909176150        856     126000  ...    436 HAYWARD AVE  42.018564 -93.651619
1  905476230       1049     139500  ...       3416 WEST ST  42.024855 -93.663671
2  911128020       1001     124900  ...       320 S 2ND ST  42.021548 -93.614068
3  535377150       1039     114000  ...   1524 DOUGLAS AVE  42.037391 -93.612207
4  534177230       1665     227000  ...  2304 FILLMORE AVE  42.044554 -93.631818
[5 rows x 85 columns]

The data type of 'SalePrice' is int64.
```

“SalePrice” 列的数据类型为 `int64`，表示它代表整数值。由于 “SalePrice” 是一个数值型（连续型）变量而非分类变量，因此预测 “SalePrice” 将是一个**回归任务**。这意味着目标是根据数据集中提供的输入特征预测一个连续的量（房屋的售价）。

## 从机器学习的角度深入回归分析

机器学习中的监督学习专注于基于输入数据预测结果。在我们的案例中，使用 Ames Housing 数据集，我们的目标是根据房屋的生活面积预测其售价——这是一项经典的回归任务。为此，我们使用 scikit-learn，该工具因其在构建预测模型方面的简单性和有效性而闻名。

首先，我们选择 “GrLivArea”（地面生活面积）作为特征，“SalePrice” 作为目标。下一步是使用 scikit-learn 的 `train_test_split()` 函数将数据集分为训练集和测试集。这一步至关重要，它使我们能够在一组数据上训练模型，并在另一组数据上评估其性能，从而确保模型的可靠性。

下面是我们如何做：

```py
# Import Linear Regression from scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Select features and target
X = Ames[['GrLivArea']]  # Feature: GrLivArea, 2D matrix
y = Ames['SalePrice']    # Target: SalePrice, 1D vector

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Scoring the model
score = round(model.score(X_test, y_test), 4)
print(f"Model R² Score: {score}")
```

这应该输出：

```py
Model R² Score: 0.4789
```

上述代码中导入的 `LinearRegression` 对象是 scikit-learn 的线性回归实现。模型的 R² 分数为 0.4789 表明我们的模型仅凭生活面积就能解释大约 48% 的售价变异——对于这样一个简单的模型来说这是一个重要的见解。这一步标志着我们初步涉足机器学习，展示了我们如何轻松地评估模型在未见或测试数据上的性能。

## 通过统计见解增强理解

在探讨了 scikit-learn 如何帮助我们评估模型在未见数据上的性能后，我们现在将注意力转向 `statsmodels`，这是一个提供不同分析角度的 Python 包。虽然 scikit-learn 在构建模型和预测结果方面表现出色，但 `statsmodels` 通过深入分析数据和模型的统计方面脱颖而出。让我们看看 `statsmodels` 如何从不同的层面为你提供见解：

```py
import statsmodels.api as sm

# Adding a constant to our independent variable for the intercept
X_with_constant = sm.add_constant(X)

# Fit the OLS model
model_stats = sm.OLS(y, X_with_constant).fit()

# Print the summary of the model
print(model_stats.summary())
```

第一个关键区别是 `statsmodels` 使用我们数据集中的所有观察值。与预测建模方法不同，预测建模方法中我们将数据分为训练集和测试集，`statsmodels` 利用整个数据集提供全面的统计见解。这种完全利用数据的方法使我们能够详细理解变量之间的关系，并提高统计估计的准确性。上述代码应该输出以下内容：

```py
OLS Regression Results                            
==============================================================================
Dep. Variable:              SalePrice   R-squared:                       0.518
Model:                            OLS   Adj. R-squared:                  0.518
Method:                 Least Squares   F-statistic:                     2774.
Date:                Sun, 31 Mar 2024   Prob (F-statistic):               0.00
Time:                        19:59:01   Log-Likelihood:                -31668.
No. Observations:                2579   AIC:                         6.334e+04
Df Residuals:                    2577   BIC:                         6.335e+04
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       1.377e+04   3283.652      4.195      0.000    7335.256    2.02e+04
GrLivArea    110.5551      2.099     52.665      0.000     106.439     114.671
==============================================================================
Omnibus:                      566.257   Durbin-Watson:                   1.926
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             3364.083
Skew:                           0.903   Prob(JB):                         0.00
Kurtosis:                       8.296   Cond. No.                     5.01e+03
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 5.01e+03\. This might indicate that there are
strong multicollinearity or other numerical problems.
```

请注意，这与 scikit-learn 中的回归不同，因为此处使用的是整个数据集，而没有进行训练-测试分割。

让我们深入 `statsmodels` 对我们 OLS 回归的输出，并解释 p 值、系数、置信区间和诊断信息告诉我们关于我们模型的什么，特别是关注于从 “GrLivArea” 预测 “SalePrice”：

### p 值和系数

+   **“GrLivArea”的系数**： “GrLivArea”的系数为 110.5551。这意味着每增加一平方英尺的生活面积，房屋的销售价格预计会增加约$110.55。这个系数量化了生活面积对房屋销售价格的影响。

+   **“GrLivArea”的 p 值**：与“GrLivArea”系数相关的 p 值基本为 0（由`P>|t|`接近 0.000 指示），表明生活面积是一个高度显著的销售价格预测因素。在统计学上，我们可以拒绝系数为零（无效应）的原假设，并且可以自信地表示生活面积与销售价格之间存在强关系（但不一定是唯一因素）。

### 信赖区间

+   **“GrLivArea”的信赖区间**： “GrLivArea”系数的信赖区间为[106.439, 114.671]。这个范围告诉我们我们可以有 95%的信心认为生活面积对销售价格的真实影响落在这个区间内。它提供了我们对系数估计值精确度的度量。

### Diagnostics

+   **R-squared (R²)**：R²值为 0.518，表示生活面积可以解释约 51.8%的销售价格变动。这是衡量模型拟合数据的程度的指标。由于数据不同，因此预计这个数值与 scikit-learn 回归中的情况不相同。

+   **F-statistic 和 Prob (F-statistic)**：F-statistic 是衡量模型整体显著性的指标。F-statistic 为 2774，Prob (F-statistic)基本为 0，这表明模型在统计上是显著的。

+   **Omnibus, Prob(Omnibus)**：这些测试评估残差的正态性。残差是预测值$\hat{y}$和实际值$y$之间的差异。线性回归算法基于残差服从正态分布的假设。Prob(Omnibus)值接近 0 表明残差不是正态分布的，这可能对某些统计测试的有效性构成担忧。

+   **Durbin-Watson**：Durbin-Watson 统计量测试残差中的自相关性。它的范围在 0 到 4 之间。接近 2 的值（1.926）表明没有强自相关。否则，这表明$X$和$y$之间的关系可能不是线性的。

`statsmodels`提供的这份全面输出深入了解了“GrLivArea”如何以及为何影响“SalePrice”，并以统计证据为基础。它强调了不仅仅依赖模型进行预测的重要性，还要对模型进行解释，以便根据扎实的统计基础做出明智的决策。这些见解对于那些希望探索数据背后统计故事的人来说极为宝贵。

## **进一步阅读**

#### APIs

+   [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) API

+   [statsmodels.api](https://www.statsmodels.org/stable/api.html) API

#### 教程

+   [Scikit-Learn 线性回归：带实例的全面指南](https://www.simplilearn.com/tutorials/scikit-learn-tutorial/sklearn-linear-regression-with-examples) 作者：Avijeet Biswal

**书籍**

+   [Python 数据科学手册](https://jakevdp.github.io/PythonDataScienceHandbook/) 作者：Jake VanderPlas

#### **Ames 住房数据集与数据字典**

+   [Ames 数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)

+   [Ames 数据字典](https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt)

## **总结**

在这篇文章中，我们探讨了监督学习的基础概念，特别是回归分析。通过使用 Ames 住房数据集，我们展示了如何使用`scikit-learn`进行模型构建和性能评估，以及使用`statsmodels`获取对数据的统计见解。从数据到见解的过程凸显了预测建模和统计分析在有效理解和利用数据中的关键作用。

具体而言，你学到了：

+   监督学习中分类任务和回归任务的区别。

+   如何根据数据的性质确定使用哪种方法。

+   如何使用`scikit-learn`实现一个简单的线性回归模型，评估其性能，并理解模型的 R²得分的重要性。

+   使用`statsmodels`探索数据的统计方面，包括系数、p 值和置信区间的解释，以及诊断测试对模型假设的重要性。

你有任何问题吗？请在下方评论中提出你的问题，我会尽力回答。
