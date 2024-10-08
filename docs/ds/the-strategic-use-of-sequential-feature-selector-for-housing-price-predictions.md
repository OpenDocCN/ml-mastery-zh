# 顺序特征选择器在住房价格预测中的战略性使用

> 原文：[`machinelearningmastery.com/the-strategic-use-of-sequential-feature-selector-for-housing-price-predictions/`](https://machinelearningmastery.com/the-strategic-use-of-sequential-feature-selector-for-housing-price-predictions/)

为了更好地理解住房价格，我们的模型需要简洁明了。我们发布此文的目的是展示如何通过简单但有效的特征选择和工程技术来创建一个有效且简单的线性回归模型。我们使用 Ames 数据集，通过顺序特征选择器（SFS）来识别最具影响力的数值特征，并通过深思熟虑的特征工程提升模型的准确性。

让我们开始吧。

![](img/7191b4fc5d8692c5c8dca04a63ddfcb3.png)

顺序特征选择器在住房价格预测中的战略性使用

图片来源：[Mahrous Houses](https://unsplash.com/photos/brown-and-black-table-lamp-on-black-wooden-shelf-kUCTWQG9IJo)。部分版权保留。

## 概述

本文分为三个部分，它们是：

+   确定最具预测性的数值特征

+   评估单个特征的预测能力

+   通过特征工程提升预测准确性

## 确定最具预测性的数值特征

在我们探索的初始阶段，我们着手确定 Ames 数据集中最具预测性的数值特征。这是通过应用顺序特征选择器（SFS）来实现的，该工具旨在筛选特征并选择能够最大化我们模型预测准确性的特征。该过程非常简单，仅关注数值列，并排除任何缺失值，以确保分析的干净和稳健：

```py
# Load only the numeric columns from the Ames dataset
import pandas as pd
Ames = pd.read_csv('Ames.csv').select_dtypes(include=['int64', 'float64'])

# Drop any columns with missing values
Ames = Ames.dropna(axis=1)

# Import Linear Regression and Sequential Feature Selector from scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector

# Initializing the Linear Regression model
model = LinearRegression()

# Perform Sequential Feature Selector
sfs = SequentialFeatureSelector(model, n_features_to_select=1)
X = Ames.drop('SalePrice', axis=1)  # Features
y = Ames['SalePrice']  # Target variable
sfs.fit(X,y)           # Uses a default of cv=5
selected_feature = X.columns[sfs.get_support()]
print("Feature selected for highest predictability:", selected_feature[0])
```

这将输出：

```py
Feature selected for highest predictability: OverallQual
```

这一结果显著挑战了最初认为面积可能是住房价格最具预测性的特征的假设。相反，它强调了**整体质量的重要性**，表明与最初的预期相反，质量是买家的主要考虑因素。需要注意的是，顺序特征选择器[利用交叉验证，默认设置为五折(cv=5)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html)来评估每个特征子集的性能。这种方法确保了所选特征——通过最高的平均交叉验证 R²得分来体现——是稳健的，并且有可能在未见过的数据上良好地泛化。

## 评估单个特征的预测能力

在初步发现的基础上，我们进一步深入分析，按特征的预测能力进行排名。通过交叉验证，我们独立评估每个特征，计算其交叉验证的平均 R²得分，以确定其对模型准确性的个体贡献。

```py
# Building on the earlier block of code:
from sklearn.model_selection import cross_val_score

# Dictionary to hold feature names and their corresponding mean CV R² scores
feature_scores = {}

# Iterate over each feature, perform CV, and store the mean R² score
for feature in X.columns:
    X_single = X[[feature]]
    cv_scores = cross_val_score(model, X_single, y, cv=5)
    feature_scores[feature] = cv_scores.mean()

# Sort features based on their mean CV R² scores in descending order
sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)

# Print the top 3 features and their scores
top_3 = sorted_features[0:3]
for feature, score in top_3:
    print(f"Feature: {feature}, Mean CV R²: {score:.4f}")
```

这将输出：

```py
Feature: OverallQual, Mean CV R²: 0.6183
Feature: GrLivArea, Mean CV R²: 0.5127
Feature: 1stFlrSF, Mean CV R²: 0.3957
```

这些发现突出了整体质量（“OverallQual”）、生活面积（“GrLivArea”）和一层面积（“1stFlrSF”）在房价预测中的关键作用。

## 通过特征工程提升预测准确性

在我们旅程的最后阶段，我们通过将‘OverallQual’与‘GrLivArea’相乘，创建了一个新特征“质量加权面积”。这种融合旨在合成一个更强大的预测因子，囊括了房产的质量和面积维度。

```py
# Building on the earlier blocks of code:
Ames['QualityArea'] = Ames['OverallQual'] * Ames['GrLivArea']

# Setting up the feature and target variable for the new 'QualityArea' feature
X = Ames[['QualityArea']]  # New feature
y = Ames['SalePrice']

# 5-Fold CV on Linear Regression
model = LinearRegression()
cv_scores = cross_val_score(model, X, y, cv=5)

# Calculating the mean of the CV scores
mean_cv_score = cv_scores.mean()
print(f"Mean CV R² score using 'Quality Weighted Area': {mean_cv_score:.4f}")
```

这将输出：

```py
Mean CV R² score using 'Quality Weighted Area': 0.7484
```

这一 R² 分数的显著提高生动展示了特征组合捕捉数据更细微方面的效果，为在预测建模中谨慎应用特征工程提供了有力的案例。

## **进一步阅读**

#### API

+   [sklearn.feature_selection.SequentialFeatureSelector](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html) API

#### 教程

+   [序列特征选择的实用介绍](https://www.yourdatateacher.com/2023/02/15/a-practical-introduction-to-sequential-feature-selection/) by Gianluca Malato

#### **Ames 房价数据集与数据字典**

+   [Ames 数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)

+   [Ames 数据字典](https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt)

## **总结**

通过这三部分的探索，你已经掌握了用简洁的方法确定和增强房价预测因子的过程。从使用序列特征选择器（SFS）识别最具预测性的特征开始，我们发现整体质量至关重要。这一步骤尤其重要，因为我们的目标是创建最佳的简单线性回归模型，从而排除分类特征以进行简化分析。这一探索从使用序列特征选择器（SFS）识别整体质量作为关键预测因子，进而评估生活面积和一层面积的影响。创建“质量加权面积”这一融合质量与面积的特征，显著提升了模型的准确性。通过特征选择和工程的旅程突显了简洁在改进房地产预测模型中的力量，深入揭示了真正影响房价的因素。这一探索强调了即使是简单模型，只要使用正确的技术，也能对像 Ames 房价这样的复杂数据集提供深刻的见解。

具体来说，你学习了：

+   序列特征选择在揭示房价最重要预测因子方面的价值。

+   在预测爱荷华州 Ames 的房价时，质量比面积更重要。

+   合并特征到“质量加权面积”如何提升模型准确性。

你是否有关于特征选择或工程方面的经验想要分享，或者对这个过程有疑问？请在下方评论中提出你的问题或给我们反馈，我会尽力回答。
