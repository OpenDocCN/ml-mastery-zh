# 一热编码：理解数据中的“Hot”

> 原文：[`machinelearningmastery.com/one-hot-encoding-understanding-the-hot-in-data/`](https://machinelearningmastery.com/one-hot-encoding-understanding-the-hot-in-data/)

正确准备类别数据是机器学习中的一个基本步骤，特别是在使用线性模型时。一热编码作为一个关键技术，使得将类别变量转换为机器可理解的格式成为可能。本文告诉你为什么不能直接使用类别变量，并展示了在寻找线性回归中最具预测性的类别特征时使用一热编码的方法。

让我们开始吧。

![](img/84a746c5b165dc1f69ee148356c2ca81.png)

一热编码：理解数据中的“Hot”

照片由 [sutirta budiman](https://unsplash.com/photos/low-angle-photography-of-yellow-hot-air-balloon-eN6c3KWNXcA) 提供。保留部分权利。

## 概述

本文分为三个部分：

+   什么是一热编码？

+   确定最具预测性的类别特征

+   评估单个特征的预测能力

## 什么是一热编码？

在线性模型的数据预处理过程中，“一热编码”是管理类别数据的关键技术。在这种方法中，“hot”表示类别的存在（编码为 1），而“cold”（或 0）表示其不存在，使用二进制向量进行表示。

从测量水平的角度看，类别数据是**名义数据**，这意味着如果我们使用数字作为标签（例如，1 代表男性，2 代表女性），加法和减法等操作将没有意义。而且如果标签不是数字，你甚至无法进行任何数学运算。

一热编码将变量的每个类别分隔成独立的特征，防止在线性回归和其他线性模型中将类别数据误解为具有某种序数意义。编码后，数字具有实际意义，并且可以直接用于数学方程。

例如，考虑一个像“颜色”这样的类别特征，其值为红色、蓝色和绿色。一热编码将其转换为三个二进制特征（“Color_Red”，“Color_Blue”和“Color_Green”），每个特征指示每个观察值的颜色的存在（1）或不存在（0）。这种表示方式向模型明确说明这些类别是不同的，没有固有的顺序。

为什么这很重要？许多机器学习模型，包括线性回归，都在数值数据上运行，并假设值之间存在数值关系。直接将类别编码为数字（例如，红色=1，蓝色=2，绿色=3）可能会暗示不存在的层次结构或定量关系，从而可能扭曲预测。一热编码避免了这个问题，以模型可以准确解释的形式保留了数据的类别性质。

让我们将这一技术应用于 Ames 数据集，通过示例展示转换过程：

```py
# Load only categorical columns without missing values from the Ames dataset
import pandas as pd
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["object"]).dropna(axis=1)
print(f"The shape of the DataFrame before One Hot Encoding is: {Ames.shape}")

# Import OneHotEncoder and apply it to Ames:
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
Ames_One_Hot = encoder.fit_transform(Ames)

# Convert the encoded result back to a DataFrame
Ames_encoded_df = pd.DataFrame(Ames_One_Hot, columns=encoder.get_feature_names_out(Ames.columns))

# Display the new DataFrame and it's expanded shape
print(Ames_encoded_df.head())
print(f"The shape of the DataFrame after One Hot Encoding is: {Ames_encoded_df.shape}")
```

这将输出：

```py
The shape of the DataFrame before One Hot Encoding is: (2579, 27)

   MSZoning_A (agr)  ...  SaleCondition_Partial
0               0.0  ...                    0.0
1               0.0  ...                    0.0
2               0.0  ...                    0.0
3               0.0  ...                    0.0
4               0.0  ...                    0.0
[5 rows x 188 columns]

The shape of the DataFrame after One Hot Encoding is: (2579, 188)
```

如图所示，Ames 数据集的分类列被转换为 188 个不同的特征，展示了独热编码提供的扩展复杂性和详细表示。虽然这种扩展增加了数据集的维度，但在建模分类特征与线性回归目标变量之间的关系时，这是一个至关重要的预处理步骤。

## 识别最具预测性的分类特征

在理解了独热编码在线性模型中的基本前提和应用后，我们分析的下一步是确定哪个分类特征对预测目标变量贡献最大。在下面的代码片段中，我们迭代数据集中的每个分类特征，应用独热编码，并结合交叉验证评估其预测能力。在这里，`drop="first"` 参数在 `OneHotEncoder` 函数中起着至关重要的作用：

```py
# Buidling on the code above to identify top categorical feature
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Set 'SalePrice' as the target variable
y = pd.read_csv("Ames.csv")["SalePrice"]

# Dictionary to store feature names and their corresponding mean CV R² scores
feature_scores = {}

for feature in Ames.columns:
    encoder = OneHotEncoder(drop="first")
    X_encoded = encoder.fit_transform(Ames[[feature]])

    # Initialize the linear regression model
    model = LinearRegression()

    # Perform 5-fold cross-validation and calculate R² scores
    scores = cross_val_score(model, X_encoded, y)
    mean_score = scores.mean()

    # Store the mean R² score
    feature_scores[feature] = mean_score

# Sort features based on their mean CV R² scores in descending order
sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
print("Feature selected for highest predictability:", sorted_features[0][0])
```

`drop="first"` 参数用于缓解完全共线性。通过丢弃第一个类别（在所有其他类别中隐式编码为零），我们减少了冗余和输入变量的数量，而不会丢失任何信息。这种做法简化了模型，使其更易于解释，并且通常提高其性能。上面的代码将输出：

```py
Feature selected for highest predictability: Neighborhood
```

我们的分析揭示了“Neighborhood”是数据集中预测能力最高的分类特征。这一发现突显了位置在 Ames 数据集中的房价上的显著影响。

## 评估单个特征的预测能力

通过对独热编码的更深入理解以及识别最具预测性的分类特征，我们现在将分析扩展到揭示对房价有显著影响的前五个分类特征。这一步对微调我们的预测模型至关重要，使我们能够专注于那些在预测结果中提供最大价值的特征。通过评估每个特征的平均交叉验证 R²分数，我们不仅可以确定这些特征的个体重要性，还能洞察不同方面的属性如何影响整体估值。

让我们*深入探讨*这个评估：

```py
# Building on the code above to determine the performance of top 5 categorical features
print("Top 5 Categorical Features:")
for feature, score in sorted_features[0:5]:
    print(f"{feature}: Mean CV R² = {score:.4f}")
```

我们分析的输出呈现了决定房价的关键因素的揭示性快照：

```py
Top 5 Categorical Features:
Neighborhood: Mean CV R² = 0.5407
ExterQual: Mean CV R² = 0.4651
KitchenQual: Mean CV R² = 0.4373
Foundation: Mean CV R² = 0.2547
HeatingQC: Mean CV R² = 0.1892
```

这一结果突显了“Neighborhood”作为顶级预测因子的作用，强化了位置对房价的显著影响。紧随其后的是“ExterQual”（外部材料质量）和“KitchenQual”（厨房质量），这表明买家对建筑质量和装修的重视。“Foundation”和“HeatingQC”（供暖质量和状态）也显得重要，尽管预测能力较低，这表明结构完整性和舒适性特征是购房者的重要考量因素。

## **进一步阅读**

#### APIs

+   [sklearn.preprocessing.OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) API

#### 教程

+   [One-hot 编码分类变量](https://www.blog.trainindata.com/one-hot-encoding-categorical-variables/) 作者 Sole Galli

#### **Ames 房屋数据集及数据字典**

+   [Ames 数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)

+   [Ames 数据字典](https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt)

## **总结**

在这篇文章中，我们重点讲解了为线性模型准备分类数据的关键过程。我们从解释 One Hot Encoding 开始，展示了该技术如何通过创建二进制向量使分类数据对线性回归变得可解释。我们的分析确定了“Neighborhood”作为对房价影响最大的分类特征，强调了位置在房地产估价中的关键作用。

具体来说，你学到了：

+   One Hot Encoding 在将分类数据转换为线性模型可用格式中的作用，防止算法误解数据的性质。

+   `drop='first'` 参数在 One Hot Encoding 中的重要性，以避免线性模型中的完全共线性。

+   如何评估单个分类特征的预测能力，并在线性模型中对其表现进行排名。

你有任何问题吗？请在下方评论中提问，我会尽力回答。
