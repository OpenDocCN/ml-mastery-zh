# 在具有数值特征的线性回归中的甜蜜点搜索

> 原文：[`machinelearningmastery.com/the-search-for-the-sweet-spot-in-a-linear-regression-with-numeric-features/`](https://machinelearningmastery.com/the-search-for-the-sweet-spot-in-a-linear-regression-with-numeric-features/)

与奥卡姆剃刀的原则一致，简单的开始往往会导致最深刻的洞见，特别是在构建预测模型时。在这篇文章中，我们将使用 Ames Housing Dataset，首先找出那些独自闪耀的关键特征。然后，逐步将这些洞察层叠起来，观察它们的综合效果如何提升我们的准确预测能力。随着我们深入探讨，我们将利用 Sequential Feature Selector (SFS)来筛选复杂性，突出特征的最佳组合。这种系统的方法将指导我们找到“甜蜜点”——一个和谐的组合，其中选定的特征在不增加不必要数据负担的情况下最大化了模型的预测精度。

让我们开始吧。

![](img/7e42c34ceb3610fcb7dc01ed7efbfea4.png)

在具有数值特征的线性回归中的甜蜜点搜索

图片来源：[Joanna Kosinska](https://unsplash.com/photos/assorted-color-candies-on-container--ayOfwsd9mY)。部分权利保留。

## 概述

本文分为三个部分；它们是：

+   从单一特征到集体影响

+   使用 SFS 深入挖掘：组合的力量

+   寻找预测的“甜蜜点”

## 从个体优势到集体影响

我们的第一步是识别在 Ames 数据集中众多可用特征中，哪些特征在单独使用时表现出强大的预测能力。我们转向简单的线性回归模型，每个模型专注于根据对房价的预测能力识别出的顶级独立特征之一。

```py
# Load the essential libraries and Ames dataset
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import pandas as pd

Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"])
Ames.dropna(axis=1, inplace=True)
X = Ames.drop("SalePrice", axis=1)
y = Ames["SalePrice"]

# Initialize the Linear Regression model
model = LinearRegression()

# Prepare to collect feature scores
feature_scores = {}

# Evaluate each feature with cross-validation
for feature in X.columns:
    X_single = X[[feature]]
    cv_scores = cross_val_score(model, X_single, y)
    feature_scores[feature] = cv_scores.mean()

# Identify the top 5 features based on mean CV R² scores
sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
top_5 = sorted_features[0:5]

# Display the top 5 features and their individual performance
for feature, score in top_5:
    print(f"Feature: {feature}, Mean CV R²: {score:.4f}")
```

这将输出可以单独用于简单线性回归的前 5 个特征：

```py
Feature: OverallQual, Mean CV R²: 0.6183
Feature: GrLivArea, Mean CV R²: 0.5127
Feature: 1stFlrSF, Mean CV R²: 0.3957
Feature: YearBuilt, Mean CV R²: 0.2852
Feature: FullBath, Mean CV R²: 0.2790
```

好奇心驱使我们更深入地思考：如果我们将这些顶级特征组合成一个多重线性回归模型，会发生什么？它们的集体力量是否会超过它们各自的贡献？

```py
# Extracting the top 5 features for our multiple linear regression
top_features = [feature for feature, score in top_5]

# Building the model with the top 5 features
X_top = Ames[top_features]

# Evaluating the model with cross-validation
cv_scores_mlr = cross_val_score(model, X_top, y, cv=5, scoring="r2")
mean_mlr_score = cv_scores_mlr.mean()

print(f"Mean CV R² Score for Multiple Linear Regression Model: {mean_mlr_score:.4f}")
```

初步发现令人鼓舞；每个特征确实都有其优点。然而，当它们结合在一个多重回归模型中时，我们观察到了一种“相当”的改善——这证明了房价预测的复杂性。

```py
Mean CV R² Score for Multiple Linear Regression Model: 0.8003
```

这个结果暗示了未被充分发掘的潜力：是否有更具战略性的方法来选择和组合特征，以提高预测准确性？

## 使用 SFS 深入挖掘：组合的力量

随着我们将 Sequential Feature Selector (SFS)的使用从$n=1$扩展到$n=5$，一个重要的概念浮现出来：组合的力量。让我们通过构建上述代码来说明：

```py
# Perform Sequential Feature Selector with n=5 and build on above code
from sklearn.feature_selection import SequentialFeatureSelector

sfs = SequentialFeatureSelector(model, n_features_to_select=5)
sfs.fit(X, y)

selected_features = X.columns[sfs.get_support()].to_list()
print(f"Features selected by SFS: {selected_features}")

scores = cross_val_score(model, Ames[selected_features], y)
print(f"Mean CV R² Score using SFS with n=5: {scores.mean():.4f}")
```

选择$n=5$不仅仅是选择五个最佳的独立特征。而是识别出五个特征的组合，这些特征组合在一起时，可以优化模型的预测能力：

```py
Features selected by SFS: ['GrLivArea', 'OverallQual', 'YearBuilt', '1stFlrSF', 'KitchenAbvGr']
Mean CV R² Score using SFS with n=5: 0.8056
```

这一结果在与基于单独预测能力选择的前五个特征进行比较时尤为令人启发。属性“FullBath”（未被 SFS 选择）被“SFS 选择中的“KitchenAbvGr”取代。这种差异突显了特征选择的一个基本原则：**关键在于组合**。SFS 不仅仅寻找强大的单一预测变量；它寻求在一起效果最好的特征。这可能意味着选择一个单独看起来不够优秀的特征，但当与其他特征组合时，会提高模型的准确性。

如果你想知道为什么会这样，特征组合中的特征应当是互补的，而不是相关的。这样，每个新特征为预测器提供了新的信息，而不是与已知信息重复。

## 寻找预测的“最佳点”

达到最佳特征选择的过程始于将我们的模型推向极限。通过最初考虑所有可能的特征，我们能够全面了解每添加一个特征后模型性能的变化。这种可视化作为我们的起点，突出了模型预测能力的递减收益，并引导我们找到“最佳点”。让我们通过对整个特征集运行顺序特征选择器（SFS），并绘制性能图来可视化每次添加的影响：

```py
# Performance of SFS from 1 feature to maximum, building on code above:
import matplotlib.pyplot as plt

# Prepare to store the mean CV R² scores for each number of features
mean_scores = []

# Iterate over a range from 1 feature to the maximum number of features available
for n_features_to_select in range(1, len(X.columns)):
    sfs = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select)
    sfs.fit(X, y)
    selected_features = X.columns[sfs.get_support()]
    score = cross_val_score(model, X[selected_features], y, cv=5, scoring="r2").mean()
    mean_scores.append(score)

# Plot the mean CV R² scores against the number of features selected
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(X.columns)), mean_scores, marker="o")
plt.title("Performance vs. Number of Features Selected")
plt.xlabel("Number of Features")
plt.ylabel("Mean CV R² Score")
plt.grid(True)
plt.show()
```

下图展示了随着更多特征的添加，模型性能如何提高，但最终趋于平稳，表明收益递减的点：

![](https://machinelearningmastery.com/?attachment_id=16806)

比较添加特征对预测器的影响

从这张图中可以看到，使用超过十个特征的收益很小。然而，使用三种或更少的特征则是不理想的。你可以使用“肘部法则”来找到曲线弯曲的地方，从而确定最佳特征数量。这是一个主观的决策。该图建议选择 5 到 9 个特征比较合适。

凭借我们初步探索中的见解，我们对特征选择过程应用了一个容差（`tol=0.005`）。这有助于我们客观而稳健地确定最佳特征数量：

```py
# Apply Sequential Feature Selector with tolerance = 0.005, building on code above
sfs_tol = SequentialFeatureSelector(model, n_features_to_select="auto", tol=0.005)
sfs_tol.fit(X, y)

# Get the number of features selected with tolerance
n_features_selected = sum(sfs_tol.get_support())

# Prepare to store the mean CV R² scores for each number of features
mean_scores_tol = []

# Iterate over a range from 1 feature to the Sweet Spot
for n_features_to_select in range(1, n_features_selected + 1):
    sfs = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select)
    sfs.fit(X, y)
    selected_features = X.columns[sfs.get_support()]
    score = cross_val_score(model, X[selected_features], y, cv=5, scoring="r2").mean()
    mean_scores_tol.append(score)

# Plot the mean CV R² scores against the number of features selected
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_features_selected + 1), mean_scores_tol, marker="o")
plt.title("The Sweet Spot: Performance vs. Number of Features Selected")
plt.xlabel("Number of Features")
plt.ylabel("Mean CV R² Score")
plt.grid(True)
plt.show()
```

这一战略举措使我们能够集中于那些提供最高预测性的特征，最终选择出 8 个最佳特征：

![](https://machinelearningmastery.com/?attachment_id=16808)

从图中找到最佳特征数量

我们现在可以通过展示 SFS 选择的特征来总结我们的发现：

```py
# Print the selected features and their performance, building on the above: 
selected_features = X.columns[sfs_tol.get_support()]
print(f"Number of features selected: {n_features_selected}")
print(f"Selected features: {selected_features.tolist()}")
print(f"Mean CV R² Score using SFS with tol=0.005: {mean_scores_tol[-1]:.4f}")
```

```py
Number of features selected: 8
Selected features: ['GrLivArea', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', '1stFlrSF', 'BedroomAbvGr', 'KitchenAbvGr']
Mean CV R² Score using SFS with tol=0.005: 0.8239
```

通过关注这 8 个特征，我们实现了一个在复杂性与高预测性之间取得平衡的模型，展示了特征选择的有效方法。

## **进一步阅读**

#### APIs

+   [sklearn.feature_selection.SequentialFeatureSelector](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html) API

#### 教程

+   [序列特征选择](https://www.youtube.com/watch?v=0vCXcGJg5Bo) 由 Sebastian Raschka 提供

#### **艾姆斯房屋数据集与数据字典**

+   [艾姆斯数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)

+   [艾姆斯数据字典](https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt)

## **总结**

通过这三部分的文章，你已经从评估单个特征的预测能力开始，逐步掌握了在精炼模型中利用它们的综合力量。我们的探索表明，虽然更多的特征可以增强模型捕捉复杂模式的能力，但也会有一个点，额外的特征不再对提高预测有所贡献。通过对序列特征选择器应用容差水平，你已经锁定了一组最佳特征，这些特征在不使预测变得过于复杂的情况下，将我们的模型性能提升至顶峰。这个被识别为八个关键特征的最佳点，体现了预测建模中简单与复杂的战略融合。

具体来说，你学到了：

+   **从简单开始的艺术**：通过从简单的线性回归模型开始，了解每个特征的单独预测价值，为更复杂的分析奠定了基础。

+   **选择的协同效应**：过渡到序列特征选择器突显了不仅要关注单个特征的强度，还要注意它们在有效结合时的协同效应。

+   **最大化模型效能**：通过 SFS 寻找预测的最佳点并设定容差值，让我们认识到特征选择中精确度的重要性，实现以最少的特征获得最大的效果。

有问题吗？请在下面的评论中提出你的问题，我会尽力回答。
