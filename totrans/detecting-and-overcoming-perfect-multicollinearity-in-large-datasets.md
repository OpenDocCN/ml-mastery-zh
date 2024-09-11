# 在大型数据集中检测和克服完美的多重共线性

> 原文：[`machinelearningmastery.com/detecting-and-overcoming-perfect-multicollinearity-in-large-datasets/`](https://machinelearningmastery.com/detecting-and-overcoming-perfect-multicollinearity-in-large-datasets/)

统计学家和数据科学家面临的一个重大挑战是多重共线性，特别是其最严重的形式——完美的多重共线性。这个问题常常在特征众多的大型数据集中未被察觉，可能伪装自己并扭曲统计模型的结果。

在这篇文章中，我们探讨了检测、解决和优化受完美多重共线性影响的模型的方法。通过实际分析和示例，我们旨在为您提供必要的工具，以增强模型的鲁棒性和可解释性，确保它们提供可靠的见解和准确的预测。

让我们开始吧。

![](img/fcc19ef4088d150c20c8db33e1c5fa32.png)

在大型数据集中检测和克服完美的多重共线性

图片由 [Ryan Stone](https://unsplash.com/photos/red-bridge-during-daytime-sOLbaTbs5mU) 提供。部分权利保留。

## 概述

本文分为三个部分；它们是：

+   探索完美的多重共线性对线性回归模型的影响

+   使用套索回归解决多重共线性问题

+   使用套索回归的见解来优化线性回归模型

## 探索完美的多重共线性对线性回归模型的影响

多重线性回归因其可解释性而特别受到重视。它可以直接理解每个预测变量对响应变量的影响。然而，它的有效性依赖于特征独立的假设。

共线性意味着一个变量可以被表示为其他变量的线性组合。因此，这些变量不是彼此独立的。

线性回归在特征集没有共线性的假设下进行。为了确保这一假设成立，理解线性代数中的一个核心概念——矩阵的秩，是至关重要的。在线性回归中，秩揭示了特征的线性独立性。本质上，没有特征应该是另一个特征的直接线性组合。这种独立性至关重要，因为特征之间的依赖关系——即秩小于特征数量——会导致完美的多重共线性。这种情况可能会扭曲回归模型的可解释性和可靠性，影响其在做出明智决策时的实用性。

让我们使用 Ames Housing 数据集来探讨这个问题。我们将检查数据集的秩和特征数量，以检测多重共线性。

```py
# Import necessary libraries to check and compare number of columns vs rank of dataset
import pandas as pd
import numpy as np

# Load the dataset
Ames = pd.read_csv('Ames.csv')

# Select numerical columns without missing values
numerical_data = Ames.select_dtypes(include=[np.number]).dropna(axis=1)

# Calculate the matrix rank
rank = np.linalg.matrix_rank(numerical_data.values)

# Number of features
num_features = numerical_data.shape[1]

# Print the rank and the number of features
print(f"Numerical features without missing values: {num_features}")
print(f"Rank: {rank}")
```

我们的初步结果显示，Ames Housing 数据集存在多重共线性，具有 27 个特征但只有 26 的秩。

```py
Numerical features without missing values: 27
Rank: 26
```

为了处理这一点，让我们使用定制的函数来识别冗余特征。这种方法有助于做出明智的特征选择或修改决策，从而提升模型的可靠性和可解释性。

```py
# Creating and using a function to identify redundant features in a dataset
import pandas as pd
import numpy as np

def find_redundant_features(data):
    """
    Identifies and returns redundant features in a dataset based on matrix rank.
    A feature is considered redundant if removing it does not decrease the rank of the dataset,
    indicating that it can be expressed as a linear combination of other features.

    Parameters:
        data (DataFrame): The numerical dataset to analyze.

    Returns:
        list: A list of redundant feature names.
    """

    # Calculate the matrix rank of the original dataset
    original_rank = np.linalg.matrix_rank(data)
    redundant_features = []

    for column in data.columns:
        # Create a new dataset without this column
        temp_data = data.drop(column, axis=1)
        # Calculate the rank of the new dataset
        temp_rank = np.linalg.matrix_rank(temp_data)

        # If the rank does not decrease, the removed column is redundant
        if temp_rank == original_rank:
            redundant_features.append(column)

    return redundant_features

# Usage of the function with the numerical data
Ames = pd.read_csv('Ames.csv')
numerical_data = Ames.select_dtypes(include=[np.number]).dropna(axis=1)
redundant_features = find_redundant_features(numerical_data)
print("Redundant features:", redundant_features)
```

以下特征已被识别为冗余，表明它们对模型的预测能力没有独特贡献：

```py
Redundant features: ['GrLivArea', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF']
```

在识别出数据集中的冗余特征后，了解它们冗余的性质至关重要。具体而言，我们怀疑‘GrLivArea’可能只是第一层面积（“1stFlrSF”）、第二层面积（“2ndFlrSF”）和低质量完工平方英尺（“LowQualFinSF”）的总和。为了验证这一点，我们将计算这三个面积的总和，并将其与“GrLivArea”直接比较，以确认它们是否确实相同。

```py
#import pandas
import pandas as pd

# Load the dataset
Ames = pd.read_csv('Ames.csv')

# Calculate the sum of '1stFlrSF', '2ndFlrSF', and 'LowQualFinSF'
Ames['CalculatedGrLivArea'] = Ames['1stFlrSF'] + Ames['2ndFlrSF'] + Ames['LowQualFinSF']

# Compare the calculated sum with the existing 'GrLivArea' column to see if they are the same
Ames['IsEqual'] = Ames['GrLivArea'] == Ames['CalculatedGrLivArea']

# Output the percentage of rows where the values match
match_percentage = Ames['IsEqual'].mean() * 100
print(f"Percentage of rows where GrLivArea equals the sum of the other three features: {int(match_percentage)}%")
```

我们的分析确认，“GrLivArea”在数据集中 100%的情况下正好是“1stFlrSF”、“2ndFlrSF”和“LowQualFinSF”的总和：

```py
Percentage of rows where GrLivArea equals the sum of the other three features: 100%
```

在通过矩阵秩分析确认了“GrLivArea”的冗余性后，我们现在的目标是可视化多重共线性对回归模型稳定性和预测能力的影响。接下来的步骤将涉及使用冗余特征运行多重线性回归，以观察系数估计的方差。这个练习将帮助以一种具体的方式展示多重共线性的实际影响，强化了在模型构建中仔细选择特征的必要性。

```py
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Load the data
Ames = pd.read_csv('Ames.csv')
features = ['GrLivArea', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF']
X = Ames[features]
y = Ames['SalePrice']

# Initialize a K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Collect coefficients and CV scores
coefficients = []
cv_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Initialize and fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    coefficients.append(model.coef_)

    # Calculate R² score using the model's score method
    score = model.score(X_test, y_test)
    # print(score)
    cv_scores.append(score)

# Plotting the coefficients
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.boxplot(np.array(coefficients), labels=features)
plt.title('Box Plot of Coefficients Across Folds (MLR)')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.grid(True)

# Plotting the CV scores
plt.subplot(1, 2, 2)
plt.plot(range(1, 6), cv_scores, marker='o', linestyle='-')  # Adjusted x-axis to start from 1
plt.title('Cross-Validation R² Scores (MLR)')
plt.xlabel('Fold')
plt.xticks(range(1, 6))  # Set x-ticks to match fold numbers
plt.ylabel('R² Score')
plt.ylim(min(cv_scores) - 0.05, max(cv_scores) + 0.05)  # Dynamically adjust y-axis limits
plt.grid(True)

# Annotate mean R² score
mean_r2 = np.mean(cv_scores)
plt.annotate(f'Mean CV R²: {mean_r2:.3f}', xy=(1.25, 0.65), color='red', fontsize=14),

plt.tight_layout()
plt.show()
```

结果可以通过下面的两个图表来演示：

![](https://machinelearningmastery.com/?attachment_id=17325)

左侧的箱线图展示了系数估计的显著方差。这些值的显著分布不仅指出了我们模型的不稳定性，还直接挑战了其可解释性。多重线性回归特别重视其可解释性，这依赖于其系数的稳定性和一致性。当系数在不同的数据子集之间变化很大时，很难得出清晰且可操作的见解，这对于根据模型的预测做出明智决策至关重要。鉴于这些挑战，需要一种更为稳健的方法来解决模型系数的变异性和不稳定性。

## 使用 Lasso 回归处理多重共线性

Lasso 回归作为一种稳健的解决方案出现。与多重线性回归不同，Lasso 可以惩罚系数的大小，并且可以将一些系数设置为零，从而有效地减少模型中的特征数量。这种特征选择在缓解多重共线性方面特别有益。让我们应用 Lasso 到之前的例子中以演示这一点。

```py
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Load the data
Ames = pd.read_csv('Ames.csv')
features = ['GrLivArea', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF']
X = Ames[features]
y = Ames['SalePrice']

# Initialize a K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Prepare to collect results
results = {}

for alpha in [1, 2]:  # Loop through both alpha values
    coefficients = []
    cv_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize and fit the Lasso regression model
        lasso_model = Lasso(alpha=alpha, max_iter=20000)
        lasso_model.fit(X_train_scaled, y_train)
        coefficients.append(lasso_model.coef_)

        # Calculate R² score using the model's score method
        score = lasso_model.score(X_test_scaled, y_test)
        cv_scores.append(score)

    results[alpha] = (coefficients, cv_scores)

# Plotting the results
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
alphas = [1, 2]

for i, alpha in enumerate(alphas):
    coefficients, cv_scores = results[alpha]

    # Plotting the coefficients
    axes[i, 0].boxplot(np.array(coefficients), labels=features)
    axes[i, 0].set_title(f'Box Plot of Coefficients (Lasso with alpha={alpha})')
    axes[i, 0].set_xlabel('Features')
    axes[i, 0].set_ylabel('Coefficient Value')
    axes[i, 0].grid(True)

    # Plotting the CV scores
    axes[i, 1].plot(range(1, 6), cv_scores, marker='o', linestyle='-')
    axes[i, 1].set_title(f'Cross-Validation R² Scores (Lasso with alpha={alpha})')
    axes[i, 1].set_xlabel('Fold')
    axes[i, 1].set_xticks(range(1, 6))
    axes[i, 1].set_ylabel('R² Score')
    axes[i, 1].set_ylim(min(cv_scores) - 0.05, max(cv_scores) + 0.05)
    axes[i, 1].grid(True)
    mean_r2 = np.mean(cv_scores)
    axes[i, 1].annotate(f'Mean CV R²: {mean_r2:.3f}', xy=(1.25, 0.65), color='red', fontsize=12)

plt.tight_layout()
plt.show()
```

通过调整正则化强度（alpha），我们可以观察到增加惩罚如何影响系数和模型的预测准确性：

![](https://machinelearningmastery.com/?attachment_id=17327)

左侧的箱型图显示，随着 alpha 的增加，系数的分布范围和幅度减少，表明估计更加稳定。特别是，当 alpha 设置为 1 时，‘2ndFlrSF’ 的系数开始接近零，并且当 alpha 增加到 2 时几乎为零。这一趋势表明，随着正则化强度的增加，‘2ndFlrSF’ 对模型的贡献最小，这表明它可能在模型中是冗余的或与其他特征存在共线性。这种稳定性直接归因于 Lasso 减少不重要特征影响的能力，这些特征可能会导致多重共线性。

‘2ndFlrSF’ 可以在对模型的预测能力影响最小的情况下被移除，这一点非常重要。它突显了 Lasso 在识别和消除不必要预测因子方面的高效性。值得注意的是，即使这个特征被有效地归零，模型的整体预测能力也保持不变，这展示了 Lasso 在简化模型复杂度的同时维持模型性能的鲁棒性。

## 使用 Lasso 回归洞见优化线性回归模型

根据 Lasso 回归获得的洞见，我们通过移除被识别为对预测能力贡献最小的特征‘2ndFlrSF’来优化我们的模型。本节评估了修订模型的系数性能和稳定性，仅使用‘GrLivArea’，‘1stFlrSF’，和‘LowQualFinSF’。

```py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# Load the data
Ames = pd.read_csv('Ames.csv')
features = ['GrLivArea', '1stFlrSF', 'LowQualFinSF']  # Remove '2ndFlrSF' after running Lasso
X = Ames[features]
y = Ames['SalePrice']

# Initialize a K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Collect coefficients and CV scores
coefficients = []
cv_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Initialize and fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    coefficients.append(model.coef_)

    # Calculate R² score using the model's score method
    score = model.score(X_test, y_test)
    # print(score)
    cv_scores.append(score)

# Plotting the coefficients
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.boxplot(np.array(coefficients), labels=features)
plt.title('Box Plot of Coefficients Across Folds (MLR)')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.grid(True)

# Plotting the CV scores
plt.subplot(1, 2, 2)
plt.plot(range(1, 6), cv_scores, marker='o', linestyle='-')  # Adjusted x-axis to start from 1
plt.title('Cross-Validation R² Scores (MLR)')
plt.xlabel('Fold')
plt.xticks(range(1, 6))  # Set x-ticks to match fold numbers
plt.ylabel('R² Score')
plt.ylim(min(cv_scores) - 0.05, max(cv_scores) + 0.05)  # Dynamically adjust y-axis limits
plt.grid(True)

# Annotate mean R² score
mean_r2 = np.mean(cv_scores)
plt.annotate(f'Mean CV R²: {mean_r2:.3f}', xy=(1.25, 0.65), color='red', fontsize=14),

plt.tight_layout()
plt.show()
```

我们优化后的多元回归模型的结果可以通过下面的两个图示展示：

![](https://machinelearningmastery.com/?attachment_id=17328)

左侧的箱型图展示了系数在不同交叉验证折中的分布情况。与包括“2ndFlrSF”的先前模型相比，系数的方差明显减少。这种方差减少突显了移除冗余特征的有效性，这可以帮助稳定模型的估计并增强其可解释性。每个特征的系数现在表现出更少的波动，这表明模型可以在不同数据子集中一致地评估这些特征的重要性。

除了保持模型的预测能力外，特征复杂性的减少显著提高了模型的可解释性。变量减少后，每个变量对结果的贡献更为明确，我们现在可以更容易地评估这些特定特征对销售价格的影响。这种清晰度允许更直接的解释，并基于模型输出做出更有信心的决策。利益相关者可以更好地理解“GrLivArea”，“1stFlrSF”和“LowQualFinSF”的变化如何影响物业价值，从而促进更清晰的沟通和更具操作性的见解。这种透明度在解释模型预测与预测本身同样重要的领域中尤其宝贵。

## **进一步**阅读

#### API

+   [sklearn.linear_model.Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) API

#### 教程

+   [应用 Lasso 回归自动特征选择](https://developer.ibm.com/tutorials/awb-lasso-regression-automatic-feature-selection/) 作者：Eda Kavlakoglu

+   [使用 Lasso 回归进行机器学习中的特征选择](https://www.yourdatateacher.com/2021/05/05/feature-selection-in-machine-learning-using-lasso-regression/) 作者：Gianluca Malato

#### **艾姆斯房屋数据集与数据字典**

+   [艾姆斯数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)

+   [艾姆斯数据字典](https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt)

## **总结**

这篇博客文章解决了回归模型中的完美多重共线性问题，从使用矩阵秩分析检测开始。随后我们探讨了如何通过减少特征数量、稳定系数估计并保持模型预测能力来缓解多重共线性。最后，通过战略性特征减少来改进线性回归模型，并提高其可解释性和可靠性。

具体来说，你学到了：

+   使用矩阵秩分析检测数据集中的完美多重共线性。

+   应用 Lasso 回归来缓解多重共线性并协助特征选择。

+   使用 Lasso 的见解来改进线性回归模型，以提高可解释性。

有任何问题吗？请在下面的评论中提出问题，我会尽力回答。
