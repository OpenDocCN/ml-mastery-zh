# 从训练-测试到交叉验证：提升模型评估

> 原文：[`machinelearningmastery.com/from-train-test-to-cross-validation-advancing-your-models-evaluation/`](https://machinelearningmastery.com/from-train-test-to-cross-validation-advancing-your-models-evaluation/)

许多初学者最初会依赖训练-测试方法来评估他们的模型。这种方法简单明了，似乎能清楚地指示模型在未见数据上的表现。然而，这种方法往往导致对模型能力的不完整理解。在这篇博客中，我们将讨论为什么超越基本的训练-测试分割是重要的，以及交叉验证如何提供对模型性能的更全面评估。加入我们，指导你完成实现对机器学习模型进行更深入、更准确评估的必要步骤。

让我们开始吧。

![](img/1a6dd8d3aeb574d9d77cc3d178743df5.png)

从训练-测试到交叉验证：提升模型评估

图片由[Belinda Fewings](https://unsplash.com/photos/man-in-yellow-polo-shirt-and-black-pants-standing-on-red-plastic-chair-gQELczXc_NA)提供。版权所有。

## 概述

本文分为三个部分；它们是：

+   模型评估：训练-测试 vs. 交叉验证

+   交叉验证的“为什么”

+   深入探讨 K 折交叉验证

## 模型评估：训练-测试 vs. 交叉验证

机器学习模型由其设计（例如线性模型与非线性模型）和其参数（例如线性回归模型中的系数）决定。在考虑如何拟合模型之前，你需要确保模型适合数据。

机器学习模型的性能通过它在以前未见过（或测试）数据上的表现来衡量。在标准的训练-测试分割中，我们将数据集分成两部分：较大的一部分用于训练模型，较小的一部分用于测试其性能。如果测试性能令人满意，则该模型是合适的。这种方法简单直接，但并不总是最有效地利用数据。![](https://machinelearningmastery.com/cross-validation-002/)

然而，使用交叉验证，我们更进一步。第二张图片显示了一个 5 折交叉验证，其中数据集被分成五个“折”。在每一轮验证中，使用一个不同的折作为测试集，其余的作为训练集。这个过程重复五次，确保每个数据点都被用于训练和测试。![](https://machinelearningmastery.com/cross-validation-003/)

下面是一个例子来说明上述内容：

```py
# Load the Ames dataset
import pandas as pd
Ames = pd.read_csv('Ames.csv')

# Import Linear Regression, Train-Test, Cross-Validation from scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score

# Select features and target
X = Ames[['GrLivArea']]  # Feature: GrLivArea, a 2D matrix
y = Ames['SalePrice']    # Target: SalePrice, a 1D vector

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model using Train-Test
model = LinearRegression()
model.fit(X_train, y_train)
train_test_score = round(model.score(X_test, y_test), 4)
print(f"Train-Test R² Score: {train_test_score}")

# Perform 5-Fold Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=5)
cv_scores_rounded = [round(score, 4) for score in cv_scores]
print(f"Cross-Validation R² Scores: {cv_scores_rounded}")
```

虽然训练-测试方法产生一个 R²评分，但交叉验证提供了五个不同的 R²评分，分别来自数据的每一个折，提供了对模型性能的更全面视图：

```py
Train-Test R² Score: 0.4789
Cross-Validation R² Scores: [0.4884, 0.5412, 0.5214, 0.5454, 0.4673]
```

五个折叠中大致相等的 R²分数表明模型稳定。然后，您可以决定该模型（即线性回归）是否提供了可接受的预测能力。

## 交叉验证的“为什么”

理解我们模型在不同数据子集上的表现变异性对机器学习至关重要。虽然训练-测试划分方法有用，但它只提供了我们模型在某一特定未见数据集上的表现快照。

交叉验证通过系统地使用多个数据折叠进行训练和测试，提供了对模型性能的更为稳健和全面的评估。每个折叠作为一个独立的测试，提供了模型在不同数据样本上的预期表现的见解。这种多样性不仅有助于识别潜在的过拟合，还确保性能指标（在本例中为 R²分数）不会过于乐观或悲观，而是更可靠地反映模型对未见数据的泛化能力。

为了直观展示这一点，我们来比较一次训练-测试划分和 5 折交叉验证过程中的 R²分数：

```py
# Import Seaborn and Matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming cv_scores_rounded contains your cross-validation scores
# And train_test_score is your single train-test R² score

# Plot the box plot for cross-validation scores
cv_scores_df = pd.DataFrame(cv_scores_rounded, columns=['Cross-Validation Scores'])
sns.boxplot(data=cv_scores_df, y='Cross-Validation Scores', width=0.3, color='lightblue', fliersize=0)

# Overlay individual scores as points
plt.scatter([0] * len(cv_scores_rounded), cv_scores_rounded, color='blue', label='Cross-Validation Scores')
plt.scatter(0, train_test_score, color='red', zorder=5, label='Train-Test Score')

# Plot the visual 
plt.title('Model Evaluation: Cross-Validation vs. Train-Test')
plt.ylabel('R² Score')
plt.xticks([0], ['Evaluation Scores'])
plt.legend(loc='lower left', bbox_to_anchor=(0, +0.1))
plt.show()
```

这个可视化突出了单次训练-测试评估与交叉验证提供的更广泛视角之间的区别：

![](https://machinelearningmastery.com/?attachment_id=16741)

通过交叉验证，我们对模型的性能有了更深入的理解，使我们更接近于开发既有效又可靠的机器学习解决方案。

## 深入了解 K 折交叉验证

交叉验证是可靠的机器学习模型评估的基石，`cross_val_score()`提供了一种快速且自动化的方式来执行此任务。现在，我们将注意力转向`KFold`类，这是 scikit-learn 的一个组件，提供了对交叉验证折叠的深入了解。`KFold`类不仅提供一个评分，还提供了对模型在不同数据片段上的表现的窗口。我们通过复制上面的示例来演示这一点：

```py
# Import K-Fold and necessary libraries
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Select features and target
X = Ames[['GrLivArea']].values  # Convert to numpy array for KFold
y = Ames['SalePrice'].values    # Convert to numpy array for KFold

# Initialize Linear Regression and K-Fold
model = LinearRegression()
kf = KFold(n_splits=5)

# Manually perform K-Fold Cross-Validation
for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
    # Split the data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the model and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate and print the R² score for the current fold
    print(f"Fold {fold}:")
    print(f"TRAIN set size: {len(train_index)}")
    print(f"TEST set size: {len(test_index)}")
    print(f"R² score: {round(r2_score(y_test, y_pred), 4)}\n")
```

这段代码块将显示每个训练集和测试集的大小及每个折叠的 R²分数：

```py
Fold 1:
TRAIN set size: 2063
TEST set size: 516
R² score: 0.4884

Fold 2:
TRAIN set size: 2063
TEST set size: 516
R² score: 0.5412

Fold 3:
TRAIN set size: 2063
TEST set size: 516
R² score: 0.5214

Fold 4:
TRAIN set size: 2063
TEST set size: 516
R² score: 0.5454

Fold 5:
TRAIN set size: 2064
TEST set size: 515
R² score: 0.4673
```

`KFold`类在交叉验证过程中提供的透明度和控制能力上表现出色。虽然`cross_val_score()`将过程简化为一行代码，但`KFold`将其展开，让我们查看数据的确切划分。这在需要时非常有价值：

+   了解您的数据是如何被划分的。

+   在每个折叠之前实现自定义预处理。

+   获得对模型性能一致性的见解。

通过使用`KFold`类，您可以手动迭代每个划分，并应用模型训练和测试过程。这不仅有助于确保您对每个阶段使用的数据完全了解，还提供了根据复杂需求修改过程的机会。

## **进一步阅读**

#### APIs

+   [sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) API

+   [sklearn.model_selection.cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) API

+   [sklearn.model_selection.KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) API

#### 教程

+   [机器学习中的交叉验证](https://www.geeksforgeeks.org/cross-validation-machine-learning/) by Geeks for Geeks

#### **Ames 房价数据集与数据字典**

+   [Ames 数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)

+   [Ames 数据字典](https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt)

## **摘要**

在这篇文章中，我们探讨了通过交叉验证和`KFold`方法进行全面模型评估的重要性。这两种技术通过保持训练数据和测试数据的独立，细致地避免了数据泄漏的陷阱，从而确保模型性能的准确测量。此外，通过对每个数据点进行一次验证并用其进行 K-1 次训练，这些方法提供了模型泛化能力的详细视角，提高了对其在现实世界应用中的信心。通过实际示例，我们展示了将这些策略整合到评估过程中如何导致更可靠、更强大的机器学习模型，准备好应对新的和未见过的数据的挑战。

具体来说，你学到了：

+   `cross_val_score()`在自动化交叉验证过程中的效率。

+   如何通过`KFold`提供详细的数据拆分控制，以便量身定制的模型评估。

+   两种方法如何确保数据的充分利用并防止数据泄漏。

你有任何问题吗？请在下方评论中提出你的问题，我将尽力回答。
