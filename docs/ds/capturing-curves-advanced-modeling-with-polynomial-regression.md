# 捕捉曲线：使用多项式回归进行高级建模

> 原文：[`machinelearningmastery.com/capturing-curves-advanced-modeling-with-polynomial-regression/`](https://machinelearningmastery.com/capturing-curves-advanced-modeling-with-polynomial-regression/)

当我们在机器学习中分析变量之间的关系时，我们常常发现直线无法全面描述情况。这时，多项式变换发挥作用，它为我们的回归模型添加了层次，同时不会使计算过程复杂化。通过将特征转换为它们的多项式对应物——平方、立方以及其他高阶项——我们赋予线性模型弯曲和扭曲的灵活性，从而更好地适应数据的潜在趋势。

本文将探讨如何超越简单的线性模型，以捕捉数据中的更复杂关系。你将了解多项式回归和立方回归技术的强大功能，这些技术使我们能够看到表面现象之外的潜在模式，这些模式可能会被直线忽略。我们还将深入讨论在模型中添加复杂性与保持预测能力之间的平衡，确保模型既强大又实用。

让我们开始吧。

![](img/e51c144c7f906e5661d9b65dc57e088f.png)

捕捉曲线：使用多项式回归进行高级建模

图片由 [Joakim Aglo](https://unsplash.com/photos/white-concrete-building-low-angle-photography-rr-euqNcCf4) 提供。保留所有权利。

## 概述

本文分为三个部分，它们是：

+   通过线性回归建立基准

+   使用多项式回归捕捉曲线

+   立方回归的实验

## 通过线性回归建立基准

当我们谈论两个变量之间的关系时，线性回归通常是第一步，因为它最简单。它通过将直线拟合到数据上来建模这种关系。这条直线由简单的方程 `y = mx + b` 描述，其中 `y` 是因变量，`x` 是自变量，`m` 是直线的斜率，`b` 是 y 轴截距。让我们通过预测 Ames 数据集中基于整体质量的“SalePrice”来演示这一点，整体质量是一个范围从 1 到 10 的整数值。

```py
# Import the necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Prepare data for linear regression
Ames = pd.read_csv("Ames.csv")
X = Ames[["OverallQual"]]  # Predictor
y = Ames["SalePrice"]      # Response

# Create and fit the linear regression model
linear_model = LinearRegression()
linear_model.fit(X, y)

# Coefficients
intercept = int(linear_model.intercept_)
slope = int(linear_model.coef_[0])
eqn = f"Fitted Line: y = {slope}x - {abs(intercept)}"

# Perform 5-fold cross-validation to evaluate model performance
cv_score = cross_val_score(linear_model, X, y).mean()

# Visualize Best Fit and display CV results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", alpha=0.5, label="Data points")
plt.plot(X, linear_model.predict(X), color="red", label=eqn)
plt.title("Linear Regression of SalePrice vs OverallQual", fontsize=16)
plt.xlabel("Overall Quality", fontsize=12)
plt.ylabel("Sale Price", fontsize=12)
plt.legend(fontsize=14)
plt.grid(True)
plt.text(1, 540000, f"5-Fold CV R²: {cv_score:.3f}", fontsize=14, color="green")
plt.show()
```

![](https://machinelearningmastery.com/?attachment_id=16977)

在基本线性回归中，我们的模型得出了以下方程：`y = 43383x - 84264`。这意味着每增加一个质量点，与销售价格的增加约为 $43,383。为了评估模型的表现，我们使用了 5 折交叉验证，得出的 R² 值为 0.618。这个值表明，通过这个简单的模型，约 61.8% 的销售价格变异性可以通过房屋的整体质量来解释。

线性回归易于理解和实现。然而，它假设自变量和因变量之间的关系是线性的，但这可能并不总是如此，如上图的散点图所示。虽然线性回归提供了一个良好的起点，但现实世界的数据通常需要更复杂的模型来捕捉曲线关系，正如我们将在下一个关于多项式回归的部分中看到的那样。

## 使用多项式回归捕捉曲线

现实世界中的关系通常不是直线而是曲线。多项式回归允许我们建模这些曲线关系。对于三次多项式，这种方法将我们的简单线性方程扩展到每个`x`的幂：`y = ax + bx² + cx³ + d`。我们可以通过使用`sklearn.preprocessing`库中的`PolynomialFeatures`类来实现，它生成一个新的特征矩阵，包括所有小于或等于指定度数的特征的多项式组合。以下是我们如何将其应用于我们的数据集：

```py
# Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Load the data
Ames = pd.read_csv("Ames.csv")
X = Ames[["OverallQual"]]
y = Ames["SalePrice"]

# Transform the predictor variable to polynomial features up to the 3rd degree
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)

# Create and fit the polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Extract model coefficients that form the polynomial equation
#intercept = np.rint(poly_model.intercept_).astype(int)
intercept = int(poly_model.intercept_)
coefs = np.rint(poly_model.coef_).astype(int)
eqn = f"Fitted Line: y = {coefs[0]}x¹ - {abs(coefs[1])}x² + {coefs[2]}x³ - {abs(intercept)}"

# Perform 5-fold cross-validation
cv_score = cross_val_score(poly_model, X_poly, y).mean()

# Generate data to plot curve
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_range_poly = poly.transform(X_range)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", alpha=0.5, label="Data points")
plt.plot(X_range, poly_model.predict(X_range_poly), color="red", label=eqn)
plt.title("Polynomial Regression (3rd Degree) of SalePrice vs OverallQual", fontsize=16)
plt.xlabel("Overall Quality", fontsize=12)
plt.ylabel("Sale Price", fontsize=12)
plt.legend(fontsize=14)
plt.grid(True)
plt.text(1, 540000, f"5-Fold CV R²: {cv_score:.3f}", fontsize=14, color="green")
plt.show()
```

首先，我们将预测变量转换为最高三次的多项式特征。这一增强将我们的特征集从仅有的`x`（整体质量）扩展到`x, x², x³`（即，每个特征变为三个不同但相关的特征），使我们的线性模型能够拟合数据中的更复杂的曲线关系。然后，我们将这些转换后的数据拟合到线性回归模型中，以捕捉整体质量和销售价格之间的非线性关系。

![](https://machinelearningmastery.com/?attachment_id=16978)

我们的新模型的方程为`y = 65966x¹ - 11619x² + 1006x³ - 31343`。该曲线比直线更贴合数据点，表明模型更优。我们的 5 折交叉验证给出了 0.681 的 R²值，相比我们的线性模型有所改进。这表明包括平方项和立方项有助于模型捕捉数据中的更多复杂性。多项式回归引入了拟合曲线的能力，但有时专注于特定的幂，例如立方项，可以揭示更深刻的见解，如我们在立方回归中将深入探讨的那样。

## 实验立方回归

有时，我们可能怀疑`x`的特定幂尤为重要。在这种情况下，我们可以专注于该幂。立方回归是一种特殊情况，我们通过独立变量的立方来建模关系：`y = ax³ + b`。为了有效地专注于这个幂，我们可以利用`sklearn.preprocessing`库中的`FunctionTransformer`类，它允许我们创建自定义转换器，将特定函数应用于数据。这种方法对于隔离和突出高阶项如`x³`对响应变量的影响非常有用，提供了立方项如何单独解释数据变异性的清晰视图。

```py
# Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import FunctionTransformer
import matplotlib.pyplot as plt

# Load data
Ames = pd.read_csv("Ames.csv")
X = Ames[["OverallQual"]]
y = Ames["SalePrice"]

# Function to apply cubic transformation
def cubic_transformation(x):
    return x ** 3

# Apply transformation
cubic_transformer = FunctionTransformer(cubic_transformation)
X_cubic = cubic_transformer.fit_transform(X)

# Fit model
cubic_model = LinearRegression()
cubic_model.fit(X_cubic, y)

# Get coefficients and intercept
intercept_cubic = int(cubic_model.intercept_)
coef_cubic = int(cubic_model.coef_[0])
eqn = f"Fitted Line: y = {coef_cubic}x³ + {intercept_cubic}"

# Cross-validation
cv_score_cubic = cross_val_score(cubic_model, X_cubic, y).mean()

# Generate data to plot curve
X_range = np.linspace(X.min(), X.max(), 300)
X_range_cubic = cubic_transformer.transform(X_range)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", alpha=0.5, label="Data points")
plt.plot(X_range, cubic_model.predict(X_range_cubic), color="red", label=eqn)
plt.title("Cubic Regression of SalePrice vs OverallQual", fontsize=16)
plt.xlabel("Overall Quality", fontsize=12)
plt.ylabel("Sale Price", fontsize=12)
plt.legend(fontsize=14)
plt.grid(True)
plt.text(1, 540000, f"5-Fold CV R²: {cv_score_cubic:.3f}", fontsize=14, color="green")
plt.show()
```

我们对自变量进行了立方变换，得到了立方模型，其方程为`y = 361x³ + 85579`。这比完整的多项式回归模型稍简单，专注于立方项的预测能力。

![](https://machinelearningmastery.com/?attachment_id=16985)

使用立方回归，我们的 5 折交叉验证得到了 0.678 的 R²。这一表现略低于完整的多项式模型，但仍显著优于线性模型。立方回归比更高阶的多项式回归更简单，并且在某些数据集中足以捕捉关系。它比高阶多项式模型更不容易过拟合，但比线性模型更灵活。立方回归模型中的系数 361，表示随着质量**立方**的增加，销售价格的增长率。这强调了非常高质量水平对价格的重大影响，表明具有卓越质量的物业，其销售价格的增长幅度不成比例。这一见解对关注高端物业的投资者或开发商尤其有价值。

正如你可能想象的，这种技术并不限制于多项式回归。如果你认为在特定场景下有意义，你可以引入更复杂的函数，如对数函数和指数函数。

## **进一步阅读**

#### API

+   [sklearn.preprocessing.PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) API

+   [sklearn.preprocessing.FunctionTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html) API

#### 教程

+   [使用 scikit-learn 进行 Python 多项式回归](https://data36.com/polynomial-regression-python-scikit-learn/) 作者：Tamas Ujhelyi

#### **Ames 房价数据集与数据字典**

+   [Ames 数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)

+   [Ames 数据字典](https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt)

## **总结**

本文探讨了适用于不同复杂度数据建模的各种回归技术。我们从**线性回归**开始，以建立预测房价的基线，基于质量评分。配套的视觉图示展示了线性模型如何试图通过数据点拟合一条直线，阐明了回归的基本概念。进一步使用**多项式回归**，我们处理了更复杂的非线性趋势，这提高了模型的灵活性和准确性。配套图表显示了多项式曲线如何比简单的线性模型更紧密地拟合数据点。最后，我们聚焦于**立方回归**，以检验预测变量特定幂次的影响，隔离高阶项对因变量的影响。立方模型被证明特别有效，以足够的精度和简单性捕捉了关系的基本特征。

具体而言，你学习了：

+   如何使用可视化技术识别非线性趋势。

+   如何使用多项式回归技术建模非线性趋势。

+   如何通过较少的模型复杂度来捕捉类似的预测性。

你有什么问题吗？请在下面的评论中提出问题，我会尽力回答。
