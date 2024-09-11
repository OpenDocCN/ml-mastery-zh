# 解释线性回归模型中的系数

> 原文：[`machinelearningmastery.com/interpreting-coefficients-in-linear-regression-models/`](https://machinelearningmastery.com/interpreting-coefficients-in-linear-regression-models/)

线性回归模型是机器学习中的基础。仅仅拟合一条直线并读取系数就能提供很多信息。但是，我们如何从这些模型中提取和解释系数，以理解它们对预测结果的影响呢？本文将展示如何通过探索各种场景来解释系数。我们将探讨单一数值特征的分析，检查类别变量的作用，并揭示当这些特征组合时引入的复杂性。通过这一探索，我们旨在为您提供有效利用线性回归模型所需的技能，提高您在不同数据驱动领域的分析能力。

![](img/20941486b0c8dd9e41aaa905d541f812.png)

解释线性回归模型中的系数

照片由 [Zac Durant](https://unsplash.com/photos/silhouette-photo-of-man-on-cliff-during-sunset-_6HzPU9Hyfg) 提供。保留所有权利。

让我们开始吧。

## 概述

本文分为三个部分；它们是：

+   使用单一数值特征解释线性模型中的系数

+   使用单一类别特征解释线性模型中的系数

+   数值特征和类别特征的组合讨论

## 使用单一数值特征解释线性模型中的系数

在这一部分，我们专注于 Ames Housing 数据集中一个单一的数值特征，“GrLivArea”（以平方英尺计的地面生活面积），以理解其对“SalePrice”的直接影响。我们使用 K-Fold 交叉验证来验证模型的性能，并提取“GrLivArea”的系数。这个系数估算了在其他所有因素保持不变的情况下，每增加一平方英尺的生活面积，房价预期增加的金额。这是线性回归分析的基本方面，确保“GrLivArea”的影响从其他变量中被隔离出来。

这是我们设置回归模型以实现这一目标的方法：

```py
# Set up to obtain CV model performance and coefficient using K-Fold
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

Ames = pd.read_csv("Ames.csv")
X = Ames[["GrLivArea"]].values  # get 2D matrix
y = Ames["SalePrice"].values    # get 1D vector

model = LinearRegression()
kf = KFold(n_splits=5)
coefs = []
scores = []

# Manually perform K-Fold Cross-Validation
for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
    # Split the data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the model, obtain fold performance and coefficient
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
    coefs.append(model.coef_)

mean_score = np.mean(scores)
print(f"Mean CV R² = {mean_score:.4f}")

mean_coefs = np.mean(coefs)
print(f"Mean Coefficient = {mean_coefs:.4f}")
```

该代码块的输出提供了两个关键的信息：各折的平均 R²得分和“GrLivArea”的平均系数。R²得分给我们一个总体了解，表明我们的模型在不同子集中的拟合程度，指示模型的一致性和可靠性。同时，平均系数量化了“GrLivArea”对“SalePrice”在所有验证折中的平均影响。

```py
Mean CV R² = 0.5127
Mean Coefficient = 110.5214
```

“GrLivArea”的系数可以直接解释为每平方英尺的价格变化。具体而言，它表明“GrLivArea”每增加一平方英尺，房屋的销售价格预计会增加约$110.52（与每平方英尺的价格不同，因为系数指的是**边际价格**）。相反，居住面积减少一平方英尺通常会使销售价格降低相同的金额。

## 线性模型中单一分类特征的系数解释

虽然像“GrLivArea”这样的数值特征可以直接用于我们的回归模型，但分类特征则需要不同的方法。对这些分类变量的正确编码对准确的模型训练至关重要，并确保结果具有可解释性。在这一部分，我们将探讨独热编码——一种将分类变量转换为模型框架内可解释格式的技术，以准备进行线性回归。我们将特别关注如何解释这些转换所产生的系数，包括选择参考类别以简化这些解释的策略。

在应用独热编码时选择合适的参考类别至关重要，因为它设定了与其他类别比较的基准。这个基准类别的平均值通常作为我们回归模型中的截距。让我们探索不同邻里的销售价格分布，以选择一个既能使我们的模型具有可解释性又有意义的参考类别：

```py
# Rank neighborhoods by their mean sale price
Ames = pd.read_csv("Ames.csv")
neighbor_stats = Ames.groupby("Neighborhood")["SalePrice"].agg(["count", "mean"]).sort_values(by="mean")
print(neighbor_stats.round(0).astype(int))
```

这一输出将通过突出显示平均价格最低和最高的邻里，以及指示具有足够数据点（计数）以确保稳健统计分析的邻里，来指导我们的选择：

```py
              count    mean
Neighborhood               
MeadowV          34   96836
BrDale           29  106095
IDOTRR           76  108103
BrkSide         103  126030
OldTown         213  126939
Edwards         165  133152
SWISU            42  133576
Landmrk           1  137000
Sawyer          139  137493
NPkVill          22  140743
Blueste          10  143590
NAmes           410  145087
Mitchel         104  162655
SawyerW         113  188102
Gilbert         143  189440
NWAmes          123  190372
Greens            8  193531
Blmngtn          23  196237
CollgCr         236  198133
Crawfor          92  202076
ClearCr          40  213981
Somerst         143  228762
Timber           54  242910
Veenker          23  251263
GrnHill           2  280000
StoneBr          43  305308
NridgHt         121  313662
NoRidge          67  326114
```

选择像“MeadowV”这样的邻里作为我们的参考设置了明确的基准，从而使得其他邻里的系数解释起来很简单：它们显示了房屋比“MeadowV”贵多少。

通过将“MeadowV”确定为我们的参考邻里，我们现在准备对“Neighborhood”特征应用独热编码，明确排除“MeadowV”以将其作为模型中的基准。这一步确保了所有后续的邻里系数都以“MeadowV”为基准进行解释，从而提供了不同区域房价的清晰对比分析。下一段代码将展示这一编码过程，使用 K 折交叉验证拟合线性回归模型，并计算平均系数和 Y 截距。这些计算将帮助量化每个邻里相对于我们基准的附加值或缺陷，为市场评估提供可操作的见解。

```py
# Build on initial set up and block of code above
# Import OneHotEncoder to preprocess a categorical feature
from sklearn.preprocessing import OneHotEncoder

# One Hot Encoding for "Neighborhood", Note: drop=["MeadowV"]
encoder = OneHotEncoder(sparse=False, drop=["MeadowV"])
X = encoder.fit_transform(Ames[["Neighborhood"]])
y = Ames["SalePrice"].values

# Setup KFold and initialize storage
kf = KFold(n_splits=5)
scores = []
coefficients = []
intercept = []

# Perform the KFold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Append the results for each fold
    scores.append(model.score(X_test, y_test))
    coefficients.append(model.coef_)
    intercept.append(model.intercept_)

mean_score = np.mean(scores)
print(f"Mean CV R² = {mean_score:.4f}")
mean_coefficients = np.mean(coefficients, axis=0)
mean_intercept = np.mean(intercept)
print(f"Mean Y-intercept = {mean_intercept:.0f}")

# Retrieve neighborhood names from the encoder, adjusting for the dropped category
neighborhoods = encoder.categories_[0]
if "MeadowV" in neighborhoods:
    neighborhoods = [name for name in neighborhoods if name != "MeadowV"]

# Create a DataFrame to nicely display neighborhoods with their average coefficients
import pandas as pd

coefficients_df = pd.DataFrame({
    "Neighborhood": neighborhoods,
    "Average Coefficient": mean_coefficients.round(0).astype(int)
})

# Print or return the DataFrame
print(coefficients_df.sort_values(by="Average Coefficient").reset_index(drop=True))
```

不论我们在进行独热编码时“丢弃”了哪个特征，平均 R²值将保持在 0.5408。

Y 截距提供了一个具体的量化基准。作为“MeadowV”中平均销售价格的代表，这个 Y 截距形成了基础价格水平，用于衡量其他邻里的溢价或折扣。

```py
Mean CV R² = 0.5408
Mean Y-intercept = 96827

   Neighborhood  Average Coefficient
0        BrDale                 9221
1        IDOTRR                11335
2       BrkSide                29235
3       OldTown                30092
4       Landmrk                31729
5       Edwards                36305
6         SWISU                36848
7        Sawyer                40645
8       NPkVill                43988
9       Blueste                46388
10        NAmes                48274
11      Mitchel                65851
12      SawyerW                91252
13      Gilbert                92627
14       NWAmes                93521
15       Greens                96641
16      Blmngtn                99318
17      CollgCr               101342
18      Crawfor               105258
19      ClearCr               116993
20      Somerst               131844
21       Timber               146216
22      Veenker               155042
23      GrnHill               183173
24      StoneBr               208096
25      NridgHt               216605
26      NoRidge               229423
```

计算相对于“MeadowV”的每个邻里的系数揭示了其在房价上的溢价或折扣。通过在我们的独热编码过程中将“MeadowV”设为参考类别，其平均销售价格有效地成为了我们模型的截距。然后，为其他邻里计算的系数则测量了相对于“MeadowV”的预期销售价格差异。例如，某个邻里的正系数表明，那里房价比“MeadowV”高出系数的数值，假设其他因素保持不变。这种安排使我们能够直接评估和比较不同邻里对“SalePrice”的影响，提供了对每个邻里相对市场价值的清晰且可量化的理解。

## 讨论数值特征和类别特征的结合

到目前为止，我们已经分别考察了数值特征和类别特征对预测的影响。然而，现实世界中的数据通常需要更复杂的模型来同时处理多种数据类型，以捕捉市场中的复杂关系。为此，熟悉像 `ColumnTransformer` 这样的工具至关重要，它可以同时处理不同的数据类型，确保每个特征都为建模做好最佳准备。接下来，我们将演示一个例子，结合居住面积（“GrLivArea”）和邻里分类，看看这些因素如何共同影响我们的模型性能。

```py
# Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load data
Ames = pd.read_csv("Ames.csv")

# Select features and target
features = Ames[["GrLivArea", "Neighborhood"]]
target = Ames["SalePrice"]

# Preprocess features using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", ["GrLivArea"]),
        ("cat", OneHotEncoder(sparse=False, drop=["MeadowV"], handle_unknown="ignore"), ["Neighborhood"])
    ])

# Fit and transform the features
X_transformed = preprocessor.fit_transform(features)
feature_names = ["GrLivArea"] + list(preprocessor.named_transformers_["cat"].get_feature_names_out())

# Initialize KFold
kf = KFold(n_splits=5)

# Initialize variables to store results
coefficients_list = []
intercepts_list = []
scores = []

# Perform the KFold cross-validation
for train_index, test_index in kf.split(X_transformed):
    X_train, X_test = X_transformed[train_index], X_transformed[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]

    # Initialize the linear regression model
    model = LinearRegression()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Store coefficients and intercepts
    coefficients_list.append(model.coef_)
    intercepts_list.append(model.intercept_)

    # Evaluate the model
    scores.append(model.score(X_test, y_test))

# Calculate the mean of scores, coefficients, and intercepts
average_score = np.mean(scores)
average_coefficients = np.mean(coefficients_list, axis=0)
average_intercept = np.mean(intercepts_list)

# Display the average R² score and Y-Intercept across all folds
# The Y-Intercept represents the baseline sale price in "MeadowV" with no additional living area
print(f"Mean CV R² Score of Combined Model: {average_score:.4f}")
print(f"Mean Y-intercept = {average_intercept:.0f}")

# Create a DataFrame for the coefficients
df_coefficients = pd.DataFrame({
    "Feature": feature_names,
    "Average Coefficient": average_coefficients
    }).sort_values(by="Average Coefficient").reset_index(drop=True)

# Display the DataFrame
print("Coefficients for Combined Model:")
print(df_coefficients)
```

上述代码应输出：

```py
Mean CV R² Score of Combined Model: 0.7375
Mean Y-intercept = 11786

Coefficients for Combined Model:
                 Feature  Average Coefficient
0     Neighborhood_SWISU         -3728.929853
1    Neighborhood_IDOTRR         -1498.971239
2              GrLivArea            78.938757
3   Neighborhood_OldTown          2363.805796
4    Neighborhood_BrDale          6551.114637
5   Neighborhood_BrkSide         16521.117849
6   Neighborhood_Landmrk         16921.529665
7   Neighborhood_Edwards         17520.110407
8   Neighborhood_NPkVill         30034.541748
9     Neighborhood_NAmes         31717.960146
10   Neighborhood_Sawyer         32009.140024
11  Neighborhood_Blueste         39908.310031
12   Neighborhood_NWAmes         44409.237736
13  Neighborhood_Mitchel         48013.229999
14  Neighborhood_SawyerW         48204.606372
15  Neighborhood_Gilbert         49255.248193
16  Neighborhood_Crawfor         55701.500795
17  Neighborhood_ClearCr         61737.497483
18  Neighborhood_CollgCr         69781.161291
19  Neighborhood_Blmngtn         72456.245569
20  Neighborhood_Somerst         90020.562168
21   Neighborhood_Greens         90219.452164
22   Neighborhood_Timber         97021.781128
23  Neighborhood_Veenker         98829.786236
24  Neighborhood_NoRidge        120717.748175
25  Neighborhood_StoneBr        147811.849406
26  Neighborhood_NridgHt        150129.579392
27  Neighborhood_GrnHill        157858.199004
```

将“GrLivArea”和“Neighborhood”结合到一个模型中显著提高了 R² 分数，从单独的 0.5127 和 0.5408 分别上升到 0.7375。这一显著的提升表明，整合多种数据类型能够更准确地反映影响房地产价格的复杂因素。

然而，这种整合给模型引入了新的复杂性。像“GrLivArea”和“Neighborhood”这样的特征之间的交互效应可以显著改变系数。例如，“GrLivArea”在单特征模型中的系数从 110.52 降至组合模型中的 78.93。这一变化表明，居住面积的价值受到不同邻里特征的影响。引入多个变量需要对系数进行调整，以考虑预测变量之间的重叠方差，从而导致系数通常与单特征模型中的不同。

我们综合模型计算得出的平均 Y 截距为 $11,786。这一数值代表了“MeadowV”邻里中一栋基础生活面积（按“GrLivArea”调整为零）的房屋的预测售价。这个截距作为一个基础价格点，增强了我们对不同邻里相对于“MeadowV”的成本比较的解释，一旦调整了生活面积的大小。因此，每个邻里的系数都能告诉我们相对于基准“MeadowV”的额外成本或节省，为我们提供了关于不同区域房产相对价值的清晰且可操作的见解。

## **进一步**阅读

#### API

+   [sklearn.compose.ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) API

#### 教程

+   [解释回归系数](https://www.theanalysisfactor.com/interpreting-regression-coefficients/) 作者：Karen Grace-Martin

#### **Ames Housing 数据集与数据字典**

+   [Ames 数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)

+   [Ames 数据字典](https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt)

## **总结**

本文引导你通过使用 Ames Housing 数据集来解释线性回归模型中的系数，提供了清晰的实际示例。我们探讨了不同类型的特征——数值型和分类型——如何影响模型的预测能力和清晰度。此外，我们还讨论了结合这些特征的挑战和好处，特别是在解释的背景下。

具体来说，你学到了：

+   **单一数值特征的直接影响：** “GrLivArea”系数如何直接量化每增加一平方英尺的“SalePrice”提升，提供了在简单模型中其预测价值的明确度量。

+   **处理分类变量：** One Hot Encoding 在处理诸如“Neighborhood”（邻里）等分类特征时的重要性，说明了选择基准类别如何影响系数的解释，并为不同区域的比较奠定了基础。

+   **结合特征以提升模型性能：** “GrLivArea”和“Neighborhood”的整合不仅提高了预测准确性（R² 分数），还引入了影响每个特征系数解释的复杂性。这部分强调了在实现高预测准确性和保持模型可解释性之间的权衡，这对于在房地产市场中做出明智决策至关重要。

你有任何问题吗？请在下方评论中提出你的问题，我会尽力回答。
