# 管道的力量

> 原文：[`machinelearningmastery.com/the-power-of-pipelines/`](https://machinelearningmastery.com/the-power-of-pipelines/)

机器学习项目通常需要执行一系列的数据预处理步骤，然后是学习算法。单独管理这些步骤可能繁琐且容易出错。这时`sklearn`管道发挥作用。这篇文章将探讨管道如何自动化机器学习工作流的关键方面，如数据预处理、特征工程和机器学习算法的整合。

让我们开始吧。

![](img/6b3185917d2454cd5c46bae0270eab6c.png)

管道的力量

照片由[Quinten de Graaf](https://unsplash.com/photos/metal-pipe-between-trees-at-daytime-L4gN0aeaPY4)拍摄。保留所有权利。

## 概述

这篇文章分为三部分；它们是：

+   什么是管道？

+   通过高级转换提升我们的模型

+   在管道中使用插补处理缺失数据

## 什么是管道？

管道用于自动化和封装各种转换步骤和最终估计器的序列为一个对象。通过定义管道，你确保相同的步骤序列应用于训练数据和测试数据，从而提高模型的可重复性和可靠性。

让我们展示管道的实现，并将其与没有使用管道的传统方法进行比较。考虑一个简单的场景，我们想基于房屋的质量来预测房价，使用‘OverallQual’特征来自 Ames Housing 数据集。这里是一个并排比较，展示使用和不使用管道的 5 折交叉验证：

```py
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Prepare data and setup for linear regression
Ames = pd.read_csv('Ames.csv')
y = Ames['SalePrice']
linear_model = LinearRegression()

# Perform 5-fold cross-validation without Pipeline
cv_score = cross_val_score(linear_model, Ames[['OverallQual']], y).mean()
print("Example Without Pipeline, Mean CV R² score for 'OverallQual': {:.3f}".format(cv_score))

# Perform 5-fold cross-validation WITH Pipeline
pipeline = Pipeline([('regressor', linear_model)])
pipeline_score = cross_val_score(pipeline, Ames[['OverallQual']], y, cv=5).mean()
print("Example With Pipeline, Mean CV R² for 'OverallQual': {:.3f}".format(pipeline_score))
```

两种方法产生完全相同的结果：

```py
Example Without Pipeline, Mean CV R² score for 'OverallQual': 0.618
Example With Pipeline, Mean CV R² for 'OverallQual': 0.618
```

这里有一个可视化图示例，以说明这个基本管道。

![](https://machinelearningmastery.com/the-power-of-pipelines/screenshot-2024-04-29-at-14-13-12/)

这个例子使用了一个简单的案例，仅包含一个特征。然而，随着模型的复杂性增加，管道可以管理多个预处理步骤，如缩放、编码和降维，然后再应用模型。

在我们对 sklearn 管道的基础理解上，让我们扩展我们的场景，包括特征工程——这是提高模型性能的关键步骤。特征工程涉及从现有数据中创建新特征，这些特征可能与目标变量有更强的关系。在我们的案例中，我们怀疑房屋质量与生活面积之间的交互可能比单独的特征更好地预测房价。这里是一个并排比较，展示使用和不使用管道的 5 折交叉验证：

```py
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Prepare data and setup for linear regression
Ames = pd.read_csv('Ames.csv')
y = Ames['SalePrice']
linear_model = LinearRegression()

# Perform 5-fold cross-validation without Pipeline
Ames['OWA'] = Ames['OverallQual'] * Ames['GrLivArea']
cv_score_2 = cross_val_score(linear_model, Ames[['OWA']], y).mean()
print("Example Without Pipeline, Mean CV R² score for 'Quality Weighted Area': {:.3f}".format(cv_score_2))

# WITH Pipeline
# Define the transformation function for 'QualityArea'
def create_quality_area(X):
    X['QualityArea'] = X['OverallQual'] * X['GrLivArea']
    return X[['QualityArea']].values

# Setup the FunctionTransformer using the function
quality_area_transformer = FunctionTransformer(create_quality_area)

# Pipeline using the engineered feature 'QualityArea'
pipeline_2 = Pipeline([
    ('quality_area_transform', quality_area_transformer),
    ('regressor', linear_model)
])
pipeline_score_2 = cross_val_score(pipeline_2, Ames[['OverallQual', 'GrLivArea']], y, cv=5).mean()

# Output the mean CV scores rounded to four decimal places
print("Example With Pipeline, Mean CV R² score for 'Quality Weighted Area': {:.3f}".format(pipeline_score_2))
```

两种方法再次产生相同的结果：

```py
Example Without Pipeline, Mean CV R² score for 'Quality Weighted Area': 0.748
Example With Pipeline, Mean CV R² score for 'Quality Weighted Area': 0.748
```

这一结果表明，通过使用管道，我们将特征工程封装在模型训练过程中，使其成为交叉验证的一个组成部分。使用管道时，每次交叉验证折叠将生成‘Quality Weighted Area’特征，确保我们的特征工程步骤得到正确验证，避免数据泄漏，从而产生更可靠的模型性能估计。

这里有一个可视化图示，展示了我们如何在这个管道的预处理步骤中使用`FunctionTransformer`。

![](https://machinelearningmastery.com/the-power-of-pipelines/screenshot-2024-04-29-at-14-23-54/)

上述管道确保了我们的特征工程和预处理工作准确反映了模型的性能指标。随着我们继续前进，我们将进入更高级的领域，展示在处理各种预处理任务和不同类型的变量时，管道的鲁棒性。

## 利用高级变换提升我们的模型

我们的下一个示例包含了立方变换、特征工程和类别编码，并且包括了未经任何变换的原始特征。这展示了一个管道如何处理各种数据类型和变换，将预处理和建模步骤流畅地整合成一个统一的过程。

```py
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

# Prepare data and setup for linear regression
Ames = pd.read_csv('Ames.csv')
y = Ames['SalePrice']
linear_model = LinearRegression()

# Function to apply cubic transformation
def cubic_transformation(x):
    return x ** 3

# Function to create 'QualityArea'
def create_quality_area(X):
    X['QualityArea'] = X['OverallQual'] * X['GrLivArea']
    return X[['QualityArea']].values

# Setup the FunctionTransformer for cubic and quality area transformations
cubic_transformer = FunctionTransformer(cubic_transformation)
quality_area_transformer = FunctionTransformer(create_quality_area)

# Setup ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cubic', cubic_transformer, ['OverallQual']),
        ('quality_area_transform', quality_area_transformer, ['OverallQual', 'GrLivArea']),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), ['Neighborhood', 'ExterQual', 'KitchenQual']),
        ('passthrough', 'passthrough', ['YearBuilt'])
    ])

# Create the pipeline with the preprocessor and linear regression
pipeline_3 = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', linear_model)
])

# Evaluate the pipeline using 5-fold cross-validation
pipeline_score_3 = cross_val_score(pipeline_3, Ames, y, cv=5).mean()

# Output the mean CV scores rounded to four decimal places
print("Mean CV R² score with enhanced transformations: {:.3f}".format(pipeline_score_3))
```

特征工程是一门艺术，通常需要创造性的触感。通过对‘OverallQual’特征应用立方变换，我们假设质量与价格之间的非线性关系可能会被更好地捕捉。此外，我们还工程化了一个‘QualityArea’特征，我们认为它可能会比单独的特征更显著地与销售价格互动。我们还通过使用独热编码来处理类别特征‘Neighborhood’、‘ExterQual’和‘KitchenQual’，这是准备文本数据用于建模的关键步骤。我们直接将其传递给模型，以确保‘YearBuilt’中的有价值的时间信息不会被不必要地转化。上述管道产生了以下结果：

```py
Mean CV R² score with enhanced transformations: 0.850
```

以令人印象深刻的平均交叉验证 R² 分数 0.850，这个管道展示了深思熟虑的特征工程和预处理对模型性能的显著影响。它突显了管道的效率和可扩展性，并强调了在构建强大的预测模型时其战略重要性。这里有一个可视化图示来说明这个管道。

![](https://machinelearningmastery.com/the-power-of-pipelines/screenshot-2024-04-29-at-14-38-48/)

这种方法的真正优势在于其统一的工作流。通过将特征工程、变换和模型评估优雅地结合到一个连贯的过程中，管道大大提高了我们预测模型的准确性和有效性。这个高级示例强调了一个概念：使用管道时，复杂性不会以牺牲清晰性或性能为代价。

## 在管道中使用插补处理缺失数据

现实中的大多数数据集，尤其是大型数据集，通常包含缺失值。不处理这些缺失值可能导致预测模型中出现显著的偏差或错误。在这一部分，我们将演示如何将数据插补无缝集成到我们的管道中，以确保我们的线性回归模型在面对这些问题时具有鲁棒性。

在之前的帖子中，我们深入探讨了缺失数据，[手动插补缺失值](https://machinelearningmastery.com/classifying_variables/)在 Ames 数据集中而未使用管道。在此基础上，我们现在介绍如何在我们的管道框架中简化和自动化插补，为新手提供更高效且无错误的处理方法。

我们选择使用`SimpleImputer`来处理数据集中‘BsmtQual’（地下室质量）特征的缺失值。`SimpleImputer`将用常数‘None’替代缺失值，以表示没有地下室。插补后，我们使用`OneHotEncoder`将这些分类数据转换为适合线性模型的数值格式。通过将此插补嵌入到我们的管道中，我们确保插补策略在训练和测试阶段都得到正确应用，从而防止数据泄漏，并通过交叉验证维护模型评估的完整性。

下面是我们如何将其集成到管道设置中的：

```py
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load data
Ames = pd.read_csv('Ames.csv')
y = Ames['SalePrice']
linear_model = LinearRegression()

# Function to apply cubic transformation
def cubic_transformation(x):
    return x ** 3

# Function to create 'QualityArea'
def create_quality_area(X):
    X['QualityArea'] = X['OverallQual'] * X['GrLivArea']
    return X[['QualityArea']].values

# Setup the FunctionTransformer for cubic and quality area transformations
cubic_transformer = FunctionTransformer(cubic_transformation)
quality_area_transformer = FunctionTransformer(create_quality_area)

# Prepare the BsmtQual imputation and encoding within a nested pipeline
bsmt_qual_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Setup ColumnTransformer for all preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cubic', cubic_transformer, ['OverallQual']),
        ('quality_area_transform', quality_area_transformer, ['OverallQual', 'GrLivArea']),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), ['Neighborhood', 'ExterQual', 'KitchenQual']),
        ('bsmt_qual', bsmt_qual_transformer, ['BsmtQual']),  # Adding BsmtQual handling
        ('passthrough', 'passthrough', ['YearBuilt'])
    ])

# Create the pipeline with the preprocessor and linear regression
pipeline_4 = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', linear_model)
])

# Evaluate the pipeline using 5-fold cross-validation
pipeline_score = cross_val_score(pipeline_4, Ames, y, cv=5).mean()

# Output the mean CV scores rounded to four decimal places
print("Mean CV R² score with imputing & transformations: {:.3f}".format(pipeline_score))
```

在我们的管道中使用`SimpleImputer`有助于有效处理缺失数据。当它与其他预处理步骤和线性回归模型结合使用时，完整的设置使我们能够评估预处理选择对模型性能的真实影响。

```py
Mean CV R² score with imputing & transformations: 0.856
```

这是一个包含缺失数据插补的管道示意图：

![](https://machinelearningmastery.com/the-power-of-pipelines/screenshot-2024-04-30-at-12-36-48/)

这种集成展示了`sklearn`管道的灵活性，并强调了诸如插补等预处理步骤在机器学习工作流中如何无缝地融入，从而提高模型的可靠性和准确性。

## **进一步阅读**

#### APIs

+   [sklearn.pipeline.Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) API

#### 教程

+   [用 Scikit-learn 管道简化你的机器学习工作流程](https://www.kdnuggets.com/streamline-your-machine-learning-workflow-with-scikit-learn-pipelines) 作者：Abid Ali Awan

#### **Ames 房屋数据集 & 数据字典**

+   [Ames 数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)

+   [Ames 数据字典](https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt)

## **总结**

在这篇文章中，我们探讨了 sklearn 管道的应用，最终实现了在线性回归背景下处理缺失值的数据插补的复杂集成。我们展示了数据预处理步骤、特征工程以及高级转换的无缝自动化，以完善模型的性能。这篇文章中强调的方法不仅关乎保持工作流程的效率，还关乎确保我们希望构建的预测模型的一致性和准确性。

具体来说，你学到了：

+   sklearn 管道的基础概念以及它们如何封装一系列数据转换和最终的估计器。

+   当集成到管道中时，特征工程可以通过创建新的、更具预测性的特征来增强模型性能。

+   在管道中战略性地使用`SimpleImputer`来有效处理缺失数据，防止数据泄漏并提高模型的可靠性。

如果你有任何问题，请在下面的评论中提问，我会尽力回答。
