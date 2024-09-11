# 在 Python 中更容易进行实验

> 原文：[`machinelearningmastery.com/easier-experimenting-in-python/`](https://machinelearningmastery.com/easier-experimenting-in-python/)

当我们在进行机器学习项目时，经常需要尝试多种替代方案。Python 中的一些特性允许我们尝试不同的选项而不需要太多的努力。在本教程中，我们将看到一些加速实验的技巧。

完成本教程后，您将学到：

+   如何利用鸭子类型特性轻松交换函数和对象

+   如何将组件变成彼此的插拔替换以帮助实验更快地运行

使用我的新书[Python for Machine Learning](https://machinelearningmastery.com/python-for-machine-learning/) **启动您的项目**，包括*逐步教程*和所有示例的*Python 源代码*文件。

让我们开始吧！[](../Images/0975618a0760ede0fe8c1345db6a7e39.png)

在 Python 中更容易进行实验。由[Jake Givens](https://unsplash.com/photos/iR8m2RRo-z4)拍摄。部分权利保留

## 概述

本教程分为三个部分；它们是：

+   机器学习项目的工作流程

+   函数作为对象

+   注意事项

## 机器学习项目的工作流程

考虑一个非常简单的机器学习项目如下：

```py
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

# Train
clf = SVC()
clf.fit(X_train, y_train)

# Test
score = clf.score(X_val, y_val)
print("Validation accuracy", score)
```

这是一个典型的机器学习项目工作流程。我们有数据预处理阶段，然后是模型训练，之后是评估我们的结果。但在每个步骤中，我们可能想尝试一些不同的东西。例如，我们可能会想知道是否归一化数据会使其更好。因此，我们可以将上面的代码重写为以下内容：

```py
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

# Train
clf = Pipeline([('scaler',StandardScaler()), ('classifier',SVC())])
clf.fit(X_train, y_train)

# Test
score = clf.score(X_val, y_val)
print("Validation accuracy", score)
```

到目前为止一切顺利。但是如果我们继续用不同的数据集、不同的模型或不同的评分函数进行实验，每次在使用缩放器和不使用之间进行切换将意味着大量的代码更改，并且很容易出错。

因为 Python 支持鸭子类型，我们可以看到以下两个分类器模型实现了相同的接口：

```py
clf = SVC()
clf = Pipeline([('scaler',StandardScaler()), ('classifier',SVC())])
```

因此，我们可以简单地在这两个版本之间选择并保持一切完整。我们可以说这两个模型是**插拔替换**。

利用此属性，我们可以创建一个切换变量来控制我们所做的设计选择：

```py
USE_SCALER = True

if USE_SCALER:
    clf = Pipeline([('scaler',StandardScaler()), ('classifier',SVC())])
else:
    clf = SVC()
```

通过在`USE_SCALER`变量之间切换`True`和`False`，我们可以选择是否应用缩放器。更复杂的例子是在不同的缩放器和分类器模型之间进行选择，例如：

```py
SCALER = "standard"
CLASSIFIER = "svc"

if CLASSIFIER == "svc":
    model = SVC()
elif CLASSIFIER == "cart":
    model = DecisionTreeClassifier()
else:
    raise NotImplementedError

if SCALER == "standard":
    clf = Pipeline([('scaler',StandardScaler()), ('classifier',model)])
elif SCALER == "maxmin":
    clf = Pipeline([('scaler',MaxMinScaler()), ('classifier',model)])
elif SCALER == None:
    clf = model
else:
    raise NotImplementedError
```

一个完整的示例如下：

```py
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# toggle between options
SCALER = "maxmin"    # "standard", "maxmin", or None
CLASSIFIER = "cart"  # "svc" or "cart"

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

# Create model
if CLASSIFIER == "svc":
    model = SVC()
elif CLASSIFIER == "cart":
    model = DecisionTreeClassifier()
else:
    raise NotImplementedError

if SCALER == "standard":
    clf = Pipeline([('scaler',StandardScaler()), ('classifier',model)])
elif SCALER == "maxmin":
    clf = Pipeline([('scaler',MinMaxScaler()), ('classifier',model)])
elif SCALER == None:
    clf = model
else:
    raise NotImplementedError

# Train
clf.fit(X_train, y_train)

# Test
score = clf.score(X_val, y_val)
print("Validation accuracy", score)
```

如果您进一步走一步，甚至可以跳过切换变量，直接使用字符串进行快速实验：

```py
import numpy as np
import scipy.stats as stats

# Covariance matrix and Cholesky decomposition
cov = np.array([[1, 0.8], [0.8, 1]])
L = np.linalg.cholesky(cov)

# Generate 100 pairs of bi-variate Gaussian random numbers
if not "USE SCIPY":
   z = np.random.randn(100,2)
   x = z @ L.T
else:
   x = stats.multivariate_normal(mean=[0, 0], cov=cov).rvs(100)

...
```

## 函数作为对象

在 Python 中，函数是一等公民。您可以将函数分配给变量。事实上，在 Python 中，函数是对象，类也是（类本身，不仅仅是类的具体实例）。因此，我们可以使用上述相似函数的相同技术进行实验。

```py
import numpy as np

DIST = "normal"

if DIST == "normal":
    rangen = np.random.normal
elif DIST == "uniform":
    rangen = np.random.uniform
else:
    raise NotImplementedError

random_data = rangen(size=(10,5))
print(random_data)
```

以上类似于调用`np.random.normal(size=(10,5))`，但我们将函数保存在变量中，以便于随时替换一个函数。请注意，由于我们使用相同的参数调用函数，我们必须确保所有变体都能接受它。如果不能，我们可能需要一些额外的代码行来创建一个包装器。例如，在生成学生 t 分布的情况下，我们需要一个额外的自由度参数：

```py
import numpy as np

DIST = "t"

if DIST == "normal":
    rangen = np.random.normal
elif DIST == "uniform":
    rangen = np.random.uniform
elif DIST == "t":
    def t_wrapper(size):
        # Student's t distribution with 3 degree of freedom
        return np.random.standard_t(df=3, size=size)
    rangen = t_wrapper
else:
    raise NotImplementedError

random_data = rangen(size=(10,5))
print(random_data)
```

这是因为在上述情况中，`np.random.normal`、`np.random.uniform`和我们定义的`t_wrapper`都可以互换使用。

### 想要开始学习 Python 进行机器学习吗？

现在免费参加我的 7 天电子邮件快速课程（附有示例代码）。

点击注册，还可以免费获得课程的 PDF 电子书版本。

## 注意事项

机器学习与其他编程项目不同，因为工作流程中存在更多的不确定性。当您构建网页或游戏时，您心中有一个目标。但在机器学习项目中，有一些探索性工作。

在其他项目中，您可能会使用像 git 或 Mercurial 这样的源代码控制系统来管理您的源代码开发历史。然而，在机器学习项目中，我们试验许多步骤的不同**组合**。使用 git 管理这些不同的变化可能并不合适，更不用说有时可能会过度。因此，使用切换变量来控制流程应该能让我们更快地尝试不同的方法。当我们在 Jupyter 笔记本上工作时，这特别方便。

然而，当我们将多个版本的代码放在一起时，程序变得笨拙且不易读。确认决策后最好进行一些清理工作。这将有助于我们将来的维护工作。

## 进一步阅读

本节提供更多关于该主题的资源，如果您希望深入了解。

#### 书籍

+   *流畅的 Python*，第二版，作者 Luciano Ramalho，[`www.amazon.com/dp/1492056359/`](https://www.amazon.com/dp/1492056359/)

## 总结

在本教程中，您已经看到 Python 中的鸭子类型属性如何帮助我们创建可互换的替代品。具体而言，您学到了：

+   鸭子类型可以帮助我们在机器学习工作流中轻松切换替代方案。

+   我们可以利用切换变量来在替代方案之间进行实验。
