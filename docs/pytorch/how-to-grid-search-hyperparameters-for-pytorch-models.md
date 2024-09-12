# 如何为 PyTorch 模型进行超参数网格搜索

> 原文：[`machinelearningmastery.com/how-to-grid-search-hyperparameters-for-pytorch-models/`](https://machinelearningmastery.com/how-to-grid-search-hyperparameters-for-pytorch-models/)

神经网络的“权重”在 PyTorch 代码中被称为“参数”，并且在训练过程中由优化器进行微调。相反，超参数是神经网络的参数，设计固定并且不通过训练进行调整。例如隐藏层数量和激活函数的选择。超参数优化是深度学习的重要部分。原因在于神经网络配置非常困难，并且需要设置很多参数。此外，单个模型的训练可能非常缓慢。

在这篇文章中，您将发现如何使用 scikit-learn Python 机器学习库的网格搜索功能来调整 PyTorch 深度学习模型的超参数。阅读完本文后，您将了解到：

+   如何将 PyTorch 模型包装以在 scikit-learn 中使用，以及如何使用网格搜索

+   如何网格搜索常见的神经网络参数，如学习率、退出率、时期和神经元数量

+   如何在自己的项目中定义自己的超参数调整实验

**用我的书 [Deep Learning with PyTorch](https://machinelearningmastery.com/deep-learning-with-pytorch/) **来** 开始你的项目 **。它提供** 自学教程 **和** 可工作的代码 **。

让我们开始吧！![](img/68cf4cc263eb39c28a29db22beb8dee9.png)

如何为 PyTorch 模型进行超参数网格搜索

照片由 [brandon siu](https://unsplash.com/photos/2ePI2R4ka0I) 提供。部分权利保留。

## 概述

在本文中，您将看到如何使用 scikit-learn 的网格搜索功能，提供一系列示例，您可以将其复制粘贴到自己的项目中作为起点。以下是我们将要涵盖的主题列表：

+   如何在 scikit-learn 中使用 PyTorch 模型

+   如何在 scikit-learn 中使用网格搜索

+   如何调整批量大小和训练时期

+   如何调整优化算法

+   如何调整学习率和动量

+   如何调整网络权重初始化

+   如何调整激活函数

+   如何调整退出正则化

+   如何调整隐藏层中神经元的数量

## 如何在 scikit-learn 中使用 PyTorch 模型

如果使用 skorch 包装，PyTorch 模型可以在 scikit-learn 中使用。这是为了利用 Python 的鸭子类型特性，使 PyTorch 模型提供类似于 scikit-learn 模型的 API，以便可以与 scikit-learn 中的所有内容一起使用。在 skorch 中，有 `NeuralNetClassifier` 用于分类神经网络和 `NeuralNetRegressor` 用于回归神经网络。您可能需要运行以下命令来安装模块。

```py
pip install skorch
```

要使用这些包装器，你必须将你的 PyTorch 模型定义为使用 `nn.Module` 的一个类，然后在构造 `NeuralNetClassifier` 类时将类的名称传递给 `module` 参数。例如：

```py
class MyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        ...

    def forward(self, x):
        ...
        return x

# create the skorch wrapper
model = NeuralNetClassifier(
    module=MyClassifier
)
```

`NeuralNetClassifier` 类的构造函数可以接受默认参数，这些参数会传递给 `model.fit()`（这是在 scikit-learn 模型中调用训练循环的方式），例如训练轮数和批次大小。例如：

```py
model = NeuralNetClassifier(
    module=MyClassifier,
    max_epochs=150,
    batch_size=10
)
```

`NeuralNetClassifier` 类的构造函数还可以接受新的参数，这些参数可以传递给你的模型类的构造函数，但你必须在参数前加上 `module__`（两个下划线）。这些新参数可能在构造函数中有默认值，但当包装器实例化模型时，它们会被覆盖。例如：

```py
import torch.nn as nn
from skorch import NeuralNetClassifier

class SonarClassifier(nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.layers = []
        self.acts = []
        for i in range(n_layers):
            self.layers.append(nn.Linear(60, 60))
            self.acts.append(nn.ReLU())
            self.add_module(f"layer{i}", self.layers[-1])
            self.add_module(f"act{i}", self.acts[-1])
        self.output = nn.Linear(60, 1)

    def forward(self, x):
        for layer, act in zip(self.layers, self.acts):
            x = act(layer(x))
        x = self.output(x)
        return x

model = NeuralNetClassifier(
    module=SonarClassifier,
    max_epochs=150,
    batch_size=10,
    module__n_layers=2
)
```

你可以通过初始化模型并打印它来验证结果：

```py
print(model.initialize())
```

在这个示例中，你应该能看到：

```py
<class 'skorch.classifier.NeuralNetClassifier'>initialized: Linear(in_features=60, out_features=60, bias=True)
    (act0): ReLU()
    (layer1): Linear(in_features=60, out_features=60, bias=True)
    (act1): ReLU()
    (output): Linear(in_features=60, out_features=1, bias=True)
  ),
)
```

### 想要开始使用 PyTorch 进行深度学习吗？

现在就参加我的免费电子邮件速成课程（附带示例代码）。

点击注册并免费获得课程的 PDF Ebook 版本。

## 如何在 scikit-learn 中使用网格搜索

网格搜索是一种模型超参数优化技术。它通过穷举所有超参数的组合，找到给出最佳分数的组合。在 scikit-learn 中，这种技术由 `GridSearchCV` 类提供。在构造这个类时，你必须在 `param_grid` 参数中提供一个超参数字典。这是模型参数名称与要尝试的值数组的映射。

默认情况下，准确率是优化的评分指标，但你可以在 `GridSearchCV` 构造函数的 score 参数中指定其他评分指标。`GridSearchCV` 过程将为每个参数组合构建和评估一个模型。交叉验证用于评估每个单独的模型，默认使用的是 3 折交叉验证，虽然你可以通过将 cv 参数指定给 `GridSearchCV` 构造函数来覆盖这一点。

下面是一个定义简单网格搜索的示例：

```py
param_grid = {
    'epochs': [10,20,30]
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
```

通过将 `GridSearchCV` 构造函数中的 `n_jobs` 参数设置为 $-1$，该过程将使用你机器上的所有核心。否则，网格搜索过程将仅在单线程中运行，这在多核 CPU 上较慢。

完成后，你可以在 `grid.fit()` 返回的结果对象中访问网格搜索的结果。`best_score_` 成员提供了在优化过程中观察到的最佳分数，而 `best_params_` 描述了获得最佳结果的参数组合。你可以在 scikit-learn API 文档中了解更多关于 `GridSearchCV` 类的信息。

**快速启动你的项目**，可以参考我的书籍 [Deep Learning with PyTorch](https://machinelearningmastery.com/deep-learning-with-pytorch/)。它提供了**自学教程**和**可运行的代码**。

## 问题描述

现在你已经知道如何将 PyTorch 模型与 scikit-learn 配合使用以及如何在 scikit-learn 中使用网格搜索，让我们看一些示例。

所有示例将在一个小型标准机器学习数据集上演示，名为[Pima Indians 糖尿病发作分类数据集](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv)。这是一个包含所有数值属性的小数据集，易于处理。

在本帖中的示例中，你将汇总最佳参数。这不是网格搜索的最佳方式，因为参数可能相互作用，但它适用于演示目的。

## 如何调整批量大小和轮数

在第一个简单的示例中，你将查看调整批量大小和训练网络时使用的轮数。

在迭代梯度下降中，批量大小是指在权重更新之前展示给网络的样本数。它也是网络训练中的一种优化，定义了每次读取多少样本并保持在内存中。

轮数是指整个训练数据集在训练过程中被展示给网络的次数。一些网络对批量大小比较敏感，比如 LSTM 递归神经网络和卷积神经网络。

在这里，你将评估从 10 到 100 的不同小批量大小，每次递增 20。

完整的代码列表如下：

```py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# PyTorch classifier
class PimaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(8, 12)
        self.act = nn.ReLU()
        self.output = nn.Linear(12, 1)
        self.prob = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.layer(x))
        x = self.prob(self.output(x))
        return x

# create model with skorch
model = NeuralNetClassifier(
    PimaClassifier,
    criterion=nn.BCELoss,
    optimizer=optim.Adam,
    verbose=False
)

# define the grid search parameters
param_grid = {
    'batch_size': [10, 20, 40, 60, 80, 100],
    'max_epochs': [10, 50, 100]
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

运行此示例将产生以下输出：

```py
Best: 0.714844 using {'batch_size': 10, 'max_epochs': 100}
0.665365 (0.020505) with: {'batch_size': 10, 'max_epochs': 10}
0.588542 (0.168055) with: {'batch_size': 10, 'max_epochs': 50}
0.714844 (0.032369) with: {'batch_size': 10, 'max_epochs': 100}
0.671875 (0.022326) with: {'batch_size': 20, 'max_epochs': 10}
0.696615 (0.008027) with: {'batch_size': 20, 'max_epochs': 50}
0.714844 (0.019918) with: {'batch_size': 20, 'max_epochs': 100}
0.666667 (0.009744) with: {'batch_size': 40, 'max_epochs': 10}
0.687500 (0.033603) with: {'batch_size': 40, 'max_epochs': 50}
0.707031 (0.024910) with: {'batch_size': 40, 'max_epochs': 100}
0.667969 (0.014616) with: {'batch_size': 60, 'max_epochs': 10}
0.694010 (0.036966) with: {'batch_size': 60, 'max_epochs': 50}
0.694010 (0.042473) with: {'batch_size': 60, 'max_epochs': 100}
0.670573 (0.023939) with: {'batch_size': 80, 'max_epochs': 10}
0.674479 (0.020752) with: {'batch_size': 80, 'max_epochs': 50}
0.703125 (0.026107) with: {'batch_size': 80, 'max_epochs': 100}
0.680990 (0.014382) with: {'batch_size': 100, 'max_epochs': 10}
0.670573 (0.013279) with: {'batch_size': 100, 'max_epochs': 50}
0.687500 (0.017758) with: {'batch_size': 100, 'max_epochs': 100}
```

你可以看到，批量大小为 10 和 100 轮数取得了约 71%准确率的最佳结果（但你还应该考虑准确率的标准差）。

## 如何调整训练优化算法

所有深度学习库应提供多种优化算法。PyTorch 也不例外。

在此示例中，你将调整用于训练网络的优化算法，每种算法都使用默认参数。

这是一个奇特的示例，因为通常你会先选择一种方法，然后专注于调整其在问题上的参数（见下一个示例）。

在这里，你将评估 PyTorch 中可用的[优化算法套件](https://pytorch.org/docs/stable/optim.html)。

完整的代码列表如下：

```py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# PyTorch classifier
class PimaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(8, 12)
        self.act = nn.ReLU()
        self.output = nn.Linear(12, 1)
        self.prob = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.layer(x))
        x = self.prob(self.output(x))
        return x

# create model with skorch
model = NeuralNetClassifier(
    PimaClassifier,
    criterion=nn.BCELoss,
    max_epochs=100,
    batch_size=10,
    verbose=False
)

# define the grid search parameters
param_grid = {
    'optimizer': [optim.SGD, optim.RMSprop, optim.Adagrad, optim.Adadelta,
                  optim.Adam, optim.Adamax, optim.NAdam],
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

运行此示例将产生以下输出：

```py
Best: 0.721354 using {'optimizer': <class 'torch.optim.adamax.Adamax'>}
0.674479 (0.036828) with: {'optimizer': <class 'torch.optim.sgd.SGD'>}
0.700521 (0.043303) with: {'optimizer': <class 'torch.optim.rmsprop.RMSprop'>}
0.682292 (0.027126) with: {'optimizer': <class 'torch.optim.adagrad.Adagrad'>}
0.572917 (0.051560) with: {'optimizer': <class 'torch.optim.adadelta.Adadelta'>}
0.714844 (0.030758) with: {'optimizer': <class 'torch.optim.adam.Adam'>}
0.721354 (0.019225) with: {'optimizer': <class 'torch.optim.adamax.Adamax'>}
0.709635 (0.024360) with: {'optimizer': <class 'torch.optim.nadam.NAdam'>}
```

结果表明，Adamax 优化算法表现最佳，准确率约为 72%。

值得一提的是，`GridSearchCV`会经常重新创建你的模型，因此每次试验都是独立的。之所以能够做到这一点，是因为`NeuralNetClassifier`封装器知道你 PyTorch 模型的类名，并在请求时为你实例化一个。

## 如何调整学习率和动量

通常，预先选择一种优化算法来训练网络并调整其参数是很常见的。

迄今为止，最常见的优化算法是传统的随机梯度下降（SGD），因为它被广泛理解。在这个示例中，你将优化 SGD 学习率和动量参数。

学习率控制每个批次结束时更新权重的幅度，动量控制前一次更新对当前权重更新的影响程度。

你将尝试一系列小的标准学习率和动量值，从 0.2 到 0.8，步长为 0.2，以及 0.9（因为它在实际中可能是一个流行的值）。在 PyTorch 中，设置学习率和动量的方法如下：

```py
optimizer = optim.SGD(lr=0.001, momentum=0.9)
```

在 skorch 包装器中，你可以使用前缀`optimizer__`将参数路由到优化器。

通常，将优化中的纪元数也包含在内是一个好主意，因为每批次的学习量（学习率）、每纪元的更新次数（批量大小）和纪元数之间存在依赖关系。

完整的代码清单如下：

```py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# PyTorch classifier
class PimaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(8, 12)
        self.act = nn.ReLU()
        self.output = nn.Linear(12, 1)
        self.prob = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.layer(x))
        x = self.prob(self.output(x))
        return x

# create model with skorch
model = NeuralNetClassifier(
    PimaClassifier,
    criterion=nn.BCELoss,
    optimizer=optim.SGD,
    max_epochs=100,
    batch_size=10,
    verbose=False
)

# define the grid search parameters
param_grid = {
    'optimizer__lr': [0.001, 0.01, 0.1, 0.2, 0.3],
    'optimizer__momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

运行此示例将产生以下输出。

```py
Best: 0.682292 using {'optimizer__lr': 0.001, 'optimizer__momentum': 0.9}
0.648438 (0.016877) with: {'optimizer__lr': 0.001, 'optimizer__momentum': 0.0}
0.671875 (0.017758) with: {'optimizer__lr': 0.001, 'optimizer__momentum': 0.2}
0.674479 (0.022402) with: {'optimizer__lr': 0.001, 'optimizer__momentum': 0.4}
0.677083 (0.011201) with: {'optimizer__lr': 0.001, 'optimizer__momentum': 0.6}
0.679688 (0.027621) with: {'optimizer__lr': 0.001, 'optimizer__momentum': 0.8}
0.682292 (0.026557) with: {'optimizer__lr': 0.001, 'optimizer__momentum': 0.9}
0.671875 (0.019918) with: {'optimizer__lr': 0.01, 'optimizer__momentum': 0.0}
0.648438 (0.024910) with: {'optimizer__lr': 0.01, 'optimizer__momentum': 0.2}
0.546875 (0.143454) with: {'optimizer__lr': 0.01, 'optimizer__momentum': 0.4}
0.567708 (0.153668) with: {'optimizer__lr': 0.01, 'optimizer__momentum': 0.6}
0.552083 (0.141790) with: {'optimizer__lr': 0.01, 'optimizer__momentum': 0.8}
0.451823 (0.144561) with: {'optimizer__lr': 0.01, 'optimizer__momentum': 0.9}
0.348958 (0.001841) with: {'optimizer__lr': 0.1, 'optimizer__momentum': 0.0}
0.450521 (0.142719) with: {'optimizer__lr': 0.1, 'optimizer__momentum': 0.2}
0.450521 (0.142719) with: {'optimizer__lr': 0.1, 'optimizer__momentum': 0.4}
0.450521 (0.142719) with: {'optimizer__lr': 0.1, 'optimizer__momentum': 0.6}
0.348958 (0.001841) with: {'optimizer__lr': 0.1, 'optimizer__momentum': 0.8}
0.348958 (0.001841) with: {'optimizer__lr': 0.1, 'optimizer__momentum': 0.9}
0.444010 (0.136265) with: {'optimizer__lr': 0.2, 'optimizer__momentum': 0.0}
0.450521 (0.142719) with: {'optimizer__lr': 0.2, 'optimizer__momentum': 0.2}
0.348958 (0.001841) with: {'optimizer__lr': 0.2, 'optimizer__momentum': 0.4}
0.552083 (0.141790) with: {'optimizer__lr': 0.2, 'optimizer__momentum': 0.6}
0.549479 (0.142719) with: {'optimizer__lr': 0.2, 'optimizer__momentum': 0.8}
0.651042 (0.001841) with: {'optimizer__lr': 0.2, 'optimizer__momentum': 0.9}
0.552083 (0.141790) with: {'optimizer__lr': 0.3, 'optimizer__momentum': 0.0}
0.348958 (0.001841) with: {'optimizer__lr': 0.3, 'optimizer__momentum': 0.2}
0.450521 (0.142719) with: {'optimizer__lr': 0.3, 'optimizer__momentum': 0.4}
0.552083 (0.141790) with: {'optimizer__lr': 0.3, 'optimizer__momentum': 0.6}
0.450521 (0.142719) with: {'optimizer__lr': 0.3, 'optimizer__momentum': 0.8}
0.450521 (0.142719) with: {'optimizer__lr': 0.3, 'optimizer__momentum': 0.9}
```

你可以看到，使用 SGD 时，最佳结果是学习率为 0.001 和动量为 0.9，准确率约为 68%。

## 如何调整网络权重初始化

神经网络权重初始化曾经很简单：使用小的随机值。

现在有一套不同的技术可供选择。你可以从[`torch.nn.init`](https://pytorch.org/docs/stable/nn.init.html)文档中获得一个[备选列表]。

在这个例子中，你将通过评估所有可用技术来调整网络权重初始化的选择。

你将在每一层上使用相同的权重初始化方法。理想情况下，根据每层使用的激活函数，使用不同的权重初始化方案可能更好。在下面的示例中，你将在隐藏层使用整流函数。由于预测是二元的，因此在输出层使用 sigmoid。权重初始化在 PyTorch 模型中是隐式的。因此，你需要在层创建后但在使用前编写自己的逻辑来初始化权重。让我们按如下方式修改 PyTorch：

```py
# PyTorch classifier
class PimaClassifier(nn.Module):
    def __init__(self, weight_init=torch.nn.init.xavier_uniform_):
        super().__init__()
        self.layer = nn.Linear(8, 12)
        self.act = nn.ReLU()
        self.output = nn.Linear(12, 1)
        self.prob = nn.Sigmoid()
        # manually init weights
        weight_init(self.layer.weight)
        weight_init(self.output.weight)

    def forward(self, x):
        x = self.act(self.layer(x))
        x = self.prob(self.output(x))
        return x
```

向`PimaClassifier`类添加了一个参数`weight_init`，它期望来自`torch.nn.init`的一个初始化器。在`GridSearchCV`中，你需要使用`module__`前缀来使`NeuralNetClassifier`将参数路由到模型类构造函数。

完整的代码清单如下：

```py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# PyTorch classifier
class PimaClassifier(nn.Module):
    def __init__(self, weight_init=init.xavier_uniform_):
        super().__init__()
        self.layer = nn.Linear(8, 12)
        self.act = nn.ReLU()
        self.output = nn.Linear(12, 1)
        self.prob = nn.Sigmoid()
        # manually init weights
        weight_init(self.layer.weight)
        weight_init(self.output.weight)

    def forward(self, x):
        x = self.act(self.layer(x))
        x = self.prob(self.output(x))
        return x

# create model with skorch
model = NeuralNetClassifier(
    PimaClassifier,
    criterion=nn.BCELoss,
    optimizer=optim.Adamax,
    max_epochs=100,
    batch_size=10,
    verbose=False
)

# define the grid search parameters
param_grid = {
    'module__weight_init': [init.uniform_, init.normal_, init.zeros_,
                           init.xavier_normal_, init.xavier_uniform_,
                           init.kaiming_normal_, init.kaiming_uniform_]
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

运行此示例将产生以下输出。

```py
Best: 0.697917 using {'module__weight_init': <function kaiming_uniform_ at 0x112020c10>}
0.348958 (0.001841) with: {'module__weight_init': <function uniform_ at 0x1120204c0>}
0.602865 (0.061708) with: {'module__weight_init': <function normal_ at 0x112020550>}
0.652344 (0.003189) with: {'module__weight_init': <function zeros_ at 0x112020820>}
0.691406 (0.030758) with: {'module__weight_init': <function xavier_normal_ at 0x112020af0>}
0.592448 (0.171589) with: {'module__weight_init': <function xavier_uniform_ at 0x112020a60>}
0.563802 (0.152971) with: {'module__weight_init': <function kaiming_normal_ at 0x112020ca0>}
0.697917 (0.013279) with: {'module__weight_init': <function kaiming_uniform_ at 0x112020c10>}
```

最佳结果是通过 He-uniform 权重初始化方案实现的，性能达到约 70%。

## 如何调整神经元激活函数

激活函数控制单个神经元的非线性及其触发时机。

一般来说，整流线性单元（ReLU）激活函数是最受欢迎的。然而，过去使用过的是 sigmoid 和 tanh 函数，这些函数可能在不同的问题上仍然更为适用。

在这个示例中，你将评估 PyTorch 中一些可用的激活函数。你只会在隐藏层中使用这些函数，因为在输出层中需要一个 sigmoid 激活函数用于二分类问题。类似于之前的示例，这个是模型类构造函数的一个参数，你将使用 `module__` 前缀来设置 `GridSearchCV` 参数网格。

一般来说，将数据准备到不同传递函数的范围是一个好主意，但在这个案例中你不会这样做。

完整的代码清单如下所示：

```py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# PyTorch classifier
class PimaClassifier(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super().__init__()
        self.layer = nn.Linear(8, 12)
        self.act = activation()
        self.output = nn.Linear(12, 1)
        self.prob = nn.Sigmoid()
        # manually init weights
        init.kaiming_uniform_(self.layer.weight)
        init.kaiming_uniform_(self.output.weight)

    def forward(self, x):
        x = self.act(self.layer(x))
        x = self.prob(self.output(x))
        return x

# create model with skorch
model = NeuralNetClassifier(
    PimaClassifier,
    criterion=nn.BCELoss,
    optimizer=optim.Adamax,
    max_epochs=100,
    batch_size=10,
    verbose=False
)

# define the grid search parameters
param_grid = {
    'module__activation': [nn.Identity, nn.ReLU, nn.ELU, nn.ReLU6,
                           nn.GELU, nn.Softplus, nn.Softsign, nn.Tanh,
                           nn.Sigmoid, nn.Hardsigmoid]
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

运行此示例将产生以下输出。

```py
Best: 0.699219 using {'module__activation': <class 'torch.nn.modules.activation.ReLU'>}
0.687500 (0.025315) with: {'module__activation': <class 'torch.nn.modules.linear.Identity'>}
0.699219 (0.011049) with: {'module__activation': <class 'torch.nn.modules.activation.ReLU'>}
0.674479 (0.035849) with: {'module__activation': <class 'torch.nn.modules.activation.ELU'>}
0.621094 (0.063549) with: {'module__activation': <class 'torch.nn.modules.activation.ReLU6'>}
0.674479 (0.017566) with: {'module__activation': <class 'torch.nn.modules.activation.GELU'>}
0.558594 (0.149189) with: {'module__activation': <class 'torch.nn.modules.activation.Softplus'>}
0.675781 (0.014616) with: {'module__activation': <class 'torch.nn.modules.activation.Softsign'>}
0.619792 (0.018688) with: {'module__activation': <class 'torch.nn.modules.activation.Tanh'>}
0.643229 (0.019225) with: {'module__activation': <class 'torch.nn.modules.activation.Sigmoid'>}
0.636719 (0.022326) with: {'module__activation': <class 'torch.nn.modules.activation.Hardsigmoid'>}
```

它显示 ReLU 激活函数在准确率约为 70% 时取得了最佳结果。

## 如何调整 dropout 正则化

在这个示例中，你将调整 [dropout 率以进行正则化](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/) 以限制过拟合并提高模型的泛化能力。

为了获得最佳结果，dropout 最好与权重约束（例如最大范数约束）结合使用，后者在前向传播函数中实现。

这涉及到拟合 dropout 百分比和权重约束。我们将尝试 0.0 到 0.9 之间的 dropout 百分比（1.0 不合适）以及 0 到 5 之间的 MaxNorm 权重约束值。

完整的代码清单如下所示。

```py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# PyTorch classifier
class PimaClassifier(nn.Module):
    def __init__(self, dropout_rate=0.5, weight_constraint=1.0):
        super().__init__()
        self.layer = nn.Linear(8, 12)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(12, 1)
        self.prob = nn.Sigmoid()
        self.weight_constraint = weight_constraint
        # manually init weights
        init.kaiming_uniform_(self.layer.weight)
        init.kaiming_uniform_(self.output.weight)

    def forward(self, x):
        # maxnorm weight before actual forward pass
        with torch.no_grad():
            norm = self.layer.weight.norm(2, dim=0, keepdim=True).clamp(min=self.weight_constraint / 2)
            desired = torch.clamp(norm, max=self.weight_constraint)
            self.layer.weight *= (desired / norm)
        # actual forward pass
        x = self.act(self.layer(x))
        x = self.dropout(x)
        x = self.prob(self.output(x))
        return x

# create model with skorch
model = NeuralNetClassifier(
    PimaClassifier,
    criterion=nn.BCELoss,
    optimizer=optim.Adamax,
    max_epochs=100,
    batch_size=10,
    verbose=False
)

# define the grid search parameters
param_grid = {
    'module__weight_constraint': [1.0, 2.0, 3.0, 4.0, 5.0],
    'module__dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

运行此示例将产生以下输出。

```py
Best: 0.701823 using {'module__dropout_rate': 0.1, 'module__weight_constraint': 2.0}
0.669271 (0.015073) with: {'module__dropout_rate': 0.0, 'module__weight_constraint': 1.0}
0.692708 (0.035132) with: {'module__dropout_rate': 0.0, 'module__weight_constraint': 2.0}
0.589844 (0.170180) with: {'module__dropout_rate': 0.0, 'module__weight_constraint': 3.0}
0.561198 (0.151131) with: {'module__dropout_rate': 0.0, 'module__weight_constraint': 4.0}
0.688802 (0.021710) with: {'module__dropout_rate': 0.0, 'module__weight_constraint': 5.0}
0.697917 (0.009744) with: {'module__dropout_rate': 0.1, 'module__weight_constraint': 1.0}
0.701823 (0.016367) with: {'module__dropout_rate': 0.1, 'module__weight_constraint': 2.0}
0.694010 (0.010253) with: {'module__dropout_rate': 0.1, 'module__weight_constraint': 3.0}
0.686198 (0.025976) with: {'module__dropout_rate': 0.1, 'module__weight_constraint': 4.0}
0.679688 (0.026107) with: {'module__dropout_rate': 0.1, 'module__weight_constraint': 5.0}
0.701823 (0.029635) with: {'module__dropout_rate': 0.2, 'module__weight_constraint': 1.0}
0.682292 (0.014731) with: {'module__dropout_rate': 0.2, 'module__weight_constraint': 2.0}
0.701823 (0.009744) with: {'module__dropout_rate': 0.2, 'module__weight_constraint': 3.0}
0.701823 (0.026557) with: {'module__dropout_rate': 0.2, 'module__weight_constraint': 4.0}
0.687500 (0.015947) with: {'module__dropout_rate': 0.2, 'module__weight_constraint': 5.0}
0.686198 (0.006639) with: {'module__dropout_rate': 0.3, 'module__weight_constraint': 1.0}
0.656250 (0.006379) with: {'module__dropout_rate': 0.3, 'module__weight_constraint': 2.0}
0.565104 (0.155608) with: {'module__dropout_rate': 0.3, 'module__weight_constraint': 3.0}
0.700521 (0.028940) with: {'module__dropout_rate': 0.3, 'module__weight_constraint': 4.0}
0.669271 (0.012890) with: {'module__dropout_rate': 0.3, 'module__weight_constraint': 5.0}
0.661458 (0.018688) with: {'module__dropout_rate': 0.4, 'module__weight_constraint': 1.0}
0.669271 (0.017566) with: {'module__dropout_rate': 0.4, 'module__weight_constraint': 2.0}
0.652344 (0.006379) with: {'module__dropout_rate': 0.4, 'module__weight_constraint': 3.0}
0.680990 (0.037783) with: {'module__dropout_rate': 0.4, 'module__weight_constraint': 4.0}
0.692708 (0.042112) with: {'module__dropout_rate': 0.4, 'module__weight_constraint': 5.0}
0.666667 (0.006639) with: {'module__dropout_rate': 0.5, 'module__weight_constraint': 1.0}
0.652344 (0.011500) with: {'module__dropout_rate': 0.5, 'module__weight_constraint': 2.0}
0.662760 (0.007366) with: {'module__dropout_rate': 0.5, 'module__weight_constraint': 3.0}
0.558594 (0.146610) with: {'module__dropout_rate': 0.5, 'module__weight_constraint': 4.0}
0.552083 (0.141826) with: {'module__dropout_rate': 0.5, 'module__weight_constraint': 5.0}
0.548177 (0.141826) with: {'module__dropout_rate': 0.6, 'module__weight_constraint': 1.0}
0.653646 (0.013279) with: {'module__dropout_rate': 0.6, 'module__weight_constraint': 2.0}
0.661458 (0.008027) with: {'module__dropout_rate': 0.6, 'module__weight_constraint': 3.0}
0.553385 (0.142719) with: {'module__dropout_rate': 0.6, 'module__weight_constraint': 4.0}
0.669271 (0.035132) with: {'module__dropout_rate': 0.6, 'module__weight_constraint': 5.0}
0.662760 (0.015733) with: {'module__dropout_rate': 0.7, 'module__weight_constraint': 1.0}
0.636719 (0.024910) with: {'module__dropout_rate': 0.7, 'module__weight_constraint': 2.0}
0.550781 (0.146818) with: {'module__dropout_rate': 0.7, 'module__weight_constraint': 3.0}
0.537760 (0.140094) with: {'module__dropout_rate': 0.7, 'module__weight_constraint': 4.0}
0.542969 (0.138144) with: {'module__dropout_rate': 0.7, 'module__weight_constraint': 5.0}
0.565104 (0.148654) with: {'module__dropout_rate': 0.8, 'module__weight_constraint': 1.0}
0.657552 (0.008027) with: {'module__dropout_rate': 0.8, 'module__weight_constraint': 2.0}
0.428385 (0.111418) with: {'module__dropout_rate': 0.8, 'module__weight_constraint': 3.0}
0.549479 (0.142719) with: {'module__dropout_rate': 0.8, 'module__weight_constraint': 4.0}
0.648438 (0.005524) with: {'module__dropout_rate': 0.8, 'module__weight_constraint': 5.0}
0.540365 (0.136861) with: {'module__dropout_rate': 0.9, 'module__weight_constraint': 1.0}
0.605469 (0.053083) with: {'module__dropout_rate': 0.9, 'module__weight_constraint': 2.0}
0.553385 (0.139948) with: {'module__dropout_rate': 0.9, 'module__weight_constraint': 3.0}
0.549479 (0.142719) with: {'module__dropout_rate': 0.9, 'module__weight_constraint': 4.0}
0.595052 (0.075566) with: {'module__dropout_rate': 0.9, 'module__weight_constraint': 5.0}
```

你可以看到 10% 的 dropout 率和 2.0 的权重约束得到了最佳的准确率，约为 70%。

## 如何调整隐藏层中的神经元数量

层中神经元的数量是一个重要的调整参数。通常，层中神经元的数量控制网络的表示能力，至少在拓扑结构的那个点上是如此。

一般来说，足够大的单层网络可以近似任何其他神经网络，这归因于 [通用逼近定理](https://en.wikipedia.org/wiki/Universal_approximation_theorem)。

在这个示例中，你将调整单个隐藏层中的神经元数量。你将尝试从 1 到 30 的值，步长为 5。

更大的网络需要更多的训练，并且批量大小和训练轮数至少应该与神经元的数量一起进行优化。

完整的代码清单如下所示。

```py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

class PimaClassifier(nn.Module):
    def __init__(self, n_neurons=12):
        super().__init__()
        self.layer = nn.Linear(8, n_neurons)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(n_neurons, 1)
        self.prob = nn.Sigmoid()
        self.weight_constraint = 2.0
        # manually init weights
        init.kaiming_uniform_(self.layer.weight)
        init.kaiming_uniform_(self.output.weight)

    def forward(self, x):
        # maxnorm weight before actual forward pass
        with torch.no_grad():
            norm = self.layer.weight.norm(2, dim=0, keepdim=True).clamp(min=self.weight_constraint / 2)
            desired = torch.clamp(norm, max=self.weight_constraint)
            self.layer.weight *= (desired / norm)
        # actual forward pass
        x = self.act(self.layer(x))
        x = self.dropout(x)
        x = self.prob(self.output(x))
        return x

# create model with skorch
model = NeuralNetClassifier(
    PimaClassifier,
    criterion=nn.BCELoss,
    optimizer=optim.Adamax,
    max_epochs=100,
    batch_size=10,
    verbose=False
)

# define the grid search parameters
param_grid = {
    'module__n_neurons': [1, 5, 10, 15, 20, 25, 30]
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

运行此示例将产生以下输出。

```py
Best: 0.708333 using {'module__n_neurons': 30}
0.654948 (0.003683) with: {'module__n_neurons': 1}
0.666667 (0.023073) with: {'module__n_neurons': 5}
0.694010 (0.014382) with: {'module__n_neurons': 10}
0.682292 (0.014382) with: {'module__n_neurons': 15}
0.707031 (0.028705) with: {'module__n_neurons': 20}
0.703125 (0.030758) with: {'module__n_neurons': 25}
0.708333 (0.015733) with: {'module__n_neurons': 30}
```

你可以看到，最佳结果是在隐藏层有 30 个神经元的网络中获得的，准确率约为 71%。

## 超参数优化的提示

本节列出了一些在调整神经网络的超参数时需要考虑的实用提示。

+   **$k$ 折交叉验证**。你可以看到本文示例中的结果存在一些变化。默认使用了 3 折交叉验证，但也许使用 $k=5$ 或 $k=10$ 会更稳定。仔细选择交叉验证配置以确保结果稳定。

+   **审查整个网格**。不要只关注最佳结果，审查整个结果网格并寻找支持配置决策的趋势。当然，会有更多的组合，评估时间更长。

+   **并行化**。如果可以的话，请使用所有核心，神经网络训练速度较慢，我们经常希望尝试许多不同的参数。考虑在云平台如 AWS 上运行它。

+   **使用数据集的样本**。由于网络训练速度较慢，请尝试在训练数据集的较小样本上进行训练，只是为了了解参数的一般方向，而不是最优配置。

+   **从粗网格开始**。从粗粒度的网格开始，并在能够缩小范围后逐渐缩放到更细粒度的网格。

+   **不要转移结果**。结果通常是特定于问题的。尽量避免在每个新问题上使用喜爱的配置。你发现的一个问题上的最优结果不太可能转移到下一个项目上。相反，要寻找像层的数量或参数之间的关系这样更广泛的趋势。

+   **可复现性是一个问题**。虽然我们在 NumPy 中设置了随机数生成器的种子，但结果并不是 100%可复现的。在网格搜索包装的 PyTorch 模型中，可复现性比本文介绍的更多。

## 进一步阅读

此部分提供了更多关于这个主题的资源，如果你想深入了解的话。

+   [skorch](https://skorch.readthedocs.io/en/latest/) 文档

+   来自 PyTorch 的 [torch.nn](https://pytorch.org/docs/stable/nn.html)

+   [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 来自 scikit-learn

## 摘要

在本文中，你了解到如何使用 PyTorch 和 scikit-learn 在 Python 中调整深度学习网络的超参数。

具体来说，您学到了：

+   如何将 PyTorch 模型包装以在 scikit-learn 中使用以及如何使用网格搜索。

+   如何为 PyTorch 模型网格搜索一套不同的标准神经网络参数。

+   如何设计您自己的超参数优化实验。
