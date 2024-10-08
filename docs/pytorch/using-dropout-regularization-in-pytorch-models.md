# 在 PyTorch 模型中使用 Dropout 正则化

> 原文：[`machinelearningmastery.com/using-dropout-regularization-in-pytorch-models/`](https://machinelearningmastery.com/using-dropout-regularization-in-pytorch-models/)

[Dropout](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/) 是一种简单而强大的神经网络和深度学习模型的正则化技术。

在这篇文章中，你将发现 Dropout 正则化技术及其如何应用于 PyTorch 模型。

阅读完这篇文章后，你将了解：

+   Dropout 正则化技术的工作原理

+   如何在输入层上使用 Dropout

+   如何在隐藏层上使用 Dropout

+   如何调整你的问题的 dropout 水平

**通过我的书籍** [Deep Learning with PyTorch](https://machinelearningmastery.com/deep-learning-with-pytorch/) **启动你的项目**。它提供了 **自学教程** 和 **有效代码**。

让我们开始吧！[](../Images/53f7b7fdd299d10c88720ea518639304.png)

在 PyTorch 模型中使用 Dropout 正则化

照片由 [Priscilla Fraire](https://unsplash.com/photos/65dCe4Zuek4) 拍摄。部分权利保留。

## 概述

本文分为六个部分；它们是

+   神经网络的 Dropout 正则化

+   PyTorch 中的 Dropout 正则化

+   在输入层上使用 Dropout

+   在隐藏层上使用 Dropout

+   评估模式中的 Dropout

+   使用 Dropout 的提示

## 神经网络的 Dropout 正则化

Dropout 是一种用于神经网络模型的正则化技术，提出于 2012 年至 2014 年间。它是神经网络中的一层。在神经网络模型的训练过程中，它会从前一层接收输出，随机选择一些神经元并将其归零，然后传递到下一层，从而有效地忽略它们。这意味着它们对下游神经元激活的贡献在前向传递时被暂时移除，并且在反向传递时不会对这些神经元应用任何权重更新。

当模型用于推断时，dropout 层只是将所有神经元的权重缩放，以补偿训练过程中丢弃的影响。

Dropout 是一种破坏性的技术，但惊人的是，它可以提高模型的准确性。当神经网络学习时，神经元权重会在网络中形成其上下文。神经元的权重被调节以适应特定特征，从而提供一些专门化。相邻的神经元会依赖这种专门化，如果过度依赖，可能会导致模型过于专门化，从而对训练数据过于脆弱。这种在训练过程中对神经元上下文的依赖被称为复杂的协同适应。

你可以想象，如果在训练过程中神经元被随机丢弃，其他神经元将不得不介入并处理缺失神经元所需的表示。这被认为会导致网络学习到多个独立的内部表示。

其效果是网络对神经元的特定权重不那么敏感。这反过来使网络能够更好地进行泛化，并且不容易对训练数据进行过拟合。

### 想要开始使用 PyTorch 进行深度学习吗？

立即参加我的免费电子邮件速成课程（包含示例代码）。

点击注册并获取免费的 PDF 电子书版本课程。

## PyTorch 中的 Dropout 正则化

你不需要从 PyTorch 张量中随机选择元素来手动实现 dropout。可以将 PyTorch 的 `nn.Dropout()` 层引入到你的模型中。它通过以给定的概率 $p$（例如 20%）随机选择要丢弃的节点来实现。在训练循环中，PyTorch 的 dropout 层进一步将结果张量按 $\dfrac{1}{1-p}$ 的因子缩放，以保持平均张量值。由于这种缩放，dropout 层在推理时会成为一个恒等函数（即无效，仅将输入张量复制为输出张量）。你应确保在评估模型时将模型转换为推理模式。

让我们看看如何在 PyTorch 模型中使用 `nn.Dropout()`。

示例将使用 [Sonar 数据集](http://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks))。这是一个二分类问题，旨在正确识别从声纳回波返回的岩石和假矿。它是一个适合神经网络的测试数据集，因为所有输入值都是数值型且具有相同的尺度。

数据集可以 [从 UCI 机器学习库下载](http://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data)。你可以将声纳数据集放在当前工作目录中，文件名为 *sonar.csv*。

你将使用 scikit-learn 进行 10 折交叉验证来评估开发的模型，以便更好地揭示结果中的差异。

数据集中有 60 个输入值和一个输出值。输入值在用于网络之前会被标准化。基线神经网络模型有两个隐藏层，第一个层有 60 个单元，第二个层有 30 个。使用随机梯度下降来训练模型，学习率和动量相对较低。

完整的基线模型如下所示：

```py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# Read data
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]

# Label encode the target from string to integer
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# Convert to 2D PyTorch tensors
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# Define PyTorch model
class SonarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(60, 60)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(60, 30)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(30, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

# Helper function to train the model and return the validation result
def model_train(model, X_train, y_train, X_val, y_val,
                n_epochs=300, batch_size=16):
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    batch_start = torch.arange(0, len(X_train), batch_size)

    model.train()
    for epoch in range(n_epochs):
        for start in batch_start:
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # evaluate accuracy after training
    model.eval()
    y_pred = model(X_val)
    acc = (y_pred.round() == y_val).float().mean()
    acc = float(acc)
    return acc

# run 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True)
accuracies = []
for train, test in kfold.split(X, y):
    # create model, train, and get accuracy
    model = SonarModel()
    acc = model_train(model, X[train], y[train], X[test], y[test])
    print("Accuracy: %.2f" % acc)
    accuracies.append(acc)

# evaluate the model
mean = np.mean(accuracies)
std = np.std(accuracies)
print("Baseline: %.2f%% (+/- %.2f%%)" % (mean*100, std*100))
```

运行示例会产生 82% 的估计分类准确率。

```py
Accuracy: 0.81
Accuracy: 0.81
Accuracy: 0.76
Accuracy: 0.86
Accuracy: 0.81
Accuracy: 0.90
Accuracy: 0.86
Accuracy: 0.95
Accuracy: 0.65
Accuracy: 0.80
Baseline: 82.12% (+/- 7.78%)
```

## 在输入层中使用 Dropout

Dropout 可以应用于称为可见层的输入神经元。

在下面的示例中，在输入层和第一个隐藏层之间添加了一个新的 Dropout 层。dropout 率设置为 20%，意味着每次更新周期中将随机排除五分之一的输入。

从上面的基线示例继续，下面的代码使用输入 dropout 对相同的网络进行测试：

```py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# Read data
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]

# Label encode the target from string to integer
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# Convert to 2D PyTorch tensors
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# Define PyTorch model, with dropout at input
class SonarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.layer1 = nn.Linear(60, 60)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(60, 30)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(30, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

# Helper function to train the model and return the validation result
def model_train(model, X_train, y_train, X_val, y_val,
                n_epochs=300, batch_size=16):
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    batch_start = torch.arange(0, len(X_train), batch_size)

    model.train()
    for epoch in range(n_epochs):
        for start in batch_start:
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # evaluate accuracy after training
    model.eval()
    y_pred = model(X_val)
    acc = (y_pred.round() == y_val).float().mean()
    acc = float(acc)
    return acc

# run 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True)
accuracies = []
for train, test in kfold.split(X, y):
    # create model, train, and get accuracy
    model = SonarModel()
    acc = model_train(model, X[train], y[train], X[test], y[test])
    print("Accuracy: %.2f" % acc)
    accuracies.append(acc)

# evaluate the model
mean = np.mean(accuracies)
std = np.std(accuracies)
print("Baseline: %.2f%% (+/- %.2f%%)" % (mean*100, std*100))
```

运行示例会导致分类准确率略微下降，至少在单次测试运行中。

```py
Accuracy: 0.62
Accuracy: 0.90
Accuracy: 0.76
Accuracy: 0.62
Accuracy: 0.67
Accuracy: 0.86
Accuracy: 0.90
Accuracy: 0.86
Accuracy: 0.90
Accuracy: 0.85
Baseline: 79.40% (+/- 11.20%)
```

## 在隐藏层中使用 Dropout

Dropout 可以应用于网络模型的隐藏神经元。这种做法更为常见。

在下面的示例中，Dropout 被应用在两个隐藏层之间以及最后一个隐藏层和输出层之间。再次使用了 20%的 Dropout 率：

```py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# Read data
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]

# Label encode the target from string to integer
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# Convert to 2D PyTorch tensors
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# Define PyTorch model, with dropout at hidden layers
class SonarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(60, 60)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(60, 30)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(30, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.dropout1(x)
        x = self.act2(self.layer2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.output(x))
        return x

# Helper function to train the model and return the validation result
def model_train(model, X_train, y_train, X_val, y_val,
                n_epochs=300, batch_size=16):
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    batch_start = torch.arange(0, len(X_train), batch_size)

    model.train()
    for epoch in range(n_epochs):
        for start in batch_start:
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # evaluate accuracy after training
    model.eval()
    y_pred = model(X_val)
    acc = (y_pred.round() == y_val).float().mean()
    acc = float(acc)
    return acc

# run 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True)
accuracies = []
for train, test in kfold.split(X, y):
    # create model, train, and get accuracy
    model = SonarModel()
    acc = model_train(model, X[train], y[train], X[test], y[test])
    print("Accuracy: %.2f" % acc)
    accuracies.append(acc)

# evaluate the model
mean = np.mean(accuracies)
std = np.std(accuracies)
print("Baseline: %.2f%% (+/- %.2f%%)" % (mean*100, std*100))
```

您可以看到，在这种情况下，添加 Dropout 层稍微提高了准确性。

```py
Accuracy: 0.86
Accuracy: 1.00
Accuracy: 0.86
Accuracy: 0.90
Accuracy: 0.90
Accuracy: 0.86
Accuracy: 0.81
Accuracy: 0.81
Accuracy: 0.70
Accuracy: 0.85
Baseline: 85.50% (+/- 7.36%)
```

## 评估模式下的 Dropout

Dropout 将随机将部分输入重置为零。如果您想知道训练结束后会发生什么，答案是什么也不会发生！当模型处于评估模式时，PyTorch 的 Dropout 层应该像一个恒等函数一样运行。这很重要，因为 Dropout 层的目标是确保网络对输入学习足够的线索以进行预测，而不是依赖于数据中的罕见现象。但是在推理时，您应尽可能向模型提供尽可能多的信息。

## 使用 Dropout 的技巧

Dropout 的原始论文提供了一系列标准机器学习问题的实验结果。因此，他们提供了一些在实践中使用 Dropout 时需要考虑的有用启发。

+   通常，使用 20%-50%的神经元的小 Dropout 值，其中 20%是一个很好的起点。概率过低几乎没有效果，而值过高会导致网络学习不足。

+   使用更大的网络。当在更大的网络上使用 Dropout 时，通常能够获得更好的性能，因为这给模型更多机会学习独立的表示。

+   在可见单元（输入）和隐藏单元上使用 Dropout。在网络的每一层应用 Dropout 已经显示出良好的结果。

+   使用大的学习率和衰减，以及大的动量。将学习率增加 10 到 100 倍，并使用 0.9 或 0.99 的高动量值。

+   约束网络权重的大小。大学习率可能导致非常大的网络权重。施加网络权重大小的约束，例如最大范数正则化，大小为 4 或 5，已被证明能够改善结果。

## 进一步阅读

以下是可以进一步了解神经网络和深度学习模型中 Dropout 的资源。

论文

+   [Dropout: 一种简单的防止神经网络过拟合的方法](http://jmlr.org/papers/v15/srivastava14a.html)

+   [通过防止特征检测器的共适应来改进神经网络](http://arxiv.org/abs/1207.0580)

在线资料

+   [深度学习中 Dropout 方法如何工作？](https://www.quora.com/How-does-the-dropout-method-work-in-deep-learning) 在 Quora 上

+   [nn.Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) 来自 PyTorch 文档

## 总结

在本文中，您将了解到用于深度学习模型的 Dropout 正则化技术。您将学到：

+   Dropout 是什么以及它如何工作

+   如何在自己的深度学习模型中使用 Dropout。

+   在您自己的模型上获得 Dropout 最佳结果的技巧。
