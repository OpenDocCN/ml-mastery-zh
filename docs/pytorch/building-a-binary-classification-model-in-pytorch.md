# 在 PyTorch 中构建二元分类模型

> 原文：[`machinelearningmastery.com/building-a-binary-classification-model-in-pytorch/`](https://machinelearningmastery.com/building-a-binary-classification-model-in-pytorch/)

PyTorch 库是用于深度学习的。深度学习模型的一些应用是解决回归或分类问题。

在本文中，您将发现如何使用 PyTorch 开发和评估用于二元分类问题的神经网络模型。

完成本文后，您将了解：

+   如何加载训练数据并使其在 PyTorch 中可用

+   如何设计和训练神经网络

+   如何使用 k 折交叉验证评估神经网络模型的性能

+   如何以推理模式运行模型

+   如何为二元分类模型创建接收器操作特性曲线

**用我的书 [Deep Learning with PyTorch](https://machinelearningmastery.com/deep-learning-with-pytorch/) 开始您的项目**。它提供**自学教程**和**可运行的代码**。

让我们开始吧！![](img/e9eeb55887fe1686c2c425077d9c631c.png)

在 PyTorch 中构建二元分类模型

照片由 [David Tang](https://unsplash.com/photos/Ufx030zbA3s) 拍摄。部分权利保留。

## 数据集描述

您在本教程中将使用的数据集是 [Sonar 数据集](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks))。

这是描述声纳回波反射不同表面的数据集。60 个输入变量是不同角度的回波强度。这是一个需要模型区分岩石和金属圆柱体的二元分类问题。

您可以在 [UCI 机器学习库](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)) 上了解更多关于这个数据集的信息。您可以免费[下载数据集](https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data)，并将其放置在工作目录中，文件名为 `sonar.csv`。

这是一个被广泛理解的数据集。所有变量都是连续的，通常在 0 到 1 的范围内。输出变量是字符串“M”表示矿石和“R”表示岩石，需要将其转换为整数 1 和 0。

使用这个数据集的一个好处是它是一个标准的基准问题。这意味着我们对一个优秀模型的预期技能有一些了解。使用交叉验证，一个神经网络应该能够达到 84% 到 88% 的准确率。[链接](http://www.is.umk.pl/projects/datasets.html#Sonar)

## 加载数据集

如果您已经以 CSV 格式下载并将数据集保存为 `sonar.csv` 在本地目录中，您可以使用 pandas 加载数据集。有 60 个输入变量 (`X`) 和一个输出变量 (`y`)。由于文件包含混合数据（字符串和数字），使用 pandas 比其他工具如 NumPy 更容易读取它们。

数据可以如下读取：

```py
import pandas as pd

# Read data
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]
```

这是一个二分类数据集。你更倾向于使用数值标签而不是字符串标签。你可以使用 scikit-learn 中的 `LabelEncoder` 进行这种转换。`LabelEncoder` 是将每个标签映射到一个整数。在这种情况下，只有两个标签，它们将变成 0 和 1。

使用它时，你需要首先调用 `fit()` 函数以让它学习可用的标签。然后调用 `transform()` 进行实际转换。下面是如何使用 `LabelEncoder` 将 `y` 从字符串转换为 0 和 1：

```py
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
```

你可以使用以下方法查看标签：

```py
print(encoder.classes_)
```

输出为：

```py
['M' 'R']
```

如果你运行 `print(y)`，你会看到以下内容

```py
[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
```

你会看到标签被转换为 0 和 1。从 `encoder.classes_` 中，你知道 0 代表“M”，1 代表“R”。在二分类的背景下，它们也分别被称为负类和正类。

之后，你应该将它们转换为 PyTorch 张量，因为这是 PyTorch 模型希望使用的格式。

```py
import torch

X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
```

### 想要开始使用 PyTorch 进行深度学习吗？

立即参加我的免费邮件速成课程（包括示例代码）。

点击注册，还可以获得免费的 PDF 电子书版本课程。

## 创建模型

现在你已经准备好进行神经网络模型训练了。

正如你在之前的一些帖子中看到的，最简单的神经网络模型是一个只有一个隐藏层的 3 层模型。深度学习模型通常指的是那些有多个隐藏层的模型。所有神经网络模型都有称为权重的参数。模型的参数越多，按照经验我们认为它就越强大。你应该使用一个层数较少但每层参数更多的模型，还是使用一个层数较多但每层参数较少的模型？让我们来探讨一下。

每层具有更多参数的模型称为更宽的模型。在这个例子中，输入数据有 60 个特征用于预测一个二分类变量。你可以假设构建一个具有 180 个神经元的单隐层宽模型（是输入特征的三倍）。这样的模型可以使用 PyTorch 构建：

```py
import torch.nn as nn

class Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(60, 180)
        self.relu = nn.ReLU()
        self.output = nn.Linear(180, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x
```

因为这是一个二分类问题，输出必须是长度为 1 的向量。然后你还希望输出在 0 和 1 之间，因此你可以将其视为概率或模型对输入属于“正类”的预测置信度。

更多层的模型称为更深的模型。考虑到之前的模型有一个包含 180 个神经元的层，你可以尝试一个具有三个层，每层 60 个神经元的模型。这样的模型可以使用 PyTorch 构建：

```py
class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(60, 60)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(60, 60)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(60, 60)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(60, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x
```

你可以确认这两个模型的参数数量是相似的，如下所示：

```py
# Compare model sizes
model1 = Wide()
model2 = Deep()
print(sum([x.reshape(-1).shape[0] for x in model1.parameters()]))  # 11161
print(sum([x.reshape(-1).shape[0] for x in model2.parameters()]))  # 11041
```

`model1.parameters()` 将返回所有模型的参数，每个参数都是 PyTorch 张量。然后你可以将每个张量重塑为向量并计算向量的长度，使用 `x.reshape(-1).shape[0]`。因此，上述方法总结了每个模型中的总参数数量。

## 使用交叉验证比较模型

你应该使用宽模型还是深度模型？一种方法是使用交叉验证来比较它们。

这是一种技术，利用“训练集”数据来训练模型，然后使用“测试集”数据来查看模型的预测准确性。测试集的结果是你应该关注的。然而，你不想只测试一次模型，因为如果你看到极端好的或坏的结果，可能是偶然的。你希望运行这个过程$k$次，使用不同的训练集和测试集，以确保你在比较**模型设计**，而不是某次训练的结果。

你可以在这里使用的技术称为 k 折交叉验证。它将较大的数据集拆分成$k$份，然后将一份作为测试集，而其他$k-1$份作为训练集。这样会有$k$种不同的组合。因此，你可以重复实验$k$次并取平均结果。

在 scikit-learn 中，你有一个用于分层 k 折的函数。分层的意思是，当数据拆分成$k$份时，算法会查看标签（即，二分类问题中的正负类），以确保每份数据中包含相等数量的各类。

运行 k 折交叉验证是微不足道的，例如以下代码：

```py
# define 5-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True)
cv_scores = []
for train, test in kfold.split(X, y):
    # create model, train, and get accuracy
    model = Wide()
    acc = model_train(model, X[train], y[train], X[test], y[test])
    print("Accuracy (wide): %.2f" % acc)
    cv_scores.append(acc)

# evaluate the model
acc = np.mean(cv_scores)
std = np.std(cv_scores)
print("Model accuracy: %.2f%% (+/- %.2f%%)" % (acc*100, std*100))
```

简单来说，你使用`StratifiedKFold()`来自 scikit-learn 来拆分数据集。这个函数会返回给你索引。因此，你可以使用`X[train]`和`X[test]`来创建拆分后的数据集，并将它们命名为训练集和验证集（以免与“测试集”混淆，测试集会在我们选择模型设计后使用）。你假设有一个函数可以在模型上运行训练循环，并给出验证集上的准确率。然后你可以找出这个得分的均值和标准差，作为这种模型设计的性能指标。请注意，在上面的 for 循环中，你需要每次创建一个新的模型，因为你不应该在 k 折交叉验证中重新训练一个已经训练好的模型。

训练循环可以定义如下：

```py
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

def model_train(model, X_train, y_train, X_val, y_val):
    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 250   # number of epochs to run
    batch_size = 10  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc
```

上述训练循环包含了通常的元素：前向传播、反向传播和梯度下降权重更新。但它扩展到每个 epoch 后有一个评估步骤：你以评估模式运行模型，并检查模型如何预测**验证集**。验证集上的准确率会被记住，并与模型权重一起保存。在训练结束时，最佳权重会被恢复到模型中，并返回最佳准确率。这个返回值是你在多次训练的 epoch 中遇到的最佳值，并且基于验证集。

注意，你在上面的`tqdm`中设置了`disable=True`。你可以将其设置为`False`，以便在训练过程中查看训练集的损失和准确率。

请记住，目标是选择最佳设计并重新训练模型。在训练中，你需要一个评估得分，以便了解生产中的预期效果。因此，你应该将获得的整个数据集拆分为训练集和测试集。然后，你可以在 k 折交叉验证中进一步拆分训练集。

有了这些，下面是你如何比较两个模型设计的方法：通过对每个模型进行 k 折交叉验证，并比较准确度：

```py
from sklearn.model_selection import StratifiedKFold, train_test_split

# train-test split: Hold out the test set for final model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# define 5-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True)
cv_scores_wide = []
for train, test in kfold.split(X_train, y_train):
    # create model, train, and get accuracy
    model = Wide()
    acc = model_train(model, X_train[train], y_train[train], X_train[test], y_train[test])
    print("Accuracy (wide): %.2f" % acc)
    cv_scores_wide.append(acc)
cv_scores_deep = []
for train, test in kfold.split(X_train, y_train):
    # create model, train, and get accuracy
    model = Deep()
    acc = model_train(model, X_train[train], y_train[train], X_train[test], y_train[test])
    print("Accuracy (deep): %.2f" % acc)
    cv_scores_deep.append(acc)

# evaluate the model
wide_acc = np.mean(cv_scores_wide)
wide_std = np.std(cv_scores_wide)
deep_acc = np.mean(cv_scores_deep)
deep_std = np.std(cv_scores_deep)
print("Wide: %.2f%% (+/- %.2f%%)" % (wide_acc*100, wide_std*100))
print("Deep: %.2f%% (+/- %.2f%%)" % (deep_acc*100, deep_std*100))
```

你可能会看到上述输出如下：

```py
Accuracy (wide): 0.72
Accuracy (wide): 0.66
Accuracy (wide): 0.83
Accuracy (wide): 0.76
Accuracy (wide): 0.83
Accuracy (deep): 0.90
Accuracy (deep): 0.72
Accuracy (deep): 0.93
Accuracy (deep): 0.69
Accuracy (deep): 0.76
Wide: 75.86% (+/- 6.54%)
Deep: 80.00% (+/- 9.61%)
```

因此，你发现较深的模型优于较宽的模型，因为其平均准确度更高且标准差更低。

## 重新训练最终模型

现在你知道选择哪个设计了，你想要重新构建模型并重新训练它。通常在 k 折交叉验证中，你会使用较小的数据集来加快训练速度。最终准确度不是问题，因为 k 折交叉验证的目的在于确定哪个设计更好。在最终模型中，你想提供更多的数据并生成更好的模型，因为这是你在生产中将使用的。

既然你已经将数据分为训练集和测试集，这些就是你将使用的数据。在 Python 代码中，

```py
# rebuild model with full set of training data
if wide_acc > deep_acc:
    print("Retrain a wide model")
    model = Wide()
else:
    print("Retrain a deep model")
    model = Deep()
acc = model_train(model, X_train, y_train, X_test, y_test)
print(f"Final model accuracy: {acc*100:.2f}%")
```

你可以重用 `model_train()` 函数，因为它执行了所有必要的训练和验证。这是因为最终模型或在 k 折交叉验证中的训练过程不会改变。

这个模型是你可以在生产中使用的。通常，与训练不同，预测是在生产中逐个数据样本进行的。以下是我们通过运行五个测试集样本来演示使用模型进行推断的方法：

```py
model.eval()
with torch.no_grad():
    # Test out inference with 5 samples
    for i in range(5):
        y_pred = model(X_test[i:i+1])
        print(f"{X_test[i].numpy()} -> {y_pred[0].numpy()} (expected {y_test[i].numpy()})")
```

它的输出应如下所示：

```py
[0.0265 0.044  0.0137 0.0084 0.0305 0.0438 0.0341 0.078  0.0844 0.0779
 0.0327 0.206  0.1908 0.1065 0.1457 0.2232 0.207  0.1105 0.1078 0.1165
 0.2224 0.0689 0.206  0.2384 0.0904 0.2278 0.5872 0.8457 0.8467 0.7679
 0.8055 0.626  0.6545 0.8747 0.9885 0.9348 0.696  0.5733 0.5872 0.6663
 0.5651 0.5247 0.3684 0.1997 0.1512 0.0508 0.0931 0.0982 0.0524 0.0188
 0.01   0.0038 0.0187 0.0156 0.0068 0.0097 0.0073 0.0081 0.0086 0.0095] -> [0.9583146] (expected [1.])
...

[0.034  0.0625 0.0381 0.0257 0.0441 0.1027 0.1287 0.185  0.2647 0.4117
 0.5245 0.5341 0.5554 0.3915 0.295  0.3075 0.3021 0.2719 0.5443 0.7932
 0.8751 0.8667 0.7107 0.6911 0.7287 0.8792 1\.     0.9816 0.8984 0.6048
 0.4934 0.5371 0.4586 0.2908 0.0774 0.2249 0.1602 0.3958 0.6117 0.5196
 0.2321 0.437  0.3797 0.4322 0.4892 0.1901 0.094  0.1364 0.0906 0.0144
 0.0329 0.0141 0.0019 0.0067 0.0099 0.0042 0.0057 0.0051 0.0033 0.0058] -> [0.01937182] (expected [0.])
```

你在 `torch.no_grad()` 上下文中运行代码，因为你确定没有必要在结果上运行优化器。因此，你希望解除涉及的张量对如何计算值的记忆。

二分类神经网络的输出介于 0 和 1 之间（由于最后的 sigmoid 函数）。从 `encoder.classes_` 中，你可以看到 0 代表“M”，1 代表“R”。对于介于 0 和 1 之间的值，你可以简单地将其四舍五入为最接近的整数并解释 0-1 结果，即，

```py
y_pred = model(X_test[i:i+1])
y_pred = y_pred.round() # 0 or 1
```

或者使用其他阈值将值量化为 0 或 1，即，

```py
threshold = 0.68
y_pred = model(X_test[i:i+1])
y_pred = (y_pred > threshold).float() # 0.0 or 1.0
```

实际上，将其四舍五入为最接近的整数等同于使用 0.5 作为阈值。一个好的模型应该对阈值的选择具有鲁棒性。这是指模型输出恰好为 0 或 1。否则，你会更喜欢一个很少报告中间值但经常返回接近 0 或接近 1 值的模型。要判断你的模型是否优秀，你可以使用**接收者操作特征曲线**（ROC），它是绘制模型在各种阈值下的真正率与假正率的图。你可以利用 scikit-learn 和 matplotlib 来绘制 ROC：

```py
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

with torch.no_grad():
    # Plot the ROC curve
    y_pred = model(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr) # ROC curve = TPR vs FPR
    plt.title("Receiver Operating Characteristics")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
```

你可能会看到以下内容。曲线总是从左下角开始，并在右上角结束。曲线越靠近左上角，模型的效果就越好。![](img/4b42ed8d731efa28450b20623da84dc2.png)

## 完整代码

将所有内容汇总，以下是上述代码的完整版本：

```py
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

# Read data
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]

# Binary encoding of labels
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# Convert to 2D PyTorch tensors
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# Define two models
class Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(60, 180)
        self.relu = nn.ReLU()
        self.output = nn.Linear(180, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(60, 60)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(60, 60)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(60, 60)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(60, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

# Compare model sizes
model1 = Wide()
model2 = Deep()
print(sum([x.reshape(-1).shape[0] for x in model1.parameters()]))  # 11161
print(sum([x.reshape(-1).shape[0] for x in model2.parameters()]))  # 11041

# Helper function to train one model
def model_train(model, X_train, y_train, X_val, y_val):
    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 300   # number of epochs to run
    batch_size = 10  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc

# train-test split: Hold out the test set for final model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# define 5-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True)
cv_scores_wide = []
for train, test in kfold.split(X_train, y_train):
    # create model, train, and get accuracy
    model = Wide()
    acc = model_train(model, X_train[train], y_train[train], X_train[test], y_train[test])
    print("Accuracy (wide): %.2f" % acc)
    cv_scores_wide.append(acc)
cv_scores_deep = []
for train, test in kfold.split(X_train, y_train):
    # create model, train, and get accuracy
    model = Deep()
    acc = model_train(model, X_train[train], y_train[train], X_train[test], y_train[test])
    print("Accuracy (deep): %.2f" % acc)
    cv_scores_deep.append(acc)

# evaluate the model
wide_acc = np.mean(cv_scores_wide)
wide_std = np.std(cv_scores_wide)
deep_acc = np.mean(cv_scores_deep)
deep_std = np.std(cv_scores_deep)
print("Wide: %.2f%% (+/- %.2f%%)" % (wide_acc*100, wide_std*100))
print("Deep: %.2f%% (+/- %.2f%%)" % (deep_acc*100, deep_std*100))

# rebuild model with full set of training data
if wide_acc > deep_acc:
    print("Retrain a wide model")
    model = Wide()
else:
    print("Retrain a deep model")
    model = Deep()
acc = model_train(model, X_train, y_train, X_test, y_test)
print(f"Final model accuracy: {acc*100:.2f}%")

model.eval()
with torch.no_grad():
    # Test out inference with 5 samples
    for i in range(5):
        y_pred = model(X_test[i:i+1])
        print(f"{X_test[i].numpy()} -> {y_pred[0].numpy()} (expected {y_test[i].numpy()})")

    # Plot the ROC curve
    y_pred = model(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr) # ROC curve = TPR vs FPR
    plt.title("Receiver Operating Characteristics")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
```

## 总结

在这篇文章中，你发现了如何使用 PyTorch 构建二分类模型。

你学会了如何使用 PyTorch 一步一步地解决二分类问题，具体包括：

+   如何加载和准备 PyTorch 中使用的数据

+   如何创建神经网络模型并使用 k 折交叉验证对其进行比较

+   如何训练二分类模型并获取其接收者操作特征曲线
