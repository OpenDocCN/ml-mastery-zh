# 如何评估 PyTorch 模型的性能

> 原文：[`machinelearningmastery.com/how-to-evaluate-the-performance-of-pytorch-models/`](https://machinelearningmastery.com/how-to-evaluate-the-performance-of-pytorch-models/)

设计深度学习模型有时是一门艺术。这里有许多决策点，很难判断哪种方案最好。设计的一种方法是通过试验和错误，并在实际数据上评估结果。因此，拥有科学的方法来评估神经网络和深度学习模型的性能非常重要。事实上，这也是比较任何机器学习模型在特定用途上的方法。

在这篇文章中，你将发现用于稳健评估模型性能的工作流程。在示例中，我们将使用 PyTorch 构建我们的模型，但该方法也适用于其他模型。完成这篇文章后，你将了解：

+   如何使用验证数据集评估 PyTorch 模型

+   如何使用 k 折交叉验证评估 PyTorch 模型

使用我的书籍 [深度学习与 PyTorch](https://machinelearningmastery.com/deep-learning-with-pytorch/) **启动你的项目**。它提供了**自学教程**和**有效代码**。

让我们开始吧！[](../Images/1dbc3c45767bd37c79ffc70d5105b3ec.png)

如何评估 PyTorch 模型的性能

图片由 [Kin Shing Lai](https://unsplash.com/photos/7qUtO7iNZ4M) 提供。保留部分权利。

## 概述

本章分为四部分，它们是：

+   模型的经验评估

+   数据拆分

+   使用验证集训练 PyTorch 模型

+   k 折交叉验证

## 模型的经验评估

在从头设计和配置深度学习模型时，需要做出很多决策。这包括设计决策，如使用多少层，每层的大小，使用什么层或激活函数。这还可能包括损失函数的选择、优化算法、训练的轮次以及模型输出的解释。幸运的是，有时你可以复制其他人的网络结构。有时，你可以通过一些启发式方法来做出选择。要判断你是否做出了正确的选择，最好的方法是通过实际数据的经验评估来比较多个备选方案。

深度学习常用于处理具有非常大数据集的问题，即数万或数十万的数据样本。这为测试提供了充足的数据。但你需要一个稳健的测试策略来估计模型在未见数据上的表现。基于此，你可以有一个指标来比较不同模型配置之间的优劣。

### 想要开始使用 PyTorch 进行深度学习？

现在就参加我的免费电子邮件速成课程（含示例代码）。

点击注册并获取课程的免费 PDF 电子书版本。

## 数据拆分

如果你有数以万计甚至更多的样本数据集，不必总是将所有数据都提供给模型进行训练。这将不必要地增加复杂性并延长训练时间。更多并不总是更好。你可能得不到最佳结果。

当你有大量数据时，应该将其中一部分作为**训练集**用于模型训练。另一部分作为**测试集**，在训练之外保留，但会用已训练或部分训练的模型进行验证。这一步通常称为“训练-测试分离”。

让我们考虑[Pima Indians Diabetes 数据集](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv)。您可以使用 NumPy 加载数据：

```py
import numpy as np
data = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
```

有 768 个数据样本。虽然不多，但足以演示分割。让我们将前 66% 视为训练集，剩余部分作为测试集。最简单的方法是通过对数组进行切片：

```py
# find the boundary at 66% of total samples
count = len(data)
n_train = int(count * 0.66)
# split the data at the boundary
train_data = data[:n_train]
test_data = data[n_train:]
```

66% 的选择是任意的，但你不希望训练集太小。有时你可能会使用 70%-30% 的分割。但如果数据集很大，你甚至可以使用 30%-70% 的分割，如果训练数据的 30% 足够大的话。

如果按此方式拆分数据，表明数据集已被洗牌，以使训练集和测试集同样多样化。如果发现原始数据集已排序，并且仅在最后取测试集，可能会导致所有测试数据属于同一类或在某个输入特征中具有相同值。这并不理想。

当然，在拆分之前可以调用`np.random.shuffle(data)`来避免这种情况。但是许多机器学习工程师通常使用 scikit-learn 来实现这一点。请参阅以下示例：

```py
import numpy as np
from sklearn.model_selection import train_test_split

data = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
train_data, test_data = train_test_split(data, test_size=0.33)
```

但更常见的是，在分开输入特征和输出标签之后进行。请注意，这个来自 scikit-learn 的函数不仅可以在 NumPy 数组上工作，还可以在 PyTorch 张量上工作：

```py
import numpy as np
import torch
from sklearn.model_selection import train_test_split

data = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = data[:, 0:8]
y = data[:, 8]
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
```

## 使用验证训练 PyTorch 模型

让我们重新审视在此数据集上构建和训练深度学习模型的代码：

```py
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

...

model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

# loss function and optimizer
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 50    # number of epochs to run
batch_size = 10  # size of each batch
batches_per_epoch = len(Xtrain) // batch_size

for epoch in range(n_epochs):
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            # take a batch
            start = i * batch_size
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
            bar.set_postfix(
                loss=float(loss)
            )
```

在这段代码中，每次迭代从训练集中提取一个批次，并在前向传播中发送到模型。然后在反向传播中计算梯度并更新权重。

虽然在这种情况下，您在训练循环中使用二元交叉熵作为损失指标，但您可能更关心预测准确性。计算准确性很容易。您将输出（在 0 到 1 的范围内）四舍五入到最接近的整数，以便获得二进制值 0 或 1。然后计算您的预测与标签匹配的百分比，这给出了准确性。

但你的预测是什么？它是上面的`y_pred`，这是您当前模型在`X_batch`上的预测。将准确性添加到训练循环变成了这样：

```py
for epoch in range(n_epochs):
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            # take a batch
            start = i * batch_size
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
            # print progress, with accuracy
            acc = (y_pred.round() == y_batch).float().mean()
            bar.set_postfix(
                loss=float(loss)
                acc=float(acc)
            )
```

然而，`X_batch`和`y_batch`被优化器使用，优化器将微调你的模型，使其能够从`X_batch`预测`y_batch`。现在你使用准确率检查`y_pred`是否与`y_batch`匹配。这就像作弊一样，因为如果你的模型以某种方式记住了解决方案，它可以直接向你报告`y_pred`，而无需真正从`X_batch`中推断`y_pred`，并获得完美的准确率。

实际上，一个深度学习模型可能复杂到你无法确定你的模型只是记住了答案还是推断了答案。因此，最好的方法是**不要**从`X_batch`或`X_train`中的任何内容计算准确率，而是从其他地方：你的测试集。让我们在每个时期结束后使用`X_test`添加准确率测量：

```py
for epoch in range(n_epochs):
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            # take a batch
            start = i * batch_size
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
    # evaluate model at end of epoch
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    acc = float(acc)
    print(f"End of {epoch}, accuracy {acc}")
```

在这种情况下，内部 for 循环中的`acc`只是一个显示进展的度量。与显示损失度量没有太大区别，只是它不参与梯度下降算法。你期望准确率随着损失度量的改善而提高。

在外部 for 循环中，每个时期结束时，你从`X_test`计算准确率。工作流程类似：你将测试集提供给模型并请求其预测，然后统计与测试集标签匹配的结果数量。但这正是你需要关注的准确率。它应该随着训练的进展而提高，但如果你没有看到它的提升（即准确率增加）甚至有所下降，你必须中断训练，因为它似乎开始过拟合。过拟合是指模型开始记住训练集而不是从中学习推断预测。一个迹象是训练集的准确率不断提高，而测试集的准确率却下降。

以下是实现上述所有内容的完整代码，从数据拆分到使用测试集进行验证：

```py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split

data = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = data[:, 0:8]
y = data[:, 8]
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

# loss function and optimizer
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 50    # number of epochs to run
batch_size = 10  # size of each batch
batches_per_epoch = len(X_train) // batch_size

for epoch in range(n_epochs):
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar: #, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            # take a batch
            start = i * batch_size
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
    # evaluate model at end of epoch
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    acc = float(acc)
    print(f"End of {epoch}, accuracy {acc}")
```

上面的代码将打印如下内容：

```py
End of 0, accuracy 0.5787401795387268
End of 1, accuracy 0.6102362275123596
End of 2, accuracy 0.6220472455024719
End of 3, accuracy 0.6220472455024719
End of 4, accuracy 0.6299212574958801
End of 5, accuracy 0.6377952694892883
End of 6, accuracy 0.6496062874794006
End of 7, accuracy 0.6535432934761047
End of 8, accuracy 0.665354311466217
End of 9, accuracy 0.6614173054695129
End of 10, accuracy 0.665354311466217
End of 11, accuracy 0.665354311466217
End of 12, accuracy 0.665354311466217
End of 13, accuracy 0.665354311466217
End of 14, accuracy 0.665354311466217
End of 15, accuracy 0.6732283234596252
End of 16, accuracy 0.6771653294563293
End of 17, accuracy 0.6811023354530334
End of 18, accuracy 0.6850393414497375
End of 19, accuracy 0.6889764070510864
End of 20, accuracy 0.6850393414497375
End of 21, accuracy 0.6889764070510864
End of 22, accuracy 0.6889764070510864
End of 23, accuracy 0.6889764070510864
End of 24, accuracy 0.6889764070510864
End of 25, accuracy 0.6850393414497375
End of 26, accuracy 0.6811023354530334
End of 27, accuracy 0.6771653294563293
End of 28, accuracy 0.6771653294563293
End of 29, accuracy 0.6692913174629211
End of 30, accuracy 0.6732283234596252
End of 31, accuracy 0.6692913174629211
End of 32, accuracy 0.6692913174629211
End of 33, accuracy 0.6732283234596252
End of 34, accuracy 0.6771653294563293
End of 35, accuracy 0.6811023354530334
End of 36, accuracy 0.6811023354530334
End of 37, accuracy 0.6811023354530334
End of 38, accuracy 0.6811023354530334
End of 39, accuracy 0.6811023354530334
End of 40, accuracy 0.6811023354530334
End of 41, accuracy 0.6771653294563293
End of 42, accuracy 0.6771653294563293
End of 43, accuracy 0.6771653294563293
End of 44, accuracy 0.6771653294563293
End of 45, accuracy 0.6771653294563293
End of 46, accuracy 0.6771653294563293
End of 47, accuracy 0.6732283234596252
End of 48, accuracy 0.6732283234596252
End of 49, accuracy 0.6732283234596252
```

## k 折交叉验证

在上面的例子中，你从测试集计算了准确率。它被用作模型在训练过程中进展的**评分**。你希望在这个分数达到最大值时停止。实际上，仅仅通过比较这个测试集的分数，你就知道你的模型在第 21 个时期之后表现最佳，并且之后开始过拟合。对吗？

如果你构建了两个不同设计的模型，是否应该仅仅比较这些模型在同一测试集上的准确率，并声称一个比另一个更好？

实际上，你可以认为即使在提取测试集之前已经打乱了数据集，测试集也不够具有代表性。你也可以认为，偶然间，一个模型可能更适合这个特定的测试集，但不一定总是更好。为了更强有力地论证哪个模型更好，不依赖于测试集的选择，你可以尝试**多个测试集**并计算准确率的平均值。

这就是 k 折交叉验证的作用。它是决定哪种**设计**效果更好的过程。它通过多次从头开始训练过程来工作，每次使用不同的训练和测试集组合。因此，您将得到$k$个模型和$k$个相应测试集的准确性分数。您不仅对平均准确率感兴趣，还对标准偏差感兴趣。标准偏差告诉您准确性分数是否一致，或者某些测试集在模型中特别好或特别差。

由于 k 折交叉验证多次从头开始训练模型，最好将训练循环包装在函数中：

```py
def model_train(X_train, y_train, X_test, y_test):
    # create new model
    model = nn.Sequential(
        nn.Linear(8, 12),
        nn.ReLU(),
        nn.Linear(12, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )

    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 25    # number of epochs to run
    batch_size = 10  # size of each batch
    batches_per_epoch = len(X_train) // batch_size

    for epoch in range(n_epochs):
        with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for i in bar:
                # take a batch
                start = i * batch_size
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
    # evaluate accuracy at end of training
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    return float(acc)
```

上面的代码故意不打印任何内容（使用`tqdm`中的`disable=True`）以保持屏幕整洁。

同样，在 scikit-learn 中，您有一个用于 k 折交叉验证的函数。您可以利用它来生成模型准确性的稳健估计：

```py
from sklearn.model_selection import StratifiedKFold

# define 5-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True)
cv_scores = []
for train, test in kfold.split(X, y):
    # create model, train, and get accuracy
    acc = model_train(X[train], y[train], X[test], y[test])
    print("Accuracy: %.2f" % acc)
    cv_scores.append(acc)
# evaluate the model
print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores)*100, np.std(cv_scores)*100))
```

运行此命令将输出：

```py
Accuracy: 0.64
Accuracy: 0.67
Accuracy: 0.68
Accuracy: 0.63
Accuracy: 0.59
64.05% (+/- 3.30%)
```

在 scikit-learn 中，有多个 k 折交叉验证函数，这里使用的是分层 k 折。它假设`y`是类标签，并考虑它们的值，以便在拆分中提供平衡的类表示。

上面的代码使用了$k=5$或 5 个拆分。这意味着将数据集分为五个相等的部分，选择其中一个作为测试集，将其余部分组合为训练集。有五种方法可以做到这一点，因此上述的 for 循环将进行五次迭代。在每次迭代中，您调用`model_train()`函数并得到准确率分数。然后将其保存到列表中，这将用于计算最终的均值和标准偏差。

`kfold`对象将返回给您**索引**。因此，您无需提前运行训练-测试分割，而是在调用`model_train()`函数时使用提供的索引动态提取训练集和测试集。

上面的结果显示，该模型的表现适中，平均准确率为 64%。由于标准偏差为 3%，这意味着大部分时间，您预期模型的准确率在 61%到 67%之间。您可以尝试更改上述模型，例如添加或删除一层，并观察均值和标准偏差的变化。您也可以尝试增加训练中使用的时期数并观察结果。

k 折交叉验证的均值和标准偏差是您应该用来评估模型设计的基准。

将所有内容综合起来，以下是完整的 k 折交叉验证代码：

```py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import StratifiedKFold

data = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = data[:, 0:8]
y = data[:, 8]
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

def model_train(X_train, y_train, X_test, y_test):
    # create new model
    model = nn.Sequential(
        nn.Linear(8, 12),
        nn.ReLU(),
        nn.Linear(12, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )

    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 25    # number of epochs to run
    batch_size = 10  # size of each batch
    batches_per_epoch = len(X_train) // batch_size

    for epoch in range(n_epochs):
        with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for i in bar:
                # take a batch
                start = i * batch_size
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
    # evaluate accuracy at end of training
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    return float(acc)

# define 5-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True)
cv_scores = []
for train, test in kfold.split(X, y):
    # create model, train, and get accuracy
    acc = model_train(X[train], y[train], X[test], y[test])
    print("Accuracy: %.2f" % acc)
    cv_scores.append(acc)
# evaluate the model
print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores)*100, np.std(cv_scores)*100))
```

## 摘要

在本文中，您了解到在深度学习模型在未见数据上估计性能时，有一个稳健的方法的重要性，并学习了如何实现。您看到：

+   如何使用 scikit-learn 将数据分割成训练集和测试集

+   如何在 scikit-learn 的帮助下进行 k 折交叉验证

+   如何修改 PyTorch 模型中的训练循环，以包括测试集验证和交叉验证
