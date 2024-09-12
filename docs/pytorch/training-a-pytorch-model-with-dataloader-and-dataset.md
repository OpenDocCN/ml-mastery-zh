# 使用 DataLoader 和 Dataset 训练 PyTorch 模型

> 原文：[`machinelearningmastery.com/training-a-pytorch-model-with-dataloader-and-dataset/`](https://machinelearningmastery.com/training-a-pytorch-model-with-dataloader-and-dataset/)

当您构建和训练一个 PyTorch 深度学习模型时，可以通过几种不同的方式提供训练数据。最终，PyTorch 模型的工作方式类似于一个接受 PyTorch 张量并返回另一个张量的函数。您在如何获取输入张量方面有很大的自由度。可能最简单的方式是准备整个数据集的大张量，并在每个训练步骤中从中提取一个小批次。但是您会发现，使用`DataLoader`可以节省一些处理数据的代码行数。

在本篇文章中，您将了解如何在 PyTorch 中使用 Data 和 DataLoader。完成本文后，您将学会：

+   如何创建和使用 DataLoader 来训练您的 PyTorch 模型

+   如何使用 Data 类动态生成数据

**用我的书[Kick-start your project](https://machinelearningmastery.com/deep-learning-with-pytorch/)**。它提供**自学教程**和**工作代码**。

让我们开始吧！[](../Images/af45383d29bec7b11a42f8b5cd0c4c39.png)

使用 DataLoader 和 Dataset 训练 PyTorch 模型

照片由[Emmanuel Appiah](https://unsplash.com/photos/vPUVQOyOtyk)提供。部分权利保留。

## 概览

本文分为三个部分；它们是：

+   什么是`DataLoader`？

+   在训练循环中使用`DataLoader`

## 什么是`DataLoader`？

要训练一个深度学习模型，您需要数据。通常数据作为数据集提供。在数据集中，有很多数据样本或实例。您可以要求模型一次处理一个样本，但通常您会让模型处理一个包含多个样本的批次。您可以通过在张量上使用切片语法从数据集中提取一个批次来创建一个批次。为了获得更高质量的训练，您可能还希望在每个 epoch 中对整个数据集进行洗牌，以确保整个训练循环中没有两个相同的批次。有时，您可能会引入**数据增强**来手动为数据引入更多的变化。这在与图像相关的任务中很常见，您可以随机倾斜或缩放图像，以从少数图像生成大量数据样本。

您可以想象需要编写大量代码来完成所有这些操作。但使用`DataLoader`会更加轻松。

以下是如何创建一个`DataLoader`并从中获取一个批次的示例。在此示例中，使用了[sonar 数据集](http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks))，并最终将其转换为 PyTorch 张量，传递给`DataLoader`：

```py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# Read data, convert to NumPy arrays
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60].values
y = data.iloc[:, 60].values

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# convert into PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# create DataLoader, then take one batch
loader = DataLoader(list(zip(X,y)), shuffle=True, batch_size=16)
for X_batch, y_batch in loader:
    print(X_batch, y_batch)
    break
```

您可以从上面的输出中看到`X_batch`和`y_batch`是 PyTorch 张量。`loader`是`DataLoader`类的一个实例，可以像可迭代对象一样工作。每次从中读取时，您都会从原始数据集中获取一个特征和目标批次。

当你创建一个`DataLoader`实例时，你需要提供一个样本对列表。每个样本对是一个特征和相应目标的数据样本。需要使用列表，因为`DataLoader`期望使用`len()`来获取数据集的总大小，并使用数组索引来检索特定样本。批处理大小是`DataLoader`的一个参数，因此它知道如何从整个数据集创建批次。你几乎总是应该使用`shuffle=True`，这样每次加载数据时样本都会被打乱。这对训练很有用，因为在每个 epoch 中，你将读取每个批次一次。当你从一个 epoch 进入另一个 epoch 时，`DataLoader`会知道你已经耗尽了所有的批次，所以会重新洗牌，这样你就会得到新的样本组合。

### 想要用 PyTorch 开始深度学习吗？

现在参加我的免费电子邮件速成课程（附有示例代码）。

点击注册，还可以获得课程的免费 PDF 电子书版本。

## 在训练循环中使用`DataLoader`

下面是一个在训练循环中使用`DataLoader`的示例：

```py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# train-test split for evaluation of the model
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# set up DataLoader for training set
loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=16)

# create model
model = nn.Sequential(
    nn.Linear(60, 60),
    nn.ReLU(),
    nn.Linear(60, 30),
    nn.ReLU(),
    nn.Linear(30, 1),
    nn.Sigmoid()
)

# Train the model
n_epochs = 200
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
model.train()
for epoch in range(n_epochs):
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# evaluate accuracy after training
model.eval()
y_pred = model(X_test)
acc = (y_pred.round() == y_test).float().mean()
acc = float(acc)
print("Model accuracy: %.2f%%" % (acc*100))
```

你可以看到一旦创建了`DataLoader`实例，训练循环就会变得更加简单。在上面的例子中，只有训练集被打包成了一个`DataLoader`，因为你需要按批次遍历它。你也可以为测试集创建一个`DataLoader`，并用它进行模型评估，但由于精度是针对整个测试集计算而不是按批次计算，因此`DataLoader`的好处并不显著。

将所有内容整合在一起，以下是完整的代码。

```py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Read data, convert to NumPy arrays
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60].values
y = data.iloc[:, 60].values

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# convert into PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# train-test split for evaluation of the model
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# set up DataLoader for training set
loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=16)

# create model
model = nn.Sequential(
    nn.Linear(60, 60),
    nn.ReLU(),
    nn.Linear(60, 30),
    nn.ReLU(),
    nn.Linear(30, 1),
    nn.Sigmoid()
)

# Train the model
n_epochs = 200
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
model.train()
for epoch in range(n_epochs):
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# evaluate accuracy after training
model.eval()
y_pred = model(X_test)
acc = (y_pred.round() == y_test).float().mean()
acc = float(acc)
print("Model accuracy: %.2f%%" % (acc*100))
```

## 创建使用`Dataset`类的数据迭代器

在 PyTorch 中，有一个`Dataset`类，可以与`DataLoader`类紧密耦合。回想一下，`DataLoader`期望其第一个参数能够使用`len()`和数组索引。`Dataset`类是这一切的基类。你可能希望使用`Dataset`类的原因是在获取数据样本之前需要进行一些特殊处理。例如，数据可能需要从数据库或磁盘读取，并且你可能只想在内存中保留少量样本而不是预取所有内容。另一个例子是对数据进行实时预处理，例如图像任务中常见的随机增强。

要使用`Dataset`类，你只需从它继承并实现两个成员函数。以下是一个示例：

```py
from torch.utils.data import Dataset

class SonarDataset(Dataset):
    def __init__(self, X, y):
        # convert into PyTorch tensors and remember them
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # this should return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.X[idx]
        target = self.y[idx]
        return features, target
```

这并不是使用`Dataset`的最强大方式，但足够简单，可以演示其工作原理。有了这个，你可以创建一个`DataLoader`并用它进行模型训练。修改自前面的示例，你会得到以下内容：

```py
...

# set up DataLoader for training set
dataset = SonarDataset(X_train, y_train)
loader = DataLoader(dataset, shuffle=True, batch_size=16)

# create model
model = nn.Sequential(
    nn.Linear(60, 60),
    nn.ReLU(),
    nn.Linear(60, 30),
    nn.ReLU(),
    nn.Linear(30, 1),
    nn.Sigmoid()
)

# Train the model
n_epochs = 200
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
model.train()
for epoch in range(n_epochs):
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# evaluate accuracy after training
model.eval()
y_pred = model(torch.tensor(X_test, dtype=torch.float32))
y_test = torch.tensor(y_test, dtype=torch.float32)
acc = (y_pred.round() == y_test).float().mean()
acc = float(acc)
print("Model accuracy: %.2f%%" % (acc*100))
```

你将`dataset`设置为`SonarDataset`的一个实例，其中你实现了`__len__()`和`__getitem__()`函数。这在前面的示例中用于设置`DataLoader`实例的列表的位置。之后，在训练循环中一切都一样。请注意，在示例中，你仍然直接使用 PyTorch 张量来处理测试集。

在`__getitem__()`函数中，你传入一个像数组索引一样的整数，返回一对数据，即特征和目标。你可以在这个函数中实现任何操作：运行一些代码生成合成数据样本，从互联网动态读取数据，或者对数据添加随机变化。当你无法将整个数据集全部加载到内存中时，这个函数非常有用，因此你可以仅加载需要的数据样本。

实际上，由于你已创建了一个 PyTorch 数据集，你不需要使用 scikit-learn 来将数据分割成训练集和测试集。在`torch.utils.data`子模块中，你可以使用`random_split()`函数来与`Dataset`类一起实现相同的目的。以下是一个完整的示例：

```py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, default_collate
from sklearn.preprocessing import LabelEncoder

# Read data, convert to NumPy arrays
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60].values
y = data.iloc[:, 60].values

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y).reshape(-1, 1)

class SonarDataset(Dataset):
    def __init__(self, X, y):
        # convert into PyTorch tensors and remember them
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # this should return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.X[idx]
        target = self.y[idx]
        return features, target

# set up DataLoader for data set
dataset = SonarDataset(X, y)
trainset, testset = random_split(dataset, [0.7, 0.3])
loader = DataLoader(trainset, shuffle=True, batch_size=16)

# create model
model = nn.Sequential(
    nn.Linear(60, 60),
    nn.ReLU(),
    nn.Linear(60, 30),
    nn.ReLU(),
    nn.Linear(30, 1),
    nn.Sigmoid()
)

# Train the model
n_epochs = 200
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
model.train()
for epoch in range(n_epochs):
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# create one test tensor from the testset
X_test, y_test = default_collate(testset)
model.eval()
y_pred = model(X_test)
acc = (y_pred.round() == y_test).float().mean()
acc = float(acc)
print("Model accuracy: %.2f%%" % (acc*100))
```

这与你之前看到的例子非常相似。请注意，PyTorch 模型仍然需要张量作为输入，而不是`Dataset`。因此，在上述情况下，你需要使用`default_collate()`函数将数据集中的样本收集成张量。

## 进一步阅读

如果你希望深入了解此主题，本节提供了更多资源。

+   [PyTorch 文档中的 torch.utils.data 模块](https://pytorch.org/docs/stable/data.html)

+   [PyTorch 教程中的数据集和数据加载器](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

## 总结

在本文中，你学习了如何使用`DataLoader`创建打乱顺序的数据批次，以及如何使用`Dataset`提供数据样本。具体来说，你学会了：

+   `DataLoader`作为向训练循环提供数据批次的便捷方式

+   如何使用`Dataset`生成数据样本

+   如何结合`Dataset`和`DataLoader`以在模型训练中动态生成数据批次
