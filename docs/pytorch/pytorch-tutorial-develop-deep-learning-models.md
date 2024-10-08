# PyTorch 教程：如何使用 Python 开发深度学习模型

> 原文：[`machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/`](https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/)

进行深度学习的预测建模是现代开发者需要掌握的一项技能。

PyTorch 是由 Facebook 开发和维护的顶级开源深度学习框架。

从本质上讲，PyTorch 是一个数学库，允许你对基于图的模型进行高效计算和自动微分。虽然直接实现这一点具有挑战性，但幸运的是，现代的 PyTorch API 提供了类和习惯用法，使你可以轻松开发一系列深度学习模型。

在本教程中，你将发现一个逐步指导，帮助你在 PyTorch 中开发深度学习模型。

完成本教程后，你将了解：

+   Torch 和 PyTorch 之间的区别以及如何安装和确认 PyTorch 是否正常工作。

+   PyTorch 模型的五个生命周期步骤，以及如何定义、训练和评估模型。

+   如何为回归、分类和预测建模任务开发 PyTorch 深度学习模型。

**通过我的书籍开始你的项目**，[《深度学习与 PyTorch》](https://machinelearningmastery.com/deep-learning-with-pytorch/)。它提供了**自学教程**和**可运行的代码**。

让我们开始吧。![PyTorch 教程 - 如何开发深度学习模型](img/13ff658685e88a714278b5d716b7a04a.png)

PyTorch 教程 – 如何开发深度学习模型

照片由 [Dimitry B](https://flickr.com/photos/ru_boff/14863560864/)。保留了一些权利。

## PyTorch 教程概述

本教程的重点是使用 PyTorch API 进行常见的深度学习模型开发任务；我们不会深入探讨深度学习的数学和理论。关于这些，我推荐[从这本优秀的书籍开始](https://amzn.to/2Y8JuBv)。

学习 Python 深度学习的最佳方式是动手实践。直接深入，你可以稍后再回顾理论。

我设计了每个代码示例以使用最佳实践，并使其独立，以便你可以直接复制并粘贴到你的项目中，并根据你的具体需求进行调整。这将使你比仅依赖官方文档快速掌握 API 领先一步。

这是一个大型教程，因此分为三个部分；它们是：

1.  如何安装 PyTorch

    1.  什么是 Torch 和 PyTorch？

    1.  如何安装 PyTorch

    1.  如何确认 PyTorch 已安装

1.  PyTorch 深度学习模型生命周期

    1.  步骤 1：准备数据

    1.  步骤 2：定义模型

    1.  步骤 3：训练模型

    1.  步骤 4：评估模型

    1.  步骤 5：进行预测

1.  如何开发 PyTorch 深度学习模型

    1.  如何为二类分类开发 MLP

    1.  如何为多类分类开发 MLP

    1.  如何为回归开发 MLP

    1.  如何为图像分类开发 CNN

### 你可以在 Python 中进行深度学习！

完成这个教程。最多花费 60 分钟！

**你不需要完全理解一切（至少现在不需要）**。你的目标是从头到尾运行教程并获得结果。你不需要在第一遍完全理解一切。在进行过程中列出你的问题。大量使用 API 文档来学习你正在使用的所有函数。

**你不需要首先了解数学**。数学是描述算法工作原理的紧凑方式，特别是线性代数、概率和微积分工具。这些不是你学习算法工作的唯一工具。你也可以使用代码，并通过不同的输入和输出探索算法行为。了解数学不会告诉你选择哪种算法或如何最佳配置它。只有通过精心控制的实验才能发现这一点。

**你不需要知道算法如何工作**。了解深度学习算法的限制和如何配置是重要的，但学习算法可以稍后再做。你需要在较长的时间内慢慢建立起这方面的知识。今天，先熟悉平台的使用。

**你不需要成为 Python 程序员**。如果你是新手，Python 语言的语法可能很直观。和其他语言一样，专注于函数调用（例如 function()）和赋值（例如 a = "b"）。这将让你快速掌握语言的基础知识。你是开发者；你知道如何快速掌握一门语言的基础知识。开始动手，详细内容稍后再深入了解。

**你不需要成为深度学习专家**。你可以稍后了解各种算法的优缺点，有很多教程可以帮助你了解深度学习项目的步骤。

## 1\. 如何安装 PyTorch

在这一节中，你将了解 PyTorch 是什么，如何安装它以及如何确认安装正确。

### 1.1\. Torch 和 PyTorch 是什么？

[PyTorch](https://github.com/pytorch/pytorch) 是一个由 Facebook 开发和维护的开源 Python 深度学习库。

该项目始于 2016 年，并迅速成为开发者和研究人员中流行的框架。

[Torch](https://github.com/torch/torch7)（*Torch7*）是一个用 C 语言编写的开源深度学习项目，通常通过 Lua 接口使用。它是 PyTorch 的前身项目，目前已不再积极开发。PyTorch 在其名称中包含了“*Torch*”，以示尊重先前的 torch 库，“*Py*”前缀表示新项目专注于 Python。

PyTorch API 简单且灵活，使其成为学术界和研究人员开发新深度学习模型和应用的最爱。广泛的使用导致了许多针对特定应用的扩展（例如文本、计算机视觉和音频数据），以及可能直接使用的预训练模型。因此，它可能是学术界使用的最受欢迎的库。

相比于更简单的接口如[Keras](https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/)，PyTorch 的灵活性以易用性为代价，特别是对于初学者。选择使用 PyTorch 而不是 Keras 会放弃一些易用性、稍微陡峭的学习曲线和更多的代码以获得更多的灵活性，也许还有一个更活跃的学术社区。

### 1.2. 如何安装 PyTorch

在安装 PyTorch 之前，请确保你已经安装了 Python，例如 Python 3.6 或更高版本。

如果你没有安装 Python，你可以使用 Anaconda 安装。这个教程将向你展示如何：

+   [如何使用 Anaconda 设置 Python 环境进行机器学习](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

有许多方法可以安装 PyTorch 开源深度学习库。

在工作站上安装 PyTorch 最常见，也许最简单的方法是使用 pip。

例如，在命令行中，你可以输入：

```py
sudo pip install torch
```

深度学习最流行的应用之一是[计算机视觉](https://machinelearningmastery.com/what-is-computer-vision/)，而 PyTorch 的计算机视觉包被称为“[torchvision](https://github.com/pytorch/vision/tree/master/torchvision)。”

强烈建议同时安装 torchvision，可以按照以下方法安装：

```py
sudo pip install torchvision
```

如果你更愿意使用更具体于你的平台或包管理器的安装方法，你可以在这里查看完整的安装说明：

+   [PyTorch 安装指南](https://pytorch.org/get-started/locally/)

现在无需设置 GPU。

本教程中的所有示例在现代 CPU 上都能正常运行。如果你想为 GPU 配置 PyTorch，可以在完成本教程后进行。不要分心！

### 1.3. 如何确认 PyTorch 已安装

一旦 PyTorch 安装完成，确认库是否成功安装并且可以开始使用是很重要的。

不要跳过这一步。

如果 PyTorch 未正确安装或在此步骤中出现错误，你将无法在后续运行示例。

创建一个名为 *versions.py* 的新文件，并将以下代码复制粘贴到文件中。

```py
# check pytorch version
import torch
print(torch.__version__)
```

保存文件，然后打开你的命令行并将目录更改为你保存文件的位置。

然后输入：

```py
python versions.py
```

然后你应该看到类似以下的输出：

```py
1.3.1
```

这确认了 PyTorch 已正确安装，并且我们都在使用相同的版本。

这还展示了如何从命令行运行 Python 脚本。我建议以这种方式从命令行运行所有代码，而不是从笔记本或 IDE 中运行。

## 2\. PyTorch 深度学习模型生命周期

在本节中，你将了解深度学习模型的生命周期和你可以用来定义模型的 PyTorch API。

模型具有生命周期，这个非常简单的知识为建模数据集和理解 PyTorch API 提供了基础。

生命周期中的五个步骤如下：

+   1\. 准备数据。

+   2\. 定义模型。

+   3\. 训练模型。

+   4\. 评估模型。

+   5\. 做出预测。

让我们逐步仔细看看每个步骤。

**注意**：有许多方法可以使用 PyTorch API 实现这些步骤，虽然我旨在展示最简单的、最常见的或最惯用的方法。

如果你发现更好的方法，请在下面的评论中告诉我。

### 第一步：准备数据

第一步是加载和准备你的数据。

神经网络模型需要数值输入数据和数值输出数据。

你可以使用标准 Python 库来加载和准备表格数据，如 CSV 文件。例如，可以使用 Pandas 加载 CSV 文件，使用 scikit-learn 的工具对类别数据（如类别标签）进行编码。

PyTorch 提供了 [Dataset 类](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)，你可以扩展和自定义它来加载你的数据集。

例如，你的数据集对象的构造函数可以加载你的数据文件（例如 CSV 文件）。然后，你可以重写 *__len__()* 函数，该函数用于获取数据集的长度（行数或样本数），以及 *__getitem__()* 函数，该函数用于通过索引获取特定样本。

在加载数据集时，你还可以执行任何所需的转换，如缩放或编码。

下面提供了一个自定义 *Dataset* 类的骨架。

```py
# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # store the inputs and outputs
        self.X = ...
        self.y = ...

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
```

加载后，PyTorch 提供了 [DataLoader 类](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)，用于在训练和评估模型期间遍历 *Dataset* 实例。

可以为训练数据集、测试数据集甚至验证数据集创建 *DataLoader* 实例。

[random_split() 函数](https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split) 可以用来将数据集拆分为训练集和测试集。拆分后，可以将 *Dataset* 的选择行提供给 DataLoader，并设置批大小和数据是否每个 epoch 应该被打乱。

例如，我们可以通过传入数据集中选择的行样本来定义 *DataLoader*。

```py
...
# create the dataset
dataset = CSVDataset(...)
# select rows from the dataset
train, test = random_split(dataset, [[...], [...]])
# create a data loader for train and test sets
train_dl = DataLoader(train, batch_size=32, shuffle=True)
test_dl = DataLoader(test, batch_size=1024, shuffle=False)
```

一旦定义了，可以对 *DataLoader* 进行枚举，每次迭代产生一批样本。

```py
...
# train the model
for i, (inputs, targets) in enumerate(train_dl):
	...
```

### 第二步：定义模型

下一步是定义模型。

在 PyTorch 中定义模型的惯用方法是定义一个扩展 [Module 类](https://pytorch.org/docs/stable/nn.html#module) 的类。

类的构造函数定义了模型的层和重写的`forward()`函数定义了如何通过模型的定义层进行前向传播的方法。

许多层都可用，例如用于全连接层的[Linear](https://pytorch.org/docs/stable/nn.html#torch.nn.Linear)，用于卷积层的[Conv2d](https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d)，以及用于池化层的[MaxPool2d](https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool2d)。

激活函数也可以作为层来定义，例如[ReLU](https://pytorch.org/docs/stable/nn.html#torch.nn.ReLU)、[Softmax](https://pytorch.org/docs/stable/nn.html#torch.nn.Softmax)和[Sigmoid](https://pytorch.org/docs/stable/nn.html#torch.nn.Sigmoid)。

下面是一个具有一个层的简单 MLP 模型的示例。

```py
# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.layer = Linear(n_inputs, 1)
        self.activation = Sigmoid()

    # forward propagate input
    def forward(self, X):
        X = self.layer(X)
        X = self.activation(X)
        return X
```

给定层的权重也可以在构造函数中定义层之后初始化。

常见的示例包括[Xavier](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_)和[He 权重](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_)初始化方案。例如：

```py
...
xavier_uniform_(self.layer.weight)
```

### 步骤 3：训练模型

训练过程要求您定义一个损失函数和一个优化算法。

常见的损失函数包括以下内容：

+   [二元交叉熵损失](https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss)：用于二元分类的二元交叉熵损失。

+   [交叉熵损失](https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss)：用于多类分类的交叉熵损失。

+   [均方误差损失](https://pytorch.org/docs/stable/nn.html#torch.nn.MSELoss)：用于回归的均方误差损失。

关于损失函数的更多信息，请参阅教程：

+   [深度学习神经网络的损失和损失函数](https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/)

使用随机梯度下降进行优化，标准算法由[SGD 类](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD)提供，尽管还有其他版本的算法可用，例如[Adam](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam)。

```py
# define the optimization
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
```

训练模型涉及枚举用于训练数据集的*DataLoader*。

首先，需要一个循环来迭代训练周期的数量。然后，需要一个内部循环来处理随机梯度下降的小批量。

```py
...
# enumerate epochs
for epoch in range(100):
    # enumerate mini batches
    for i, (inputs, targets) in enumerate(train_dl):
    	...
```

模型的每次更新都遵循相同的一般模式，包括：

+   清除最后的误差梯度。

+   通过模型的输入进行前向传递。

+   计算模型输出的损失。

+   通过模型进行误差反向传播。

+   更新模型以尝试减少损失。

例如：

```py
...
# clear the gradients
optimizer.zero_grad()
# compute the model output
yhat = model(inputs)
# calculate loss
loss = criterion(yhat, targets)
# credit assignment
loss.backward()
# update model weights
optimizer.step()
```

### 步骤 4：评估模型

一旦模型适合，就可以在测试数据集上进行评估。

这可以通过使用*DataLoader*来处理测试数据集并收集测试集的预测值来实现，然后将预测值与测试集的预期值进行比较，并计算性能指标。

```py
...
for i, (inputs, targets) in enumerate(test_dl):
    # evaluate the model on the test set
    yhat = model(inputs)
    ...
```

### 步骤 5：进行预测

适合的模型可以用来对新数据进行预测。

例如，您可能有一张单独的图片或一行数据，想要进行预测。

这要求您将数据封装在[PyTorch Tensor](https://pytorch.org/docs/stable/tensors.html)数据结构中。

Tensor 只是 PyTorch 版本的 NumPy 数组，用于保存数据。它还允许您执行模型图中的自动微分任务，比如在训练模型时调用*backward()*。

预测结果也将是一个 Tensor，尽管您可以通过[分离 Tensor](https://pytorch.org/docs/stable/autograd.html#torch.Tensor.detach)来从自动微分图中获取 NumPy 数组，并调用 NumPy 函数。

```py
...
# convert row to data
row = Variable(Tensor([row]).float())
# make prediction
yhat = model(row)
# retrieve numpy array
yhat = yhat.detach().numpy()
```

现在我们已经熟悉了 PyTorch API 的高级别和模型生命周期，让我们看看如何从头开始开发一些标准的深度学习模型。

## 3\. 如何开发 PyTorch 深度学习模型

在本节中，您将了解如何开发、评估和预测使用标准深度学习模型（包括多层感知器（MLP）和卷积神经网络（CNN））的方法。

多层感知器模型（简称 MLP）是一种标准的全连接神经网络模型。

它由节点层组成，其中每个节点与前一层的所有输出连接，并且每个节点的输出与下一层节点的所有输入连接。

MLP 是一个具有一个或多个全连接层的模型。这种模型适用于表格数据，即数据在表格或电子表格中的形式，每个变量对应一列，每个观测对应一行。您可能想用 MLP 探索三种预测建模问题，它们分别是二元分类、多类分类和回归。

让我们为每种情况在真实数据集上拟合一个模型。

**注意**：本节中的模型是有效的，但尚未经过优化。请尝试提升它们的性能。在下方评论区分享您的发现。

### 3.1\. 如何开发用于二元分类的 MLP

我们将使用电离层二元（两类）分类数据集来演示 MLP 进行二元分类。

此数据集涉及根据雷达返回预测大气中是否存在结构。

数据集将使用 Pandas 自动下载，但您也可以在此处了解更多信息。

+   [电离层数据集（csv）](https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv)。

+   [电离层数据集描述](https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.names)。

我们将使用一个[LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)将字符串标签编码为整数值 0 和 1。模型将在 67%的数据上进行拟合，其余 33%将用于评估，使用[train_test_split()函数](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)进行拆分。

使用‘*relu*’激活函数和‘*He Uniform*’权重初始化是一种良好的实践。这种组合有助于克服[梯度消失](https://machinelearningmastery.com/how-to-fix-vanishing-gradients-using-the-rectified-linear-activation-function/)的问题，尤其是在训练深度神经网络模型时。有关 ReLU 的更多信息，请参阅教程：

+   [对修正线性单元（ReLU）的温和介绍](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)

该模型预测类别 1 的概率，并使用 sigmoid 激活函数。模型使用随机梯度下降进行优化，并力求最小化[二元交叉熵损失](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)。

完整示例列在下方。

```py
# pytorch mlp for binary classification
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path, header=None)
        # store the inputs and outputs
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(10, 8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(8, 1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
         # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        return X

# prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl

# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(100):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

# prepare the data
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
model = MLP(34)
# train the model
train_model(train_dl, model)
# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)
# make a single prediction (expect class=1)
row = [1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300]
yhat = predict(row, model)
print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))
```

运行示例首先报告训练和测试数据集的形状，然后拟合模型并在测试数据集上进行评估。最后，对单行数据进行预测。

**注意**：由于算法或评估过程的随机性质，或者数值精度的差异，你的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑多次运行示例，并比较平均结果。

## 进一步阅读

本节提供了更多关于该主题的资源，以便你能够深入了解。

### 书籍

+   [Deep Learning](https://amzn.to/2Y8JuBv)，2016 年。

+   [Programming PyTorch for Deep Learning: Creating and Deploying Deep Learning Applications](https://amzn.to/2LA71Gq)，2018 年。

+   [Deep Learning with PyTorch](https://amzn.to/2Yw2s5q)，2020 年。

+   [Deep Learning for Coders with fastai and PyTorch: AI Applications Without a PhD](https://amzn.to/2P0MQDM)，2020 年。

### PyTorch 项目

+   [PyTorch 主页](https://pytorch.org/)。

+   [PyTorch 文档](https://pytorch.org/docs/stable/index.html)

+   [PyTorch 安装指南](https://pytorch.org/get-started/locally/)

+   [PyTorch，维基百科](https://en.wikipedia.org/wiki/PyTorch)。

+   [PyTorch 在 GitHub 上](https://github.com/pytorch/pytorch)。

### API

+   [torch.utils.data API](https://pytorch.org/docs/stable/data.html)。

+   [torch.nn API](https://pytorch.org/docs/stable/nn.html)。

+   [torch.nn.init API](https://pytorch.org/docs/stable/nn.init.html)。

+   [torch.optim API](https://pytorch.org/docs/stable/optim.html)。

+   [torch.Tensor API](https://pytorch.org/docs/stable/tensors.html)

## 总结

在本教程中，你发现了一个逐步指南，帮助你在 PyTorch 中开发深度学习模型。

具体来说，你学到了：

+   Torch 和 PyTorch 的区别，以及如何安装和确认 PyTorch 是否正常工作。

+   PyTorch 模型的五步生命周期以及如何定义、拟合和评估模型。

+   如何开发用于回归、分类和预测建模任务的 PyTorch 深度学习模型。

你有什么问题吗？

在下方评论中提出你的问题，我会尽力回答。
