# 保存和加载你的 PyTorch 模型

> 原文：[`machinelearningmastery.com/save-and-load-your-pytorch-models/`](https://machinelearningmastery.com/save-and-load-your-pytorch-models/)

深度学习模型是数据的数学抽象，其中涉及大量参数。训练这些参数可能需要数小时、数天甚至数周，但之后，你可以利用结果在新数据上应用。这在机器学习中称为推理。了解如何将训练好的模型保存在磁盘上，并在以后加载以进行推理是很重要的。在这篇文章中，你将学习如何将 PyTorch 模型保存到文件中，并重新加载以进行预测。阅读完这一章后，你将知道：

+   PyTorch 模型中的状态和参数是什么

+   如何保存模型状态

+   如何加载模型状态

**启动你的项目**，参阅我的书籍[《PyTorch 深度学习》](https://machinelearningmastery.com/deep-learning-with-pytorch/)。它提供了**自学教程**和**可运行的代码**。

开始吧。![](img/b616b0be4904ab3e5846c25d2e4373fc.png)

保存和加载你的 PyTorch 模型

照片由[Joseph Chan](https://unsplash.com/photos/Wwtq9Lvk_ZE)提供。保留一些权利。

## 概述

本文分为三部分，它们是

+   构建一个示例模型

+   PyTorch 模型内部包含什么

+   访问模型的`state_dict`

## 构建一个示例模型

让我们从一个非常简单的 PyTorch 模型开始。这是一个基于鸢尾花数据集的模型。你将使用 scikit-learn 加载数据集（目标是整数标签 0、1 和 2），并为这个多类分类问题训练一个神经网络。在这个模型中，你使用了 log softmax 作为输出激活函数，以便与负对数似然损失函数结合。这相当于没有输出激活函数结合交叉熵损失函数。

```py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data into NumPy arrays
data = load_iris()
X, y = data["data"], data["target"]

# convert NumPy array into PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# PyTorch model
class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 8)
        self.act = nn.ReLU()
        self.output = nn.Linear(8, 3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.logsoftmax(self.output(x))
        return x

model = Multiclass()

# loss metric and optimizer
loss_fn = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# prepare model and training parameters
n_epochs = 100
batch_size = 5
batch_start = torch.arange(0, len(X), batch_size)

# training loop
for epoch in range(n_epochs):
    for start in batch_start:
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
```

使用如此简单的模型和小数据集，不应该花费太长时间完成训练。之后，我们可以通过使用测试集来验证该模型是否有效：

```py
...
y_pred = model(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
print("Accuracy: %.2f" % acc)
```

例如，它会打印出

```py
Accuracy: 0.96
```

### 想要开始使用 PyTorch 进行深度学习吗？

现在立即注册我的免费电子邮件速成课程（附示例代码）。

点击注册，并获取课程的免费 PDF 电子书版本。

## PyTorch 模型内部包含什么

PyTorch 模型是 Python 中的一个对象。它包含一些深度学习构建块，例如各种层和激活函数。它还知道如何将它们连接起来，以便从输入张量中生成输出。模型的算法在创建时是固定的，但它有可训练的参数，这些参数在训练循环中应被修改，以使模型更加准确。

你看到如何在设置优化器以进行训练循环时获取模型参数，即

```py
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

函数`model.parameters()`为你提供了一个生成器，依次引用每一层的可训练参数，形式为 PyTorch 张量。因此，你可以复制这些参数或覆盖它们，例如：

```py
# create a new model
newmodel = Multiclass()
# ask PyTorch to ignore autograd on update and overwrite parameters
with torch.no_grad():
    for newtensor, oldtensor in zip(newmodel.parameters(), model.parameters()):
        newtensor.copy_(oldtensor)
# test with new model using copied tensor
y_pred = newmodel(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
print("Accuracy: %.2f" % acc)
```

结果应该与之前完全相同，因为你通过复制参数使两个模型变得完全相同。

然而，情况并非总是如此。一些模型具有**不可训练的参数**。一个例子是许多卷积神经网络中常见的批归一化层。它的作用是在前一层产生的张量上应用归一化，并将归一化后的张量传递给下一层。它有两个参数：均值和标准差，这些参数在训练循环中从输入数据中学习，但不能被优化器训练。因此，这些参数不是`model.parameters()`的一部分，但同样重要。

## 访问模型的`state_dict`

要访问模型的所有参数，无论是否可训练，你可以从`state_dict()`函数中获取。从上面的模型中，你可以得到以下内容：

```py
import pprint
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(model.state_dict())
```

上面的模型产生了以下结果：

```py
OrderedDict([   (   'hidden.weight',
                    tensor([[ 0.1480,  0.0336,  0.3425,  0.2832],
        [ 0.5265,  0.8587, -0.7023, -1.1149],
        [ 0.1620,  0.8440, -0.6189, -0.6513],
        [-0.1559,  0.0393, -0.4701,  0.0825],
        [ 0.6364, -0.6622,  1.1150,  0.9162],
        [ 0.2081, -0.0958, -0.2601, -0.3148],
        [-0.0804,  0.1027,  0.7363,  0.6068],
        [-0.4101, -0.3774, -0.1852,  0.1524]])),
                (   'hidden.bias',
                    tensor([ 0.2057,  0.7998, -0.0578,  0.1041, -0.3903, -0.4521, -0.5307, -0.1532])),
                (   'output.weight',
                    tensor([[-0.0954,  0.8683,  1.0667,  0.2382, -0.4245, -0.0409, -0.2587, -0.0745],
        [-0.0829,  0.8642, -1.6892, -0.0188,  0.0420, -0.1020,  0.0344, -0.1210],
        [-0.0176, -1.2809, -0.3040,  0.1985,  0.2423,  0.3333,  0.4523, -0.1928]])),
                ('output.bias', tensor([ 0.0998,  0.6360, -0.2990]))])
```

它被称为`state_dict`，因为模型的所有状态变量都在这里。它是来自 Python 内置`collections`模块的一个`OrderedDict`对象。PyTorch 模型中的所有组件都有一个名称，参数也是如此。`OrderedDict`对象允许你通过匹配名称将权重正确地映射回参数。

这就是你应该如何保存和加载模型：将模型状态提取到`OrderedDict`中，序列化并保存到磁盘。在推理时，你首先创建一个模型（不进行训练），然后加载状态。在 Python 中，序列化的本机格式是 pickle：

```py
import pickle

# Save model
with open("iris-model.pickle", "wb") as fp:
    pickle.dump(model.state_dict(), fp)

# Create new model and load states
newmodel = Multiclass()
with open("iris-model.pickle", "rb") as fp:
    newmodel.load_state_dict(pickle.load(fp))

# test with new model using copied tensor
y_pred = newmodel(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
print("Accuracy: %.2f" % acc)
```

你知道它有效，因为你没有训练的模型产生了与训练过的模型相同的结果。

确实，推荐的方式是使用 PyTorch API 来保存和加载状态，而不是手动使用 pickle：

```py
# Save model
torch.save(model.state_dict(), "iris-model.pth")

# Create new model and load states
newmodel = Multiclass()
newmodel.load_state_dict(torch.load("iris-model.pth"))

# test with new model using copied tensor
y_pred = newmodel(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
print("Accuracy: %.2f" % acc)
```

`*.pth`文件实际上是由 PyTorch 创建的一些 pickle 文件的压缩文件。推荐这样做，因为 PyTorch 可以在其中存储额外的信息。请注意，你仅保存了状态而不是模型。你仍然需要使用 Python 代码创建模型并将状态加载到其中。如果你还希望保存模型本身，你可以传入整个模型，而不是状态：

```py
# Save model
torch.save(model, "iris-model-full.pth")

# Load model
newmodel = torch.load("iris-model-full.pth")

# test with new model using copied tensor
y_pred = newmodel(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
print("Accuracy: %.2f" % acc)
```

但请记住，由于 Python 语言的特性，这并不会免除你需要保存模型代码的责任。上面的`newmodel`对象是你之前定义的`Multiclass`类的一个实例。当你从磁盘加载模型时，Python 需要详细知道这个类是如何定义的。如果你仅运行`torch.load()`这一行脚本，你将看到以下错误信息：

```py
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
File "/.../torch/serialization.py", line 789, in load
return _load(opened_zipfile, map_location, pickle_module, **pickle_load_args)
File "/.../torch/serialization.py", line 1131, in _load
result = unpickler.load()
File "/.../torch/serialization.py", line 1124, in find_class
return super().find_class(mod_name, name)
AttributeError: Can't get attribute 'Multiclass' on <module '__main__' (built-in)>
```

这就是为什么推荐只保存状态字典而不是整个模型的原因。

将所有内容整合在一起，以下是展示如何创建模型、训练它并保存到磁盘的完整代码：

```py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data into NumPy arrays
data = load_iris()
X, y = data["data"], data["target"]

# convert NumPy array into PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# PyTorch model
class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 8)
        self.act = nn.ReLU()
        self.output = nn.Linear(8, 3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.logsoftmax(self.output(x))
        return x

model = Multiclass()

# loss metric and optimizer
loss_fn = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# prepare model and training parameters
n_epochs = 100
batch_size = 5
batch_start = torch.arange(0, len(X), batch_size)

# training loop
for epoch in range(n_epochs):
    for start in batch_start:
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

# Save model
torch.save(model.state_dict(), "iris-model.pth")
```

以下是如何从磁盘加载模型并进行推理的步骤：

```py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data into NumPy arrays
data = load_iris()
X, y = data["data"], data["target"]

# convert NumPy array into PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# PyTorch model
class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 8)
        self.act = nn.ReLU()
        self.output = nn.Linear(8, 3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.logsoftmax(self.output(x))
        return x

# Create new model and load states
model = Multiclass()
with open("iris-model.pickle", "rb") as fp:
    model.load_state_dict(pickle.load(fp))

# Run model for inference
y_pred = model(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
print("Accuracy: %.2f" % acc)
```

## 进一步阅读

本节提供了更多关于此主题的资源，帮助你深入了解。

+   [PyTorch 教程中的保存和加载模型](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

## 总结

在这篇文章中，你学习了如何将训练好的 PyTorch 模型保存在磁盘中以及如何重新使用它。特别是，你学到了

+   在 PyTorch 模型中，什么是参数和状态

+   如何将模型的所有必要状态保存到磁盘

+   如何从保存的状态重建一个可用的模型
