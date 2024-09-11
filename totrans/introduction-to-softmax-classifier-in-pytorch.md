# PyTorch 中的 Softmax 分类器介绍

> 原文：[`machinelearningmastery.com/introduction-to-softmax-classifier-in-pytorch/`](https://machinelearningmastery.com/introduction-to-softmax-classifier-in-pytorch/)

虽然逻辑回归分类器用于二类分类，softmax 分类器是一种监督学习算法，主要用于多类别情况。

Softmax 分类器通过为每个类别分配概率分布来工作。具有最高概率的类别的概率分布被归一化为 1，其他所有概率则相应地缩放。

类似地，softmax 函数将神经元的输出转换为类别上的概率分布。它具有以下属性：

1.  它与逻辑 sigmoid 有关，逻辑 sigmoid 用于概率建模，并具有类似的属性。

1.  它的取值范围在 0 到 1 之间，0 表示不可能发生的事件，1 表示必然发生的事件。

1.  对输入`x`的 softmax 的导数可以解释为预测给定输入`x`时某个特定类别被选择的可能性。

在本教程中，我们将构建一个一维的 softmax 分类器并探索其功能。特别地，我们将学习：

+   如何使用 Softmax 分类器进行多类分类。

+   如何在 PyTorch 中构建和训练 Softmax 分类器。

+   如何分析模型在测试数据上的结果。

**启动你的项目**，参考我的书籍[《深度学习与 PyTorch》](https://machinelearningmastery.com/deep-learning-with-pytorch/)。它提供了**自学教程**和**可运行的代码**。

让我们开始吧！[](../Images/57fde22c2688d158fe47a133006f4715.png)

PyTorch 中的 Softmax 分类器介绍。

图片由[Julia Caesar](https://unsplash.com/photos/HTSpgMng5ys)提供。版权所有。

## 概述

本教程分为四个部分；它们是

+   准备数据集

+   将数据集加载到 DataLoader 中

+   使用`nn.Module`构建模型

+   训练分类器

## 准备数据集

首先，让我们构建我们的数据集类以生成一些数据样本。与之前的实验不同，你将为多个类别生成数据。然后你将训练 softmax 分类器并在这些数据样本上进行预测，之后使用它对测试数据进行预测。

以下内容，我们基于一个输入变量生成四个类别的数据：

```py
import torch
from torch.utils.data import Dataset

class toy_data(Dataset):
    "The data for multi-class classification"
    def __init__(self):
        # single input
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        # multi-class output
        self.y = torch.zeros(self.x.shape[0])
        self.y[(self.x > -2.0)[:, 0] * (self.x < 0.0)[:, 0]] = 1 self.y[(self.x >= 0.0)[:, 0] * (self.x < 2.0)[:, 0]] = 2 self.y[(self.x >= 2.0)[:, 0]] = 3
        self.y = self.y.type(torch.LongTensor)
        self.len = self.x.shape[0]

    def __getitem__(self, idx):
        "accessing one element in the dataset by index"
        return self.x[idx], self.y[idx] 

    def __len__(self):
        "size of the entire dataset"
        return self.len
```

让我们创建数据对象并检查前十个数据样本及其标签。

```py
# Create the dataset object and check a few samples
data = toy_data()
print("first ten data samples: ", data.x[0:10])
print("first ten data labels: ", data.y[0:10])
```

这将打印：

```py
first ten data samples:  tensor([[-3.0000],
        [-2.9000],
        [-2.8000],
        [-2.7000],
        [-2.6000],
        [-2.5000],
        [-2.4000],
        [-2.3000],
        [-2.2000],
        [-2.1000]])
first ten data labels:  tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```

## 使用`nn.Module`构建 Softmax 模型

你将使用 PyTorch 中的`nn.Module`来构建自定义的 softmax 模块。这与之前教程中为逻辑回归构建的自定义模块类似。那么这里有什么不同呢？之前你使用`1`代替`n_outputs`进行二分类，而在这里我们将定义四个类别进行多分类。其次，在`forward()`函数中，模型不使用逻辑函数进行预测。

```py
class Softmax(torch.nn.Module):
    "custom softmax module"
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        pred = self.linear(x)
        return pred
```

现在，让我们创建模型对象。它接受一个一维向量作为输入，并对四个不同的类别进行预测。我们还来检查一下参数是如何初始化的。

```py
# call Softmax Classifier
model_softmax = Softmax(1, 4)
model_softmax.state_dict()
```

这将打印

```py
OrderedDict([('linear.weight',
              tensor([[-0.0075],
                      [ 0.5364],
                      [-0.8230],
                      [-0.7359]])),
             ('linear.bias', tensor([-0.3852,  0.2682, -0.0198,  0.7929]))])
```

### 想要开始使用 PyTorch 进行深度学习吗？

现在参加我的免费电子邮件速成课程（附带示例代码）。

点击注册，还可以获得课程的免费 PDF 电子书版本。

## 训练模型

结合随机梯度下降，你将使用交叉熵损失进行模型训练，并将学习率设置为 0.01。你将数据加载到数据加载器中，并将批量大小设置为 2。

```py
...
from torch.utils.dataimport DataLoader

# define loss, optimizier, and dataloader
optimizer = torch.optim.SGD(model_softmax.parameters(), lr = 0.01)
criterion = torch.nn.CrossEntropyLoss()
train_loader = DataLoader(dataset = data, batch_size = 2)
```

既然一切都已设置好，我们来训练我们的模型 100 次迭代。

```py
# Train the model
Loss = []
epochs = 100
for epoch in range(epochs):
    for x, y in train_loader:
        optimizer.zero_grad()
        y_pred = model_softmax(x)
        loss = criterion(y_pred, y)
        Loss.append(loss)
        loss.backward()
        optimizer.step()
print("Done!")
```

训练循环完成后，你调用模型上的 `max()` 方法来进行预测。参数 `1` 返回相对于轴一的最大值，即从每列返回最大值的索引。

```py
# Make predictions on test data
pred_model =  model_softmax(data.x)
_, y_pred = pred_model.max(1)
print("model predictions on test data:", y_pred)
```

从上面，你应该可以看到：

```py
model predictions on test data: tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
```

这些是模型在测试数据上的预测结果。

让我们也检查一下模型的准确性。

```py
# check model accuracy
correct = (data.y == y_pred).sum().item()
acc = correct / len(data)
print("model accuracy: ", acc)
```

在这种情况下，你可能会看到

```py
model accuracy:  0.9833333333333333
```

在这个简单的模型中，如果你训练得更久，你会看到准确率接近 1。

将所有内容整合在一起，以下是完整的代码：

```py
import torch
from torch.utils.data import Dataset, DataLoader

class toy_data(Dataset):
    "The data for multi-class classification"
    def __init__(self):
        # single input
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        # multi-class output
        self.y = torch.zeros(self.x.shape[0])
        self.y[(self.x > -2.0)[:, 0] * (self.x < 0.0)[:, 0]] = 1
        self.y[(self.x >= 0.0)[:, 0] * (self.x < 2.0)[:, 0]] = 2
        self.y[(self.x >= 2.0)[:, 0]] = 3
        self.y = self.y.type(torch.LongTensor)
        self.len = self.x.shape[0]

    def __getitem__(self, idx):
        "accessing one element in the dataset by index"
        return self.x[idx], self.y[idx] 

    def __len__(self):
        "size of the entire dataset"
        return self.len

# Create the dataset object and check a few samples
data = toy_data()
print("first ten data samples: ", data.x[0:10])
print("first ten data labels: ", data.y[0:10])

class Softmax(torch.nn.Module):
    "custom softmax module"
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        pred = self.linear(x)
        return pred

# call Softmax Classifier
model_softmax = Softmax(1, 4)
model_softmax.state_dict()

# define loss, optimizier, and dataloader
optimizer = torch.optim.SGD(model_softmax.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
train_loader = DataLoader(dataset=data, batch_size=2)

# Train the model
Loss = []
epochs = 100
for epoch in range(epochs):
    for x, y in train_loader:
        optimizer.zero_grad()
        y_pred = model_softmax(x)
        loss = criterion(y_pred, y)
        Loss.append(loss)
        loss.backward()
        optimizer.step()
print("Done!")

# Make predictions on test data
pred_model =  model_softmax(data.x)
_, y_pred = pred_model.max(1)
print("model predictions on test data:", y_pred)

# check model accuracy
correct = (data.y == y_pred).sum().item()
acc = correct / len(data)
print("model accuracy: ", acc)
```

## 总结

在本教程中，你学习了如何构建一个简单的一维 Softmax 分类器。特别地，你学习了：

+   如何使用 Softmax 分类器进行多类分类。

+   如何在 PyTorch 中构建和训练 Softmax 分类器。

+   如何分析模型在测试数据上的结果。
