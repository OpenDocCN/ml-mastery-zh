# 逐步开发你的第一个 PyTorch 神经网络

> 原文：[`machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/`](https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/)

PyTorch 是一个强大的用于构建深度学习模型的 Python 库。它提供了定义和训练神经网络以及用于推理的一切所需工具。你不需要写很多代码就能完成所有这些。在这篇文章中，你将了解如何使用 PyTorch 在 Python 中创建你的第一个深度学习神经网络模型。完成本文后，你将了解到：

+   如何加载 CSV 数据集并准备用于 PyTorch 使用

+   如何在 PyToch 中定义多层感知器模型

+   如何在验证数据集上训练和评估 PyToch 模型

**使用我的书 [Deep Learning with PyTorch](https://machinelearningmastery.com/deep-learning-with-pytorch/) 快速启动你的项目**。它提供了**带有工作代码的自学教程**。

让我们开始吧！![](img/c98e1f3e5f9db5608610e28f74622e06.png)

逐步开发你的第一个 PyTorch 神经网络

照片由 [drown_ in_city](https://unsplash.com/photos/V2DylCx9kkc) 拍摄。部分权利保留。

## 概述

需要的代码不多。你将会慢慢过一遍，这样你将来就会知道如何创建自己的模型。本文你将学到的步骤如下：

+   加载数据

+   定义 PyToch 模型

+   定义损失函数和优化器

+   运行训练循环

+   评估模型

+   进行预测

## 加载数据

第一步是定义本文中打算使用的函数和类。你将使用 NumPy 库加载你的数据集，并使用 PyTorch 库进行深度学习模型。

下面列出了所需的导入：

```py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
```

现在你可以加载你的数据集了。

在本文中，你将使用 Pima Indians 糖尿病发作数据集。这是自该领域早期以来的标准机器学习数据集。它描述了 Pima 印第安人的患者医疗记录数据及其在五年内是否有糖尿病发作。

这是一个二元分类问题（糖尿病的发作为 1，否则为 0）。描述每个患者的所有输入变量都被转换为数值。这使得它可以直接与期望数值输入和输出的神经网络一起使用，并且是我们在 PyTorch 中首次尝试神经网络的理想选择。

你也可以在这里下载它 [here](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)。

下载数据集并将其放在本地工作目录中，与你的 Python 文件位于同一位置。将其保存为文件名 `pima-indians-diabetes.csv`。打开文件后，你应该看到类似以下的数据行：

```py
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,66,23,94,28.1,0.167,21,0
0,137,40,35,168,43.1,2.288,33,1
...
```

你现在可以使用 NumPy 函数`loadtxt()`将文件作为数字矩阵加载。共有八个输入变量和一个输出变量（最后一列）。你将学习一个模型来将输入变量的行（$X$）映射到输出变量（$y$），这通常总结为$y = f(X)$。变量总结如下：

输入变量（$X$）：

1.  怀孕次数

1.  口服葡萄糖耐量测试中 2 小时的血浆葡萄糖浓度

1.  舒张压（mm Hg）

1.  三头肌皮肤褶皱厚度（mm）

1.  2 小时血清胰岛素（μIU/ml）

1.  身体质量指数（体重 kg/（身高 m）²）

1.  糖尿病家族史功能

1.  年龄（岁）

输出变量（$y$）：

+   类别标签（0 或 1）

一旦 CSV 文件被加载到内存中，你可以将数据列拆分为输入变量和输出变量。

数据将存储在一个二维数组中，其中第一个维度是行，第二个维度是列，例如（行，列）。你可以通过使用标准的 NumPy 切片操作符“`:`”将数组分割成两个数组。你可以通过切片`0:8`从索引 0 到索引 7 选择前八列。然后，你可以通过索引 8 选择输出列（第 9 个变量）。

```py
...

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
```

但这些数据应该先转换为 PyTorch 张量。一个原因是 PyTorch 通常使用 32 位浮点数，而 NumPy 默认使用 64 位浮点数。大多数操作中不允许混用。转换为 PyTorch 张量可以避免可能引起问题的隐式转换。你也可以借此机会纠正形状以符合 PyTorch 的预期，例如，优选$n\times 1$矩阵而不是$n$-向量。

要转换，请从 NumPy 数组创建张量：

```py
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
```

你现在已经准备好定义你的神经网络模型了。

### 想要开始使用 PyTorch 进行深度学习吗？

现在就来参加我的免费电子邮件速成课程（附有示例代码）。

点击注册并获取课程的免费 PDF 电子书版本。

## 定义模型

确实，在 PyTorch 中有两种定义模型的方法。目标是将其制作成一个接受输入并返回输出的函数。

一个模型可以定义为一系列层。你可以创建一个`Sequential`模型，其中列出这些层。为了确保正确，首先需要确认第一层具有正确数量的输入特征。在这个例子中，你可以为八个输入变量指定输入维度`8`作为一个向量。

确定层的其他参数或模型需要多少层并不是一个简单的问题。你可以使用启发式方法来帮助你设计模型，或者参考其他人在处理类似问题时的设计。通常，最佳的神经网络结构是通过试错实验过程找到的。一般来说，你需要一个足够大的网络来捕捉问题的结构，但又要足够小以提高速度。在这个例子中，我们使用一个具有三层的全连接网络结构。

在 PyTorch 中使用`Linear`类定义全连接层或密集层。它简单地意味着类似于矩阵乘法的操作。您可以将输入的数量指定为第一个参数，将输出的数量指定为第二个参数。输出的数量有时被称为层中的神经元数或节点数。

在该层之后，您还需要一个激活函数**after**。如果未提供，您只需将矩阵乘法的输出传递给下一步，或者有时您称之为线性激活，因此该层的名称如此。

在这个例子中，您将在前两个层上使用修正线性单元激活函数（称为 ReLU），并在输出层上使用 sigmoid 函数。

输出层上的 sigmoid 函数确保输出在 0 和 1 之间，这很容易映射到类 1 的概率或通过 0.5 的截止阈值划分为任一类的硬分类。过去，您可能已经在所有层上使用了 sigmoid 和 tanh 激活函数，但事实证明，sigmoid 激活可能导致深度神经网络中的梯度消失问题，而 ReLU 激活则在速度和准确性方面表现更佳。

您可以通过添加每一层来将所有这些部分组合在一起，例如：

+   该模型期望具有 8 个变量的数据行（第一层的第一个参数设置为`8`）

+   第一个隐藏层有 12 个神经元，后面跟着一个 ReLU 激活函数

+   第二个隐藏层有 8 个神经元，后面跟着另一个 ReLU 激活函数

+   输出层有一个神经元，后面跟着一个 sigmoid 激活函数

```py
...

model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
```

你可以通过以下方式打印模型：

```py
print(model)
```

您将看到：

```py
Sequential(
  (0): Linear(in_features=8, out_features=12, bias=True)
  (1): ReLU()
  (2): Linear(in_features=12, out_features=8, bias=True)
  (3): ReLU()
  (4): Linear(in_features=8, out_features=1, bias=True)
  (5): Sigmoid()
)
```

您可以自由更改设计并查看是否比本文后续部分获得更好或更差的结果。

但请注意，在 PyTorch 中，有一种更冗长的创建模型的方式。上面的模型可以作为从`nn.Module`继承的 Python `class`来创建：

```py
...

class PimaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 12)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(12, 8)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(8, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x

model = PimaClassifier()
print(model)
```

在这种情况下，打印出的模型将是：

```py
PimaClassifier(
  (hidden1): Linear(in_features=8, out_features=12, bias=True)
  (act1): ReLU()
  (hidden2): Linear(in_features=12, out_features=8, bias=True)
  (act2): ReLU()
  (output): Linear(in_features=8, out_features=1, bias=True)
  (act_output): Sigmoid()
)
```

在这种方法中，一个类需要在构造函数中定义所有的层，因为在创建时需要准备所有的组件，但是输入尚未提供。请注意，您还需要调用父类的构造函数（`super().__init__()`行）来启动您的模型。您还需要在类中定义一个`forward()`函数，以告诉输入张量`x`如何生成返回的输出张量。

您可以从上面的输出中看到，模型记住了您如何调用每一层。

## 训练准备

一个定义好的模型已经准备好进行训练，但你需要指定训练的目标。在这个例子中，数据有输入特征$X$和输出标签$y$。你希望神经网络模型产生一个尽可能接近$y$的输出。训练网络意味着找到将输入映射到数据集中输出的最佳权重集。损失函数是用来衡量预测距离$y$的指标。在这个例子中，你应该使用二元交叉熵，因为这是一个二分类问题。

一旦你决定了损失函数，你还需要一个优化器。优化器是你用来逐步调整模型权重以产生更好输出的算法。可以选择许多优化器，在这个例子中使用的是 Adam。这个流行的梯度下降版本可以自动调整自己，并在广泛的问题中提供良好的结果。

```py
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

优化器通常具有一些配置参数。最重要的是学习率`lr`。但所有优化器都需要知道优化的内容。因此，你需要传递`model.parameters()`，这是你创建的模型中所有参数的生成器。

## 训练模型

你已经定义了你的模型、损失度量和优化器。通过在一些数据上执行模型，它已经准备好进行训练。

训练神经网络模型通常需要轮次和批次。这些术语用于描述数据如何传递给模型：

+   **轮次**：将整个训练数据集传递给模型一次。

+   **批次**：一个或多个传递给模型的样本，从中梯度下降算法将执行一次迭代。

简而言之，整个数据集被分成批次，你通过训练循环将批次一个一个传递给模型。一旦你用完了所有批次，你就完成了一轮。然后，你可以用相同的数据集重新开始，开始第二轮，继续优化模型。这个过程会重复，直到你对模型的输出感到满意为止。

批次的大小受系统内存的限制。此外，所需的计算量与批次的大小成线性比例。多个轮次中的批次数量决定了你进行梯度下降以优化模型的次数。这是一个权衡，你希望有更多的梯度下降迭代以便产生更好的模型，但同时又不希望训练时间过长。轮次和批次的大小可以通过试验和错误的方法来选择。

训练模型的目标是确保它学习到一个足够好的输入数据到输出分类的映射。它不会是完美的，错误是不可避免的。通常，你会看到在后期轮次中错误的数量减少，但最终会趋于平稳。这被称为模型收敛。

构建训练循环的最简单方法是使用两个嵌套的 for 循环，一个用于轮次，一个用于批次：

```py
n_epochs = 100
batch_size = 10

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')
```

当运行时，它将打印以下内容：

```py
Finished epoch 0, latest loss 0.6271069645881653
Finished epoch 1, latest loss 0.6056771874427795
Finished epoch 2, latest loss 0.5916517972946167
Finished epoch 3, latest loss 0.5822567939758301
Finished epoch 4, latest loss 0.5682642459869385
Finished epoch 5, latest loss 0.5640913248062134
...
```

## 评估模型

你已经在整个数据集上训练了我们的神经网络，你可以在相同的数据集上评估网络的性能。这将只给你一个关于你如何建模数据集的概念（例如，训练准确率），但无法了解算法在新数据上的表现。这是为了简化，但理想情况下，你可以将数据分为训练和测试数据集，用于模型的训练和评估。

你可以按照训练时调用模型的方式，在训练数据集上评估你的模型。这将为每个输入生成预测，但你仍然需要计算一个评价分数。这个分数可以与你的损失函数相同，也可以不同。因为你正在进行二分类，你可以通过将输出（范围在 0 到 1 之间的浮点数）转换为整数（0 或 1）来使用准确率作为评价分数，并与我们已知的标签进行比较。

这可以如下进行：

```py
# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(X)

accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")
```

`round()` 函数将浮点数舍入到最接近的整数。`==` 操作符进行比较并返回一个布尔张量，这可以转换为浮点数 1.0 和 0.0。`mean()` 函数将提供 1 的数量（即，预测与标签匹配）除以样本总数。`no_grad()` 上下文是可选的，但建议使用，这样你可以让 `y_pred` 不用记住它是如何得出这个数的，因为你不会对其进行微分。

把所有内容整合在一起，以下是完整的代码。

```py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# define the model
model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)
print(model)

# train the model
loss_fn   = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 10

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')

# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")
```

你可以将所有代码复制到你的 Python 文件中，并将其保存为“`pytorch_network.py`”在与你的数据文件“`pima-indians-diabetes.csv`”相同的目录下。然后，你可以从命令行运行 Python 文件作为脚本。

运行这个示例时，你应该会看到每个时期的训练循环进展，最后打印出最终的准确率。理想情况下，你希望损失降低到零，准确率达到 1.0（例如，100%）。但对于大多数非平凡的机器学习问题，这是不可能的。相反，你的模型总会有一些误差。目标是选择一个模型配置和训练配置，以实现给定数据集上最低的损失和最高的准确率。

神经网络是随机算法，这意味着相同的数据上，相同的算法每次运行代码时都可以训练出不同的模型，具有不同的技能。这是一种特性，而不是错误。模型性能的差异意味着，为了获得对模型性能的合理近似，你可能需要多次训练，并计算准确率分数的平均值。例如，下面是重新运行示例五次得到的准确率分数：

```py
Accuracy: 0.7604166865348816
Accuracy: 0.7838541865348816
Accuracy: 0.7669270634651184
Accuracy: 0.7721354365348816
Accuracy: 0.7669270634651184
```

你可以看到所有的准确率分数大约在 77%左右。

## 做出预测

你可以修改上述示例，并将其用于生成训练数据集上的预测，假装这是一个你之前未见过的新数据集。进行预测就像调用模型作为一个函数一样简单。你在输出层上使用了 sigmoid 激活函数，因此预测值将在 0 和 1 之间的范围内表示概率。你可以通过四舍五入将它们轻松转换为这个分类任务的明确二元预测。例如：

```py
...

# make probability predictions with the model
predictions = model(X)
# round predictions
rounded = predictions.round()
```

另外，你可以将概率转换为 0 或 1，直接预测明确的类别；例如：

```py
...
# make class predictions with the model
predictions = (model(X) > 0.5).int()
```

下面的完整示例对数据集中的每个示例进行预测，然后打印出数据集前五个示例的输入数据、预测类别和预期类别。

```py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# define the model
class PimaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 12)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(12, 8)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(8, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x

model = PimaClassifier()
print(model)

# train the model
loss_fn   = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 10

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# compute accuracy
y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")

# make class predictions with the model
predictions = (model(X) > 0.5).int()
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
```

这段代码使用了不同的构建模型的方法，但功能上应该与之前相同。模型训练完成后，会对数据集中所有示例进行预测，并打印出前五个示例的输入行和预测类别值，并与预期类别值进行比较。你可以看到大多数行的预测是正确的。实际上，根据你在上一节对模型性能的估计，你可以预期大约 77%的行会被正确预测。

```py
[6.0, 148.0, 72.0, 35.0, 0.0, 33.599998474121094, 0.6269999742507935, 50.0] => 1 (expected 1)
[1.0, 85.0, 66.0, 29.0, 0.0, 26.600000381469727, 0.35100001096725464, 31.0] => 0 (expected 0)
[8.0, 183.0, 64.0, 0.0, 0.0, 23.299999237060547, 0.671999990940094, 32.0] => 1 (expected 1)
[1.0, 89.0, 66.0, 23.0, 94.0, 28.100000381469727, 0.16699999570846558, 21.0] => 0 (expected 0)
[0.0, 137.0, 40.0, 35.0, 168.0, 43.099998474121094, 2.2880001068115234, 33.0] => 1 (expected 1)
```

## 进一步阅读

要了解更多关于深度学习和 PyTorch 的信息，可以查看以下内容：

### 书籍

+   Ian Goodfellow, Yoshua Bengio 和 Aaron Courville. [深度学习](https://www.amazon.com/dp/0262035618)。MIT Press, 2016。

    ([在线版本](http://www.deeplearningbook.org))。

### API

+   [PyTorch 文档](https://pytorch.org/docs/stable/index.html)

## 总结

在这篇文章中，你了解了如何使用 PyTorch 创建你的第一个神经网络模型。具体来说，你学习了使用 PyTorch 一步一步创建神经网络或深度学习模型的关键步骤，包括：

+   如何加载数据

+   如何在 PyTorch 中定义神经网络

+   如何在数据上训练模型

+   如何评估模型

+   如何使用模型进行预测
