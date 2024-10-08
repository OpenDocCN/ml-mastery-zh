# 在 PyTorch 中构建回归模型

> 原文：[`machinelearningmastery.com/building-a-regression-model-in-pytorch/`](https://machinelearningmastery.com/building-a-regression-model-in-pytorch/)

PyTorch 库用于深度学习。深度学习模型的一些应用是解决回归或分类问题。

在这篇文章中，你将发现如何使用 PyTorch 开发和评估回归问题的神经网络模型。

完成这篇文章后，你将了解：

+   如何从 scikit-learn 加载数据并将其调整为 PyTorch 模型

+   如何使用 PyTorch 创建一个回归问题的神经网络

+   如何通过数据准备技术提高模型性能

**启动你的项目**，参考我的书[《使用 PyTorch 的深度学习》](https://machinelearningmastery.com/deep-learning-with-pytorch/)。它提供了**自学教程**和**可运行的代码**。

让我们开始吧。![](img/918e52e81c4c7df61a4008105b43a255.png)

在 PyTorch 中构建回归模型，照片由[Sam Deng](https://unsplash.com/photos/2bJGj7sIclQ)提供。保留部分权利。

## 数据集描述

本教程中使用的数据集是[加州住房数据集](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)。

这是一个描述加州地区中位数房价的数据集。每个数据样本是一个普查街区组。目标变量是 1990 年每 100,000 美元的中位数房价，共有 8 个输入特征，每个特征描述房子的某一方面。它们分别是：

+   中位数收入：街区组的中位数收入

+   房屋年龄：街区组中位数房龄

+   平均房间数：每户家庭的平均房间数

+   平均卧室数：每户家庭的平均卧室数

+   人口：街区组人口

+   平均家庭成员数：每户家庭的平均成员数

+   纬度：街区组中心纬度

+   经度：街区组中心经度

这些数据很特殊，因为输入数据的尺度差异很大。例如，每栋房子的房间数通常很少，但每个街区的居民数通常很大。此外，大多数特征应该是正数，但经度必须是负数（因为这是关于加州的）。处理这种数据多样性对某些机器学习模型而言是一个挑战。

你可以从 scikit-learn 获取数据集，scikit-learn 是从互联网实时下载的：

```py
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
print(data.feature_names)

X, y = data.data, data.target
```

## 构建模型并训练

这是一个回归问题。与分类问题不同，输出变量是连续值。在神经网络中，通常在输出层使用线性激活（即没有激活），使得理论上输出范围可以是从负无穷到正无穷。

对于回归问题，你不应该期望模型完美预测值。因此，你应该关注预测值与实际值的接近程度。你可以使用均方误差（MSE）或平均绝对误差（MAE）作为损失度量。但你也可能对均方根误差（RMSE）感兴趣，因为它与输出变量具有相同的单位。

让我们尝试传统的神经网络设计，即金字塔结构。金字塔结构是使每一层中的神经元数量随着网络到达输出层而减少。输入特征数量是固定的，但你可以在第一隐藏层设置大量的神经元，并逐渐减少后续层中的数量。由于数据集中只有一个目标，最终层应该仅输出一个值。

设计如下：

```py
import torch.nn as nn

# Define the model
model = nn.Sequential(
    nn.Linear(8, 24),
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
    nn.Linear(6, 1)
)
```

要训练这个网络，你需要定义一个损失函数。MSE 是一个合理的选择。你还需要一个优化器，例如 Adam。

```py
import torch.nn as nn
import torch.optim as optim

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.0001)
```

要训练这个模型，你可以使用你常用的训练循环。为了获得一个评估分数以确保模型有效，你需要将数据分为训练集和测试集。你可能还需要通过跟踪测试集的均方误差（MSE）来避免过拟合。以下是带有训练-测试拆分的训练循环：

```py
import copy
import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split

# train-test split of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# training parameters
n_epochs = 100   # number of epochs to run
batch_size = 10  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mse = np.inf   # init to infinity
best_weights = None
history = []

# training loop
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
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch
    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    history.append(mse)
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())

# restore model and return best accuracy
model.load_state_dict(best_weights)
```

在训练循环中，`tqdm`用于设置进度条，在每次迭代步骤中，计算并报告 MSE。你可以通过将`tqdm`参数`disable`设置为`False`来查看 MSE 的变化情况。

注意，在训练循环中，每个周期是用训练集运行前向和反向步骤几次以优化模型权重，在周期结束时，使用测试集评估模型。测试集的 MSE 被记在`history`列表中。它也是评估模型的指标，最佳的模型存储在变量`best_weights`中。

运行完这个，你将得到恢复的最佳模型，并将最佳 MSE 存储在变量`best_mse`中。注意，均方误差是预测值与实际值之间差异平方的平均值。它的平方根，即 RMSE，可以视为平均差异，数值上更有用。

下面，你可以展示 MSE 和 RMSE，并绘制 MSE 的历史记录。它应该随着周期的增加而减少。

```py
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
plt.plot(history)
plt.show()
```

这个模型产生了：

```py
MSE: 0.47
RMSE: 0.68
```

MSE 图形如下所示。![](img/0c33b9cf0623b8a856785b5c34dad411.png)

将所有内容整合在一起，以下是完整的代码。

```py
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# Read data
data = fetch_california_housing()
X, y = data.data, data.target

# train-test split for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Define the model
model = nn.Sequential(
    nn.Linear(8, 24),
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
    nn.Linear(6, 1)
)

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 100   # number of epochs to run
batch_size = 10  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mse = np.inf   # init to infinity
best_weights = None
history = []

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
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch
    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    history.append(mse)
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())

# restore model and return best accuracy
model.load_state_dict(best_weights)
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
plt.plot(history)
plt.show()
```

### 想开始使用 PyTorch 进行深度学习吗？

现在就来参加我的免费邮件速成课程（包含示例代码）。

点击注册，还可以免费获得课程的 PDF 电子书版本。

## 通过预处理改进模型

在上述中，你看到 RMSE 是 0.68。实际上，通过在训练之前对数据进行打磨，RMSE 是容易改善的。这个数据集的问题在于特征的多样性：有些特征范围狭窄，有些特征范围很宽。还有一些特征是小的但正值，有些则是非常负值。这确实对大多数机器学习模型来说并不是很好。

改进的一种方法是应用**标准缩放器**。这就是将每个特征转换为其标准分数。换句话说，对于每个特征$x$，你将其替换为

$$

z = \frac{x – \bar{x}}{\sigma_x}

$$

其中$\bar{x}$是$x$的均值，而$\sigma_x$是标准差。通过这种方式，每个转换后的特征都围绕 0 进行中心化，并且在一个窄范围内，大约 70%的样本在-1 到+1 之间。这可以帮助机器学习模型收敛。

你可以使用 scikit-learn 中的标准缩放器。以下是你应如何修改上述代码的数据准备部分：

```py
import torch
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Read data
data = fetch_california_housing()
X, y = data.data, data.target

# train-test split for model evaluation
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# Standardizing data
scaler = StandardScaler()
scaler.fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
```

请注意，标准缩放器在训练-测试拆分后应用。上面的`StandardScaler`是在训练集上进行拟合，但在训练集和测试集上都进行应用。你必须避免将标准缩放器应用于所有数据，因为测试集中的信息不应泄露给模型。否则，你会引入**数据泄露**。

除此之外，几乎没有任何变化：你仍然有 8 个特征（只不过它们的值不同）。你仍然使用相同的训练循环。如果你用缩放后的数据训练模型，你应该会看到 RMSE 有所改善，例如：

```py
MSE: 0.29
RMSE: 0.54
```

尽管 MSE 的历史变化形状类似，但 y 轴显示出在缩放后效果确实更好：![](img/8550b66a243d506ee34fe176c6b78113.png)

然而，你需要在最后小心：当你使用训练好的模型并应用到新数据时，你应该在将输入数据输入模型之前应用缩放器。也就是说，推理应按如下方式进行：

```py
model.eval()
with torch.no_grad():
    # Test out inference with 5 samples from the original test set
    for i in range(5):
        X_sample = X_test_raw[i: i+1]
        X_sample = scaler.transform(X_sample)
        X_sample = torch.tensor(X_sample, dtype=torch.float32)
        y_pred = model(X_sample)
        print(f"{X_test_raw[i]} -> {y_pred[0].numpy()} (expected {y_test[i].numpy()})")
```

以下是完整的代码：

```py
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Read data
data = fetch_california_housing()
X, y = data.data, data.target

# train-test split for model evaluation
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# Standardizing data
scaler = StandardScaler()
scaler.fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Define the model
model = nn.Sequential(
    nn.Linear(8, 24),
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
    nn.Linear(6, 1)
)

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 100   # number of epochs to run
batch_size = 10  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mse = np.inf   # init to infinity
best_weights = None
history = []

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
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch
    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    history.append(mse)
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())

# restore model and return best accuracy
model.load_state_dict(best_weights)
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
plt.plot(history)
plt.show()

model.eval()
with torch.no_grad():
    # Test out inference with 5 samples
    for i in range(5):
        X_sample = X_test_raw[i: i+1]
        X_sample = scaler.transform(X_sample)
        X_sample = torch.tensor(X_sample, dtype=torch.float32)
        y_pred = model(X_sample)
        print(f"{X_test_raw[i]} -> {y_pred[0].numpy()} (expected {y_test[i].numpy()})")
```

当然，模型仍有改进的空间。一个方法是将目标呈现为对数尺度，或者等效地，使用平均绝对百分比误差（MAPE）作为损失函数。这是因为目标变量是房价，它的范围很广。对于相同的误差幅度，低价房更容易出现问题。你可以修改上述代码以生成更好的预测，这也是你的练习。

## 总结

在这篇文章中，你发现了使用 PyTorch 构建回归模型的方法。

你学会了如何通过 PyTorch 逐步解决回归问题，具体包括：

+   如何加载和准备数据以供 PyTorch 使用

+   如何创建神经网络模型并选择回归的损失函数

+   如何通过应用标准缩放器提高模型的准确性
