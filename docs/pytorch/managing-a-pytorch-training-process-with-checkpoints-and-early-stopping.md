# 使用检查点和早停管理 PyTorch 训练过程

> 原文：[`machinelearningmastery.com/managing-a-pytorch-training-process-with-checkpoints-and-early-stopping/`](https://machinelearningmastery.com/managing-a-pytorch-training-process-with-checkpoints-and-early-stopping/)

大型深度学习模型可能需要很长时间来训练。如果训练过程在中间被中断，你会丢失大量工作。但有时，你实际上会想在中间中断训练过程，因为你知道继续下去不会给你更好的模型。在这篇文章中，你将发现如何控制 PyTorch 中的训练循环，以便你可以恢复被中断的过程或提前停止训练循环。

完成这篇文章后，你将知道：

+   训练时检查点的重要性

+   如何在训练过程中创建检查点并在之后恢复

+   如何通过检查点提前终止训练循环

**通过我的书 [Deep Learning with PyTorch](https://machinelearningmastery.com/deep-learning-with-pytorch/) 快速启动你的项目**。它提供了**自学教程**和**可运行的代码**。

让我们开始吧！[](../Images/05afbb1ba9774944b954833a96310b08.png)

使用检查点和早停管理 PyTorch 训练过程

图片由 [Arron Choi](https://unsplash.com/photos/7VJyD8tODfc) 提供。保留所有权利。

## 概述

本章分为两部分；它们是：

+   检查点神经网络模型

+   具有早停的检查点

## 检查点神经网络模型

许多系统都有状态。如果你可以保存系统的所有状态并在以后恢复，你可以随时回到系统行为的特定时间点。如果你在 Microsoft Word 上工作并保存了多个版本的文档，因为你不知道是否要恢复编辑，这里也是同样的想法。

同样适用于长时间运行的过程。应用程序检查点是一种容错技术。在这种方法中，系统状态的快照会被拍摄以防系统故障。如果出现问题，你可以从快照中恢复。检查点可以直接使用，也可以作为新运行的起点，从中断的地方继续。当训练深度学习模型时，检查点捕获模型的权重。这些权重可以直接用于预测或作为持续训练的基础。

PyTorch 不提供任何检查点功能，但它有检索和恢复模型权重的功能。因此，你可以利用这些功能实现检查点逻辑。让我们创建一个检查点和恢复函数，这些函数简单地保存模型的权重并将其加载回来：

```py
import torch

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

def resume(model, filename):
    model.load_state_dict(torch.load(filename))
```

以下是你通常训练 PyTorch 模型的方式。使用的数据集从 OpenML 平台获取。它是一个二分类数据集。这个示例中使用了 PyTorch DataLoader，使训练循环更简洁。

```py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split, default_collate
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

data = fetch_openml("electricity", version=1, parser="auto")

# Label encode the target, convert to float tensors
X = data['data'].astype('float').values
y = data['target']
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# train-test split for model evaluation
trainset, testset = random_split(TensorDataset(X, y), [0.7, 0.3])

# Define the model
model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 12),
    nn.ReLU(),
    nn.Linear(12, 1),
    nn.Sigmoid(),
)

# Train the model
n_epochs = 100
loader = DataLoader(trainset, shuffle=True, batch_size=32)
X_test, y_test = default_collate(testset)
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    print(f"End of epoch {epoch}: accuracy = {float(acc)*100:.2f}%")
```

如果您希望在上述训练循环中添加检查点，您可以在外部 for 循环结束时执行，这时模型通过测试集进行验证。例如，以下内容：

```py
...
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    print(f"End of epoch {epoch}: accuracy = {float(acc)*100:.2f}%")
    checkpoint(model, f"epoch-{epoch}.pth")
```

您将在工作目录中看到创建的多个文件。这段代码将从第 7 轮保存模型的示例文件`epoch-7.pth`。每一个这样的文件都是一个带有序列化模型权重的 ZIP 文件。不禁止在内部 for 循环中设置检查点，但由于引入的开销，频繁设置检查点并不是一个好主意。

作为一种容错技术，在训练循环之前添加几行代码，您可以从特定的轮次恢复：

```py
start_epoch = 0
if start_epoch > 0:
    resume_epoch = start_epoch - 1
    resume(model, f"epoch-{resume_epoch}.pth")

for epoch in range(start_epoch, n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    print(f"End of epoch {epoch}: accuracy = {float(acc)*100:.2f}%")
    checkpoint(model, f"epoch-{epoch}.pth")
```

也就是说，如果训练循环在第 8 轮中间被中断，因此最后一个检查点是从第 7 轮开始的，则在上面设置`start_epoch = 8`。

注意，如果这样做，生成训练集和测试集的`random_split()`函数可能由于随机性而导致不同的分割。如果这对你来说是个问题，你应该有一种一致的方法来创建数据集（例如，保存分割的数据以便重用）。

有时，模型外部存在状态，您可能希望对其进行检查点。一个特别的例子是优化器，在像 Adam 这样的情况下，具有动态调整的动量。如果重新启动训练循环，您可能还希望恢复优化器中的动量。这并不难做到。关键是使您的`checkpoint()`函数更复杂，例如：

```py
torch.save({
    'optimizer': optimizer.state_dict(),
    'model': model.state_dict(),
}, filename)
```

并相应地更改您的`resume()`函数：

```py
checkpoint = torch.load(filename)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
```

这有效是因为在 PyTorch 中，`torch.save()`和`torch.load()`函数都由`pickle`支持，因此您可以在包含`list`或`dict`容器的情况下使用它。

为了将所有内容整合在一起，下面是完整的代码：

```py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split, default_collate
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

data = fetch_openml("electricity", version=1, parser="auto")

# Label encode the target, convert to float tensors
X = data['data'].astype('float').values
y = data['target']
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# train-test split for model evaluation
trainset, testset = random_split(TensorDataset(X, y), [0.7, 0.3])

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

def resume(model, filename):
    model.load_state_dict(torch.load(filename))

# Define the model
model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 12),
    nn.ReLU(),
    nn.Linear(12, 1),
    nn.Sigmoid(),
)

# Train the model
n_epochs = 100
start_epoch = 0
loader = DataLoader(trainset, shuffle=True, batch_size=32)
X_test, y_test = default_collate(testset)
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

if start_epoch > 0:
    resume_epoch = start_epoch - 1
    resume(model, f"epoch-{resume_epoch}.pth")

for epoch in range(start_epoch, n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    print(f"End of epoch {epoch}: accuracy = {float(acc)*100:.2f}%")
    checkpoint(model, f"epoch-{epoch}.pth")
```

### 想要开始使用 PyTorch 进行深度学习吗？

立即参加我的免费电子邮件崩溃课程（附有示例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## 使用早停技术进行检查点

检查点不仅用于容错。您还可以使用它来保持最佳模型。如何定义最佳模型是主观的，但考虑来自测试集的分数是一个明智的方法。假设只保留找到的最佳模型，您可以修改训练循环如下：

```py
...
best_accuracy = -1
for epoch in range(start_epoch, n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    acc = float(acc) * 100
    print(f"End of epoch {epoch}: accuracy = {acc:.2f}%")
    if acc > best_accuracy:
        best_accuracy = acc
        checkpoint(model, "best_model.pth")

resume(model, "best_model.pth")
```

变量`best_accuracy`用于跟踪迄今为止获得的最高准确率（`acc`），其范围在 0 到 100 之间的百分比。每当观察到更高的准确率时，模型将被保存到文件`best_model.pth`。在整个训练循环之后，通过您之前创建的`resume()`函数恢复最佳模型。之后，您可以使用模型对未见数据进行预测。请注意，如果您使用不同的指标进行检查点，例如交叉熵损失，更好的模型应该伴随着更低的交叉熵。因此，您应该跟踪获取的最低交叉熵。

你还可以在每个训练周期无条件地保存模型检查点，并与最佳模型检查点保存一起，因为你可以创建多个检查点文件。由于上面的代码是找到最佳模型并复制它，你通常会看到对训练循环的进一步优化，如果希望看到模型改善的希望很小，可以提前停止训练。这就是可以节省训练时间的早停技术。

上面的代码在每个训练周期结束时使用测试集验证模型，并将找到的最佳模型保存在检查点文件中。最简单的早停策略是设置一个 $k$ 训练周期的阈值。如果你没有看到模型在过去 $k$ 个训练周期内有所改善，你可以在中间终止训练循环。实现方式如下：

```py
early_stop_thresh = 5
best_accuracy = -1
best_epoch = -1
for epoch in range(start_epoch, n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    acc = float(acc) * 100
    print(f"End of epoch {epoch}: accuracy = {acc:.2f}%")
    if acc > best_accuracy:
        best_accuracy = acc
        best_epoch = epoch
        checkpoint(model, "best_model.pth")
    elif epoch - best_epoch > early_stop_thresh:
        print("Early stopped training at epoch %d" % epoch)
        break  # terminate the training loop

resume(model, "best_model.pth")
```

阈值 `early_stop_thresh` 上面设置为 5。还有一个变量 `best_epoch` 用于记住最佳模型的周期。如果模型在足够长的时间内没有改善，外部 for 循环将被终止。

这个设计减轻了一个设计参数 `n_epochs` 的压力。你现在可以将 `n_epochs` 设置为训练模型的**最大**周期数，因此可以比需要的周期数更大，并且通常可以确保你的训练循环会更早停止。这也是避免过拟合的一种策略：如果模型在测试集上表现更差，早停逻辑将中断训练并恢复最佳检查点。

总结所有内容，以下是带有早停的检查点的完整代码：

```py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split, default_collate
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

data = fetch_openml("electricity", version=1, parser="auto")

# Label encode the target, convert to float tensors
X = data['data'].astype('float').values
y = data['target']
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# train-test split for model evaluation
trainset, testset = random_split(TensorDataset(X, y), [0.7, 0.3])

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

def resume(model, filename):
    model.load_state_dict(torch.load(filename))

# Define the model
model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 12),
    nn.ReLU(),
    nn.Linear(12, 1),
    nn.Sigmoid(),
)

# Train the model
n_epochs = 10000  # more than we needed
loader = DataLoader(trainset, shuffle=True, batch_size=32)
X_test, y_test = default_collate(testset)
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

early_stop_thresh = 5
best_accuracy = -1
best_epoch = -1

for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    acc = float(acc) * 100
    print(f"End of epoch {epoch}: accuracy = {acc:.2f}%")
    if acc > best_accuracy:
        best_accuracy = acc
        best_epoch = epoch
        checkpoint(model, "best_model.pth")
    elif epoch - best_epoch > early_stop_thresh:
        print("Early stopped training at epoch %d" % epoch)
        break  # terminate the training loop

resume(model, "best_model.pth")
```

你可能会看到上述代码产生：

```py
End of epoch 0: accuracy = 61.84%
End of epoch 1: accuracy = 55.90%
End of epoch 2: accuracy = 63.95%
End of epoch 3: accuracy = 66.87%
End of epoch 4: accuracy = 64.77%
End of epoch 5: accuracy = 60.03%
End of epoch 6: accuracy = 67.16%
End of epoch 7: accuracy = 66.01%
End of epoch 8: accuracy = 62.88%
End of epoch 9: accuracy = 64.28%
End of epoch 10: accuracy = 68.63%
End of epoch 11: accuracy = 70.56%
End of epoch 12: accuracy = 64.62%
End of epoch 13: accuracy = 65.63%
End of epoch 14: accuracy = 66.81%
End of epoch 15: accuracy = 65.11%
End of epoch 16: accuracy = 55.81%
End of epoch 17: accuracy = 54.59%
Early stopped training at epoch 17
```

它在第 17 个训练周期结束时停止，使用了从第 11 个周期获得的最佳模型。由于算法的随机性，你可能会看到结果有所不同。但可以肯定的是，即使将最大训练周期数设置为 10000，训练循环确实会更早停止。

当然，你可以设计更复杂的早停策略，例如，至少运行 $N$ 个周期，然后在 $k$ 个周期后允许提前停止。你可以自由调整上述代码，以使最佳训练循环满足你的需求。

## 总结

在本章中，你发现了在长时间训练过程中保存深度学习模型检查点的重要性。你学习了：

+   什么是检查点，为什么它很有用

+   如何保存模型检查点以及如何恢复检查点

+   使用检查点的不同策略

+   如何实现带有检查点的早停
