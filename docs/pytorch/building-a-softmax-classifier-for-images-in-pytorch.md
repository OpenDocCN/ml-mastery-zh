# 在 PyTorch 中为图像构建 Softmax 分类器

> 原文：[`machinelearningmastery.com/building-a-softmax-classifier-for-images-in-pytorch/`](https://machinelearningmastery.com/building-a-softmax-classifier-for-images-in-pytorch/)

Softmax 分类器是监督学习中的一种分类器。它是深度学习网络中的重要构建模块，也是深度学习从业者中最受欢迎的选择。

Softmax 分类器适用于多类分类，它为每个类别输出概率。

本教程将教你如何为图像数据构建一个 Softmax 分类器。你将学习如何准备数据集，然后学习如何使用 PyTorch 实现 Softmax 分类器。特别是，你将学习：

+   关于 Fashion-MNIST 数据集。

+   如何在 PyTorch 中使用 Softmax 分类器处理图像。

+   如何在 PyTorch 中构建和训练一个多类图像分类器。

+   如何在模型训练后绘制结果。

**启动你的项目**，参考我的书籍 [Deep Learning with PyTorch](https://machinelearningmastery.com/deep-learning-with-pytorch/)。它提供了 **自学教程** 和 **可用代码**。

让我们开始吧。![](img/1c11e009d1de000b1f5b02bf7c9745cb.png)

在 PyTorch 中为图像构建 Softmax 分类器。

图片来自 [Joshua J. Cotten](https://unsplash.com/photos/Ge1t87lvyRM)。保留所有权利。

## 概述

本教程分为三个部分：

+   +   准备数据集

    +   构建模型

    +   训练模型

## 准备数据集

你将在这里使用的数据集是 Fashion-MNIST。它是一个经过预处理和良好组织的数据集，包含 70,000 张图像，其中 60,000 张用于训练数据，10,000 张用于测试数据。

数据集中的每个示例是一个 $28\times 28$ 像素的灰度图像，总像素数为 784。数据集有 10 个类别，每张图像被标记为一个时尚项目，并与从 0 到 9 的整数标签相关联。

该数据集可以从 `torchvision` 中加载。为了加快训练速度，我们将数据集限制为 4000 个样本：

```py
from torchvision import datasets

train_data = datasets.FashionMNIST('data', train=True, download=True)
train_data = list(train_data)[:4000]
```

当你第一次获取 fashion-MNIST 数据集时，你会看到 PyTorch 从互联网下载它并保存到名为 `data` 的本地目录中：

```py
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz
  0%|          | 0/26421880 [00:00<?, ?it/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz
  0%|          | 0/29515 [00:00<?, ?it/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
  0%|          | 0/4422102 [00:00<?, ?it/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
  0%|          | 0/5148 [00:00<?, ?it/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```

上述数据集 `train_data` 是一个元组列表，每个元组包含一个图像（以 Python Imaging Library 对象的形式）和一个整数标签。

让我们用 matplotlib 绘制数据集中的前 10 张图像。

```py
import matplotlib.pyplot as plt

# plot the first 10 images in the training data
for i, (img, label) in enumerate(train_data[:10]):
    plt.subplot(4, 3, i+1)
    plt.imshow(img, cmap="gray")

plt.show()
```

你应该能看到类似以下的图像：

![](img/c04cd8b1b12cba01ef7be6dd805a3558.png)

PyTorch 需要数据集为 PyTorch 张量。因此，你将通过应用转换，使用 PyTorch transforms 中的 `ToTensor()` 方法来转换这些数据。此转换可以在 torchvision 的数据集 API 中透明地完成：

```py
from torchvision import datasets, transforms

# download and apply the transform
train_data = datasets.FashionMNIST('data', train=True, download=True, transform=transforms.ToTensor())
train_data = list(train_data)[:4000]
```

在继续模型之前，我们还将数据拆分为训练集和验证集，其中前 3500 张图像为训练集，其余的为验证集。通常我们希望在拆分之前打乱数据，但为了简洁起见，我们可以跳过这一步。

```py
# splitting the dataset into train and validation sets
train_data, val_data = train_data[:3500], train_data[3500:]
```

### 想开始使用 PyTorch 进行深度学习吗？

现在就来参加我的免费电子邮件速成课程（包含示例代码）。

点击注册，还可以获得课程的免费 PDF 电子书版。

## 构建模型

为了构建一个用于图像分类的自定义 softmax 模块，我们将使用来自 PyTorch 库的 `nn.Module`。为了简化起见，我们只构建一个层的模型。

```py
import torch

# build custom softmax module
class Softmax(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        pred = self.linear(x)
        return pred
```

现在，让我们实例化我们的模型对象。它接受一个一维向量作为输入，并对 10 个不同的类别进行预测。我们还要检查一下参数的初始化情况。

```py
# call Softmax Classifier
model_softmax = Softmax(784, 10)
print(model_softmax.state_dict())
```

你应该会看到模型的权重是随机初始化的，但它的形状应类似于以下：

```py
OrderedDict([('linear.weight',
              tensor([[-0.0344,  0.0334, -0.0278,  ..., -0.0232,  0.0198, -0.0123],
                      [-0.0274, -0.0048, -0.0337,  ..., -0.0340,  0.0274, -0.0091],
                      [ 0.0078, -0.0057,  0.0178,  ..., -0.0013,  0.0322, -0.0219],
                      ...,
                      [ 0.0158, -0.0139, -0.0220,  ..., -0.0054,  0.0284, -0.0058],
                      [-0.0142, -0.0268,  0.0172,  ...,  0.0099, -0.0145, -0.0154],
                      [-0.0172, -0.0224,  0.0016,  ...,  0.0107,  0.0147,  0.0252]])),
             ('linear.bias',
              tensor([-0.0156,  0.0061,  0.0285,  0.0065,  0.0122, -0.0184, -0.0197,  0.0128,
                       0.0251,  0.0256]))])
```

## 训练模型

你将使用随机梯度下降来训练模型，并结合交叉熵损失。让我们将学习率固定为 0.01。为了帮助训练，我们还将数据加载到数据加载器中，包括训练集和验证集，并将批量大小设置为 16。

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

现在，让我们将所有内容结合起来，并训练我们的模型 200 个周期。

```py
epochs = 200
Loss = []
acc = []
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model_softmax(images.view(-1, 28*28))
        loss = criterion(outputs, labels)
        # Loss.append(loss.item())
        loss.backward()
        optimizer.step()
    Loss.append(loss.item())
    correct = 0
    for images, labels in val_loader:
        outputs = model_softmax(images.view(-1, 28*28))
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum()
    accuracy = 100 * (correct.item()) / len(val_data)
    acc.append(accuracy)
    if epoch % 10 == 0:
        print('Epoch: {}. Loss: {}. Accuracy: {}'.format(epoch, loss.item(), accuracy))
```

你应该会看到每 10 个周期打印一次进度：

```py
Epoch: 0\. Loss: 1.0223602056503296\. Accuracy: 67.2
Epoch: 10\. Loss: 0.5806267857551575\. Accuracy: 78.4
Epoch: 20\. Loss: 0.5087125897407532\. Accuracy: 81.2
Epoch: 30\. Loss: 0.46658074855804443\. Accuracy: 82.0
Epoch: 40\. Loss: 0.4357391595840454\. Accuracy: 82.4
Epoch: 50\. Loss: 0.4111904203891754\. Accuracy: 82.8
Epoch: 60\. Loss: 0.39078089594841003\. Accuracy: 83.4
Epoch: 70\. Loss: 0.37331104278564453\. Accuracy: 83.4
Epoch: 80\. Loss: 0.35801735520362854\. Accuracy: 83.4
Epoch: 90\. Loss: 0.3443795442581177\. Accuracy: 84.2
Epoch: 100\. Loss: 0.33203184604644775\. Accuracy: 84.2
Epoch: 110\. Loss: 0.32071244716644287\. Accuracy: 84.0
Epoch: 120\. Loss: 0.31022894382476807\. Accuracy: 84.2
Epoch: 130\. Loss: 0.30044111609458923\. Accuracy: 84.4
Epoch: 140\. Loss: 0.29124370217323303\. Accuracy: 84.6
Epoch: 150\. Loss: 0.28255513310432434\. Accuracy: 84.6
Epoch: 160\. Loss: 0.2743147313594818\. Accuracy: 84.4
Epoch: 170\. Loss: 0.26647457480430603\. Accuracy: 84.2
Epoch: 180\. Loss: 0.2589966356754303\. Accuracy: 84.2
Epoch: 190\. Loss: 0.2518490254878998\. Accuracy: 84.2
```

正如你所见，模型的准确率在每个周期后都会增加，而损失则会减少。在这里，你为 softmax 图像分类器取得的准确率大约是 85%。如果你使用更多的数据并增加训练周期数，准确率可能会大大提高。现在让我们看看损失和准确率的图表。

首先是损失图表：

```py
plt.plot(Loss)
plt.xlabel("no. of epochs")
plt.ylabel("total loss")
plt.show()
```

它应该类似于以下:![](img/d73205d0bd1c939a2ef02fb9c20601ee.png)

这里是模型准确率的图表：

```py
plt.plot(acc)
plt.xlabel("no. of epochs")
plt.ylabel("total accuracy")
plt.show()
```

它类似于下面的样子:![](img/cdd1308460c6a5ac3b348f0f26ca2fda.png)

将所有内容整合起来，以下是完整的代码：

```py
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import datasets

# download and apply the transform
train_data = datasets.FashionMNIST('data', train=True, download=True, transform=transforms.ToTensor())
train_data = list(train_data)[:4000]

# splitting the dataset into train and validation sets
train_data, val_data = train_data[:3500], train_data[3500:]

# build custom softmax module
class Softmax(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(Softmax, self).__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        pred = self.linear(x)
        return pred

# call Softmax Classifier
model_softmax = Softmax(784, 10)
model_softmax.state_dict()

# define loss, optimizier, and dataloader for train and validation sets
optimizer = torch.optim.SGD(model_softmax.parameters(), lr = 0.01)
criterion = torch.nn.CrossEntropyLoss()
batch_size = 16
train_loader = DataLoader(dataset = train_data, batch_size = batch_size)
val_loader = DataLoader(dataset = val_data, batch_size = batch_size)

epochs = 200
Loss = []
acc = []
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model_softmax(images.view(-1, 28*28))
        loss = criterion(outputs, labels)
        # Loss.append(loss.item())
        loss.backward()
        optimizer.step()
    Loss.append(loss.item())
    correct = 0
    for images, labels in val_loader:
        outputs = model_softmax(images.view(-1, 28*28))
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum()
    accuracy = 100 * (correct.item()) / len(val_data)
    acc.append(accuracy)
    if epoch % 10 == 0:
        print('Epoch: {}. Loss: {}. Accuracy: {}'.format(epoch, loss.item(), accuracy))

plt.plot(Loss)
plt.xlabel("no. of epochs")
plt.ylabel("total loss")
plt.show()

plt.plot(acc)
plt.xlabel("no. of epochs")
plt.ylabel("total accuracy")
plt.show()
```

## 总结

在本教程中，你学习了如何为图像数据构建 softmax 分类器。特别是，你学到了：

+   关于 Fashion-MNIST 数据集。

+   如何在 PyTorch 中使用 softmax 分类器进行图像分类。

+   如何在 PyTorch 中构建和训练一个多类别图像分类器。

+   如何在模型训练后绘制结果。
