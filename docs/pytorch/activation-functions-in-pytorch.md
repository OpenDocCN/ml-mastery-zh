# PyTorch 中的激活函数

> 原文：[`machinelearningmastery.com/activation-functions-in-pytorch/`](https://machinelearningmastery.com/activation-functions-in-pytorch/)

随着神经网络在机器学习领域的日益普及，了解激活函数在其实现中的作用变得越来越重要。在本文中，您将探讨应用于神经网络每个神经元输出的激活函数的概念，以引入模型的非线性。没有激活函数，神经网络将仅仅是一系列线性变换，这将限制它们学习复杂模式和数据间关系的能力。

PyTorch 提供了多种激活函数，每种都具有独特的特性和用途。在 PyTorch 中一些常见的激活函数包括 ReLU、sigmoid 和 tanh。选择适合特定问题的正确激活函数对于在神经网络中实现最佳性能至关重要。您将学习如何使用不同的激活函数在 PyTorch 中训练神经网络，并分析它们的性能。

在本教程中，您将学习：

+   关于在神经网络架构中使用的各种激活函数。

+   如何在 PyTorch 中实现激活函数。

+   如何在实际问题中比较激活函数的效果。

让我们开始吧。

![](img/aa9028c6e30f12c6fd6d1a76c859a9f1.png)

PyTorch 中的激活函数

Adrian Tam 使用稳定扩散生成的图像。部分权利保留。

## 概述

本教程分为四个部分；它们分别是：

+   Logistic 激活函数

+   双曲正切激活函数

+   ReLU 激活函数

+   探索神经网络中的激活函数

## Logistic 激活函数

您将从逻辑函数开始，这是神经网络中常用的激活函数，也称为 sigmoid 函数。它接受任何输入并将其映射到 0 到 1 之间的值，可以被解释为概率。这使得它特别适用于二元分类任务，其中网络需要预测输入属于两个类别之一的概率。

Logistic 函数的主要优势之一是它是可微分的，这意味着它可以用于反向传播算法来训练神经网络。此外，它具有平滑的梯度，有助于避免梯度爆炸等问题。然而，在训练过程中它也可能引入梯度消失的问题。

现在，让我们使用 PyTorch 对张量应用 logistic 函数，并绘制出它的图像看看。

```py
# importing the libraries
import torch
import matplotlib.pyplot as plt

# create a PyTorch tensor
x = torch.linspace(-10, 10, 100)

# apply the logistic activation function to the tensor
y = torch.sigmoid(x)

# plot the results with a custom color
plt.plot(x.numpy(), y.numpy(), color='purple')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Logistic Activation Function')
plt.show()
```

![](img/f9a455e3db55a513296b0d8ff4d46ff1.png)

在上述示例中，您使用了 PyTorch 库中的 `torch.sigmoid()` 函数将 logistic 激活函数应用到了张量 `x` 上。您还使用了 matplotlib 库创建了一个具有自定义颜色的图表。

## 双曲正切激活函数

接下来，你将研究 tanh 激活函数，该函数输出值介于$-1$和$1$之间，平均输出为 0。这有助于确保神经网络层的输出保持在 0 附近，从而对归一化目的有用。Tanh 是一个平滑且连续的激活函数，这使得在梯度下降过程中更容易优化。

与逻辑激活函数类似，tanh 函数在深度神经网络中尤其容易受到梯度消失问题的影响。这是因为函数的斜率在大或小的输入值下变得非常小，使得梯度在网络中传播变得困难。

此外，由于使用了指数函数，tanh 在计算上可能比较昂贵，尤其是在大张量或用于多层深度神经网络时。

下面是如何在张量上应用 tanh 并可视化的示例。

```py
# apply the tanh activation function to the tensor
y = torch.tanh(x)

# plot the results with a custom color
plt.plot(x.numpy(), y.numpy(), color='blue')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Tanh Activation Function')
plt.show()
```

![](img/4c891af66131389e617d327300dc2ef6.png)

## ReLU 激活函数

ReLU（修正线性单元）是神经网络中另一种常用的激活函数。与 sigmoid 和 tanh 函数不同，ReLU 是一个非饱和函数，这意味着它在输入范围的极值处不会变得平坦。相反，如果输入值为正，ReLU 直接输出该值；如果为负，则输出 0。

这个简单的分段线性函数相比于 sigmoid 和 tanh 激活函数有几个优势。首先，它在计算上更高效，非常适合大规模神经网络。其次，ReLU 显示出对梯度消失问题的抗性较强，因为它没有平坦的斜率。此外，ReLU 可以帮助稀疏化网络中神经元的激活，从而可能提高泛化能力。

下面是如何将 ReLU 激活函数应用于 PyTorch 张量`x`并绘制结果的示例。

```py
# apply the ReLU activation function to the tensor
y = torch.relu(x)

# plot the results with a custom color
plt.plot(x.numpy(), y.numpy(), color='green')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('ReLU Activation Function')
plt.show()
```

![](img/35477c1bd7e0c8a93689771a7feda52f.png)

以下是打印上述所有激活函数的完整代码。

```py
# importing the libraries
import torch
import matplotlib.pyplot as plt

# create a PyTorch tensor
x = torch.linspace(-10, 10, 100)

# apply the logistic activation function to the tensor and plot
y = torch.sigmoid(x)
plt.plot(x.numpy(), y.numpy(), color='purple')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Logistic Activation Function')
plt.show()

# apply the tanh activation function to the tensor and plot
y = torch.tanh(x)
plt.plot(x.numpy(), y.numpy(), color='blue')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Tanh Activation Function')
plt.show()

# apply the ReLU activation function to the tensor and plot
y = torch.relu(x)
plt.plot(x.numpy(), y.numpy(), color='green')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('ReLU Activation Function')
plt.show()
```

## 探索神经网络中的激活函数

激活函数在深度学习模型的训练中起着至关重要的作用，因为它们向网络引入了非线性，使其能够学习复杂的模式。

让我们使用流行的 MNIST 数据集，它包含 70000 张 28×28 像素的灰度手写数字图像。你将创建一个简单的前馈神经网络来分类这些数字，并实验不同的激活函数，如 ReLU、Sigmoid、Tanh 和 Leaky ReLU。

```py
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Load the MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='data/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='data/', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
```

让我们创建一个继承自`nn.Module`的`NeuralNetwork`类。该类有三个线性层和一个激活函数作为输入参数。前向传播方法定义了网络的前向传递过程，在每个线性层之后应用激活函数，最后一个线性层除外。

```py
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation_function):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.activation_function = activation_function

    def forward(self, x):
        x = self.activation_function(self.layer1(x))
        x = self.activation_function(self.layer2(x))
        x = self.layer3(x)
        return x
```

你已将`activation_function`参数添加到`NeuralNetwork`类中，这使得你可以插入任何你想实验的激活函数。

## 使用不同激活函数训练和测试模型

让我们创建一些函数来帮助训练。`train()` 函数在一个 epoch 中训练网络。它遍历训练数据加载器，计算损失，并进行反向传播和优化。`test()` 函数在测试数据集上评估网络，计算测试损失和准确度。

```py
def train(network, data_loader, criterion, optimizer, device):
    network.train()
    running_loss = 0.0

    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.shape[0], -1)

        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)

    return running_loss / len(data_loader.dataset)

def test(network, data_loader, criterion, device):
    network.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)

            output = network(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return test_loss / len(data_loader.dataset), 100 * correct / total
```

为了进行比较，让我们创建一个激活函数的字典并进行迭代。对于每种激活函数，你实例化 `NeuralNetwork` 类，定义损失函数（`CrossEntropyLoss`），并设置优化器（`Adam`）。然后，训练模型若干个 epoch，每个 epoch 中调用 `train()` 和 `test()` 函数来评估模型的性能。你将每个 epoch 的训练损失、测试损失和测试准确度存储在结果字典中。

```py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 784
hidden_size = 128
num_classes = 10
num_epochs = 10
learning_rate = 0.001

activation_functions = {
    'ReLU': nn.ReLU(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'LeakyReLU': nn.LeakyReLU()
}

results = {}

# Train and test the model with different activation functions
for name, activation_function in activation_functions.items():
    print(f"Training with {name} activation function...")

    model = NeuralNetwork(input_size, hidden_size, num_classes, activation_function).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_history = []
    test_loss_history = []
    test_accuracy_history = []

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    results[name] = {
        'train_loss_history': train_loss_history,
        'test_loss_history': test_loss_history,
        'test_accuracy_history': test_accuracy_history
    }
```

当你运行上述代码时，它会输出：

```py
Training with ReLU activation function...
Epoch [1/10], Test Loss: 0.1589, Test Accuracy: 95.02%
Epoch [2/10], Test Loss: 0.1138, Test Accuracy: 96.52%
Epoch [3/10], Test Loss: 0.0886, Test Accuracy: 97.15%
Epoch [4/10], Test Loss: 0.0818, Test Accuracy: 97.50%
Epoch [5/10], Test Loss: 0.0783, Test Accuracy: 97.47%
Epoch [6/10], Test Loss: 0.0754, Test Accuracy: 97.80%
Epoch [7/10], Test Loss: 0.0832, Test Accuracy: 97.56%
Epoch [8/10], Test Loss: 0.0783, Test Accuracy: 97.78%
Epoch [9/10], Test Loss: 0.0789, Test Accuracy: 97.75%
Epoch [10/10], Test Loss: 0.0735, Test Accuracy: 97.99%
Training with Sigmoid activation function...
Epoch [1/10], Test Loss: 0.2420, Test Accuracy: 92.81%
Epoch [2/10], Test Loss: 0.1718, Test Accuracy: 94.99%
Epoch [3/10], Test Loss: 0.1339, Test Accuracy: 96.06%
Epoch [4/10], Test Loss: 0.1141, Test Accuracy: 96.42%
Epoch [5/10], Test Loss: 0.1004, Test Accuracy: 97.00%
Epoch [6/10], Test Loss: 0.0909, Test Accuracy: 97.10%
Epoch [7/10], Test Loss: 0.0846, Test Accuracy: 97.28%
Epoch [8/10], Test Loss: 0.0797, Test Accuracy: 97.42%
Epoch [9/10], Test Loss: 0.0785, Test Accuracy: 97.58%
Epoch [10/10], Test Loss: 0.0795, Test Accuracy: 97.58%
Training with Tanh activation function...
Epoch [1/10], Test Loss: 0.1660, Test Accuracy: 95.17%
Epoch [2/10], Test Loss: 0.1152, Test Accuracy: 96.47%
Epoch [3/10], Test Loss: 0.1057, Test Accuracy: 96.86%
Epoch [4/10], Test Loss: 0.0865, Test Accuracy: 97.21%
Epoch [5/10], Test Loss: 0.0760, Test Accuracy: 97.61%
Epoch [6/10], Test Loss: 0.0856, Test Accuracy: 97.23%
Epoch [7/10], Test Loss: 0.0735, Test Accuracy: 97.66%
Epoch [8/10], Test Loss: 0.0790, Test Accuracy: 97.67%
Epoch [9/10], Test Loss: 0.0805, Test Accuracy: 97.47%
Epoch [10/10], Test Loss: 0.0834, Test Accuracy: 97.82%
Training with LeakyReLU activation function...
Epoch [1/10], Test Loss: 0.1587, Test Accuracy: 95.14%
Epoch [2/10], Test Loss: 0.1084, Test Accuracy: 96.37%
Epoch [3/10], Test Loss: 0.0861, Test Accuracy: 97.22%
Epoch [4/10], Test Loss: 0.0883, Test Accuracy: 97.06%
Epoch [5/10], Test Loss: 0.0870, Test Accuracy: 97.37%
Epoch [6/10], Test Loss: 0.0929, Test Accuracy: 97.26%
Epoch [7/10], Test Loss: 0.0824, Test Accuracy: 97.54%
Epoch [8/10], Test Loss: 0.0785, Test Accuracy: 97.77%
Epoch [9/10], Test Loss: 0.0908, Test Accuracy: 97.92%
Epoch [10/10], Test Loss: 0.1012, Test Accuracy: 97.76%
```

你可以使用 Matplotlib 创建图表，以比较各个激活函数的性能。你可以创建三个独立的图表，以可视化每种激活函数在各个 epoch 上的训练损失、测试损失和测试准确度。

```py
import matplotlib.pyplot as plt

# Plot the training loss
plt.figure()
for name, data in results.items():
    plt.plot(data['train_loss_history'], label=name)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend()
plt.show()

# Plot the testing loss
plt.figure()
for name, data in results.items():
    plt.plot(data['test_loss_history'], label=name)
plt.xlabel('Epoch')
plt.ylabel('Testing Loss')
plt.legend()
plt.show()

# Plot the testing accuracy
plt.figure()
for name, data in results.items():
    plt.plot(data['test_accuracy_history'], label=name)
plt.xlabel('Epoch')
plt.ylabel('Testing Accuracy')
plt.legend()
plt.show()
```

![](img/acc9423f783521ee1e33b8bd5826aa72.png)![](img/bcab524bcca50d07878b3fb9e17e0169.png)![](img/7eff026adbf6d4c238a43c919c1957cb.png)

这些图表提供了各个激活函数性能的视觉比较。通过分析结果，你可以确定哪种激活函数最适合本示例中的特定任务和数据集。

## 总结

在本教程中，你已经实现了 PyTorch 中一些最流行的激活函数。你还学习了如何使用流行的 MNIST 数据集在 PyTorch 中训练神经网络，尝试了 ReLU、Sigmoid、Tanh 和 Leaky ReLU 激活函数，并通过绘制训练损失、测试损失和测试准确度来分析它们的性能。

正如你所见，激活函数的选择在模型性能中起着至关重要的作用。然而，请记住，最佳的激活函数可能会根据任务和数据集的不同而有所变化。
