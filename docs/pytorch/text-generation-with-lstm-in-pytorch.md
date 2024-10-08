# 使用 PyTorch 进行 LSTM 文本生成

> 原文：[`machinelearningmastery.com/text-generation-with-lstm-in-pytorch/`](https://machinelearningmastery.com/text-generation-with-lstm-in-pytorch/)

循环神经网络可以用于时间序列预测。在其中，创建了一个回归神经网络。它也可以被用作生成模型，通常是一个分类神经网络模型。生成模型的目标是从数据中学习某种模式，这样当它被提供一些提示时，它可以创建一个完整的输出，与学习的模式风格相同。

在本文中，你将发现如何使用 PyTorch 中的 LSTM 循环神经网络构建一个文本生成模型。完成本文后，你将了解：

+   从哪里下载可以用来训练文本生成模型的免费语料库

+   如何将文本序列问题框定为循环神经网络生成模型

+   如何开发一个 LSTM 来生成给定问题的合理文本序列

**启动你的项目**，使用我的书籍[Deep Learning with PyTorch](https://machinelearningmastery.com/deep-learning-with-pytorch/)。它提供**自学教程**和**可工作的代码**。

让我们开始吧！[](../Images/9fc2bb253eb3136eefd33aa7128fdfee.png)

使用 PyTorch 进行 LSTM 文本生成

照片由[Egor Lyfar](https://unsplash.com/photos/tfBlExFIVTw)提供。部分权利保留。

## 概述

本文分为六个部分；它们是：

+   生成模型是什么

+   获取文本数据

+   一个小的 LSTM 网络来预测下一个字符

+   使用 LSTM 模型生成文本

+   使用更大的 LSTM 网络

+   使用 GPU 加速更快的训练

## 生成模型是什么

生成模型确实只是另一个能够创造新事物的机器学习模型。生成对抗网络（GAN）是其自身的一类。使用注意机制的 Transformer 模型也被发现对生成文本段落有用。

这只是一个机器学习模型，因为模型已经通过现有数据进行了训练，所以它从中学到了一些东西。取决于如何训练它，它们可以有很大不同的工作方式。在本文中，创建了一个基于字符的生成模型。这意味着训练一个模型，它将一系列字符（字母和标点符号）作为输入，下一个即时字符作为目标。只要它能够预测接下来的字符是什么，给定前面的内容，你就可以在循环中运行模型以生成一段长文本。

这个模型可能是最简单的一个。然而，人类语言是复杂的。你不应该期望它能产生非常高质量的输出。即便如此，你需要大量的数据并且长时间训练模型，才能看到合理的结果。

### 想要开始使用 PyTorch 进行深度学习吗？

现在参加我的免费电子邮件速成课程（附有示例代码）。

点击注册，并免费获取课程的 PDF 电子书版本。

## 获取文本数据

获取高质量的数据对成功的生成模型至关重要。幸运的是，许多经典文本已经不再受版权保护。这意味着你可以免费下载这些书籍的所有文本，并在实验中使用它们，例如创建生成模型。或许获取不再受版权保护的免费书籍的最佳地方是古腾堡计划。

在这篇文章中，你将使用童年时期喜欢的一本书作为数据集，刘易斯·卡罗尔的《爱丽丝梦游仙境》：

+   [`www.gutenberg.org/ebooks/11`](https://www.gutenberg.org/ebooks/11)

你的模型将学习字符之间的依赖关系和字符序列中的条件概率，这样你就可以生成全新且原创的字符序列。这个过程非常有趣，推荐用古腾堡计划中的其他书籍重复这些实验。这些实验不限于文本；你还可以尝试其他 ASCII 数据，如计算机源代码、LATEX、HTML 或 Markdown 中的标记文档等。

你可以免费下载这本书的完整 ASCII 格式文本（纯文本 UTF-8），并将其放置在你的工作目录中，文件名为`wonderland.txt`。现在，你需要准备数据集以进行建模。古腾堡计划为每本书添加了标准的页眉和页脚，这不是原始文本的一部分。在文本编辑器中打开文件并删除页眉和页脚。页眉是明显的，并以如下文本结束：

```py
*** START OF THIS PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***
```

页脚是指在如下文本行之后的所有文本：

```py
THE END
```

你应该剩下一个大约有 3,400 行文本的文本文件。

## 一个小型 LSTM 网络来预测下一个字符

首先，你需要对数据进行一些预处理，才能构建模型。神经网络模型只能处理数字，而不能处理文本。因此，你需要将字符转换为数字。为了简化问题，你还需要将所有大写字母转换为小写字母。

在下面，你打开文本文件，将所有字母转换为小写，并创建一个 Python 字典`char_to_int`来将字符映射为不同的整数。例如，书中的唯一已排序小写字符列表如下：

```py
['\n', '\r', ' ', '!', '"', "'", '(', ')', '*', ',', '-', '.', ':', ';', '?', '[', ']',
'_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\xbb', '\xbf', '\xef']
```

由于这个问题是基于字符的，“词汇表”是文本中曾用到的不同字符。

```py
import numpy as np

# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
```

这应该打印出：

```py
Total Characters: 144574
Total Vocab: 50
```

你可以看到这本书大约有 150,000 个字符，并且当转换为小写时，词汇中只有 50 个不同的字符供网络学习——比字母表中的 26 个字符要多得多。

接下来，你需要将文本分为输入和目标。这里使用了 100 个字符的窗口。也就是说，使用字符 1 到 100 作为输入，你的模型将预测字符 101。如果使用 5 个字符的窗口，那么单词“chapter”将变成两个数据样本：

```py
chapt -> e
hapte -> r
```

在这样的长文本中，可以创建无数窗口，这会生成一个包含大量样本的数据集：

```py
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
```

运行上述代码，你可以看到总共创建了 144,474 个样本。每个样本现在都是整数形式，使用`char_to_int`映射进行转换。然而，PyTorch 模型更喜欢浮点张量。因此，你应该将这些转换为 PyTorch 张量。由于模型将使用 LSTM 层，因此输入张量应为（样本，时间步，特征）的维度。为了帮助训练，规范化输入到 0 到 1 也是一个好主意。因此你有如下内容：

```py
import torch
import torch.nn as nn
import torch.optim as optim

# reshape X to be [samples, time steps, features]
X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, seq_length, 1)
X = X / float(n_vocab)
y = torch.tensor(dataY)
print(X.shape, y.shape)
```

你现在可以定义你的 LSTM 模型。在这里，你定义了一个具有 256 个隐藏单元的单层 LSTM。输入是单一特征（即，一个字符对应一个整数）。在 LSTM 层之后添加了一个概率为 0.2 的 dropout 层。LSTM 层的输出是一个元组，其中第一个元素是每个时间步的 LSTM 单元的隐藏状态。这是隐藏状态如何随着 LSTM 单元接受每个时间步输入而演变的历史。假设最后的隐藏状态包含了最多的信息，因此仅将最后的隐藏状态传递到输出层。输出层是一个全连接层，用于为 50 个词汇产生 logits。通过 softmax 函数，logits 可以转换为类似概率的预测。

```py
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, n_vocab)
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x
```

这是一个用于 50 类单字符分类的模型。因此应使用交叉熵损失函数。该模型使用 Adam 优化器进行优化。训练循环如下所示。为简化起见，没有创建测试集，但模型在每个 epoch 结束时会再次使用训练集进行评估，以跟踪进度。

这个程序可能会运行很长时间，尤其是在 CPU 上！为了保留工作的成果，保存了迄今为止找到的最佳模型以备将来使用。

```py
n_epochs = 40
batch_size = 128
model = CharModel()

optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss(reduction="sum")
loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)

best_model = None
best_loss = np.inf
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss += loss_fn(y_pred, y_batch)
        if loss < best_loss:
            best_loss = loss
            best_model = model.state_dict()
        print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))

torch.save([best_model, char_to_dict], "single-char.pth")
```

运行上述代码可能会产生以下结果：

```py
...
Epoch 35: Cross-entropy: 245745.2500
Epoch 36: Cross-entropy: 243908.7031
Epoch 37: Cross-entropy: 238833.5000
Epoch 38: Cross-entropy: 239069.0000
Epoch 39: Cross-entropy: 234176.2812
```

交叉熵几乎在每个 epoch 中都在下降。这意味着模型可能没有完全收敛，你可以训练更多的 epochs。当训练循环完成后，你应该会创建一个文件`single-char.pth`，其中包含迄今为止找到的最佳模型权重，以及此模型使用的字符到整数映射。

为了完整性，下面是将上述所有内容结合到一个脚本中的示例：

```py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, seq_length, 1)
X = X / float(n_vocab)
y = torch.tensor(dataY)

class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, n_vocab)
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x

n_epochs = 40
batch_size = 128
model = CharModel()

optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss(reduction="sum")
loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)

best_model = None
best_loss = np.inf
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss += loss_fn(y_pred, y_batch)
        if loss < best_loss:
            best_loss = loss
            best_model = model.state_dict()
        print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))

torch.save([best_model, char_to_int], "single-char.pth")
```

## 使用 LSTM 模型生成文本

由于模型已经经过良好的训练，使用训练好的 LSTM 网络生成文本相对简单。首先，你需要重新创建网络并从保存的检查点加载训练好的模型权重。然后你需要为模型创建一些提示以开始生成。提示可以是模型能够理解的任何内容。它是一个种子序列，用于给模型提供一个生成字符的起点。然后，将生成的字符添加到序列的末尾，并修剪掉第一个字符以保持一致的长度。这个过程会重复进行，直到你想要预测新的字符（例如，一段长度为 1000 个字符的序列）。你可以选择一个随机输入模式作为你的种子序列，然后在生成字符时打印它们。

生成提示的一个简单方法是从原始数据集中随机选择一个样本，例如，使用前一节获得的 `raw_text`，可以创建如下的提示：

```py
seq_length = 100
start = np.random.randint(0, len(raw_text)-seq_length)
prompt = raw_text[start:start+seq_length]
```

但你需要提醒自己，你需要对其进行转换，因为这个提示是一个字符串，而模型期望的是一个整数向量。

整个代码仅如下所示：

```py
import numpy as np
import torch
import torch.nn as nn

best_model, char_to_int = torch.load("single-char.pth")
n_vocab = len(char_to_int)
int_to_char = dict((i, c) for c, i in char_to_int.items())

# reload the model
class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, n_vocab)
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x
model = CharModel()
model.load_state_dict(best_model)

# randomly generate a prompt
filename = "wonderland.txt"
seq_length = 100
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
start = np.random.randint(0, len(raw_text)-seq_length)
prompt = raw_text[start:start+seq_length]
pattern = [char_to_int[c] for c in prompt]

model.eval()
print('Prompt: "%s"' % prompt)
with torch.no_grad():
    for i in range(1000):
        # format input array of int into PyTorch tensor
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        x = torch.tensor(x, dtype=torch.float32)
        # generate logits as output from the model
        prediction = model(x)
        # convert logits into one character
        index = int(prediction.argmax())
        result = int_to_char[index]
        print(result, end="")
        # append the new character into the prompt for the next iteration
        pattern.append(index)
        pattern = pattern[1:]
print()
print("Done.")
```

运行此示例首先输出所使用的提示，然后输出每个生成的字符。例如，下面是此文本生成器的一次运行结果。提示是：

```py
Prompt: "nother rush at the stick, and tumbled head
over heels in its hurry to get hold of it; then alice, th"
```

生成的文本是：

```py
e was qot a litule soteet of thet was sh the thiee harden an the courd, and was tuitk a little toaee th thite ththe and said to the suher, and the whrtght the pacbit sese tha woode of the soeee, and the white rabbit ses ani thr gort to the thite rabbit, and then she was aoiinnene th the three baaed of the sueen and saed “ota turpe ”hun mot,”

“i don’t know the ter ano _enend to mere,” said the maccht ar a sore of great roaee. “ie you don’t teink if thet soued to soeed to the boeie the mooer, io you bane thing it wo
tou het bn the crur,
“h whsh you cen not,” said the manch hare.

“wes, it aadi,” said the manch hare.

“weat you tail to merer ae in an a gens if gre” ”he were thing,” said the maccht ar a sore of geeaghen asd tothe to the thieg harden an the could.
“h dan tor toe taie thing,” said the manch hare.

“wes, it aadi,” said the manch hare.

“weat you tail to merer ae in an a gens if gre” ”he were thing,” said the maccht ar a sore of geeaghen asd tothe to the thieg harden an t
```

让我们记录一些关于生成文本的观察。

+   它可以发出换行符。原始文本将行宽限制为 80 个字符，而生成模型尝试复制这一模式。

+   字符被分成类似单词的组，其中一些组是实际的英语单词（例如，“the”，“said”，和“rabbit”），但许多则不是（例如，“thite”，“soteet”，和“tha”）。

+   有些词序列是有意义的（例如，“i don’t know the”），但许多词序列则没有意义（例如，“he were thing”）。

这个基于字符的模型产生这样的输出非常令人印象深刻。它让你感受到 LSTM 网络的学习能力。然而，结果并不完美。在下一节中，你将通过开发一个更大的 LSTM 网络来提高结果的质量。

## 使用更大的 LSTM 网络

记住，LSTM 是一种递归神经网络。它将一个序列作为输入，在序列的每一步中，输入与其内部状态混合以产生输出。因此，LSTM 的输出也是一个序列。在上述情况中，来自最后一个时间步的输出用于神经网络的进一步处理，而早期步骤的输出则被丢弃。然而，这并不一定是唯一的情况。你可以将一个 LSTM 层的序列输出视为另一个 LSTM 层的输入。这样，你就可以构建一个更大的网络。

类似于卷积神经网络，堆叠 LSTM 网络应该让早期的 LSTM 层学习低层次特征，而后期的 LSTM 层学习高层次特征。虽然这种方法可能并不总是有效，但你可以尝试一下，看看模型是否能产生更好的结果。

在 PyTorch 中，制作堆叠 LSTM 层很简单。让我们将上述模型修改为以下形式：

```py
class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=2, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, n_vocab)
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x
```

唯一的变化是在 `nn.LSTM()` 的参数上：你将 `num_layers=2` 设为 2，而不是 1，以添加另一个 LSTM 层。但在这两个 LSTM 层之间，你还通过参数 `dropout=0.2` 添加了一个 dropout 层。用这个模型替换之前的模型就是你需要做的所有更改。重新运行训练，你应该会看到以下结果：

```py
...
Epoch 34: Cross-entropy: 203763.0312
Epoch 35: Cross-entropy: 204002.5938
Epoch 36: Cross-entropy: 210636.5625
Epoch 37: Cross-entropy: 199619.6875
Epoch 38: Cross-entropy: 199240.2969
Epoch 39: Cross-entropy: 196966.1250
```

你应该会看到此处的交叉熵低于前一节。这意味着这个模型的表现更好。实际上，使用这个模型，你可以看到生成的文本看起来更有意义：

```py
Prompt: "ll
say that ‘i see what i eat’ is the same thing as ‘i eat what i see’!”

“you might just as well sa"
y it to sea,” she katter said to the jury. and the thoee hardeners vhine she was seady to alice the was a long tay of the sooe of the court, and she was seady to and taid to the coor and the court.
“well you see what you see, the mookee of the soog of the season of the shase of the court!”

“i don’t know the rame thing is it?” said the caterpillar.

“the cormous was it makes he it was it taie the reason of the shall bbout it, you know.”

“i don’t know the rame thing i can’t gelp the sea,” the hatter went on, “i don’t know the peally was in the shall sereat it would be a teally.
the mookee of the court ”

“i don’t know the rame thing is it?” said the caterpillar.

“the cormous was it makes he it was it taie the reason of the shall bbout it, you know.”

“i don’t know the rame thing i can’t gelp the sea,” the hatter went on, “i don’t know the peally was in the shall sereat it would be a teally.
the mookee of the court ”

“i don’t know the rame thing is it?” said the caterpillar.

“the
Done.
```

不仅单词拼写正确，文本也更符合英语。由于交叉熵损失在你训练模型时仍在下降，你可以认为模型尚未收敛。如果你增加训练轮次，可以期待使模型变得更好。

为了完整性，下面是使用这个新模型的完整代码，包括训练和文本生成。

```py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, seq_length, 1)
X = X / float(n_vocab)
y = torch.tensor(dataY)

class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=2, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, n_vocab)
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x

n_epochs = 40
batch_size = 128
model = CharModel()

optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss(reduction="sum")
loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)

best_model = None
best_loss = np.inf
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss += loss_fn(y_pred, y_batch)
        if loss < best_loss:
            best_loss = loss
            best_model = model.state_dict()
        print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))

torch.save([best_model, char_to_int], "single-char.pth")

# Generation using the trained model
best_model, char_to_int = torch.load("single-char.pth")
n_vocab = len(char_to_int)
int_to_char = dict((i, c) for c, i in char_to_int.items())
model.load_state_dict(best_model)

# randomly generate a prompt
filename = "wonderland.txt"
seq_length = 100
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
start = np.random.randint(0, len(raw_text)-seq_length)
prompt = raw_text[start:start+seq_length]
pattern = [char_to_int[c] for c in prompt]

model.eval()
print('Prompt: "%s"' % prompt)
with torch.no_grad():
    for i in range(1000):
        # format input array of int into PyTorch tensor
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        x = torch.tensor(x, dtype=torch.float32)
        # generate logits as output from the model
        prediction = model(x)
        # convert logits into one character
        index = int(prediction.argmax())
        result = int_to_char[index]
        print(result, end="")
        # append the new character into the prompt for the next iteration
        pattern.append(index)
        pattern = pattern[1:]
print()
print("Done.")
```

## 使用 GPU 加速训练

本文中的程序运行可能会非常慢。即使你有 GPU，也不会立刻看到改善。这是因为 PyTorch 的设计，它可能不会自动使用你的 GPU。然而，如果你有支持 CUDA 的 GPU，通过将重计算任务从 CPU 移开，你可以大大提高性能。

PyTorch 模型是一个张量计算程序。张量可以存储在 GPU 或 CPU 中。只要所有操作符在同一个设备上，就可以执行操作。在这个特定的示例中，模型权重（即 LSTM 层和全连接层的权重）可以移动到 GPU 上。这样，输入也应该在执行前移动到 GPU 上，输出也将存储在 GPU 中，除非你将其移动回去。

在 PyTorch 中，你可以使用以下函数检查是否有支持 CUDA 的 GPU：

```py
torch.cuda.is_available()
```

它返回一个布尔值，指示你是否可以使用 GPU，这取决于你拥有的硬件模型、你的操作系统是否安装了适当的库以及你的 PyTorch 是否编译了相应的 GPU 支持。如果一切正常，你可以创建一个设备并将模型分配给它：

```py
device = torch.device("cuda:0")
model.to(device)
```

如果你的模型在 CUDA 设备上运行，但输入张量不在，你会看到 PyTorch 抱怨并无法继续。要将张量移动到 CUDA 设备，你应该运行如下代码：

```py
y_pred = model(X_batch.to(device))
```

`.to(device)` 部分将会起到魔法作用。但请记住，上述程序产生的`y_pred`也将位于 CUDA 设备上。因此，在运行损失函数时，你也应该做同样的操作。修改上述程序，使其能够在 GPU 上运行，将变成以下形式：

```py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, seq_length, 1)
X = X / float(n_vocab)
y = torch.tensor(dataY)

class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=2, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, n_vocab)
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x

n_epochs = 40
batch_size = 128
model = CharModel()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss(reduction="sum")
loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)

best_model = None
best_loss = np.inf
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch.to(device))
        loss = loss_fn(y_pred, y_batch.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch.to(device))
            loss += loss_fn(y_pred, y_batch.to(device))
        if loss < best_loss:
            best_loss = loss
            best_model = model.state_dict()
        print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))

torch.save([best_model, char_to_int], "single-char.pth")

# Generation using the trained model
best_model, char_to_int = torch.load("single-char.pth")
n_vocab = len(char_to_int)
int_to_char = dict((i, c) for c, i in char_to_int.items())
model.load_state_dict(best_model)

# randomly generate a prompt
filename = "wonderland.txt"
seq_length = 100
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
start = np.random.randint(0, len(raw_text)-seq_length)
prompt = raw_text[start:start+seq_length]
pattern = [char_to_int[c] for c in prompt]

model.eval()
print('Prompt: "%s"' % prompt)
with torch.no_grad():
    for i in range(1000):
        # format input array of int into PyTorch tensor
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        x = torch.tensor(x, dtype=torch.float32)
        # generate logits as output from the model
        prediction = model(x.to(device))
        # convert logits into one character
        index = int(prediction.argmax())
        result = int_to_char[index]
        print(result, end="")
        # append the new character into the prompt for the next iteration
        pattern.append(index)
        pattern = pattern[1:]
print()
print("Done.")
```

与前一节中的代码进行比较，你应该能看到它们基本相同。除了 CUDA 设备检测行：

```py
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

这将是你的 GPU，如果没有 CUDA 设备，则会回退到 CPU。随后，在几个关键位置添加了`.to(device)`以将计算移动到 GPU。

## 进一步阅读

这种字符文本模型是使用递归神经网络生成文本的流行方式。如果你有兴趣深入了解，下面还有更多资源和教程。

#### 文章

+   Andrej Karpathy。[递归神经网络的非合理有效性。](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) 2015 年 5 月。

+   Lars Eidnes。[使用递归神经网络自动生成点击诱饵标题](https://larseidnes.com/2015/10/13/auto-generating-clickbait-with-recurrent-neural-networks/)。2015 年。

+   PyTorch 教程。[序列模型与长短期记忆网络](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)

#### 论文

+   Ilya Sutskever、James Martens 和 Geoffrey Hinton。"[使用递归神经网络生成文本](https://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf)"。在：第 28 届国际机器学习会议论文集。2011 年，美国华盛顿州贝尔维尤。

#### API

+   [`PyTorch 文档中的 nn.LSTM`](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

## 总结

在本文中，您了解了如何在 PyTorch 中开发 LSTM 递归神经网络进行文本生成。完成本文后，您将了解到：

+   如何免费获取经典书籍文本作为机器学习模型的数据集

+   如何训练 LSTM 网络处理文本序列

+   如何使用 LSTM 网络生成文本序列如何使用 CUDA 设备优化 PyTorch 中的深度学习训练
