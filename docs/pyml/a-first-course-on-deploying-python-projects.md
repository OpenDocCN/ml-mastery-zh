# 部署 Python 项目的第一课

> 原文：[`machinelearningmastery.com/a-first-course-on-deploying-python-projects/`](https://machinelearningmastery.com/a-first-course-on-deploying-python-projects/)

在用 Python 开发项目的艰苦工作之后，我们想与其他人分享我们的项目。可以是你的朋友或同事。也许他们对你的代码不感兴趣，但他们希望运行并实际使用它。例如，你创建了一个回归模型，可以根据输入特征预测一个值。你的朋友希望提供自己的特征，看看你的模型预测了什么值。但随着你的 Python 项目变大，发送一个小脚本给朋友就不那么简单了。可能有许多支持文件、多重脚本，还依赖于一个库列表。正确处理这些问题可能是一个挑战。

完成本教程后，你将学习到：

+   如何通过将代码模块化来增强其部署的简易性

+   如何为你的模块创建一个包，以便我们可以依赖 `pip` 来管理依赖

+   如何使用 venv 模块创建可重复的运行环境

**快速启动你的项目**，请参考我的新书 [Python for Machine Learning](https://machinelearningmastery.com/python-for-machine-learning/)，包括 *逐步教程* 和 *Python 源代码* 文件，涵盖所有示例。

开始吧！！[](../Images/6c832d8443a1f25dddb3bb7e7d7cde92.png)

部署 Python 项目的第一课

图片来源于 [Kelly L](https://www.pexels.com/photo/tanker-ship-unloading-containers-in-port-6595774/)。版权所有。

## 概述

本教程分为四个部分，它们是：

+   从开发到部署

+   创建模块

+   从模块到包

+   为你的项目使用 venv

## 从开发到部署

当我们完成一个 Python 项目时，有时我们不想将其搁置，而是希望将其转变为常规工作。我们可能完成了一个机器学习模型的训练，并积极使用训练好的模型进行预测。我们可能构建了一个时间序列模型，并用它进行下一步预测。然而，新数据每天都在进入，所以我们需要重新训练模型，以适应发展，并保持未来预测的准确性。

无论原因如何，我们需要确保程序按预期运行。然而，这可能比我们想象的要困难得多。一个简单的 Python 脚本可能不是什么大问题，但随着程序变大，依赖增多，许多事情可能会出错。例如，我们使用的库的新版可能会破坏工作流程。或者我们的 Python 脚本可能运行某个外部程序，而在操作系统升级后，该程序可能停止工作。另一种情况是程序依赖于位于特定路径的文件，但我们可能会不小心删除或重命名文件。

我们的程序总是有可能执行失败的。但我们有一些技巧可以使它更稳健，更可靠。

## 创建模块

在之前的文章中，我们演示了如何使用以下命令检查代码片段的完成时间：

```py
python -m timeit -s 'import numpy as np' 'np.random.random()'
```

同时，我们还可以将其作为脚本的一部分来使用，并执行以下操作：

```py
import timeit
import numpy as np

time = timeit.timeit("np.random.random()", globals=globals())
print(time)
```

Python 中的`import`语句允许你重用定义在另一个文件中的函数，将其视为模块。你可能会想知道我们如何让一个模块不仅提供函数，还能成为一个可执行程序。这是帮助我们部署代码的第一步。如果我们能让模块可执行，用户将无需理解我们的代码结构即可使用它。

如果我们的程序足够大，有多个文件，最好将其打包成一个模块。在 Python 中，模块通常是一个包含 Python 脚本的文件夹，并且有一个明确的入口点。因此，这样更方便传递给其他人，并且更容易理解程序的流程。此外，我们可以为模块添加版本，并让`pip`跟踪安装的版本。

一个简单的单文件程序可以如下编写：

```py
import random

def main():
    n = random.random()
    print(n)

if __name__ == "__main__":
    main()
```

如果我们将其保存为`randomsample.py`在本地目录中，我们可以通过以下方式运行它：

```py
python randomsample.py
```

或：

```py
python -m randomsample
```

我们可以通过以下方式在另一个脚本中重用这些函数：

```py
import randomsample

randomsample.main()
```

这样有效是因为魔法变量`__name__` 只有在脚本作为主程序运行时才会是`"__main__"`，而在从另一个脚本导入时不会是。这样，你的机器学习项目可以可能被打包成如下形式：

```py
regressor/
    __init__.py
    data.json
    model.pickle
    predict.py
    train.py
```

现在，`regressor`是一个包含这五个文件的目录。`__init__.py`是一个**空文件**，仅用于表示该目录是一个可以`import`的 Python 模块。脚本`train.py`如下所示：

```py
import os
import json
import pickle
from sklearn.linear_model import LinearRegression

def load_data():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(current_dir, "data.json")
    data = json.load(open(filepath))
    return data

def train():
    reg = LinearRegression()
    data = load_data()
    reg.fit(data["data"], data["target"])
    return reg
```

`predict.py`的脚本是：

```py
import os
import pickle
import sys
import numpy as np

def predict(features):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(current_dir, "model.pickle")
    with open(filepath, "rb") as fp:
        reg = pickle.load(fp)
    return reg.predict(features)

if __name__ == "__main__":
    arr = np.asarray(sys.argv[1:]).astype(float).reshape(1,-1)
    y = predict(arr)
    print(y[0])
```

然后，我们可以在`regressor/`的父目录下运行以下命令来加载数据并训练线性回归模型。然后，我们可以使用 pickle 保存模型：

```py
import pickle
from regressor.train import train

model = train()
with open("model.pickle", "wb") as fp:
    pickle.save(model, fp)
```

如果我们将这个 pickle 文件移动到`regressor/`目录中，我们还可以在命令行中执行以下操作来运行模型：

```py
python -m regressor.predict 0.186 0 8.3 0 0.62 6.2 58 1.96 6 400 18.1 410 11.5
```

这里的数值参数是输入特征的向量。如果我们进一步移除`if`块，即创建一个文件`regressor/__main__.py`，并使用以下代码：

```py
import sys
import numpy as np
from .predict import predict

if __name__ == "__main__":
    arr = np.asarray(sys.argv[1:]).astype(float).reshape(1,-1)
    y = predict(arr)
    print(y[0])
```

然后我们可以直接从模块运行模型：

```py
python -m regressor 0.186 0 8.3 0 0.62 6.2 58 1.96 6 400 18.1 410 11.5
```

注意上例中的`form .predict import predict`行使用了 Python 的[相对导入语法](https://docs.python.org/3/reference/import.html#package-relative-imports)。这应该在模块内部用于从同一模块的其他脚本中导入组件。

### 想要开始使用 Python 进行机器学习吗？

立即参加我的免费 7 天电子邮件速成课程（附示例代码）。

点击注册，还可以获得课程的免费 PDF 电子书版本。

## 从模块到包

如果你想将你的 Python 项目作为最终产品进行分发，能够将项目作为包用 `pip install` 命令安装会很方便。这很容易做到。既然你已经从项目中创建了一个模块，你需要补充一些简单的设置说明。现在你需要创建一个项目目录，并将你的模块放在其中，配上一个 `pyproject.toml` 文件，一个 `setup.cfg` 文件和一个 `MANIFEST.in` 文件。文件结构应如下所示：

```py
project/
    pyproject.toml
    setup.cfg
    MANIFEST.in
    regressor/
        __init__.py
        data.json
        model.pickle
        predict.py
        train.py
```

我们将使用 `setuptools`，因为它已成为这项任务的标准。文件 `pyproject.toml` 用于指定 `setuptools`：

```py
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
```

关键信息在 `setup.cfg` 中提供。我们需要指定模块的名称、版本、一些可选描述、包含的内容和依赖项，例如以下内容：

```py
[metadata]
name = mlm_demo
version = 0.0.1
description = a simple linear regression model

[options]
packages = regressor
include_package_data = True
python_requires = >=3.6
install_requires =
    scikit-learn==1.0.2
    numpy>=1.22, <1.23
    h5py
```

`MANIFEST.in` 只是用来指定我们需要包含哪些额外的文件。在没有包含非 Python 脚本的项目中，这个文件可以省略。但在我们的情况下，我们需要包含训练好的模型和数据文件：

```py
include regressor/data.json
include regressor/model.pickle
```

然后在项目目录中，我们可以使用以下命令将其作为模块安装到我们的 Python 系统中：

```py
pip install .
```

随后，以下代码在**任何地方**都能正常工作，因为 `regressor` 是我们 Python 安装中的一个可访问模块：

```py
import numpy as np
from regressor.predict import predict

X = np.asarray([[0.186,0,8.3,0,0.62,6.2,58,1.96,6,400,18.1,410,11.5]])
y = predict(X)
print(y[0])
```

在 `setup.cfg` 中有一些细节值得解释：`metadata` 部分是为 `pip` 系统准备的。因此我们将包命名为 `mlm_demo`，你可以在 `pip list` 命令的输出中看到这个名称。然而，Python 的模块系统会将模块名称识别为 `regressor`，如 `options` 部分所指定。因此，这是你在 `import` 语句中应使用的名称。通常，为了用户的方便，这两个名称是相同的，这就是为什么人们会互换使用“包”和“模块”这两个术语。类似地，版本 0.0.1 出现在 `pip` 中，但代码中并未显示。通常将其放在模块目录中的 `__init__.py` 中，因此你可以在使用它的其他脚本中检查版本：

```py
__version__ = '0.0.1'
```

`options` 部分中的 `install_requires` 是让我们的项目运行的关键。这意味着在安装此模块时，我们还需要安装那些其他模块（如果指定的话）。这可能会创建一个依赖树，但当你运行 `pip install` 命令时，`pip` 会处理它。正如你所预期的，我们使用 Python 的比较运算符 `==` 来指定特定版本。但如果我们可以接受多个版本，我们使用逗号（`, `）来分隔条件，例如在 `numpy` 的情况中。

现在你可以将整个项目目录发送给其他人（例如，打包成 ZIP 文件）。他们可以在项目目录中使用 `pip install` 安装它，然后使用 `python -m regressor` 运行你的代码，前提是提供了适当的命令行参数。

最后一点：也许你听说过 Python 项目中的`requirements.txt`文件。它只是一个文本文件，通常放在一个 Python 模块或一些 Python 脚本所在的目录中。它的格式类似于上述提到的依赖项规范。例如，它可能是这样：

```py
scikit-learn==1.0.2
numpy>=1.22, <1.23
h5py
```

目的是你**不想**将你的项目做成一个包，但仍希望给出项目所需库及其版本的提示。这个文件可以被`pip`理解，我们可以用它来设置系统以准备项目：

```py
pip install -r requirements.txt
```

但这仅适用于开发中的项目，这就是`requirements.txt`能够提供的所有便利。

## 使用 venv 管理你的项目

上述方法可能是发布和部署项目的最有效方式，因为你仅包含最关键的文件。这也是推荐的方法，因为它不依赖于平台。如果我们更改 Python 版本或转移到不同的操作系统，这种方法仍然有效（除非某些特定的依赖项禁止我们这样做）。

但有时我们可能希望为项目运行重现一个精确的环境。例如，我们希望一些**不能**安装的包，而不是要求安装某些包。另外，还有些情况下，我们用`pip`安装了一个包后，另一个包的安装会打破版本依赖。我们可以用 Python 的`venv`模块解决这个问题。

`venv`模块来自 Python 的标准库，用于创建**虚拟环境**。它不是像 Docker 提供的虚拟机或虚拟化；相反，它会大量修改 Python 操作的路径位置。例如，我们可以在操作系统中安装多个版本的 Python，但虚拟环境总是假设`python`命令意味着特定版本。另一个例子是，在一个虚拟环境中，我们可以运行`pip install`来设置一些包在虚拟环境目录中，这不会干扰系统外部的环境。

要开始使用`venv`，我们可以简单地找到一个合适的位置并运行以下命令：

```py
$ python -m venv myproject
```

然后将创建一个名为`myproject`的目录。虚拟环境应该在 shell 中运行（以便可以操作环境变量）。要**激活**虚拟环境，我们执行以下命令的激活 shell 脚本（例如，在 Linux 和 macOS 的 bash 或 zsh 下）：

```py
$ source myproject/bin/activate
```

此后，你将处于 Python 虚拟环境中。命令`python`将是你在虚拟环境中创建的命令（如果你在操作系统中安装了多个 Python 版本）。已安装的包将位于`myproject/lib/python3.9/site-packages` （假设使用 Python 3.9）。当你运行`pip install`或`pip list`时，你只会看到虚拟环境中的包。

要离开虚拟环境，我们在 shell 命令行中运行**deactivate**：

```py
$ deactivate
```

这被定义为一个 shell 函数。

如果你有多个项目正在开发，并且它们需要不同版本的包（比如 TensorFlow 的不同版本），使用虚拟环境将特别有用。你可以简单地创建一个虚拟环境，激活它，使用 `pip install` 命令安装所有需要的库的正确版本，然后将你的项目代码放入虚拟环境中。你的虚拟环境目录可能会很大（例如，仅安装 TensorFlow 及其依赖项就会占用接近 1GB 的磁盘空间）。但是，随后将整个虚拟环境目录发送给其他人可以保证执行你的代码的确切环境。如果你不想运行 Docker 服务器，这可以作为 Docker 容器的一种替代方案。

## 进一步阅读

确实，还有其他工具可以帮助我们整洁地部署项目。前面提到的 Docker 可以是其中之一。Python 标准库中的 `zipapp` 包也是一个有趣的工具。如果你想深入了解，下面是关于这个主题的资源。

#### 文章

+   Python 教程，[第六章，模块](https://docs.python.org/3/tutorial/modules.html)

+   [Python 模块分发](https://docs.python.org/3/distributing/index.html)

+   [如何打包你的 Python 代码](https://python-packaging.readthedocs.io/en/latest/index.html)

+   [关于各种与 venv 相关的包](https://stackoverflow.com/questions/41573587) 在 StackOverflow 上的问题

#### APIs 和软件

+   [Setuptools](https://setuptools.pypa.io/en/latest/)

+   [venv](https://docs.python.org/3/library/venv.html) 来自 Python 标准库

## 总结

在本教程中，你已经看到如何确信地完成我们的项目并交付给另一个用户来运行。具体来说，你学到了：

+   将一组 Python 脚本变成模块的最小改动

+   如何将一个模块转换成用于 `pip` 的包

+   Python 中虚拟环境的概念及其使用方法
