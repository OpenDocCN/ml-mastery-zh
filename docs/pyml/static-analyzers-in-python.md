# Python 中的静态分析器

> 原文：[`machinelearningmastery.com/static-analyzers-in-python/`](https://machinelearningmastery.com/static-analyzers-in-python/)

静态分析器是帮助你检查代码而不实际运行代码的工具。最基本的静态分析器形式是你最喜欢的编辑器中的语法高亮器。如果你需要编译代码（比如在 C++中），你的编译器，如 LLVM，可能还会提供一些静态分析器功能，以警告你潜在的问题（例如，C++中的误用赋值“`=`”代替等于“`==`”）。在 Python 中，我们有一些工具来识别潜在错误或指出代码标准的违反。

完成本教程后，你将学习一些这些工具。具体来说，

+   工具 Pylint、Flake8 和 mypy 能做什么？

+   什么是编码风格违规？

+   我们如何使用类型提示来帮助分析器识别潜在的错误？

**通过我的新书[Python for Machine Learning](https://machinelearningmastery.com/python-for-machine-learning/)**来**启动你的项目**，包括*逐步教程*和所有示例的*Python 源代码*文件。

让我们开始吧！[](../Images/af4ac8df86aea43068b7641185d5001c.png)

Python 中的静态分析器

图片由[Skylar Kang](https://www.pexels.com/photo/blooming-sea-lavender-flowers-on-rough-surface-6044187/)提供。一些权利保留

## 概述

本教程分为三个部分；它们是：

+   **Pylint 简介**

+   **Flake8 简介**

+   **mypy 简介**

## Pylint

Lint 是很久以前为 C 创建的静态分析器的名称。Pylint 借用了这个名字，并且是最广泛使用的静态分析器之一。它作为一个 Python 包提供，我们可以通过`pip`安装：

Shell

```py
$ pip install pylint
```

然后我们在系统中有命令`pylint`可用。

Pylint 可以检查一个脚本或整个目录。例如，如果我们将以下脚本保存为`lenet5-notworking.py`：

```py
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Load MNIST digits
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Reshape data to (n_samples, height, wiedth, n_channel)
X_train = np.expand_dims(X_train, axis=3).astype("float32")
X_test = np.expand_dims(X_test, axis=3).astype("float32")

# One-hot encode the output
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# LeNet5 model
def createmodel(activation):
    model = Sequential([
        Conv2D(6, (5,5), input_shape=(28,28,1), padding="same", activation=activation),
        AveragePooling2D((2,2), strides=2),
        Conv2D(16, (5,5), activation=activation),
        AveragePooling2D((2,2), strides=2),
        Conv2D(120, (5,5), activation=activation),
        Flatten(),
        Dense(84, activation=activation),
        Dense(10, activation="softmax")
    ])
    return model

# Train the model
model = createmodel(tanh)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
earlystopping = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[earlystopping])

# Evaluate the model
print(model.evaluate(X_test, y_test, verbose=0))
model.save("lenet5.h5")
```

我们可以在运行代码之前请 Pylint 告诉我们代码的质量如何：

Shell

```py
$ pylint lenet5-notworking.py
```

输出如下：

```py
************* Module lenet5-notworking
lenet5-notworking.py:39:0: C0301: Line too long (115/100) (line-too-long)
lenet5-notworking.py:1:0: C0103: Module name "lenet5-notworking" doesn't conform to snake_case naming style (invalid-name)
lenet5-notworking.py:1:0: C0114: Missing module docstring (missing-module-docstring)
lenet5-notworking.py:4:0: E0611: No name 'datasets' in module 'LazyLoader' (no-name-in-module)
lenet5-notworking.py:5:0: E0611: No name 'models' in module 'LazyLoader' (no-name-in-module)
lenet5-notworking.py:6:0: E0611: No name 'layers' in module 'LazyLoader' (no-name-in-module)
lenet5-notworking.py:7:0: E0611: No name 'utils' in module 'LazyLoader' (no-name-in-module)
lenet5-notworking.py:8:0: E0611: No name 'callbacks' in module 'LazyLoader' (no-name-in-module)
lenet5-notworking.py:18:25: E0601: Using variable 'y_train' before assignment (used-before-assignment)
lenet5-notworking.py:19:24: E0601: Using variable 'y_test' before assignment (used-before-assignment)
lenet5-notworking.py:23:4: W0621: Redefining name 'model' from outer scope (line 36) (redefined-outer-name)
lenet5-notworking.py:22:0: C0116: Missing function or method docstring (missing-function-docstring)
lenet5-notworking.py:36:20: E0602: Undefined variable 'tanh' (undefined-variable)
lenet5-notworking.py:2:0: W0611: Unused import h5py (unused-import)
lenet5-notworking.py:3:0: W0611: Unused tensorflow imported as tf (unused-import)
lenet5-notworking.py:6:0: W0611: Unused Dropout imported from tensorflow.keras.layers (unused-import)

-------------------------------------
Your code has been rated at -11.82/10
```

如果你将模块的根目录提供给 Pylint，Pylint 将检查该模块的所有组件。在这种情况下，你会看到每行开头的不同文件路径。

这里有几点需要注意。首先，Pylint 的抱怨分为不同的类别。最常见的是我们会看到关于规范（即风格问题）、警告（即代码可能以与预期不同的方式运行）和错误（即代码可能无法运行并抛出异常）的问题。它们通过像 E0601 这样的代码来标识，其中第一个字母是类别。

Pylint 可能会出现误报。在上面的例子中，我们看到 Pylint 将从 `tensorflow.keras.datasets` 的导入标记为错误。这是由于 Tensorflow 包中的优化，导致在导入 Tensorflow 时，Python 并不会扫描和加载所有内容，而是创建了一个 LazyLoader 以仅加载大型包的必要部分。这可以显著节省程序启动时间，但也会使 Pylint 误以为我们导入了不存在的东西。

此外，Pylint 的一个关键特性是帮助我们使代码符合 PEP8 编码风格。例如，当我们定义一个没有文档字符串的函数时，即使代码没有任何错误，Pylint 也会抱怨我们没有遵循编码规范。

但 Pylint 最重要的用途是帮助我们识别潜在的问题。例如，我们将 `y_train` 拼写为大写的 `Y_train`。Pylint 会告诉我们我们在使用一个未赋值的变量。它不会直接告诉我们出了什么问题，但肯定会指向我们审校代码的正确位置。类似地，当我们在第 23 行定义变量 `model` 时，Pylint 告诉我们在外部范围内有一个同名变量。因此，稍后的 `model` 引用可能不是我们想的那样。类似地，未使用的导入可能只是因为我们拼错了模块名称。

这些都是 Pylint 提供的 **提示**。我们仍然需要运用判断来修正代码（或忽略 Pylint 的抱怨）。

但如果你知道 Pylint 应该停止抱怨的内容，你可以要求忽略这些。例如，我们知道 `import` 语句是可以的，所以我们可以用以下命令调用 Pylint：

Shell

```py
$ pylint -d E0611 lenet5-notworking.py
```

现在，所有代码 E0611 的错误将被 Pylint 忽略。你可以通过逗号分隔的列表禁用多个代码，例如：

Shell

```py
$ pylint -d E0611,C0301 lenet5-notworking.py
```

如果你想在特定的行或代码的特定部分禁用某些问题，可以在代码中添加特殊注释，如下所示：

```py
...
from tensorflow.keras.datasets import mnist  # pylint: disable=no-name-in-module
from tensorflow.keras.models import Sequential # pylint: disable=E0611
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
```

魔法关键字 `pylint:` 将引入 Pylint 特定的指令。代码 E0611 和名称 `no-name-in-module` 是相同的。在上面的例子中，由于这些特殊注释，Pylint 会对最后两个导入语句提出抱怨，但不会对前两个提出抱怨。

## Flake8

工具 Flake8 实际上是 PyFlakes、McCabe 和 pycodestyle 的封装器。当你使用以下命令安装 flake8 时：

Shell

```py
$ pip install flake8
```

你将安装所有这些依赖项。

与 Pylint 类似，安装此软件包后，我们可以使用 `flake8` 命令，并可以传递一个脚本或目录进行分析。但 Flake8 的重点倾向于编码风格。因此，对于上述相同的代码，我们会看到以下输出：

Shell

```py
$ flake8 lenet5-notworking.py
lenet5-notworking.py:2:1: F401 'h5py' imported but unused
lenet5-notworking.py:3:1: F401 'tensorflow as tf' imported but unused
lenet5-notworking.py:6:1: F401 'tensorflow.keras.layers.Dropout' imported but unused
lenet5-notworking.py:6:80: E501 line too long (85 > 79 characters)
lenet5-notworking.py:18:26: F821 undefined name 'y_train'
lenet5-notworking.py:19:25: F821 undefined name 'y_test'
lenet5-notworking.py:22:1: E302 expected 2 blank lines, found 1
lenet5-notworking.py:24:21: E231 missing whitespace after ','
lenet5-notworking.py:24:41: E231 missing whitespace after ','
lenet5-notworking.py:24:44: E231 missing whitespace after ','
lenet5-notworking.py:24:80: E501 line too long (87 > 79 characters)
lenet5-notworking.py:25:28: E231 missing whitespace after ','
lenet5-notworking.py:26:22: E231 missing whitespace after ','
lenet5-notworking.py:27:28: E231 missing whitespace after ','
lenet5-notworking.py:28:23: E231 missing whitespace after ','
lenet5-notworking.py:36:1: E305 expected 2 blank lines after class or function definition, found 1
lenet5-notworking.py:36:21: F821 undefined name 'tanh'
lenet5-notworking.py:37:80: E501 line too long (86 > 79 characters)
lenet5-notworking.py:38:80: E501 line too long (88 > 79 characters)
lenet5-notworking.py:39:80: E501 line too long (115 > 79 characters)
```

以字母 E 开头的错误代码来自 pycodestyle，以字母 F 开头的错误代码来自 PyFlakes。我们可以看到它抱怨代码风格问题，例如使用 `(5,5)` 而逗号后没有空格。我们还可以看到它可以识别变量在赋值之前的使用。但它没有捕捉到一些*代码异味*，例如函数 `createmodel()` 重新使用了在外部作用域中已经定义的变量 `model`。

### 想要开始使用 Python 进行机器学习吗？

立即获取我的免费 7 天电子邮件速成课程（包含示例代码）。

点击注册，还可以获得课程的免费 PDF 电子书版本。

与 Pylint 类似，我们也可以要求 Flake8 忽略一些警告。例如，

Shell

```py
flake8 --ignore E501,E231 lenet5-notworking.py
```

这些行不会被打印在输出中：

```py
lenet5-notworking.py:2:1: F401 'h5py' imported but unused
lenet5-notworking.py:3:1: F401 'tensorflow as tf' imported but unused
lenet5-notworking.py:6:1: F401 'tensorflow.keras.layers.Dropout' imported but unused
lenet5-notworking.py:18:26: F821 undefined name 'y_train'
lenet5-notworking.py:19:25: F821 undefined name 'y_test'
lenet5-notworking.py:22:1: E302 expected 2 blank lines, found 1
lenet5-notworking.py:36:1: E305 expected 2 blank lines after class or function definition, found 1
lenet5-notworking.py:36:21: F821 undefined name 'tanh'
```

我们还可以使用魔法注释来禁用一些警告，例如，

```py
...
import tensorflow as tf  # noqa: F401
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
```

Flake8 将查找注释 `# noqa:` 来跳过这些特定行上的一些警告。

## Mypy

Python 不是一种强类型语言，因此，与 C 或 Java 不同，你不需要在使用之前声明一些函数或变量的类型。但最近，Python 引入了类型提示符号，因此我们可以指定一个函数或变量**意图**是什么类型，而不强制遵守像强类型语言那样。

### 想要开始使用 Python 进行机器学习吗？

立即获取我的免费 7 天电子邮件速成课程（包含示例代码）。

点击注册，还可以获得课程的免费 PDF 电子书版本。

在 Python 中使用类型提示的最大好处之一是为静态分析工具提供额外的信息进行检查。 Mypy 是能够理解类型提示的工具。 即使没有类型提示，Mypy 仍然可以提供类似于 Pylint 和 Flake8 的警告。

我们可以从 PyPI 安装 Mypy：

Shell

```py
$ pip install mypy
```

然后可以将上述示例提供给 `mypy` 命令：

```py
$ mypy lenet5-notworking.py
lenet5-notworking.py:2: error: Skipping analyzing "h5py": module is installed, but missing library stubs or py.typed marker
lenet5-notworking.py:2: note: See https://mypy.readthedocs.io/en/stable/running_mypy.html#missing-imports
lenet5-notworking.py:3: error: Skipping analyzing "tensorflow": module is installed, but missing library stubs or py.typed marker
lenet5-notworking.py:4: error: Skipping analyzing "tensorflow.keras.datasets": module is installed, but missing library stubs or py.typed marker
lenet5-notworking.py:5: error: Skipping analyzing "tensorflow.keras.models": module is installed, but missing library stubs or py.typed marker
lenet5-notworking.py:6: error: Skipping analyzing "tensorflow.keras.layers": module is installed, but missing library stubs or py.typed marker
lenet5-notworking.py:7: error: Skipping analyzing "tensorflow.keras.utils": module is installed, but missing library stubs or py.typed marker
lenet5-notworking.py:8: error: Skipping analyzing "tensorflow.keras.callbacks": module is installed, but missing library stubs or py.typed marker
lenet5-notworking.py:18: error: Cannot determine type of "y_train"
lenet5-notworking.py:19: error: Cannot determine type of "y_test"
lenet5-notworking.py:36: error: Name "tanh" is not defined
Found 10 errors in 1 file (checked 1 source file)
```

我们看到与上面的 Pylint 相似的错误，尽管有时不如 Pylint 精确（例如，变量 `y_train` 的问题）。然而，我们在上面看到的一个 mypy 特点是：它期望我们使用的所有库都附带一个存根，以便进行类型检查。这是因为类型提示是**可选**的。如果库中的代码未提供类型提示，代码仍然可以正常工作，但 mypy 无法验证。一些库提供了**类型存根**，使 mypy 可以更好地检查它们。

让我们考虑另一个例子：

```py
import h5py

def dumphdf5(filename: str) -> int:
    """Open a HDF5 file and print all the dataset and attributes stored

    Args:
        filename: The HDF5 filename

    Returns:
        Number of dataset found in the HDF5 file
    """
    count: int = 0

    def recur_dump(obj) -> None:
        print(f"{obj.name} ({type(obj).__name__})")
        if obj.attrs.keys():
            print("\tAttribs:")
            for key in obj.attrs.keys():
                print(f"\t\t{key}: {obj.attrs[key]}")
        if isinstance(obj, h5py.Group):
            # Group has key-value pairs
            for key, value in obj.items():
                recur_dump(value)
        elif isinstance(obj, h5py.Dataset):
            count += 1
            print(obj[()])

    with h5py.File(filename) as obj:
        recur_dump(obj)
        print(f"{count} dataset found")

with open("my_model.h5") as fp:
    dumphdf5(fp)
```

这个程序应该加载一个 HDF5 文件（例如一个 Keras 模型），并打印其中存储的每个属性和数据。我们使用了 `h5py` 模块（它没有类型存根，因此 mypy 无法识别它使用的类型），但我们为我们定义的函数 `dumphdf5()` 添加了类型提示。这个函数期望一个 HDF5 文件的文件名并打印其中存储的所有内容。最后，将返回存储的数据集数量。

当我们将此脚本保存为 `dumphdf5.py` 并传递给 mypy 时，我们将看到如下内容：

Shell

```py
$ mypy dumphdf5.py
dumphdf5.py:1: error: Skipping analyzing "h5py": module is installed, but missing library stubs or py.typed marker
dumphdf5.py:1: note: See https://mypy.readthedocs.io/en/stable/running_mypy.html#missing-imports
dumphdf5.py:3: error: Missing return statement
dumphdf5.py:33: error: Argument 1 to "dumphdf5" has incompatible type "TextIO"; expected "str"
Found 3 errors in 1 file (checked 1 source file)
```

我们误用了函数，导致一个打开的文件对象被传递给`dumphdf5()`，而不是仅仅传递文件名（作为字符串）。Mypy 可以识别这个错误。我们还声明了该函数应该返回一个整数，但函数中没有返回语句。

然而，还有一个错误是 mypy 没有识别出来的。也就是说，内函数`recur_dump()`中使用的变量`count`应该声明为`nonlocal`，因为它是在作用域之外定义的。这个错误可以被 Pylint 和 Flake8 捕获，但 mypy 漏掉了它。

以下是完整的、修正过的代码，没有更多错误。注意，我们在第一行添加了魔法注释“`# type: ignore`”以抑制 mypy 的类型提示警告：

```py
import h5py # type: ignore

def dumphdf5(filename: str) -> int:
    """Open a HDF5 file and print all the dataset and attributes stored

    Args:
        filename: The HDF5 filename

    Returns:
        Number of dataset found in the HDF5 file
    """
    count: int = 0

    def recur_dump(obj) -> None:
        nonlocal count
        print(f"{obj.name} ({type(obj).__name__})")
        if obj.attrs.keys():
            print("\tAttribs:")
            for key in obj.attrs.keys():
                print(f"\t\t{key}: {obj.attrs[key]}")
        if isinstance(obj, h5py.Group):
            # Group has key-value pairs
            for key, value in obj.items():
                recur_dump(value)
        elif isinstance(obj, h5py.Dataset):
            count += 1
            print(obj[()])

    with h5py.File(filename) as obj:
        recur_dump(obj)
        print(f"{count} dataset found")
    return count

dumphdf5("my_model.h5")
```

总之，我们上面介绍的三种工具可以互补。你可以考虑运行所有这些工具，以查找代码中的任何潜在错误或改善编码风格。每个工具都允许一些配置，无论是通过命令行还是配置文件，以适应你的需求（例如，什么样的行长度应该引发警告？）。使用静态分析器也是帮助自己提高编程技能的一种方式。

## 进一步阅读

本节提供了更多关于这个主题的资源，如果你想深入了解。

**文章**

+   PEP8，[`peps.python.org/pep-0008/`](https://peps.python.org/pep-0008/)

+   Google Python 风格指南，[`google.github.io/styleguide/pyguide.html`](https://google.github.io/styleguide/pyguide.html)

**软件包**

+   Pylint 用户手册，[`pylint.pycqa.org/en/latest/index.html`](https://pylint.pycqa.org/en/latest/index.html)

+   Flake8，[`flake8.pycqa.org/en/latest/`](https://flake8.pycqa.org/en/latest/)

+   mypy，[`mypy.readthedocs.io/en/stable/`](https://mypy.readthedocs.io/en/stable/)

## 总结

在本教程中，你已经看到一些常见的静态分析器如何帮助你编写更好的 Python 代码。具体来说，你学习了：

+   三个工具（Pylint、Flake8 和 mypy）的优缺点

+   如何自定义这些工具的行为

+   如何理解这些分析器提出的投诉
