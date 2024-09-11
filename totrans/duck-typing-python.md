# Python 中的鸭子类型、作用域和探索性函数

> 原文：[`machinelearningmastery.com/duck-typing-python/`](https://machinelearningmastery.com/duck-typing-python/)

Python 是一种鸭子类型的语言。这意味着变量的数据类型可以随着语法的兼容性而改变。Python 也是一种动态编程语言。这意味着我们可以在程序运行时更改它，包括定义新函数和名称解析的作用域。这不仅为编写 Python 代码提供了新的范式，还为调试提供了新的工具集。接下来，我们将看到在 Python 中可以做到的，而许多其他语言无法做到的事情。

完成本教程后，你将了解到：

+   Python 是如何管理你定义的变量的

+   Python 代码如何使用你定义的变量以及为什么我们不需要像在 C 或 Java 中那样定义其类型

**用我的新书 [Python 机器学习](https://machinelearningmastery.com/python-for-machine-learning/) 来启动你的项目**，包括*逐步教程*和所有示例的*Python 源代码*文件。

让我们开始吧！![](img/d8f668f859e9d752510aecd2adad80f4.png)

Python 中的鸭子类型、作用域和探索性函数。照片由[朱莉莎·赫尔穆特](https://www.pexels.com/photo/flock-of-yellow-baby-ducks-in-grass-4381480/)拍摄。部分权利保留

## 概述

这个教程分为三部分；它们是

+   编程语言中的鸭子类型

+   Python 中的作用域和命名空间

+   调查类型和作用域

## 编程语言中的鸭子类型

鸭子类型是一些现代编程语言的特性，允许数据类型是动态的。

> 编程风格不查看对象类型以确定其接口是否正确；而是直接调用或使用方法或属性（“如果它看起来像鸭子并且嘎嘎叫，那它肯定是只鸭子。”）。通过强调接口而不是特定类型，设计良好的代码通过允许多态替换来提高灵活性。

— [Python 词汇表](https://docs.python.org/3/glossary.html)

简单来说，只要相同的语法仍然有意义，程序应该允许你交换数据结构。例如，在 C 语言中，你必须像以下这样定义函数：

C 语言

```py
float fsquare(float x)
{
    return x * x;
};

int isquare(int x)
{
    return x * x;
};
```

虽然操作 `x * x` 对于整数和浮点数来说是相同的，但接受整数参数和接受浮点数参数的函数并不相同。因为在 C 语言中类型是静态的，所以尽管它们执行相同的逻辑，我们必须定义两个函数。在 Python 中，类型是动态的；因此，我们可以定义相应的函数为：

```py
def square(x):
    return x * x
```

这个特性确实给我们带来了巨大的力量和便利。例如，从 scikit-learn 中，我们有一个做交叉验证的函数：

```py
# evaluate a perceptron model on the dataset
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import Perceptron
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
# define model
model = Perceptron()
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize result
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

但是在上述示例中，`model` 是 scikit-learn 模型对象的一个变量。无论它是像上述中的感知器模型，决策树，还是支持向量机模型，都无关紧要。重要的是在 `cross_val_score()` 函数内部，数据将通过其 `fit()` 函数传递给模型。因此，模型必须实现 `fit()` 成员函数，并且 `fit()` 函数的行为相同。其结果是 `cross_val_score()` 函数不需要特定的模型类型，只要它看起来像一个模型即可。如果我们使用 Keras 构建神经网络模型，我们可以通过包装使 Keras 模型看起来像一个 scikit-learn 模型：

```py
# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_diabetes
import numpy

# Function to create model, required for KerasClassifier
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

在上述中，我们使用了来自 Keras 的包装器。还有其他的包装器，比如 scikeras。它所做的就是确保 Keras 模型的接口看起来像一个 scikit-learn 分类器，这样你就可以利用 `cross_val_score()` 函数。如果我们用以下内容替换上述的 `model`：

```py
model = create_model()
```

那么 scikit-learn 函数会抱怨找不到 `model.score()` 函数。

同样地，由于鸭子类型，我们可以重用一个期望列表的函数来处理 NumPy 数组或 pandas series，因为它们都支持相同的索引和切片操作。例如，我们可以如下拟合时间序列与 ARIMA 模型：

```py
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import pandas as pd

data = [266.0,145.9,183.1,119.3,180.3,168.5,231.8,224.5,192.8,122.9,336.5,185.9,
        194.3,149.5,210.1,273.3,191.4,287.0,226.0,303.6,289.9,421.6,264.5,342.3,
        339.7,440.4,315.9,439.3,401.3,437.4,575.5,407.6,682.0,475.3,581.3,646.9]
model = SARIMAX(y, order=(5,1,0))
res = model.fit(disp=False)
print("AIC = ", res.aic)

data = np.array(data)
model = SARIMAX(y, order=(5,1,0))
res = model.fit(disp=False)
print("AIC = ", res.aic)

data = pd.Series(data)
model = SARIMAX(y, order=(5,1,0))
res = model.fit(disp=False)
print("AIC = ", res.aic)
```

上述应该为每次拟合生成相同的 AIC 分数。

## Python 中的作用域和命名空间

在大多数语言中，变量是在有限的作用域内定义的。例如，函数内部定义的变量只能在该函数内部访问：

```py
from math import sqrt

def quadratic(a,b,c):
    discrim = b*b - 4*a*c
    x = -b/(2*a)
    y = sqrt(discrim)/(2*a)
    return x-y, x+y
```

**局部变量** `discrim` 如果不在函数 `quadratic()` 内部，是无法访问的。这对某些人来说可能会有所惊讶：

```py
a = 1

def f(x):
    a = 2 * x
    return a

b = f(3)
print(a, b)
```

```py
1 6
```

我们在函数 `f` 外部定义了变量 `a`，但是在函数 `f` 内部，变量 `a` 被赋值为 `2 * x`。然而，函数内部的 `a` 和外部的 `a` 是无关的，除了名称。因此，当我们退出函数时，变量 `a` 的值没有改变。为了在函数 `f` 内部使其可修改，我们需要声明名称 `a` 是 `global` 的，以明确指出这个名称应该来自**全局作用域**，而不是**局部作用域**：

```py
a = 1

def f(x):
    global a
    a = 2 * x
    return a

b = f(3)
print(a, b)
```

```py
6 6
```

然而，当在函数中引入**嵌套作用域**时，问题可能进一步复杂化。考虑以下示例：

```py
a = 1

def f(x):
    a = x
    def g(x):
        return a * x
    return g(3)

b = f(2)
print(b)
```

```py
6
```

函数 `f` 内部的变量 `a` 与全局的变量 `a` 是不同的。然而，在函数 `g` 中，由于没有对 `a` 进行写入操作，而只是读取，Python 将从最近的作用域即函数 `f` 中找到相同的 `a`。变量 `x` 则是作为函数 `g` 的参数定义，并在调用 `g(3)` 时取值为 `3`，而不是假定来自函数 `f` 中的 `x` 的值。

**注意：** 如果变量在函数的**任何地方**有值被赋予，它就被定义在局部作用域中。如果在赋值之前从中读取变量的值，则会引发错误，而不是使用外部或全局作用域中同名变量的值。

此属性有多种用途。Python 中许多记忆化装饰器的实现巧妙地利用了函数作用域。另一个例子是以下内容：

```py
import numpy as np

def datagen(X, y, batch_size, sampling_rate=0.7):
    """A generator to produce samples from input numpy arrays X and y
    """
    # Select rows from arrays X and y randomly
    indexing = np.random.random(len(X)) < sampling_rate
    Xsam, ysam = X[indexing], y[indexing]

    # Actual logic to generate batches
    def _gen(batch_size):
        while True:
            Xbatch, ybatch = [], []
            for _ in range(batch_size):
                i = np.random.randint(len(Xsam))
                Xbatch.append(Xsam[i])
                ybatch.append(ysam[i])
            yield np.array(Xbatch), np.array(ybatch)

    # Create and return a generator
    return _gen(batch_size)
```

这是一个创建从输入 NumPy 数组 `X` 和 `y` 中批量样本的**生成器函数**。这样的生成器在 Keras 模型的训练中是可接受的。然而，出于诸如交叉验证等原因，我们不希望从整个输入数组 `X` 和 `y` 中采样，而是从它们的一个**固定**子集中随机选择行。我们通过在 `datagen()` 函数的开头随机选择一部分行并将它们保留在 `Xsam`、`ysam` 中来实现这一点。然后在内部函数 `_gen()` 中，从 `Xsam` 和 `ysam` 中对行进行采样，直到创建一个批次。虽然列表 `Xbatch` 和 `ybatch` 在函数 `_gen()` 内部被定义和创建，但数组 `Xsam` 和 `ysam` 不是 `_gen()` 的局部变量。更有趣的是生成器被创建时的情况：

```py
X = np.random.random((100,3))
y = np.random.random(100)

gen1 = datagen(X, y, 3)
gen2 = datagen(X, y, 4)
print(next(gen1))
print(next(gen2))
```

```py
(array([[0.89702235, 0.97516228, 0.08893787],
       [0.26395301, 0.37674529, 0.1439478 ],
       [0.24859104, 0.17448628, 0.41182877]]), array([0.2821138 , 0.87590954, 0.96646776]))
(array([[0.62199772, 0.01442743, 0.4897467 ],
       [0.41129379, 0.24600387, 0.53640666],
       [0.02417213, 0.27637708, 0.65571031],
       [0.15107433, 0.11331674, 0.67000849]]), array([0.91559533, 0.84886957, 0.30451455, 0.5144225 ]))
```

函数 `datagen()` 被调用两次，因此创建了两组不同的 `Xsam`、`yam`。但由于内部函数 `_gen()` 依赖于它们，这两组 `Xsam`、`ysam` 同时存在于内存中。技术上来说，我们称当调用 `datagen()` 时，会创建一个具有特定 `Xsam`、`ysam` 的**闭包**，并且调用 `_gen()` 会访问该闭包。换句话说，两次调用 `datagen()` 的作用域是共存的。

总结一下，每当一行代码引用一个名称（无论是变量、函数还是模块），名称都按照 LEGB 规则的顺序解析：

1.  首先是局部作用域，即在同一函数中定义的名称

1.  闭包或“非局部”作用域。如果我们在嵌套函数内部，这是上一级函数。

1.  全局作用域，即在同一脚本顶层定义的名称（但不跨不同程序文件）

1.  内置作用域，即由 Python 自动创建的作用域，例如变量 `__name__` 或函数 `list()`

### 想要开始使用 Python 进行机器学习吗？

立即注册我的免费 7 天电子邮件速成课程（附有示例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## 调查类型和作用域

因为 Python 中类型不是静态的，有时我们想知道我们在处理什么，但从代码中并不容易看出。一种方法是使用 `type()` 或 `isinstance()` 函数。例如：

```py
import numpy as np

X = np.random.random((100,3))
print(type(X))
print(isinstance(X, np.ndarray))
```

```py
<class 'numpy.ndarray'>
True
```

`type()` 函数返回一个类型对象。`isinstance()` 函数返回一个布尔值，允许我们检查某个对象是否匹配特定类型。这在我们需要知道变量的类型时非常有用。如果我们将 pandas 数据框传递给我们上面定义的 `datagen()` 函数：

```py
import pandas as pd
import numpy as np

def datagen(X, y, batch_size, sampling_rate=0.7):
    """A generator to produce samples from input numpy arrays X and y
    """
    # Select rows from arrays X and y randomly
    indexing = np.random.random(len(X)) < sampling_rate
    Xsam, ysam = X[indexing], y[indexing]

    # Actual logic to generate batches
    def _gen(batch_size):
        while True:
            Xbatch, ybatch = [], []
            for _ in range(batch_size):
                i = np.random.randint(len(Xsam))
                Xbatch.append(Xsam[i])
                ybatch.append(ysam[i])
            yield np.array(Xbatch), np.array(ybatch)

    # Create and return a generator
    return _gen(batch_size)

X = pd.DataFrame(np.random.random((100,3)))
y = pd.DataFrame(np.random.random(100))

gen3 = datagen(X, y, 3)
print(next(gen3))
```

在 Python 的调试器 `pdb` 下运行上述代码将得到如下结果：

```py
> /Users/MLM/ducktype.py(1)<module>()
-> import pandas as pd
(Pdb) c
Traceback (most recent call last):
  File "/usr/local/lib/python3.9/site-packages/pandas/core/indexes/range.py", line 385, in get_loc
    return self._range.index(new_key)
ValueError: 1 is not in range

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/Cellar/python@3.9/3.9.9/Frameworks/Python.framework/Versions/3.9/lib/python3.9/pdb.py", line 1723, in main
    pdb._runscript(mainpyfile)
  File "/usr/local/Cellar/python@3.9/3.9.9/Frameworks/Python.framework/Versions/3.9/lib/python3.9/pdb.py", line 1583, in _runscript
    self.run(statement)
  File "/usr/local/Cellar/python@3.9/3.9.9/Frameworks/Python.framework/Versions/3.9/lib/python3.9/bdb.py", line 580, in run
    exec(cmd, globals, locals)
  File "<string>", line 1, in <module>
  File "/Users/MLM/ducktype.py", line 1, in <module>
    import pandas as pd
  File "/Users/MLM/ducktype.py", line 18, in _gen
    ybatch.append(ysam[i])
  File "/usr/local/lib/python3.9/site-packages/pandas/core/frame.py", line 3458, in __getitem__
    indexer = self.columns.get_loc(key)
  File "/usr/local/lib/python3.9/site-packages/pandas/core/indexes/range.py", line 387, in get_loc
    raise KeyError(key) from err
KeyError: 1
Uncaught exception. Entering post mortem debugging
Running 'cont' or 'step' will restart the program
> /usr/local/lib/python3.9/site-packages/pandas/core/indexes/range.py(387)get_loc()
-> raise KeyError(key) from err
(Pdb)
```

从追踪信息中我们看到出了问题，因为我们无法获取 `ysam[i]`。我们可以使用以下方法来验证 `ysam` 确实是一个 Pandas DataFrame 而不是一个 NumPy 数组：

```py
(Pdb) up
> /usr/local/lib/python3.9/site-packages/pandas/core/frame.py(3458)__getitem__()
-> indexer = self.columns.get_loc(key)
(Pdb) up
> /Users/MLM/ducktype.py(18)_gen()
-> ybatch.append(ysam[i])
(Pdb) type(ysam)
<class 'pandas.core.frame.DataFrame'>
```

因此，我们不能使用 `ysam[i]` 从 `ysam` 中选择行 `i`。我们在调试器中可以做什么来验证我们应该如何修改代码？有几个有用的函数可以用来调查变量和作用域：

+   `dir()` 用于查看作用域中定义的名称或对象中定义的属性

+   `locals()` 和 `globals()` 用于查看本地和全局定义的名称和值。

例如，我们可以使用 `dir(ysam)` 来查看 `ysam` 内部定义了哪些属性或函数：

```py
(Pdb) dir(ysam)
['T', '_AXIS_LEN', '_AXIS_ORDERS', '_AXIS_REVERSED', '_AXIS_TO_AXIS_NUMBER', 
...
'iat', 'idxmax', 'idxmin', 'iloc', 'index', 'infer_objects', 'info', 'insert',
'interpolate', 'isin', 'isna', 'isnull', 'items', 'iteritems', 'iterrows',
'itertuples', 'join', 'keys', 'kurt', 'kurtosis', 'last', 'last_valid_index',
...
'transform', 'transpose', 'truediv', 'truncate', 'tz_convert', 'tz_localize',
'unstack', 'update', 'value_counts', 'values', 'var', 'where', 'xs']
(Pdb)
```

其中一些是属性，如 `shape`，还有一些是函数，如 `describe()`。你可以在 `pdb` 中读取属性或调用函数。通过仔细阅读这个输出，我们回忆起从 DataFrame 中读取行 `i` 的方法是通过 `iloc`，因此我们可以用以下语法进行验证：

```py
(Pdb) ysam.iloc[i]
0    0.83794
Name: 2, dtype: float64
(Pdb)
```

如果我们调用 `dir()` 而不带任何参数，它将给出当前作用域中定义的所有名称，例如，

```py
(Pdb) dir()
['Xbatch', 'Xsam', '_', 'batch_size', 'i', 'ybatch', 'ysam']
(Pdb) up
> /Users/MLM/ducktype.py(1)<module>()
-> import pandas as pd
(Pdb) dir()
['X', '__builtins__', '__file__', '__name__', 'datagen', 'gen3', 'np', 'pd', 'y']
(Pdb)
```

由于作用域会随着你在调用栈中移动而变化。类似于没有参数的 `dir()`，我们可以调用 `locals()` 来显示所有本地定义的变量，例如，

```py
(Pdb) locals()
{'batch_size': 3, 'Xbatch': ...,
 'ybatch': ..., '_': 0, 'i': 1, 'Xsam': ...,
 'ysam': ...}
(Pdb)
```

确实，`locals()` 返回一个 `dict`，允许你查看所有的名称和值。因此，如果我们需要读取变量 `Xbatch`，可以通过 `locals()["Xbatch"]` 来获取相同的内容。类似地，我们可以使用 `globals()` 来获取全局作用域中定义的名称字典。

这种技术有时是有益的。例如，我们可以通过使用 `dir(model)` 来检查一个 Keras 模型是否“编译”了。在 Keras 中，编译模型是为训练设置损失函数，并建立前向和反向传播的流程。因此，已编译的模型将具有额外定义的属性 `loss`：

```py
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(5, input_shape=(3,)),
    Dense(1)
])

has_loss = "loss" in dir(model)
print("Before compile, loss function defined:", has_loss)

model.compile()
has_loss = "loss" in dir(model)
print("After compile, loss function defined:", has_loss)
```

```py
Before compile, loss function defined: False
After compile, loss function defined: True
```

这使我们在代码运行之前添加了额外的保护，以防止出现错误。

## 进一步阅读

本节提供了更多关于该主题的资源，如果你想深入了解。

#### 文章

+   鸭子类型，[`en.wikipedia.org/wiki/Duck_typing`](https://en.wikipedia.org/wiki/Duck_typing)

+   Python 术语表（鸭子类型），[`docs.python.org/3/glossary.html#term-duck-typing`](https://docs.python.org/3/glossary.html#term-duck-typing)

+   Python 内置函数，[`docs.python.org/3/library/functions.html`](https://docs.python.org/3/library/functions.html)

#### 书籍

+   *流畅的 Python*，第二版，作者 Luciano Ramalho，[`www.amazon.com/dp/1492056359/`](https://www.amazon.com/dp/1492056359/)

## 概述

在本教程中，您已经看到了 Python 如何组织命名空间以及变量如何与代码交互。具体来说，您学到了：

+   Python 代码通过它们的接口使用变量；因此，变量的数据类型通常不重要。

+   Python 变量是在它们的命名空间或闭包中定义的，同名变量可以在不同的作用域中共存，因此它们不会互相干扰。

+   我们有一些来自 Python 的内置函数，允许我们检查当前作用域中定义的名称或变量的数据类型。
