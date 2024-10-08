# Python 中装饰器的温和介绍

> 原文：[`machinelearningmastery.com/a-gentle-introduction-to-decorators-in-python/`](https://machinelearningmastery.com/a-gentle-introduction-to-decorators-in-python/)

在编写代码时，无论我们是否意识到，我们常常会遇到装饰器设计模式。这是一种编程技术，可以在不修改类或函数的情况下扩展它们的功能。装饰器设计模式允许我们轻松混合和匹配扩展。Python 具有根植于装饰器设计模式的装饰器语法。了解如何制作和使用装饰器可以帮助你编写更强大的代码。

在这篇文章中，你将发现装饰器模式和 Python 的函数装饰器。

完成本教程后，你将学到：

+   什么是装饰器模式，为什么它有用

+   Python 的函数装饰器及其使用方法

**通过我的新书** [《Python 机器学习》](https://machinelearningmastery.com/python-for-machine-learning/)，*逐步教程* 和所有示例的 *Python 源代码* 文件来**快速启动你的项目**。

让我们开始吧！！[](../Images/a67e1d8293610400f6d7b5fd8ea10829.png)

Python 中装饰器的温和介绍

图片由 [Olya Kobruseva](https://www.pexels.com/photo/a-book-beside-a-cup-of-coffee-6560995/) 提供。保留部分权利。

## 概述

本教程分为四部分：

+   什么是装饰器模式，为什么它有用？

+   Python 中的函数装饰器

+   装饰器的使用案例

+   一些实用的装饰器示例

## 什么是装饰器模式，为什么它有用？

装饰器模式是一种软件设计模式，允许我们动态地向类添加功能，而无需创建子类并影响同一类的其他对象的行为。通过使用装饰器模式，我们可以轻松生成我们可能需要的不同功能排列，而无需创建指数增长数量的子类，从而使我们的代码变得越来越复杂和臃肿。

装饰器通常作为我们想要实现的主要接口的子接口来实现，并存储主要接口类型的对象。然后，它将通过覆盖原始接口中的方法并调用存储对象的方法来修改它希望添加某些功能的方法。

![](https://machinelearningmastery.com/wp-content/uploads/2022/03/Decorator-UML-Class-Diagram.png)

装饰器模式的 UML 类图

上图是装饰器设计模式的 UML 类图。装饰器抽象类包含一个`OriginalInterface`类型的对象；这是装饰器将修改其功能的对象。要实例化我们的具体`DecoratorClass`，我们需要传入一个实现了`OriginalInterface`的具体类，然后当我们调用`DecoratorClass.method1()`方法时，我们的`DecoratorClass`应修改该对象的`method1()`的输出。

然而，通过 Python，我们能够简化许多这些设计模式，因为动态类型以及函数和类是头等对象。虽然在不改变实现的情况下修改类或函数仍然是装饰器的关键思想，但我们将在下面探讨 Python 的装饰器语法。

## Python 中的函数装饰器

函数装饰器是 Python 中一个极其有用的功能。它建立在函数和类在 Python 中是头等对象的概念之上。

让我们考虑一个简单的例子，即调用一个函数两次。由于 Python 函数是对象，并且我们可以将函数作为参数传递给另一个函数，因此这个任务可以如下完成：

```py
def repeat(fn):
    fn()
    fn()

def hello_world():
    print("Hello world!")

repeat(hello_world)
```

同样，由于 Python 函数是对象，我们可以创建一个函数来返回另一个函数，即执行另一个函数两次。这可以如下完成：

```py
def repeat_decorator(fn):
    def decorated_fn():
        fn()
        fn()
    # returns a function
    return decorated_fn

def hello_world():
    print ("Hello world!")

hello_world_twice = repeat_decorator(hello_world)

# call the function
hello_world_twice()
```

上述`repeat_decorator()`返回的函数是在调用时创建的，因为它依赖于提供的参数。在上述代码中，我们将`hello_world`函数作为参数传递给`repeat_decorator()`函数，它返回`decorated_fn`函数，该函数被分配给`hello_world_twice`。之后，我们可以调用`hello_world_twice()`，因为它现在是一个函数。

装饰器模式的理念在这里适用。但我们不需要显式地定义接口和子类。事实上，`hello_world`是在上述示例中定义为一个函数的名称。没有什么阻止我们将这个名称重新定义为其他名称。因此我们也可以这样做：

```py
def repeat_decorator(fn):
    def decorated_fn():
        fn()
        fn()
    # returns a function
    return decorated_fn

def hello_world():
    print ("Hello world!")

hello_world = repeat_decorator(hello_world)

# call the function
hello_world()
```

也就是说，我们不是将新创建的函数分配给`hello_world_twice`，而是覆盖了`hello_world`。虽然`hello_world`的名称被重新分配给另一个函数，但之前的函数仍然存在，只是不对我们公开。

实际上，上述代码在功能上等同于以下代码：

```py
# function decorator that calls the function twice
def repeat_decorator(fn):
    def decorated_fn():
        fn()
        fn()
    # returns a function
    return decorated_fn

# using the decorator on hello_world function
@repeat_decorator
def hello_world():
    print ("Hello world!")

# call the function
hello_world()
```

在上述代码中，`@repeat_decorator`在函数定义之前意味着将函数传递给`repeat_decorator()`并将其名称重新分配给输出。也就是说，相当于`hello_world = repeat_decorator(hello_world)`。`@`行是 Python 中的装饰器语法。

**注意：** `@` 语法在 Java 中也被使用，但含义不同，它是注解，基本上是元数据而不是装饰器。

我们还可以实现接受参数的装饰器，但这会稍微复杂一些，因为我们需要再多一层嵌套。如果我们扩展上面的例子以定义重复函数调用的次数：

```py
def repeat_decorator(num_repeats = 2):
    # repeat_decorator should return a function that's a decorator
    def inner_decorator(fn):
        def decorated_fn():
            for i in range(num_repeats):
                fn()
        # return the new function
        return decorated_fn
    # return the decorator that actually takes the function in as the input
    return inner_decorator

# use the decorator with num_repeats argument set as 5 to repeat the function call 5 times
@repeat_decorator(5)
def hello_world():
    print("Hello world!")

# call the function
hello_world()
```

`repeat_decorator()` 接受一个参数并返回一个函数，这个函数是 `hello_world` 函数的实际装饰器（即，调用 `repeat_decorator(5)` 返回的是 `inner_decorator`，其中本地变量 `num_repeats = 5` 被设置）。上述代码将打印如下内容：

```py
Hello world!
Hello world!
Hello world!
Hello world!
Hello world!
```

在我们结束本节之前，我们应该记住，装饰器不仅可以应用于函数，也可以应用于类。由于 Python 中的类也是一个对象，我们可以用类似的方式重新定义一个类。

### 想开始学习 Python 机器学习吗？

现在就来获取我的免费 7 天电子邮件速成课程（附有示例代码）。

点击注册，并免费获得课程的 PDF 电子书版本。

## 装饰器的使用案例

Python 中的装饰器语法使得装饰器的使用变得更简单。我们使用装饰器的原因有很多，其中一个最常见的用例是隐式地转换数据。例如，我们可以定义一个假设所有操作都基于 numpy 数组的函数，然后创建一个装饰器来确保这一点，通过修改输入：

```py
# function decorator to ensure numpy input
def ensure_numpy(fn):
    def decorated_function(data):
        # converts input to numpy array
        array = np.asarray(data)
        # calls fn on input numpy array
        return fn(array)
    return decorated_function
```

我们可以进一步修改装饰器，通过调整函数的输出，例如对浮点值进行四舍五入：

```py
# function decorator to ensure numpy input
# and round off output to 4 decimal places
def ensure_numpy(fn):
    def decorated_function(data):
        array = np.asarray(data)
        output = fn(array)
        return np.around(output, 4)
    return decorated_function
```

让我们考虑一个求数组和的例子。一个 numpy 数组有内置的 `sum()` 方法，pandas DataFrame 也是如此。但是，后者是对列求和，而不是对所有元素求和。因此，一个 numpy 数组会得到一个浮点值的和，而 DataFrame 则会得到一个值的向量。但通过上述装饰器，我们可以编写一个函数，使得在这两种情况下都能得到一致的输出：

```py
import numpy as np
import pandas as pd

# function decorator to ensure numpy input
# and round off output to 4 decimal places
def ensure_numpy(fn):
    def decorated_function(data):
        array = np.asarray(data)
        output = fn(array)
        return np.around(output, 4)
    return decorated_function

@ensure_numpy
def numpysum(array):
    return array.sum()

x = np.random.randn(10,3)
y = pd.DataFrame(x, columns=["A", "B", "C"])

# output of numpy .sum() function
print("x.sum():", x.sum())
print()

# output of pandas .sum() funuction
print("y.sum():", y.sum())
print(y.sum())
print()

# calling decorated numpysum function
print("numpysum(x):", numpysum(x))
print("numpysum(y):", numpysum(y))
```

运行上述代码会得到如下输出：

```py
x.sum(): 0.3948331694737762

y.sum(): A   -1.175484
B    2.496056
C   -0.925739
dtype: float64
A   -1.175484
B    2.496056
C   -0.925739
dtype: float64

numpysum(x): 0.3948
numpysum(y): 0.3948
```

这是一个简单的例子。但是想象一下，如果我们定义一个新函数来计算数组中元素的标准差。我们可以简单地使用相同的装饰器，这样函数也会接受 pandas DataFrame。因此，所有的输入处理代码都被移到了装饰器中。这就是我们如何高效重用代码的方法。

## 一些实际的装饰器示例

既然我们学习了 Python 中的装饰器语法，那我们来看看可以用它做些什么吧！

### 备忘录化

有些函数调用我们会重复进行，但这些值很少甚至几乎不变。这可能是对数据相对静态的服务器的调用，或者作为动态编程算法或计算密集型数学函数的一部分。我们可能想要**备忘录化**这些函数调用，即将它们的输出值存储在虚拟备忘录中以便后续重用。

装饰器是实现备忘录化函数的最佳方式。我们只需要记住函数的输入和输出，但保持函数的行为不变。下面是一个例子：

```py
import pickle
import hashlib

MEMO = {} # To remember the function input and output

def memoize(fn):
    def _deco(*args, **kwargs):
        # pickle the function arguments and obtain hash as the store keys
        key = (fn.__name__, hashlib.md5(pickle.dumps((args, kwargs), 4)).hexdigest())
        # check if the key exists
        if key in MEMO:
            ret = pickle.loads(MEMO[key])
        else:
            ret = fn(*args, **kwargs)
            MEMO[key] = pickle.dumps(ret)
        return ret
    return _deco

@memoize
def fibonacci(n):
    if n in [0, 1]:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(40))
print(MEMO)
```

在这个示例中，我们实现了`memoize()`函数以便与全局字典`MEMO`一起工作，使得函数名与参数组成键，函数的返回值成为值。当调用函数时，装饰器会检查对应的键是否存在于`MEMO`中，如果存在，则返回存储的值。否则，将调用实际的函数，并将其返回值添加到字典中。

我们使用`pickle`来序列化输入和输出，并使用`hashlib`来创建输入的哈希，因为并不是所有东西都可以作为 Python 字典的键（例如，`list`是不可哈希的类型，因此不能作为键）。将任何任意结构序列化为字符串可以克服这个问题，并确保返回数据是不可变的。此外，对函数参数进行哈希处理可以避免在字典中存储异常长的键（例如，当我们将一个巨大的 numpy 数组传递给函数时）。

上述示例使用`fibonacci()`演示了记忆化的强大功能。调用`fibonacci(n)`将生成第 n 个斐波那契数。运行上述示例将产生以下输出，其中我们可以看到第 40 个斐波那契数是 102334155，以及字典`MEMO`是如何用于存储对函数的不同调用的。

```py
102334155
{('fibonacci', '635f1664f168e2a15b8e43f20d45154b'): b'\x80\x04K\x01.',
('fibonacci', 'd238998870ae18a399d03477dad0c0a8'): b'\x80\x04K\x00.',
('fibonacci', 'dbed6abf8fcf4beec7fc97f3170de3cc'): b'\x80\x04K\x01.',
...
('fibonacci', 'b9954ff996a4cd0e36fffb09f982b08e'): b'\x80\x04\x95\x06\x00\x00\x00\x00\x00\x00\x00J)pT\x02.',
('fibonacci', '8c7aba62def8063cf5afe85f42372f0d'): b'\x80\x04\x95\x06\x00\x00\x00\x00\x00\x00\x00J\xa2\x0e\xc5\x03.',
('fibonacci', '6de8535f23d756de26959b4d6e1f66f6'): b'\x80\x04\x95\x06\x00\x00\x00\x00\x00\x00\x00J\xcb~\x19\x06.'}
```

你可以尝试去掉上述代码中的`@memoize`行。你会发现程序运行时间显著增加（因为每次函数调用都会调用两个额外的函数调用，因此它的运行复杂度是 O(2^n)，而记忆化情况下为 O(n)），或者你可能会遇到内存不足的问题。

记忆化对那些输出不经常变化的昂贵函数非常有帮助，例如，下面的函数从互联网读取一些股市数据：

```py
...

import pandas_datareader as pdr

@memoize
def get_stock_data(ticker):
    # pull data from stooq
    df = pdr.stooq.StooqDailyReader(symbols=ticker, start="1/1/00", end="31/12/21").read()
    return df

#testing call to function
import cProfile as profile
import pstats

for i in range(1, 3):
    print(f"Run {i}")
    run_profile = profile.Profile()
    run_profile.enable()
    get_stock_data("^DJI")
    run_profile.disable()
    pstats.Stats(run_profile).print_stats(0)
```

如果实现正确，第一次调用`get_stock_data()`应该会更昂贵，而后续调用则会便宜得多。上述代码片段的输出结果是：

```py
Run 1
         17492 function calls (17051 primitive calls) in 1.452 seconds

Run 2
         221 function calls (218 primitive calls) in 0.001 seconds
```

如果你正在使用 Jupyter notebook，这特别有用。如果需要下载一些数据，将其包装在 memoize 装饰器中。由于开发机器学习项目意味着多次更改代码以查看结果是否有所改善，使用记忆化下载函数可以节省大量不必要的等待时间。

你可以通过将数据存储在数据库中（例如，像 GNU dbm 这样的键值存储或像 memcached 或 Redis 这样的内存数据库）来创建一个更强大的记忆化装饰器。但如果你只需要上述功能，Python 3.2 或更高版本的内置库`functools`中已经提供了装饰器`lru_cache`，因此你不需要自己编写：

```py
import functools

import pandas_datareader as pdr

# memoize using lru_cache
@functools.lru_cache
def get_stock_data(ticker):
    # pull data from stooq
    df = pdr.stooq.StooqDailyReader(symbols=ticker, start="1/1/00", end="31/12/21").read()
    return df

# testing call to function
import cProfile as profile
import pstats

for i in range(1, 3):
    print(f"Run {i}")
    run_profile = profile.Profile()
    run_profile.enable()
    get_stock_data("^DJI")
    run_profile.disable()
    pstats.Stats(run_profile).print_stats(0)
```

**注意：** `lru_cache`实现了 LRU 缓存，它将其大小限制为对函数的最新调用（默认 128）。在 Python 3.9 中，还有一个`@functools.cache`，其大小无限制，不进行 LRU 清除。

### 函数目录

另一个我们可能希望考虑使用函数装饰器的例子是用于在目录中注册函数。它允许我们将函数与字符串关联，并将这些字符串作为其他函数的参数传递。这是构建一个允许用户提供插件的系统的开始。让我们用一个例子来说明。以下是一个装饰器和我们稍后将使用的函数`activate()`。假设以下代码保存于文件`activation.py`中：

```py
# activation.py

ACTIVATION = {}

def register(name):
    def decorator(fn):
        # assign fn to "name" key in ACTIVATION
        ACTIVATION[name] = fn
        # return fn unmodified
        return fn
    return decorator

def activate(x, kind):
    try:
        fn = ACTIVATION[kind]
        return fn(x)
    except KeyError:
        print("Activation function %s undefined" % kind)
```

在上面的代码中定义了`register`装饰器之后，我们现在可以使用它来注册函数并将字符串与之关联。让我们来看一下`funcs.py`文件：

```py
# funcs.py

from activation import register
import numpy as np

@register("relu")
def relu(x):
    return np.where(x>0, x, 0)

@register("sigmoid")
def sigm(x):
    return 1/(1+np.exp(-x))

@register("tanh")
def tanh(x):
    return np.tanh(x)
```

我们通过在`ACTIVATION`字典中建立这种关联，将“relu”，“sigmoid”和“tanh”函数注册到各自的字符串。

现在，让我们看看如何使用我们新注册的函数。

```py
import numpy as np
from activation import activate

# create a random matrix
x = np.random.randn(5,3)
print(x)

# try ReLU activation on the matrix
relu_x = activate(x, "relu")
print(relu_x)

# load the functions, and call ReLU activation again
import funcs
relu_x = activate(x, "relu")
print(relu_x)
```

这将给我们输出：

```py
[[-0.81549502 -0.81352867  1.41539545]
 [-0.28782853 -1.59323543 -0.19824959]
 [ 0.06724466 -0.26622761 -0.41893662]
 [ 0.47927331 -1.84055276 -0.23147207]
 [-0.18005588 -1.20837815 -1.34768876]]
Activation function relu undefined
None
[[0\.         0\.         1.41539545]
 [0\.         0\.         0\.        ]
 [0.06724466 0\.         0\.        ]
 [0.47927331 0\.         0\.        ]
 [0\.         0\.         0\.        ]]
```

请注意，在我们到达`import func`这一行之前，ReLU 激活函数并不存在。因此调用该函数会打印错误信息，结果为`None`。然后在我们运行那一行`import`之后，我们就像加载插件模块一样加载了那些定义的函数。之后同样的函数调用给出了我们预期的结果。

请注意，我们从未显式调用模块`func`中的任何内容，也没有修改`activate()`的调用。仅仅导入`func`就使得那些新函数注册并扩展了`activate()`的功能。使用这种技术允许我们在开发非常大的系统时，只关注一小部分，而不必担心其他部分的互操作性。如果没有注册装饰器和函数目录，添加新的激活函数将需要修改**每一个**使用激活的函数。

如果你对 Keras 很熟悉，你应该能将上述内容与以下语法产生共鸣：

```py
layer = keras.layers.Dense(128, activation="relu")

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["sparse_categorical_accuracy"])
```

Keras 几乎使用类似性质的装饰器定义了所有组件。因此我们可以通过名称引用构建块。如果没有这种机制，我们必须一直使用以下语法，这让我们需要记住很多组件的位置：

```py
layer = keras.layers.Dense(128, activation=keras.activations.relu)

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), 
              optimizer=keras.optimizers.Adam(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
```

## 进一步阅读

本节提供了更多关于该主题的资源，如果你希望深入了解。

### 文章

+   [装饰器模式](https://blogs.oracle.com/javamagazine/post/the-decorator-pattern-in-depth)

+   Python 语言参考，第 8.7 节，[函数定义](https://docs.python.org/3/reference/compound_stmts.html#function)

+   [PEP 318 – 函数和方法的装饰器](https://peps.python.org/pep-0318/)

### 书籍

+   [Fluent Python](https://www.amazon.com/dp/1492056359/)，第二版，作者 Luciano Ramalho

### API

+   Python 标准库中的[functools 模块](https://docs.python.org/3/library/functools.html)

## 总结

在这篇文章中，你了解了装饰器设计模式和 Python 的装饰器语法。你还看到了一些装饰器的具体使用场景，这些可以帮助你的 Python 程序运行得更快或更易扩展。

具体来说，你学习了：

+   装饰器模式的概念以及 Python 中的装饰器语法

+   如何在 Python 中实现装饰器，以便使用装饰器语法

+   使用装饰器来适配函数输入输出、实现记忆化以及在目录中注册函数
