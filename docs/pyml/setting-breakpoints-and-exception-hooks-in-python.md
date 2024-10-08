# 在 Python 中设置断点和异常钩子

> 原文：[`machinelearningmastery.com/setting-breakpoints-and-exception-hooks-in-python/`](https://machinelearningmastery.com/setting-breakpoints-and-exception-hooks-in-python/)

在 Python 中调试代码有多种方法，其中之一是在代码中引入断点，以便在希望调用 Python 调试器的地方设置断点。不同调用点使用的语句取决于你所使用的 Python 解释器版本，正如我们在本教程中将看到的那样。

在本教程中，你将发现设置断点的各种方法，适用于不同版本的 Python。

完成本教程后，你将了解：

+   如何在早期版本的 Python 中调用 `pdb` 调试器

+   如何使用 Python 3.7 中引入的新内置 `breakpoint()` 函数

+   如何编写自己的 `breakpoint()` 函数，以简化早期版本 Python 中的调试过程

+   如何使用事后调试器

**通过我的新书 [《Python 机器学习》](https://machinelearningmastery.com/python-for-machine-learning/)** 启动你的项目，其中包括 *逐步教程* 和 *所有示例的 Python 源代码* 文件。

开始吧。![](https://machinelearningmastery.com/wp-content/uploads/2022/01/cover_breakpoints-scaled.jpg)

在不同版本的 Python 中设置断点

照片由 [Josh Withers](https://unsplash.com/photos/OfBDvcXuA88) 提供，部分权利保留。

## **教程概述**

本教程分为三个部分，它们是：

+   在 Python 代码中设置断点

    +   在早期版本的 Python 中调用 pdb 调试器

    +   在 Python 3.7 中使用 `breakpoint()` 函数

+   为早期版本的 Python 编写自己的 `breakpoint()` 函数

+   `breakpoint()` 函数的限制

## **在 Python 代码中设置断点**

[我们之前已经看到](https://machinelearningmastery.com/python-debugging-tools/) 调试 Python 脚本的一种方法是使用 Python 调试器在命令行中运行它。

为此，我们需要使用 `-m pdb` 命令，该命令在执行 Python 脚本之前加载 pdb 模块。在相同的命令行界面中，我们可以跟随一个特定的调试器命令，例如 `n` 以移动到下一行，或 `s` 如果我们打算进入一个函数。

随着代码长度的增加，这种方法可能会变得繁琐。解决这个问题并更好地控制代码断点的一种方法是直接在代码中插入断点。

### **在早期版本的 Python 中调用 pdb 调试器**

在 Python 3.7 之前调用 pdb 调试器，需要导入 pdb 并在代码中希望进入交互调试会话的地方调用 `pdb.set_trace()`。

如果我们重新考虑，比如说，代码用于 [实现通用注意力机制](https://machinelearningmastery.com/the-attention-mechanism-from-scratch/)，我们可以按如下方式进入代码：

Python

```py
from numpy import array
from numpy import random
from numpy import dot
from scipy.special import softmax

# importing the Python debugger module
import pdb

# encoder representations of four different words
word_1 = array([1, 0, 0])
word_2 = array([0, 1, 0])
word_3 = array([1, 1, 0])
word_4 = array([0, 0, 1])

# stacking the word embeddings into a single array
words = array([word_1, word_2, word_3, word_4])

# generating the weight matrices
random.seed(42)
W_Q = random.randint(3, size=(3, 3))
W_K = random.randint(3, size=(3, 3))
W_V = random.randint(3, size=(3, 3))

# generating the queries, keys and values
Q = dot(words, W_Q)
K = dot(words, W_K)
V = dot(words, W_V)

# inserting a breakpoint
pdb.set_trace()

# scoring the query vectors against all key vectors
scores = dot(Q, K.transpose())

# computing the weights by a softmax operation
weights = softmax(scores / K.shape[1] ** 0.5, axis=1)

# computing the attention by a weighted sum of the value vectors
attention = dot(weights, V)

print(attention)
```

现在执行脚本会在计算变量 `scores` 之前打开 `pdb` 调试器，我们可以继续发出任何调试器命令，例如 n 以移动到下一行或 c 以继续执行：

Python

```py
/Users/mlm/main.py(33)<module>()
-> scores = dot(Q, K.transpose())
(Pdb) n
> /Users/mlm/main.py(36)<module>()
-> weights = softmax(scores / K.shape[1] ** 0.5, axis=1)
(Pdb) c
[[0.98522025 1.74174051 0.75652026]
 [0.90965265 1.40965265 0.5       ]
 [0.99851226 1.75849334 0.75998108]
 [0.99560386 1.90407309 0.90846923]]
```

尽管功能正常，但这不是将断点插入代码的最优雅和直观的方法。Python 3.7 实现了一种更直接的方法，接下来我们将看到。

### **在 Python 3.7 中使用 `breakpoint()` 函数**

Python 3.7 附带了一个内置的 `breakpoint()` 函数，该函数在调用站点（即 `breakpoint()` 语句所在的代码点）进入 Python 调试器。

当调用时，`breakpoint()` 函数的默认实现会调用 `sys.breakpointhook()`，而 `sys.breakpointhook()` 进而调用 `pdb.set_trace()` 函数。这很方便，因为我们不需要自己显式地导入 `pdb` 并调用 `pdb.set_trace()`。

让我们重新考虑实现通用注意力机制的代码，并通过 `breakpoint()` 语句引入一个断点：

Python

```py
from numpy import array
from numpy import random
from scipy.special import softmax

# encoder representations of four different words
word_1 = array([1, 0, 0])
word_2 = array([0, 1, 0])
word_3 = array([1, 1, 0])
word_4 = array([0, 0, 1])

# stacking the word embeddings into a single array
words = array([word_1, word_2, word_3, word_4])

# generating the weight matrices
random.seed(42)
W_Q = random.randint(3, size=(3, 3))
W_K = random.randint(3, size=(3, 3))
W_V = random.randint(3, size=(3, 3))

# generating the queries, keys and values
Q = words @ W_Q
K = words @ W_K
V = words @ W_V

# inserting a breakpoint
breakpoint()

# scoring the query vectors against all key vectors
scores = Q @ K.transpose()

# computing the weights by a softmax operation
weights = softmax(scores / K.shape[1] ** 0.5, axis=1)

# computing the attention by a weighted sum of the value vectors
attention = weights @ V

print(attention)
```

使用 `breakpoint()` 函数的一个优点是，调用 `sys.breakpointhook()` 的默认实现时，会查阅一个新的环境变量 PYTHONBREAKPOINT 的值。这个环境变量可以取不同的值，根据这些值可以执行不同的操作。

例如，将 PYTHONBREAKPOINT 的值设置为 0 会禁用所有断点。因此，您的代码可以包含尽可能多的断点，但这些断点可以很容易地被停止，而无需实际删除它们。如果（例如）包含代码的脚本名称为 *main.py*，我们可以通过在命令行界面中如下调用来禁用所有断点：

Python

```py
PYTHONBREAKPOINT=0 python main.py
```

否则，我们可以通过在代码中设置环境变量来实现相同的结果：

Python

```py
from numpy import array
from numpy import random
from scipy.special import softmax

# setting the value of the PYTHONBREAKPOINT environment variable
import os
os.environ['PYTHONBREAKPOINT'] = '0'

# encoder representations of four different words
word_1 = array([1, 0, 0])
word_2 = array([0, 1, 0])
word_3 = array([1, 1, 0])
word_4 = array([0, 0, 1])

# stacking the word embeddings into a single array
words = array([word_1, word_2, word_3, word_4])

# generating the weight matrices
random.seed(42)
W_Q = random.randint(3, size=(3, 3))
W_K = random.randint(3, size=(3, 3))
W_V = random.randint(3, size=(3, 3))

# generating the queries, keys and values
Q = words @ W_Q
K = words @ W_K
V = words @ W_V

# inserting a breakpoint
breakpoint()

# scoring the query vectors against all key vectors
scores = Q @ K.transpose()

# computing the weights by a softmax operation
weights = softmax(scores / K.shape[1] ** 0.5, axis=1)

# computing the attention by a weighted sum of the value vectors
attention = weights @ V

print(attention)
```

每次调用 `sys.breakpointhook()` 时，都会查阅 PYTHONBREAKPOINT 的值。这意味着该环境变量的值在代码执行期间可以更改，而 `breakpoint()` 函数会相应地作出反应。

PYTHONBREAKPOINT 环境变量也可以设置为其他值，例如可调用对象的名称。例如，如果我们想使用除了 `pdb` 之外的其他 Python 调试器，如 `ipdb`（如果调试器尚未安装，请先运行 `pip install ipdb`）。在这种情况下，我们可以在命令行界面中调用 *main.py* 脚本并挂钩调试器，而无需对代码本身进行任何更改：

Python

```py
PYTHONBREAKPOINT=ipdb.set_trace python main.py
```

这样，`breakpoint()` 函数会在下一个调用站点进入 `ipdb` 调试器：

Python

```py
> /Users/Stefania/Documents/PycharmProjects/BreakpointPy37/main.py(33)<module>()
     32 # scoring the query vectors against all key vectors
---> 33 scores = Q @ K.transpose()
     34 

ipdb> n
> /Users/Stefania/Documents/PycharmProjects/BreakpointPy37/main.py(36)<module>()
     35 # computing the weights by a softmax operation
---> 36 weights = softmax(scores / K.shape[1] ** 0.5, axis=1)
     37 

ipdb> c
[[0.98522025 1.74174051 0.75652026]
 [0.90965265 1.40965265 0.5       ]
 [0.99851226 1.75849334 0.75998108]
 [0.99560386 1.90407309 0.90846923]]
```

该函数还可以接受输入参数，如 breakpoint(*args, **kws)，这些参数会传递给 sys.breakpointhook()。这是因为任何可调用对象（如第三方调试模块）可能接受可选参数，这些参数可以通过 breakpoint() 函数传递。

### 想要开始使用 Python 进行机器学习？

现在获取我的免费 7 天电子邮件速成课程（附示例代码）。

点击注册并免费获得课程的 PDF 电子书版本。

## **在早期版本的 Python 中编写自己的 breakpoint() 函数**

让我们回到 Python 3.7 之前的版本不自带 breakpoint() 函数的事实。我们可以编写自己的函数。

与从 Python 3.7 开始实现的 breakpoint() 函数类似，我们可以实现一个检查环境变量值的函数，并：

+   如果环境变量的值设置为 0，则会跳过代码中的所有断点。

+   如果环境变量为空字符串，则进入默认的 Python pdb 调试器。

+   根据环境变量的值进入另一个调试器。

Python

```py
...

# defining our breakpoint() function
def breakpoint(*args, **kwargs):
    import importlib
    # reading the value of the environment variable
    val = os.environ.get('PYTHONBREAKPOINT')
    # if the value has been set to 0, skip all breakpoints
    if val == '0':
        return None
    # else if the value is an empty string, invoke the default pdb debugger
    elif len(val) == 0:
        hook_name = 'pdb.set_trace'
    # else, assign the value of the environment variable
    else:
        hook_name = val
    # split the string into the module name and the function name
    mod, dot, func = hook_name.rpartition('.')
    # get the function from the module
    module = importlib.import_module(mod)
    hook = getattr(module, func)

    return hook(*args, **kwargs)

...
```

我们可以将这个函数包含到代码中并运行（在此例中使用 Python 2.7 解释器）。如果我们将环境变量的值设置为空字符串，我们会发现 pdb 调试器会停在我们放置了 breakpoint() 函数的代码点。然后，我们可以从那里开始在命令行中输入调试器命令：

Python

```py
from numpy import array
from numpy import random
from numpy import dot
from scipy.special import softmax

# setting the value of the environment variable
import os
os.environ['PYTHONBREAKPOINT'] = ''

# defining our breakpoint() function
def breakpoint(*args, **kwargs):
    import importlib
    # reading the value of the environment variable
    val = os.environ.get('PYTHONBREAKPOINT')
    # if the value has been set to 0, skip all breakpoints
    if val == '0':
        return None
    # else if the value is an empty string, invoke the default pdb debugger
    elif len(val) == 0:
        hook_name = 'pdb.set_trace'
    # else, assign the value of the environment variable
    else:
        hook_name = val
    # split the string into the module name and the function name
    mod, dot, func = hook_name.rpartition('.')
    # get the function from the module
    module = importlib.import_module(mod)
    hook = getattr(module, func)

    return hook(*args, **kwargs)

# encoder representations of four different words
word_1 = array([1, 0, 0])
word_2 = array([0, 1, 0])
word_3 = array([1, 1, 0])
word_4 = array([0, 0, 1])

# stacking the word embeddings into a single array
words = array([word_1, word_2, word_3, word_4])

# generating the weight matrices
random.seed(42)
W_Q = random.randint(3, size=(3, 3))
W_K = random.randint(3, size=(3, 3))
W_V = random.randint(3, size=(3, 3))

# generating the queries, keys and values
Q = dot(words, W_Q)
K = dot(words, W_K)
V = dot(words, W_V)

# inserting a breakpoint
breakpoint()

# scoring the query vectors against all key vectors
scores = dot(Q, K.transpose())

# computing the weights by a softmax operation
weights = softmax(scores / K.shape[1] ** 0.5, axis=1)

# computing the attention by a weighted sum of the value vectors
attention = dot(weights, V)

print(attention)
```

Python

```py
> /Users/Stefania/Documents/PycharmProjects/BreakpointPy27/main.py(32)breakpoint()->None
-> return hook(*args, **kwargs)
(Pdb) n
> /Users/Stefania/Documents/PycharmProjects/BreakpointPy27/main.py(59)<module>()
-> scores = dot(Q, K.transpose())
(Pdb) n
> /Users/Stefania/Documents/PycharmProjects/BreakpointPy27/main.py(62)<module>()
-> weights = softmax(scores / K.shape[1] ** 0.5, axis=1)
(Pdb) c
[[0.98522025 1.74174051 0.75652026]
 [0.90965265 1.40965265 0.5       ]
 [0.99851226 1.75849334 0.75998108]
 [0.99560386 1.90407309 0.90846923]]
```

同样地，如果我们将环境变量设置为：

Python

```py
os.environ['PYTHONBREAKPOINT'] = 'ipdb.set_trace'
```

我们现在实现的 `breakpoint()` 函数会进入 ipdb 调试器并停在调用点：

Python

```py
> /Users/Stefania/Documents/PycharmProjects/BreakpointPy27/main.py(31)breakpoint()
     30 
---> 31     return hook(*args, **kwargs)
     32 

ipdb> n
> /Users/Stefania/Documents/PycharmProjects/BreakpointPy27/main.py(58)<module>()
     57 # scoring the query vectors against all key vectors
---> 58 scores = dot(Q, K.transpose())
     59 

ipdb> n
> /Users/Stefania/Documents/PycharmProjects/BreakpointPy27/main.py(61)<module>()
     60 # computing the weights by a softmax operation
---> 61 weights = softmax(scores / K.shape[1] ** 0.5, axis=1)
     62 

ipdb> c
[[0.98522025 1.74174051 0.75652026]
 [0.90965265 1.40965265 0.5       ]
 [0.99851226 1.75849334 0.75998108]
 [0.99560386 1.90407309 0.90846923]]
```

将环境变量设置为 0 会跳过所有断点，计算得到的注意力输出会按预期返回到命令行：

Python

```py
os.environ['PYTHONBREAKPOINT'] = '0'
```

Python

```py
[[0.98522025 1.74174051 0.75652026]
 [0.90965265 1.40965265 0.5       ]
 [0.99851226 1.75849334 0.75998108]
 [0.99560386 1.90407309 0.90846923]]
```

这简化了 Python 3.7 之前版本的代码调试过程，因为现在只需设置环境变量的值，而不必手动在代码中的不同调用点引入（或移除）import pdb; pdb.set_trace() 语句。

## `breakpoint()` 函数的限制

`breakpoint()` 函数允许你在程序的某个点引入调试器。你需要找到需要调试器放置断点的确切位置。如果你考虑以下代码：

```py
try:
    func()
except:
    breakpoint()
    print("exception!")
```

当函数 `func()` 引发异常时，这将带来调试器。它可以由函数自身或它调用的其他函数中的深处触发。但调试器会在上述 `print("exception!")` 这一行开始，这可能不是很有用。

当事后调试器启动时，我们可以在异常点调试器处打印回溯和异常。这种方式被称为**事后调试器**。当未捕获异常被引发时，它会请求 Python 将调试器`pdb.pm()`注册为异常处理程序。当调用它时，它将查找最后引发的异常并从那一点开始启动调试器。要使用事后调试器，我们只需在运行程序之前添加以下代码：

```py
import sys
import pdb

def debughook(etype, value, tb):
    pdb.pm() # post-mortem debugger
sys.excepthook = debughook
```

这很方便，因为程序中不需要进行任何其他更改。例如，假设我们想要使用以下程序评估$1/x$的平均值。很容易忽视一些边界情况，但是当引发异常时，我们可以捕获问题：

```py
import sys
import pdb
import random

def debughook(etype, value, tb):
    pdb.pm() # post-mortem debugger
sys.excepthook = debughook

# Experimentally find the average of 1/x where x is a random integer in 0 to 9999
N = 1000
randomsum = 0
for i in range(N):
    x = random.randint(0,10000)
    randomsum += 1/x

print("Average is", randomsum/N)
```

当我们运行上述程序时，程序可能会终止，或者在循环中的随机数生成器生成零时可能会引发除零异常。在这种情况下，我们可能会看到以下内容：

```py
> /Users/mlm/py_pmhook.py(17)<module>()
-> randomsum += 1/x
(Pdb) p i
16
(Pdb) p x
0
```

我们找到了异常引发的位置以及我们可以像通常在`pdb`中做的那样检查变量的值。

实际上，在启动事后调试器时，打印回溯和异常更加方便：

```py
import sys
import pdb
import traceback

def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    print() # make a new line before launching post-mortem
    pdb.pm() # post-mortem debugger
sys.excepthook = debughook
```

调试器会话将如下启动：

```py
Traceback (most recent call last):
  File "/Users/mlm/py_pmhook.py", line 17, in <module>
    randomsum += 1/x
ZeroDivisionError: division by zero

> /Users/mlm/py_pmhook.py(17)<module>()
-> randomsum += 1/x
(Pdb)
```

## **进一步阅读**

如果你希望深入了解，本节提供了更多关于这个主题的资源。

### **网站**

+   Python pdb 模块，[`docs.python.org/3/library/pdb.html`](https://docs.python.org/3/library/pdb.html)

+   Python 内置断点函数`breakpoint()`，[`www.python.org/dev/peps/pep-0553/`](https://www.python.org/dev/peps/pep-0553/)

## **总结**

在本教程中，你了解了在不同版本的 Python 中设置断点的各种方法。

具体来说，你学到了：

+   如何在早期版本的 Python 中调用 pdb 调试器。

+   如何使用 Python 3.7 中引入的新内置断点函数`breakpoint()`。

+   如何编写自己的断点函数以简化早期版本 Python 中的调试过程

你有任何问题吗？

在下面的评论中提出你的问题，我会尽力回答。
