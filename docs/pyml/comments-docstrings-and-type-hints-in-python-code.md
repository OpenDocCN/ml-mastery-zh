# Python 代码中的注释、文档字符串和类型提示

> 原文：[`machinelearningmastery.com/comments-docstrings-and-type-hints-in-python-code/`](https://machinelearningmastery.com/comments-docstrings-and-type-hints-in-python-code/)

程序的源代码应该对人类可读。使其正确运行只是其目的的一半。如果没有适当的注释代码，任何人，包括未来的你，将很难理解代码背后的理由和意图。这样也会使代码无法维护。在 Python 中，有多种方式可以向代码添加描述，使其更具可读性或使意图更加明确。在接下来的内容中，我们将看到如何正确使用注释、文档字符串和类型提示，使我们的代码更易于理解。完成本教程后，你将了解：

+   什么是 Python 中注释的正确使用方法

+   字符串字面量或文档字符串在某些情况下如何替代注释

+   Python 中的类型提示是什么，它们如何帮助我们更好地理解代码

**快速启动你的项目**，参考我新书 [《Python 机器学习》](https://machinelearningmastery.com/python-for-machine-learning/)，包括 *逐步教程* 和 *所有示例的 Python 源代码* 文件。

让我们开始吧！[](../Images/376dbd33f3cff1575c682e181db413ae.png)

Python 代码中的注释、文档字符串和类型提示。照片由 [Rhythm Goyal](https://unsplash.com/photos/_-Ofoh09q_o) 提供。版权所有

## 概述

本教程分为三部分，它们是：

+   向 Python 代码添加注释

+   使用文档字符串

+   在 Python 代码中使用类型提示

## 向 Python 代码添加注释

几乎所有编程语言都有专门的注释语法。注释会被编译器或解释器忽略，因此它们不会影响编程流程或逻辑。但通过注释，可以更容易地阅读代码。

在像 C++ 这样的语言中，我们可以使用前导双斜杠（`//`）添加“行内注释”或使用 `/*` 和 `*/` 包围的注释块。然而，在 Python 中，我们只有“行内”版本，它们由前导井号字符（`#`）引入。

编写注释以解释每一行代码是很容易的，但这通常是一种浪费。当人们阅读源代码时，注释往往很容易引起他们的注意，因此放太多注释会分散阅读注意力。例如，以下内容是不必要且具有干扰性的：

```py
import datetime

timestamp = datetime.datetime.now()  # Get the current date and time
x = 0    # initialize x to zero
```

这样的注释仅仅是重复代码的功能。除非代码非常晦涩，这些注释不会为代码增添价值。下面的例子可能是一个边际情况，其中名称“ppf”（百分比点函数）比术语“CDF”（累积分布函数）更不为人知：

```py
import scipy.stats

z_alpha = scipy.stats.norm.ppf(0.975)  # Call the inverse CDF of standard normal
```

优秀的注释应该说明我们为什么要做某件事。让我们来看一个例子：

```py
def adadelta(objective, derivative, bounds, n_iter, rho, ep=1e-3):
    # generate an initial point
    solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    # lists to hold the average square gradients for each variable and
    # average parameter updates
    sq_grad_avg = [0.0 for _ in range(bounds.shape[0])]
    sq_para_avg = [0.0 for _ in range(bounds.shape[0])]
    # run the gradient descent
    for it in range(n_iter):
        gradient = derivative(solution[0], solution[1])
        # update the moving average of the squared partial derivatives
        for i in range(gradient.shape[0]):
            sg = gradient[i]**2.0
            sq_grad_avg[i] = (sq_grad_avg[i] * rho) + (sg * (1.0-rho))
        # build a solution one variable at a time
        new_solution = list()
        for i in range(solution.shape[0]):
            # calculate the step size for this variable
            alpha = (ep + sqrt(sq_para_avg[i])) / (ep + sqrt(sq_grad_avg[i]))
            # calculate the change and update the moving average of the squared change
            change = alpha * gradient[i]
            sq_para_avg[i] = (sq_para_avg[i] * rho) + (change**2.0 * (1.0-rho))
            # calculate the new position in this variable and store as new solution
            value = solution[i] - change
            new_solution.append(value)
        # evaluate candidate point
        solution = asarray(new_solution)
        solution_eval = objective(solution[0], solution[1])
        # report progress
        print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
    return [solution, solution_eval]
```

上面的函数实现了 AdaDelta 算法。在第一行中，当我们将某事物分配给变量`solution`时，我们不会写像“在`bounds[:,0]`和`bounds[:,1]`之间的随机插值”这样的评论，因为那只是重复的代码。我们说这行的意图是“生成一个初始点”。类似地，在函数中的其他注释中，我们标记一个 for 循环作为梯度下降算法，而不仅仅是说迭代若干次。

在编写评论或修改代码时我们要记住的一个重要问题是确保评论准确描述代码。如果它们相矛盾，对读者来说会很困惑。因此，当你打算在上面的例子的第一行放置评论以“将初始解设为下界”时，而代码显然是随机化初始解时，或者反之，你应该同时更新评论和代码。

一个例外是“待办事项”评论。不时地，当我们有改进代码的想法但尚未更改时，我们可以在代码上加上待办事项评论。我们也可以用它来标记未完成的实现。例如，

```py
# TODO replace Keras code below with Tensorflow
from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(1, (3,3), strides=(2, 2), input_shape=(8, 8, 1)))
model.summary()
...
```

这是一个常见的做法，当关键字`TODO`被发现时，许多 IDE 会以不同的方式突出显示评论块。然而，它应该是临时的，我们不应滥用它作为问题跟踪系统。

总之，有关编写注释代码的一些常见“最佳实践”列举如下：

+   评论不应重复代码而应该解释它

+   评论不应造成混淆而应消除它

+   在不易理解的代码上放置注释；例如，说明语法的非典型使用，命名正在使用的算法，或者解释意图或假设

+   评论应该简洁明了

+   保持一致的风格和语言在评论中使用

+   总是更喜欢写得更好的代码，而不需要额外的注释

## 使用文档字符串

在 C++中，我们可以编写大块的评论，如下所示：

C++

```py
TcpSocketBase::~TcpSocketBase (void)
{
  NS_LOG_FUNCTION (this);
  m_node = nullptr;
  if (m_endPoint != nullptr)
    {
      NS_ASSERT (m_tcp != nullptr);
      /*
       * Upon Bind, an Ipv4Endpoint is allocated and set to m_endPoint, and
       * DestroyCallback is set to TcpSocketBase::Destroy. If we called
       * m_tcp->DeAllocate, it will destroy its Ipv4EndpointDemux::DeAllocate,
       * which in turn destroys my m_endPoint, and in turn invokes
       * TcpSocketBase::Destroy to nullify m_node, m_endPoint, and m_tcp.
       */
      NS_ASSERT (m_endPoint != nullptr);
      m_tcp->DeAllocate (m_endPoint);
      NS_ASSERT (m_endPoint == nullptr);
    }
  if (m_endPoint6 != nullptr)
    {
      NS_ASSERT (m_tcp != nullptr);
      NS_ASSERT (m_endPoint6 != nullptr);
      m_tcp->DeAllocate (m_endPoint6);
      NS_ASSERT (m_endPoint6 == nullptr);
    }
  m_tcp = 0;
  CancelAllTimers ();
}
```

但在 Python 中，我们没有`/*`和`*/`这样的界定符的等价物，但我们可以用以下方式写多行注释：

```py
async def main(indir):
    # Scan dirs for files and populate a list
    filepaths = []
    for path, dirs, files in os.walk(indir):
        for basename in files:
            filepath = os.path.join(path, basename)
            filepaths.append(filepath)

    """Create the "process pool" of 4 and run asyncio.
    The processes will execute the worker function
    concurrently with each file path as parameter
    """
    loop = asyncio.get_running_loop()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [loop.run_in_executor(executor, func, f) for f in filepaths]
        for fut in asyncio.as_completed(futures):
            try:
                filepath = await fut
                print(filepath)
            except Exception as exc:
                print("failed one job")
```

这是因为 Python 支持声明跨多行的字符串字面量，如果它用三引号（`"""`）界定。而在代码中，字符串字面量仅仅是一个没有影响的声明。因此，它在功能上与评论没有任何区别。

我们希望使用字符串字面量的一个原因是注释掉一大块代码。例如，

```py
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
"""
X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           n_clusters_per_class=1,
                           weights=[0.01, 0.05, 0.94],
                           class_sep=0.8, random_state=0)
"""
import pickle
with open("dataset.pickle", "wb") as fp:
    X, y = pickle.load(fp)

clf = LogisticRegression(random_state=0).fit(X, y)
...
```

以上是我们可能通过尝试机器学习问题而开发的样本代码。虽然我们在开始时随机生成了一个数据集（上面的`make_classification()`调用），但我们可能希望在以后的某个时间切换到另一个数据集并重复相同的过程（例如上面的 pickle 部分）。我们可以简单地注释这些行而不是删除代码块，以便稍后存储代码。尽管它不适合最终代码的形式，但在开发解决方案时非常方便。

在 Python 中，作为注释的字符串字面量如果位于函数下的第一行，则具有特殊目的。在这种情况下，该字符串字面量被称为函数的“docstring”。例如，

```py
def square(x):
    """Just to compute the square of a value

    Args:
        x (int or float): A numerical value

    Returns:
        int or float: The square of x
    """
    return x * x
```

我们可以看到函数下的第一行是一个字面字符串，它与注释具有相同的作用。它使代码更易读，但同时我们可以从代码中检索到它：

```py
print("Function name:", square.__name__)
print("Docstring:", square.__doc__)
```

```py
Function name: square
Docstring: Just to compute the square of a value

    Args:
        x (int or float): A numerical value

    Returns:
        int or float: The square of x
```

由于 docstring 的特殊地位，有几种关于如何编写适当的 docstring 的约定。

在 C++中，我们可以使用 Doxygen 从注释中生成代码文档，类似地，Java 中有 Javadoc。Python 中最接近的匹配工具将是来自 Sphinx 或 pdoc 的“autodoc”。两者都会尝试解析 docstring 以自动生成文档。

没有标准的 docstring 编写方式，但通常我们期望它们将解释函数（或类或模块）的目的以及参数和返回值。一个常见的风格如上所述，由 Google 推崇。另一种风格来自 NumPy：

```py
def square(x):
    """Just to compupte the square of a value

    Parameters
    ----------
    x : int or float
        A numerical value

    Returns
    -------
    int or float
        The square of `x`
    """
    return x * x
```

类似 autodoc 这样的工具可以解析这些 docstring 并生成 API 文档。但即使这不是目的，使用一个描述函数性质、函数参数和返回值数据类型的 docstring 肯定可以使您的代码更易于阅读。这一点特别重要，因为 Python 不像 C++或 Java 那样是一种**鸭子类型**语言，其中变量和函数参数不声明为特定类型。我们可以利用 docstring 来说明数据类型的假设，以便人们更容易地跟踪或使用您的函数。

### 想要开始 Python 机器学习吗？

现在就参加我的免费 7 天电子邮件速成课程（附带示例代码）。

点击注册并获取免费的课程 PDF 电子书版本。

## 在 Python 代码中使用类型提示

自 Python 3.5 以来，允许类型提示语法。顾名思义，它的目的是提示类型而不是其他任何内容。因此，即使看起来将 Python 更接近 Java，它也不意味着限制要存储在变量中的数据。上面的示例可以使用类型提示进行重写：

```py
def square(x: int) -> int:
    return x * x
```

在函数中，参数后面可以跟着`: type`的语法来说明*预期*的类型。函数的返回值通过冒号前的`-> type`语法来标识。事实上，变量也可以声明类型提示，例如，

```py
def square(x: int) -> int:
    value: int = x * x
    return value
```

类型提示的好处是双重的：我们可以用它来消除一些注释，如果我们需要明确描述正在使用的数据类型。我们还可以帮助*静态分析器*更好地理解我们的代码，以便它们能够帮助识别代码中的潜在问题。

有时类型可能很复杂，因此 Python 在其标准库中提供了`typing`模块来帮助简化语法。例如，我们可以使用`Union[int,float]`表示`int`类型或`float`类型，`List[str]`表示每个元素都是字符串的列表，并使用`Any`表示任何类型。如下所示：

```py
from typing import Any, Union, List

def square(x: Union[int, float]) -> Union[int, float]:
    return x * x

def append(x: List[Any], y: Any) -> None:
    x.append(y)
```

然而，重要的是要记住，类型提示只是*提示*。它不对代码施加任何限制。因此，以下对读者来说可能很困惑，但完全合法：

```py
n: int = 3.5
n = "assign a string"
```

使用类型提示可以提高代码的可读性。然而，类型提示最重要的好处是允许像**mypy**这样的*静态分析器*告诉我们我们的代码是否有潜在的 bug。如果你用 mypy 处理以上代码行，我们会看到以下错误：

```py
test.py:1: error: Incompatible types in assignment (expression has type "float", variable has type "int")
test.py:2: error: Incompatible types in assignment (expression has type "str", variable has type "int")
Found 2 errors in 1 file (checked 1 source file)
```

静态分析器的使用将在另一篇文章中介绍。

为了说明注释、文档字符串和类型提示的使用，以下是一个例子，定义了一个生成器函数，该函数在固定宽度窗口上对 pandas DataFrame 进行采样。这对训练 LSTM 网络非常有用，其中需要提供几个连续的时间步骤。在下面的函数中，我们从 DataFrame 的随机行开始，并裁剪其后的几行。只要我们能成功获取一个完整的窗口，我们就将其作为样本。一旦我们收集到足够的样本以组成一个批次，批次就会被分发。

如果我们能够为函数参数提供类型提示，那么代码会更清晰，例如我们知道`data`应该是一个 pandas DataFrame。但是我们进一步描述了预期在文档字符串中携带一个日期时间索引。我们描述了如何从输入数据中提取一行窗口的算法以及内部 while 循环中“if”块的意图。通过这种方式，代码变得更容易理解和维护，也更容易修改以供其他用途使用。

```py
from typing import List, Tuple, Generator
import pandas as pd
import numpy as np

TrainingSampleGenerator = Generator[Tuple[np.ndarray,np.ndarray], None, None]

def lstm_gen(data: pd.DataFrame,
             timesteps: int,
             batch_size: int) -> TrainingSampleGenerator:
    """Generator to produce random samples for LSTM training

    Args:
        data: DataFrame of data with datetime index in chronological order,
              samples are drawn from this
        timesteps: Number of time steps for each sample, data will be
                   produced from a window of such length
        batch_size: Number of samples in each batch

    Yields:
        ndarray, ndarray: The (X,Y) training samples drawn on a random window
        from the input data
    """
    input_columns = [c for c in data.columns if c != "target"]
    batch: List[Tuple[pd.DataFrame, pd.Series]] = []
    while True:
        # pick one start time and security
        while True:
            # Start from a random point from the data and clip a window
            row = data["target"].sample()
            starttime = row.index[0]
            window: pd.DataFrame = data[starttime:].iloc[:timesteps]
            # If we are at the end of the DataFrame, we can't get a full
            # window and we must start over
            if len(window) == timesteps:
                break
        # Extract the input and output
        y = window["target"]
        X = window[input_columns]
        batch.append((X, y))
        # If accumulated enough for one batch, dispatch
        if len(batch) == batch_size:
            X, y = zip(*batch)
            yield np.array(X).astype("float32"), np.array(y).astype("float32")
            batch = []
```

## 进一步阅读

如果您希望深入了解这个主题，本节提供了更多资源。

#### 文章

+   编写代码注释的最佳实践，[`stackoverflow.blog/2021/12/23/best-practices-for-writing-code-comments/`](https://stackoverflow.blog/2021/12/23/best-practices-for-writing-code-comments/)

+   PEP483，类型提示理论，[`www.python.org/dev/peps/pep-0483/`](https://www.python.org/dev/peps/pep-0483/)

+   Google Python 风格指南，[`google.github.io/styleguide/pyguide.html`](https://google.github.io/styleguide/pyguide.html)

#### 软件

+   Sphinx 文档，[`www.sphinx-doc.org/en/master/index.html`](https://www.sphinx-doc.org/en/master/index.html)

+   Sphinx 的 Napoleon 模块，[`sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html`](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html)

    +   Google 风格的文档字符串示例：[`sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html`](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)

    +   NumPy 风格的文档字符串示例：[`sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html`](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html)

+   `pdoc`，[`pdoc.dev/`](https://pdoc.dev/)

+   Python 的 `typing` 模块，[`docs.python.org/3/library/typing.html`](https://docs.python.org/3/library/typing.html)

## 总结

在本教程中，你已经看到我们如何在 Python 中使用注释、文档字符串和类型提示。具体来说，你现在知道：

+   如何写好有用的注释

+   解释函数使用文档字符串的约定

+   如何使用类型提示来解决 Python 中鸭子类型的可读性问题
