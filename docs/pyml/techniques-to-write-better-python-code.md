# 编写更好 Python 代码的技术

> 原文：[`machinelearningmastery.com/techniques-to-write-better-python-code/`](https://machinelearningmastery.com/techniques-to-write-better-python-code/)

我们编写程序是为了问题解决或者制作一个可以重复解决类似问题的工具。对于后者，我们难免会再次回到之前编写的程序中，或者其他人会重用我们编写的程序。也有可能会遇到我们在编写程序时没有预见的数据。毕竟，我们仍然希望我们的程序能够**正常运行**。有一些技术和心态可以帮助我们编写更健壮的代码。

完成本教程后，你将学到

+   如何为意外情况准备代码

+   如何为代码无法处理的情况提供适当的信号

+   编写更健壮程序的最佳实践是什么

**通过我的新书[《机器学习 Python 编程》](https://machinelearningmastery.com/python-for-machine-learning/)**来**快速启动你的项目**，书中包括*逐步教程*和所有示例的*Python 源代码*文件。

让我们开始吧！！[](../Images/d62bb542e948384af4cdc4d5efc759e4.png)

编写更好 Python 代码的技术

图片由[Anna Shvets](https://www.pexels.com/photo/crop-woodworker-making-patterns-on-wooden-board-5711877/)提供。保留所有权利。

## 概述

本教程分为三个部分，分别是：

+   数据清理和自我检测编程

+   保护措施和防御性编程

+   避免错误的最佳实践

## 数据清理和自我检测编程

当我们在 Python 中编写一个函数时，我们通常会接收一些参数并返回一些值。毕竟，这就是函数的本质。由于 Python 是一种鸭子类型语言，很容易看到一个接受数字的函数被字符串调用。例如：

```py
def add(a, b):
    return a + b

c = add("one", "two")
```

这段代码完全正常，因为 Python 字符串中的 `+` 运算符表示连接。因此没有语法错误；只是它不是我们想要的函数行为。

这本不应该是个大问题，但如果函数较长，我们不应该只在后期才发现有问题。例如，我们的程序因为这样一个错误而失败和终止，这只发生在训练机器学习模型和浪费了几个小时的等待之后。如果我们能主动验证我们所假设的情况，那将会更好。这也是一个很好的实践，有助于我们向阅读我们代码的其他人传达我们在代码中期望的内容。

一个常见的做法是**清理输入**。例如，我们可以将上面的函数重写如下：

```py
def add(a, b):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise ValueError("Input must be numbers")
    return a + b
```

或者，更好的是，在可能的情况下将输入转换为浮点数：

```py
def add(a, b):
    try:
        a = float(a)
        b = float(b)
    except ValueError:
        raise ValueError("Input must be numbers")
    return a + b
```

这里的关键是在函数开始时进行一些“清理”，这样后续我们可以假设输入是某种格式。这样不仅可以更有信心地认为我们的代码按预期工作，而且可能使我们的主要算法更简单，因为我们通过清理排除了某些情况。为了说明这个想法，我们可以看看如何重新实现内置的`range()`函数：

```py
def range(a, b=None, c=None):
    if c is None:
        c = 1
    if b is None:
        b = a
        a = 0
    values = []
    n = a
    while n < b:
        values.append(n)
        n = n + c
    return values
```

这是我们可以从 Python 的内置库中获取的`range()`的简化版本。但是，通过函数开始的两个`if`语句，我们知道变量`a`、`b`和`c`总是有值的。然后，`while`循环可以像这样编写。否则，我们必须考虑调用`range()`的三种不同情况，即`range(10)`、`range(2,10)`和`range(2,10,3)`，这会使我们的`while`循环变得更复杂且容易出错。

清理输入的另一个原因是为了**标准化**。这意味着我们应该将输入格式化为标准化格式。例如，URL 应以“http://”开头，而文件路径应始终是完整的绝对路径，如`/etc/passwd`，而不是像`/tmp/../etc/././passwd`这样的路径。标准化后的输入更容易检查其一致性（例如，我们知道`/etc/passwd`包含敏感的系统数据，但对`/tmp/../etc/././passwd`不太确定）。

你可能会想知道是否有必要通过添加这些清理操作来使代码变得更长。确实，这是一个你需要决定的平衡。通常，我们不会在每个函数上都这样做，以节省精力并且不影响计算效率。我们只在可能出错的地方这样做，即在我们作为 API 向其他用户公开的接口函数或在从用户命令行获取输入的主要函数中。

然而，我们要指出的是，以下是一种错误但常见的清理方式：

```py
def add(a, b):
    assert isinstance(a, (int, float)), "`a` must be a number"
    assert isinstance(b, (int, float)), "`b` must be a number"
    return a + b
```

Python 中的`assert`语句如果第一个参数不为`True`，将引发`AssertError`异常（如果提供了可选消息）。尽管引发`AssertError`与引发`ValueError`在处理意外输入时没有实际上的不同，但不推荐使用`assert`，因为我们可以通过使用`-O`选项运行 Python 命令来“优化”我们的代码，即，

```py
$ python -O script.py
```

在这种情况下，代码`script.py`中的所有`assert`都将被忽略。因此，如果我们的意图是停止代码的执行（包括你想在更高层次捕获异常），你应该使用`if`并明确地引发异常，而不是使用`assert`。

使用`assert`的正确方式是帮助我们在开发代码时进行调试。例如，

```py
def evenitems(arr):
    newarr = []
    for i in range(len(arr)):
        if i % 2 == 0:
            newarr.append(arr[i])
    assert len(newarr) * 2 >= len(arr)
    return newarr
```

在我们开发这个函数时，我们不能确定算法是否正确。有许多事情需要检查，但在这里我们希望确保如果我们从输入中提取了每个偶数索引的项，它的长度应该至少是输入数组长度的一半。当我们尝试优化算法或修饰代码时，这个条件必须不会被破坏。我们在关键位置保留 `assert` 语句，以确保在修改后代码没有被破坏。你可以将这看作是另一种单元测试方法。但通常，当我们检查函数的输入和输出是否符合预期时，我们称之为单元测试。以这种方式使用 `assert` 是为了检查函数内部的步骤。

如果我们编写复杂的算法，添加 `assert` 来检查**循环不变量**是很有帮助的，即循环应该遵守的条件。考虑以下对排序数组进行二分查找的代码：

```py
def binary_search(array, target):
    """Binary search on array for target

    Args:
        array: sorted array
        target: the element to search for
    Returns:
        index n on the array such that array[n]==target
        if the target not found, return -1
    """
    s,e = 0, len(array)
    while s < e:
        m = (s+e)//2
        if array[m] == target:
            return m
        elif array[m] > target:
            e = m
        elif array[m] < target:
            s = m+1
        assert m != (s+e)//2, "we didn't move our midpoint"
    return -1
```

最后的 `assert` 语句是为了维护我们的循环不变量。这是为了确保我们在更新起始游标 `s` 和结束游标 `e` 时没有逻辑错误，使得中点 `m` 在下一次迭代中不会更新。如果我们在最后的 `elif` 分支中将 `s = m+1` 替换为 `s = m` 并在数组中不存在的特定目标上使用该函数，断言语句将会警告我们这个错误。这就是为什么这种技术可以帮助我们编写更好的代码。

## 保护机制与进攻性编程

看到 Python 内置了一个 `NotImplementedError` 异常真是令人惊讶。这对于我们所说的**进攻性编程**非常有用。

虽然输入清理旨在将输入对齐到我们的代码期望的格式，有时候清理所有内容并不容易，或者对我们未来的开发不方便。以下是一个例子，其中我们定义了一个注册装饰器和一些函数：

```py
import math

REGISTRY = {}

def register(name):
    def _decorator(fn):
        REGISTRY[name] = fn
        return fn
    return _decorator

@register("relu")
def rectified(x):
    return x if x > 0 else 0

@register("sigmoid")
def sigmoid(x):
    return 1/(1 + math.exp(-x))

def activate(x, funcname):
    if funcname not in REGISTRY:
        raise NotImplementedError(f"Function {funcname} is not implemented")
    else:
        func = REGISTRY[funcname]
        return func(x)

print(activate(1.23, "relu"))
print(activate(1.23, "sigmoid"))
print(activate(1.23, "tanh"))
```

我们在函数 `activate()` 中引发了 `NotImplementedError` 并附带了自定义错误消息。运行这段代码将会打印前两个调用的结果，但在第三个调用时失败，因为我们还没有定义 `tanh` 函数：

```py
1.23
0.7738185742694538
Traceback (most recent call last):
  File "/Users/MLM/offensive.py", line 28, in <module>
    print(activate(1.23, "tanh"))
  File "/Users/MLM/offensive.py", line 21, in activate
    raise NotImplementedError(f"Function {funcname} is not implemented")
NotImplementedError: Function tanh is not implemented
```

正如你所想象的，我们可以在条件不是完全无效的地方引发 `NotImplementedError`，只是因为我们还没有准备好处理这些情况。当我们逐步开发程序时，这一点特别有用，我们可以一次实现一个案例，并在稍后处理一些边角情况。设置这些保护机制可以确保我们的半成品代码不会以不应该的方式被使用。这也是一种让代码更难被滥用的好做法，即不让变量在未被察觉的情况下超出我们的预期范围。

事实上，Python 中的异常处理系统非常成熟，我们应该使用它。当你从未预期输入为负值时，应引发一个带有适当消息的`ValueError`。类似地，当发生意外情况时，例如，你创建的临时文件在中途消失，引发一个`RuntimeError`。在这些情况下你的代码无论如何都无法正常工作，抛出适当的异常有助于未来的重用。从性能角度来看，你还会发现抛出异常比使用 if 语句检查要更快。这就是为什么在 Python 中，我们更倾向于使用“请宽恕而不是许可”（EAFP）而不是“跃前先看”（LBYL）。

这里的原则是你绝不应让异常静默地进行，因为你的算法将无法正确运行，有时还会产生危险的效果（例如，删除错误的文件或产生网络安全问题）。

### 想要开始使用 Python 进行机器学习？

现在就参加我的免费 7 天电子邮件速成课程（附带示例代码）。

点击注册，还可以获得课程的免费 PDF 电子书版本。

## 避免错误的良好实践

无法说我们写的代码没有错误。它就像我们测试过的一样好，但我们不知道自己不知道什么。总有潜在的方式会意外地破坏代码。然而，有一些实践可以促进良好的代码并减少错误。

首先是使用函数式编程范式。虽然我们知道 Python 有允许我们用函数式语法编写算法的构造，但函数式编程背后的原则是函数调用不产生副作用。我们从不改变任何东西，也不使用函数外部声明的变量。“无副作用”原则在避免大量错误方面非常强大，因为我们永远不会错误地改变任何东西。

当我们在 Python 中编程时，经常会发现数据结构无意中被修改。考虑以下情况：

```py
def func(a=[]):
    a.append(1)
    return a
```

这个函数的作用很简单。然而，当我们在没有任何参数的情况下调用这个函数时，使用了默认值，并返回了`[1]`。当我们再次调用它时，使用了不同的默认值，返回了`[1,1]`。这是因为我们在函数声明时创建的列表`[]`作为参数`a`的默认值是一个初始化的对象。当我们向其中添加一个值时，这个对象会发生变化。下次调用函数时会看到这个变化后的对象。

除非我们明确想这样做（例如，原地排序算法），否则我们不应该将函数参数用作变量，而应将其作为只读使用。如果合适，我们应当对其进行复制。例如，

```py
LOGS = []

def log(action):
    LOGS.append(action)

data = {"name": None}
for n in ["Alice", "Bob", "Charlie"]:
    data["name"] = n
    ...  # do something with `data`
    log(data)  # keep a record of what we did
```

这段代码原本是为了记录我们在列表`LOGS`中所做的操作，但它并没有实现。当我们处理名字“Alice”、“Bob”以及“Charlie”时，`LOGS`中的三条记录都会是“Charlie”，因为我们在其中保留了可变的字典对象。应将其修改如下：

```py
import copy

def log(action):
    copied_action = copy.deepcopy(action)
    LOGS.append(copied_action)
```

然后我们将在日志中看到三个不同的名称。总的来说，如果我们函数的参数是一个可变对象，我们应该小心。

避免错误的另一种方法是不要重复造轮子。在 Python 中，我们有许多优秀的容器和优化过的操作。你不应该尝试自己创建一个栈数据结构，因为列表支持`append()`和`pop()`。你的实现不会更快。同样，如果你需要一个队列，我们在标准库的`collections`模块中有`deque`。Python 没有平衡搜索树或链表，但字典是高度优化的，我们应该在可能的情况下使用字典。函数也是如此，我们有 JSON 库，不应自行编写。如果我们需要一些数值算法，可以检查一下 NumPy 是否有合适的实现。

避免错误的另一种方法是使用更好的逻辑。一个包含大量循环和分支的算法很难跟踪，甚至可能让我们自己感到困惑。如果我们能使代码更清晰，就更容易发现错误。例如，创建一个检查矩阵上三角部分是否包含负数的函数，可以这样做：

```py
def neg_in_upper_tri(matrix):
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    for i in range(n_rows):
        for j in range(n_cols):
            if i > j:
                continue  # we are not in upper triangular
            if matrix[i][j] < 0:
                return True
    return False
```

但我们也使用 Python 生成器将其拆分成两个函数：

```py
def get_upper_tri(matrix):
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    for i in range(n_rows):
        for j in range(n_cols):
            if i > j:
                continue  # we are not in upper triangular
            yield matrix[i][j]

def neg_in_upper_tri(matrix):
    for element in get_upper_tri(matrix):
        if element[i][j] < 0:
            return True
    return False
```

我们多写了几行代码，但保持了每个函数专注于一个主题。如果函数更复杂，将嵌套循环拆分成生成器可能有助于使代码更易于维护。

让我们考虑另一个例子：我们想编写一个函数来检查输入字符串是否看起来像一个有效的浮点数或整数。我们要求字符串是“`0.12`”，而不接受“`.12`”。我们需要整数像“`12`”，而不是“`12.`”。我们也不接受科学记数法，比如“`1.2e-1`”或千位分隔符，如“`1,234.56`”。为了简化，我们也不考虑符号，比如“`+1.23`”或“`-1.23`”。

我们可以编写一个函数，从第一个字符扫描到最后一个字符，并记住到目前为止看到的内容。然后检查我们看到的内容是否符合预期。代码如下：

```py
def isfloat(floatstring):
    if not isinstance(floatstring, str):
        raise ValueError("Expects a string input")
    seen_integer = False
    seen_dot = False
    seen_decimal = False
    for char in floatstring:
        if char.isdigit():
            if not seen_integer:
                seen_integer = True
            elif seen_dot and not seen_decimal:
                seen_decimal = True
        elif char == ".":
            if not seen_integer:
                return False  # e.g., ".3456"
            elif not seen_dot:
                seen_dot = True
            else:
                return False  # e.g., "1..23"
        else:
            return False  # e.g. "foo"
    if not seen_integer:
        return False   # e.g., ""
    if seen_dot and not seen_decimal:
        return False  # e.g., "2."
    return True

print(isfloat("foo"))       # False
print(isfloat(".3456"))     # False
print(isfloat("1.23"))      # True
print(isfloat("1..23"))     # False
print(isfloat("2"))         # True
print(isfloat("2."))        # False
print(isfloat("2,345.67"))  # False
```

上面的函数`isfloat()`在 for 循环内部有许多嵌套分支，显得很杂乱。即使在 for 循环之后，逻辑也不完全清晰如何确定布尔值。实际上，我们可以用不同的方法来编写代码，以减少错误的可能性，比如使用状态机模型：

```py
def isfloat(floatstring):
    if not isinstance(floatstring, str):
        raise ValueError("Expects a string input")
    # States: "start", "integer", "dot", "decimal"
    state = "start"
    for char in floatstring:
        if state == "start":
            if char.isdigit():
                state = "integer"
            else:
                return False  # bad transition, can't continue
        elif state == "integer":
            if char.isdigit():
                pass  # stay in the same state
            elif char == ".":
                state = "dot"
            else:
                return False  # bad transition, can't continue
        elif state == "dot":
            if char.isdigit():
                state = "decimal"
            else:
                return False  # bad transition, can't continue
        elif state == "decimal":
            if not char.isdigit():
                return False  # bad transition, can't continue
    if state in ["integer", "decimal"]:
        return True
    else:
        return False

print(isfloat("foo"))       # False
print(isfloat(".3456"))     # False
print(isfloat("1.23"))      # True
print(isfloat("1..23"))     # False
print(isfloat("2"))         # True
print(isfloat("2."))        # False
print(isfloat("2,345.67"))  # False
```

从视觉上，我们将下面的图示转化为代码。我们维护一个状态变量，直到扫描完输入字符串。状态将决定接受输入中的一个字符并移动到另一个状态，还是拒绝字符并终止。该函数只有在停留在可接受状态，即“整数”或“小数”，时才返回 True。这段代码更容易理解且结构更清晰。

![](img/422b81bc004d88bc9ce584b2ea2eb512.png)

实际上，更好的方法是使用正则表达式来匹配输入字符串，即，

```py
import re

def isfloat(floatstring):
    if not isinstance(floatstring, str):
        raise ValueError("Expects a string input")
    m = re.match(r"\d+(\.\d+)?$", floatstring)
    return m is not None

print(isfloat("foo"))       # False
print(isfloat(".3456"))     # False
print(isfloat("1.23"))      # True
print(isfloat("1..23"))     # False
print(isfloat("2"))         # True
print(isfloat("2."))        # False
print(isfloat("2,345.67"))  # False
```

然而，正则表达式匹配器在背后也在运行一个状态机。

这个主题还有很多值得探索的内容。例如，我们如何更好地分离函数和对象的职责，以使代码更易于维护和理解。有时，使用不同的数据结构可以让我们编写更简单的代码，从而使代码更强健。这不是一种科学，但几乎总是，如果代码更简单，就能避免错误。

最后，考虑为你的项目采用一种**编码风格**。保持一致的编码方式是你将来阅读自己编写代码时减少心理负担的第一步。这也使你更容易发现错误。

## 进一步阅读

本节提供了更多关于该主题的资源，如果你想深入了解。

#### 文章

+   [Google Python 风格指南](https://google.github.io/styleguide/pyguide.html)

#### 书籍

+   [构建安全软件](https://www.amazon.com/dp/0321425235/) 由 John Viega 和 Gary R. McGraw 编著

+   [构建安全可靠的系统](https://www.amazon.com/dp/1492083127/) 由 Heather Adkins 等人编著

+   [Python 编程指南](https://www.amazon.com/dp/1491933178/) 由 Kenneth Reitz 和 Tanya Schlusser 编著

+   [编程实践](https://www.amazon.com/dp/020161586X/) 由 Brian Kernighan 和 Rob Pike 编著

+   [重构](https://www.amazon.com/dp/0134757599/)，第 2 版，由 Martin Fowler 编著

## 总结

在本教程中，你已经看到了提升代码质量的高级技巧。这些技巧可以让代码为不同的情况做好更好的准备，使其更为稳健。它们还可以使代码更易于阅读、维护和扩展，从而适合未来的重用。这里提到的一些技巧在其他编程语言中也很通用。

具体来说，你学到了：

+   为什么我们希望清理输入，以及这如何帮助简化程序

+   `assert` 作为开发工具的正确使用方法

+   如何适当地使用 Python 异常来在意外情况下发出信号

+   处理可变对象时 Python 编程中的陷阱
