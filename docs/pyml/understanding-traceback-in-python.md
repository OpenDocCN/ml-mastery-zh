# 理解 Python 中的 Traceback

> 原文：[`machinelearningmastery.com/understanding-traceback-in-python/`](https://machinelearningmastery.com/understanding-traceback-in-python/)

当 Python 程序中发生异常时，通常会打印 traceback。知道如何阅读 traceback 可以帮助你轻松识别错误并进行修复。在本教程中，我们将看到 traceback 可以告诉你什么。

完成本教程后，你将了解：

+   如何读取 traceback

+   如何在没有异常的情况下打印调用栈

+   Traceback 中未显示的内容

使用我的新书 [Python for Machine Learning](https://machinelearningmastery.com/python-for-machine-learning/) **启动你的项目**，包括*逐步教程*和所有示例的*Python 源代码*文件。

让我们开始吧！[](../Images/bac7f8e495a1a3172f09a36f73585547.png)

理解 Python 中的 Traceback

图片由 [Marten Bjork](https://unsplash.com/photos/GM9Xpgb0g98) 提供，部分权利保留

## 教程概述

本教程分为四部分；它们是：

1.  简单程序的调用层次结构

1.  异常时的 Traceback

1.  手动触发 traceback

1.  模型训练中的一个示例

## 简单程序的调用层次结构

让我们考虑一个简单的程序：

```py
def indentprint(x, indent=0, prefix="", suffix=""):
    if isinstance(x, dict):
        printdict(x, indent, prefix, suffix)
    elif isinstance(x, list):
        printlist(x, indent, prefix, suffix)
    elif isinstance(x, str):
        printstring(x, indent, prefix, suffix)
    else:
        printnumber(x, indent, prefix, suffix)

def printdict(x, indent, prefix, suffix):
    spaces = " " * indent
    print(spaces + prefix + "{")
    for n, key in enumerate(x):
        comma = "," if n!=len(x)-1 else ""
        indentprint(x[key], indent+2, str(key)+": ", comma)
    print(spaces + "}" + suffix)

def printlist(x, indent, prefix, suffix):
    spaces = " " * indent
    print(spaces + prefix + "[")
    for n, item in enumerate(x):
        comma = "," if n!=len(x)-1 else ""
        indentprint(item, indent+2, "", comma)
    print(spaces + "]" + suffix)

def printstring(x, indent, prefix, suffix):
    spaces = " " * indent
    print(spaces + prefix + '"' + str(x) + '"' + suffix)

def printnumber(x, indent, prefix, suffix):
    spaces = " " * indent
    print(spaces + prefix + str(x) + suffix)

data = {
    "a": [{
        "p": 3, "q": 4,
        "r": [3,4,5],
    },{
        "f": "foo", "g": 2.71
    },{
        "u": None, "v": "bar"
    }],
    "c": {
        "s": ["fizz", 2, 1.1],
        "t": []
    },
}

indentprint(data)
```

这个程序将带有缩进的 Python 字典 `data` 打印出来。它的输出如下：

```py
{
  a: [
    {
      p: 3,
      q: 4,
      r: [
        3,
        4,
        5
      ]
    },
    {
      f: "foo",
      g: 2.71
    },
    {
      u: None,
      v: "bar"
    }
  ],
  c: {
    s: [
      "fizz",
      2,
      1.1
    ],
    t: [
    ]
  }
}
```

这是一个短程序，但函数之间相互调用。如果我们在每个函数的开头添加一行，我们可以揭示输出是如何随着控制流产生的：

```py
def indentprint(x, indent=0, prefix="", suffix=""):
    print(f'indentprint(x, {indent}, "{prefix}", "{suffix}")')
    if isinstance(x, dict):
        printdict(x, indent, prefix, suffix)
    elif isinstance(x, list):
        printlist(x, indent, prefix, suffix)
    elif isinstance(x, str):
        printstring(x, indent, prefix, suffix)
    else:
        printnumber(x, indent, prefix, suffix)

def printdict(x, indent, prefix, suffix):
    print(f'printdict(x, {indent}, "{prefix}", "{suffix}")')
    spaces = " " * indent
    print(spaces + prefix + "{")
    for n, key in enumerate(x):
        comma = "," if n!=len(x)-1 else ""
        indentprint(x[key], indent+2, str(key)+": ", comma)
    print(spaces + "}" + suffix)

def printlist(x, indent, prefix, suffix):
    print(f'printlist(x, {indent}, "{prefix}", "{suffix}")')
    spaces = " " * indent
    print(spaces + prefix + "[")
    for n, item in enumerate(x):
        comma = "," if n!=len(x)-1 else ""
        indentprint(item, indent+2, "", comma)
    print(spaces + "]" + suffix)

def printstring(x, indent, prefix, suffix):
    print(f'printstring(x, {indent}, "{prefix}", "{suffix}")')
    spaces = " " * indent
    print(spaces + prefix + '"' + str(x) + '"' + suffix)

def printnumber(x, indent, prefix, suffix):
    print(f'printnumber(x, {indent}, "{prefix}", "{suffix}")')
    spaces = " " * indent
    print(spaces + prefix + str(x) + suffix)
```

输出将被更多信息搞乱：

```py
indentprint(x, 0, "", "")
printdict(x, 0, "", "")
{
indentprint(x, 2, "a: ", ",")
printlist(x, 2, "a: ", ",")
  a: [
indentprint(x, 4, "", ",")
printdict(x, 4, "", ",")
    {
indentprint(x, 6, "p: ", ",")
printnumber(x, 6, "p: ", ",")
      p: 3,
indentprint(x, 6, "q: ", ",")
printnumber(x, 6, "q: ", ",")
      q: 4,
indentprint(x, 6, "r: ", "")
printlist(x, 6, "r: ", "")
      r: [
indentprint(x, 8, "", ",")
printnumber(x, 8, "", ",")
        3,
indentprint(x, 8, "", ",")
printnumber(x, 8, "", ",")
        4,
indentprint(x, 8, "", "")
printnumber(x, 8, "", "")
        5
      ]
    },
indentprint(x, 4, "", ",")
printdict(x, 4, "", ",")
    {
indentprint(x, 6, "f: ", ",")
printstring(x, 6, "f: ", ",")
      f: "foo",
indentprint(x, 6, "g: ", "")
printnumber(x, 6, "g: ", "")
      g: 2.71
    },
indentprint(x, 4, "", "")
printdict(x, 4, "", "")
    {
indentprint(x, 6, "u: ", ",")
printnumber(x, 6, "u: ", ",")
      u: None,
indentprint(x, 6, "v: ", "")
printstring(x, 6, "v: ", "")
      v: "bar"
    }
  ],
indentprint(x, 2, "c: ", "")
printdict(x, 2, "c: ", "")
  c: {
indentprint(x, 4, "s: ", ",")
printlist(x, 4, "s: ", ",")
    s: [
indentprint(x, 6, "", ",")
printstring(x, 6, "", ",")
      "fizz",
indentprint(x, 6, "", ",")
printnumber(x, 6, "", ",")
      2,
indentprint(x, 6, "", "")
printnumber(x, 6, "", "")
      1.1
    ],
indentprint(x, 4, "t: ", "")
printlist(x, 4, "t: ", "")
    t: [
    ]
  }
}
```

现在我们知道了每个函数调用的顺序。这就是调用栈的概念。在我们运行函数中的一行代码时，我们想知道是什么调用了这个函数。

## 异常时的 Traceback

如果我们在代码中犯了一个错别字，例如：

```py
def printdict(x, indent, prefix, suffix):
    spaces = " " * indent
    print(spaces + prefix + "{")
    for n, key in enumerate(x):
        comma = "," if n!=len(x)-1 else ""
        indentprint(x[key], indent+2, str(key)+": ", comma)
    print(spaces + "}") + suffix
```

错误在最后一行，其中闭合括号应该在行末，而不是在任何 `+` 之前。`print()` 函数的返回值是 Python 的 `None` 对象。将内容添加到 `None` 会触发异常。

如果你使用 Python 解释器运行这个程序，你将看到：

```py
{
  a: [
    {
      p: 3,
      q: 4,
      r: [
        3,
        4,
        5
      ]
    }
Traceback (most recent call last):
  File "tb.py", line 52, in 
    indentprint(data)
  File "tb.py", line 3, in indentprint
    printdict(x, indent, prefix, suffix)
  File "tb.py", line 16, in printdict
    indentprint(x[key], indent+2, str(key)+": ", comma)
  File "tb.py", line 5, in indentprint
    printlist(x, indent, prefix, suffix)
  File "tb.py", line 24, in printlist
    indentprint(item, indent+2, "", comma)
  File "tb.py", line 3, in indentprint
    printdict(x, indent, prefix, suffix)
  File "tb.py", line 17, in printdict
    print(spaces + "}") + suffix
TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'
```

以“Traceback (most recent call last):”开头的行是 traceback。它是你的程序在遇到异常时的**栈**。在上述示例中，traceback 以“最近的调用最后”顺序显示。因此你的主函数在顶部，而触发异常的函数在底部。所以我们知道问题出在函数 `printdict()` 内部。

通常，你会在 traceback 的末尾看到错误消息。在这个例子中，是由于将 `None` 和字符串相加触发的 `TypeError`。但 traceback 的帮助到此为止。你需要弄清楚哪个是 `None`，哪个是字符串。通过阅读 traceback，我们也知道触发异常的函数 `printdict()` 是由 `indentprint()` 调用的，`indentprint()` 又由 `printlist()` 调用，依此类推。

如果你在 Jupyter notebook 中运行这段代码，输出如下：

```py
{
  a: [
    {
      p: 3,
      q: 4,
      r: [
        3,
        4,
        5
      ]
    }
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/var/folders/6z/w0ltb1ss08l593y5xt9jyl1w0000gn/T/ipykernel_37031/2508041071.py in 
----> 1 indentprint(x)

/var/folders/6z/w0ltb1ss08l593y5xt9jyl1w0000gn/T/ipykernel_37031/2327707064.py in indentprint(x, indent, prefix, suffix)
      1 def indentprint(x, indent=0, prefix="", suffix=""):
      2     if isinstance(x, dict):
----> 3         printdict(x, indent, prefix, suffix)
      4     elif isinstance(x, list):
      5         printlist(x, indent, prefix, suffix)

/var/folders/6z/w0ltb1ss08l593y5xt9jyl1w0000gn/T/ipykernel_37031/2327707064.py in printdict(x, indent, prefix, suffix)
     14     for n, key in enumerate(x):
     15         comma = "," if n!=len(x)-1 else ""
---> 16         indentprint(x[key], indent+2, str(key)+": ", comma)
     17     print(spaces + "}") + suffix
     18 

/var/folders/6z/w0ltb1ss08l593y5xt9jyl1w0000gn/T/ipykernel_37031/2327707064.py in indentprint(x, indent, prefix, suffix)
      3         printdict(x, indent, prefix, suffix)
      4     elif isinstance(x, list):
----> 5         printlist(x, indent, prefix, suffix)
      6     elif isinstance(x, str):
      7         printstring(x, indent, prefix, suffix)

/var/folders/6z/w0ltb1ss08l593y5xt9jyl1w0000gn/T/ipykernel_37031/2327707064.py in printlist(x, indent, prefix, suffix)
     22     for n, item in enumerate(x):
     23         comma = "," if n!=len(x)-1 else ""
---> 24         indentprint(item, indent+2, "", comma)
     25     print(spaces + "]" + suffix)
     26 

/var/folders/6z/w0ltb1ss08l593y5xt9jyl1w0000gn/T/ipykernel_37031/2327707064.py in indentprint(x, indent, prefix, suffix)
      1 def indentprint(x, indent=0, prefix="", suffix=""):
      2     if isinstance(x, dict):
----> 3         printdict(x, indent, prefix, suffix)
      4     elif isinstance(x, list):
      5         printlist(x, indent, prefix, suffix)

/var/folders/6z/w0ltb1ss08l593y5xt9jyl1w0000gn/T/ipykernel_37031/2327707064.py in printdict(x, indent, prefix, suffix)
     15         comma = "," if n!=len(x)-1 else ""
     16         indentprint(x[key], indent+2, str(key)+": ", comma)
---> 17     print(spaces + "}") + suffix
     18 
     19 def printlist(x, indent, prefix, suffix):

TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'
```

信息本质上是相同的，但它提供了每个函数调用前后的行。

### 想要开始学习 Python 进行机器学习吗？

现在立即报名我的免费 7 天电子邮件速成课程（包含示例代码）。

点击注册并获得免费 PDF 电子书版课程。

## 手动触发 traceback

打印 traceback 最简单的方法是添加 `raise` 语句来手动创建异常。但这也会终止你的程序。如果我们希望在任何时间打印栈，即使没有任何异常，我们可以使用以下方法：

```py
import traceback

def printdict(x, indent, prefix, suffix):
    spaces = " " * indent
    print(spaces + prefix + "{")
    for n, key in enumerate(x):
        comma = "," if n!=len(x)-1 else ""
        indentprint(x[key], indent+2, str(key)+": ", comma)
    traceback.print_stack()    # print the current call stack
    print(spaces + "}" + suffix)
```

行 `traceback.print_stack()` 将打印当前调用栈。

但确实，我们通常只在出现错误时才打印栈（以便了解为什么会这样）。更常见的用例如下：

```py
import traceback
import random

def compute():
    n = random.randint(0, 10)
    m = random.randint(0, 10)
    return n/m

def compute_many(n_times):
    try:
        for _ in range(n_times):
            x = compute()
        print(f"Completed {n_times} times")
    except:
        print("Something wrong")
        traceback.print_exc()

compute_many(100)
```

这是重复计算函数的典型模式，例如蒙特卡洛模拟。但如果我们不够小心，可能会遇到一些错误，如上例中的除零错误。问题是，在更复杂的计算情况下，你不能轻易发现缺陷。例如上面的情况，问题隐藏在 `compute()` 的调用中。因此，理解错误的产生方式是有帮助的。但同时，我们希望处理错误的情况，而不是让整个程序终止。如果我们使用 `try-catch` 构造，traceback 默认不会打印。因此，我们需要使用 `traceback.print_exc()` 语句手动打印。

实际上，我们可以使 traceback 更加详细。由于 traceback 是调用栈，我们可以检查调用栈中的每个函数，并检查每一层中的变量。在这种复杂的情况下，这是我通常用来做更详细跟踪的函数：

```py
def print_tb_with_local():
    """Print stack trace with local variables. This does not need to be in
    exception. Print is using the system's print() function to stderr.
    """
    import traceback, sys
    tb = sys.exc_info()[2]
    stack = []
    while tb:
        stack.append(tb.tb_frame)
        tb = tb.tb_next()
    traceback.print_exc()
    print("Locals by frame, most recent call first", file=sys.stderr)
    for frame in stack:
        print("Frame {0} in {1} at line {2}".format(
            frame.f_code.co_name,
            frame.f_code.co_filename,
            frame.f_lineno), file=sys.stderr)
        for key, value in frame.f_locals.items():
            print("\t%20s = " % key, file=sys.stderr)
            try:
                if '__repr__' in dir(value):
                    print(value.__repr__(), file=sys.stderr)
                elif '__str__' in dir(value):
                    print(value.__str__(), file=sys.stderr)
                else:
                    print(value, file=sys.stderr)
            except:
                print("", file=sys.stderr)
```

## 模型训练的一个示例

traceback 中报告的调用栈有一个限制：你只能看到 Python 函数。这对于你编写的程序应该没问题，但许多大型 Python 库的一部分是用其他语言编写并编译成二进制的。例如 Tensorflow。所有底层操作都是以二进制形式存在以提升性能。因此，如果你运行以下代码，你会看到不同的内容：

```py
import numpy as np

sequence = np.arange(0.1, 1.0, 0.1)  # 0.1 to 0.9
n_in = len(sequence)
sequence = sequence.reshape((1, n_in, 1))

# define model
import tensorflow as tf
from tensorflow.keras.layers import LSTM, RepeatVector, Dense, TimeDistributed, Input
from tensorflow.keras import Sequential, Model

model = Sequential([
    LSTM(100, activation="relu", input_shape=(n_in+1, 1)),
    RepeatVector(n_in),
    LSTM(100, activation="relu", return_sequences=True),
    TimeDistributed(Dense(1))
])
model.compile(optimizer="adam", loss="mse")

model.fit(sequence, sequence, epochs=300, verbose=0)
```

模型中第一个 LSTM 层的 `input_shape` 参数应该是 `(n_in, 1)` 以匹配输入数据，而不是 `(n_in+1, 1)`。这段代码在你调用最后一行时将打印以下错误：

```py
Traceback (most recent call last):
  File "trback3.py", line 20, in 
    model.fit(sequence, sequence, epochs=300, verbose=0)
  File "/usr/local/lib/python3.9/site-packages/keras/utils/traceback_utils.py", line 67, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/usr/local/lib/python3.9/site-packages/tensorflow/python/framework/func_graph.py", line 1129, in autograph_handler
    raise e.ag_error_metadata.to_exception(e)
ValueError: in user code:

    File "/usr/local/lib/python3.9/site-packages/keras/engine/training.py", line 878, in train_function  *
        return step_function(self, iterator)
    File "/usr/local/lib/python3.9/site-packages/keras/engine/training.py", line 867, in step_function  **
        outputs = model.distribute_strategy.run(run_step, args=(data,))
    File "/usr/local/lib/python3.9/site-packages/keras/engine/training.py", line 860, in run_step  **
        outputs = model.train_step(data)
    File "/usr/local/lib/python3.9/site-packages/keras/engine/training.py", line 808, in train_step
        y_pred = self(x, training=True)
    File "/usr/local/lib/python3.9/site-packages/keras/utils/traceback_utils.py", line 67, in error_handler
        raise e.with_traceback(filtered_tb) from None
    File "/usr/local/lib/python3.9/site-packages/keras/engine/input_spec.py", line 263, in assert_input_compatibility
        raise ValueError(f'Input {input_index} of layer "{layer_name}" is '

    ValueError: Input 0 of layer "sequential" is incompatible with the layer: expected shape=(None, 10, 1), found shape=(None, 9, 1)
```

如果你查看追溯信息，你无法真正看到完整的调用栈。例如，你知道你调用了 `model.fit()`，但第二个帧来自一个名为 `error_handler()` 的函数。在这里，你无法看到 `fit()` 函数如何触发了这个函数。这是因为 Tensorflow 被高度优化了。许多内容隐藏在编译代码中，Python 解释器无法看到。

在这种情况下，耐心阅读追溯信息并找出原因的线索是至关重要的。当然，错误信息通常也会给你一些有用的提示。

## 进一步阅读

如果你想更深入了解该主题，本节提供了更多资源。

### 书籍

+   [Python Cookbook, 第三版](https://www.amazon.com/dp/1449340377) 作者 David Beazley 和 Brian K. Jones

### Python 官方文档

+   [追溯模块](https://docs.python.org/3/library/traceback.html) 在 Python 标准库中

## 总结

在本教程中，你学习了如何读取和打印 Python 程序的追溯信息。

具体来说，你学习了：

+   追溯信息告诉你什么

+   如何在程序的任何点打印追溯信息而不引发异常

在下一篇文章中，我们将学习如何在 Python 调试器中导航调用栈。
