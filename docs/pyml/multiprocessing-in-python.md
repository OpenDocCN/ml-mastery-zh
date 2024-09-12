# Python 中的多处理

> 原文：[`machinelearningmastery.com/multiprocessing-in-python/`](https://machinelearningmastery.com/multiprocessing-in-python/)

当你在进行计算机视觉项目时，你可能需要处理大量图像数据。这是耗时的，如果能够并行处理多个图像会更好。多处理是指系统能够同时运行多个处理器。如果你有一台单处理器的计算机，它会在多个进程之间切换以保持它们的运行。然而，如今大多数计算机至少配备了多核处理器，可以同时执行多个进程。Python 的多处理模块是一个工具，可以通过将任务分配给不同的进程来提高脚本的效率。

完成本教程后，你将会知道：

+   为什么我们要使用多处理

+   如何使用 Python 多处理模块中的基本工具

**通过我的新书《机器学习的 Python》**启动你的项目，其中包括*一步步的教程*和所有示例的*Python 源代码*文件。

开始吧.![](img/9a88deccc690ace4b32ff7249d00cfa4.png)

Python 中的多处理

图片来源：[Thirdman](https://www.pexels.com/photo/wooden-interior-of-a-piano-6193847/)。保留部分权利。

## 概述

本教程分为四个部分，它们是：

+   多处理的好处

+   基本多处理

+   实际应用中的多处理

+   使用 joblib

## 多处理的好处

你可能会问，“为什么要使用多处理？”多处理可以通过并行运行多个任务而不是顺序运行，显著提高程序的效率。一个类似的术语是多线程，但它们是不同的。

进程是加载到内存中运行的程序，并且不与其他进程共享内存。线程是进程中的一个执行单元。多个线程在一个进程中运行，并且共享进程的内存空间。

Python 的全局解释器锁（GIL）只允许解释器下的一个线程同时运行，这意味着如果需要 Python 解释器，你不能享受多线程的性能提升。这就是为什么多处理在 Python 中优于多线程。多个进程可以并行运行，因为每个进程都有自己的解释器来执行分配给它的指令。此外，操作系统会将你的程序视为多个进程，并分别调度它们，即你的程序总的来说获得了更多的计算机资源。因此，当程序是 CPU 绑定时，多处理更快。在程序中有大量 I/O 的情况下，线程可能更有效，因为大多数时候，程序在等待 I/O 完成。然而，多处理通常更高效，因为它是并发运行的。

## 基本多处理

让我们使用 Python 的 Multiprocessing 模块编写一个基本程序，演示如何进行并发编程。

让我们看看这个函数`task()`，它在休眠 0.5 秒钟后打印休眠前后的内容：

```py
import time

def task():
    print('Sleeping for 0.5 seconds')
    time.sleep(0.5)
    print('Finished sleeping')
```

要创建一个进程，我们只需使用 multiprocessing 模块：

```py
...
import multiprocessing
p1 = multiprocessing.Process(target=task)
p2 = multiprocessing.Process(target=task)
```

`Process()`的`target`参数指定了进程运行的目标函数。但这些进程不会立即运行，直到我们启动它们：

```py
...
p1.start()
p2.start()
```

完整的并发程序如下：

```py
import multiprocessing
import time

def task():
    print('Sleeping for 0.5 seconds')
    time.sleep(0.5)
    print('Finished sleeping')

if __name__ == "__main__":
    start_time = time.perf_counter()

    # Creates two processes
    p1 = multiprocessing.Process(target=task)
    p2 = multiprocessing.Process(target=task)

    # Starts both processes
    p1.start()
    p2.start()

    finish_time = time.perf_counter()

    print(f"Program finished in {finish_time-start_time} seconds")
```

我们必须将主程序放在`if __name__ == "__main__"`下，否则`multiprocessing`模块会报错。这种安全结构确保 Python 在创建子进程之前完成程序分析。

但是，代码有个问题，因为程序计时器在我们创建的进程执行之前就已打印出来。以下是上面代码的输出：

```py
Program finished in 0.012921249988721684 seconds
Sleeping for 0.5 seconds
Sleeping for 0.5 seconds
Finished sleeping
Finished sleeping
```

我们需要对两个进程调用`join()`函数，以确保它们在时间打印之前运行。这是因为有三个进程在运行：`p1`、`p2`和主进程。主进程负责跟踪时间并打印执行所需的时间。我们应该确保`finish_time`的那一行在`p1`和`p2`进程完成之前不会运行。我们只需在`start()`函数调用后立即添加这段代码：

```py
...
p1.join()
p2.join()
```

`join()`函数允许我们使其他进程等待，直到对其调用了`join()`的进程完成。以下是添加了`join()`语句后的输出：

```py
Sleeping for 0.5 seconds
Sleeping for 0.5 seconds
Finished sleeping
Finished sleeping
Program finished in 0.5688213340181392 seconds
```

使用类似的推理，我们可以运行更多的进程。以下是从上面修改的完整代码，设置为 10 个进程：

```py
import multiprocessing
import time

def task():
    print('Sleeping for 0.5 seconds')
    time.sleep(0.5)
    print('Finished sleeping')

if __name__ == "__main__": 
    start_time = time.perf_counter()
    processes = []

    # Creates 10 processes then starts them
    for i in range(10):
        p = multiprocessing.Process(target = task)
        p.start()
        processes.append(p)

    # Joins all the processes 
    for p in processes:
        p.join()

    finish_time = time.perf_counter()

    print(f"Program finished in {finish_time-start_time} seconds")
```

### 想要开始学习 Python 进行机器学习吗？

现在立即参加我的免费 7 天邮件速成课程（附有示例代码）。

点击注册，并获取课程的免费 PDF 电子书版本。

## 实际应用中的多进程

启动一个新进程然后将其合并回主进程是 Python（以及许多其他语言）中多进程的工作方式。我们想要运行多进程的原因可能是为了并行执行多个不同的任务以提高速度。这可以是一个图像处理函数，我们需要对数千张图像进行处理。也可以是将 PDF 转换为文本以进行后续的自然语言处理任务，我们需要处理一千个 PDF。通常，我们会创建一个接收参数（例如，文件名）的函数来完成这些任务。

让我们考虑一个函数：

```py
def cube(x):
    return x**3
```

如果我们想要运行从 1 到 1,000 的参数，我们可以创建 1,000 个进程并行运行：

```py
import multiprocessing

def cube(x):
    return x**3

if __name__ == "__main__":
    # this does not work
    processes = [multiprocessing.Process(target=cube, args=(x,)) for x in range(1,1000)]
    [p.start() for p in processes]
    result = [p.join() for p in processes]
    print(result)
```

但是，这不会有效，因为你可能只有少量的核心。运行 1,000 个进程会造成过多的开销，超出操作系统的容量。此外，你可能已经耗尽了内存。更好的方法是运行进程池，以限制同时运行的进程数量：

```py
import multiprocessing
import time

def cube(x):
    return x**3

if __name__ == "__main__":
    pool = multiprocessing.Pool(3)
    start_time = time.perf_counter()
    processes = [pool.apply_async(cube, args=(x,)) for x in range(1,1000)]
    result = [p.get() for p in processes]
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
    print(result)
```

`multiprocessing.Pool()`的参数是要在池中创建的进程数。如果省略，Python 将其设置为你计算机中的核心数量。

我们使用`apply_async()`函数将参数传递给`cube`函数的列表推导。这将为池创建任务。它被称为“`async`”（异步），因为我们没有等待任务完成，主进程可能会继续运行。因此，`apply_async()`函数不会返回结果，而是一个我们可以使用的对象`get()`，以等待任务完成并检索结果。由于我们在列表推导中获取结果，因此结果的顺序对应于我们在异步任务中创建的参数。然而，这并不意味着进程在池中按此顺序启动或完成。

如果你认为编写代码行来启动进程并将其连接起来过于显式，你可以考虑使用`map()`代替：

```py
import multiprocessing
import time

def cube(x):
    return x**3

if __name__ == "__main__":
    pool = multiprocessing.Pool(3)
    start_time = time.perf_counter()
    result = pool.map(cube, range(1,1000))
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
    print(result)
```

我们在这里没有使用 start 和 join，因为它们被隐藏在`pool.map()`函数后面。`pool.map()`的作用是将可迭代对象`range(1,1000)`拆分成块，并在池中运行每个块。map 函数是列表推导的并行版本：

```py
result = [cube(x) for x in range(1,1000)]
```

但现代的替代方法是使用`concurrent.futures`中的`map`，如下所示：

```py
import concurrent.futures
import time

def cube(x):
    return x**3

if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(3) as executor:
        start_time = time.perf_counter()
        result = list(executor.map(cube, range(1,1000)))
        finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
    print(result)
```

这段代码在后台运行了`multiprocessing`模块。这样做的好处是，我们可以通过将`ProcessPoolExecutor`替换为`ThreadPoolExecutor`，将程序从多进程改为多线程。当然，你必须考虑全局解释器锁是否对你的代码构成问题。

## 使用 joblib

`joblib`包是一组使并行计算更简单的工具。它是用于多进程的常见第三方库。它还提供缓存和序列化功能。要安装`joblib`包，请在终端中使用以下命令：

```py
pip install joblib
```

我们可以将之前的例子转换为如下，以使用`joblib`：

```py
import time
from joblib import Parallel, delayed

def cube(x):
    return x**3

start_time = time.perf_counter()
result = Parallel(n_jobs=3)(delayed(cube)(i) for i in range(1,1000))
finish_time = time.perf_counter()
print(f"Program finished in {finish_time-start_time} seconds")
print(result)
```

实际上，看到它的作用是直观的。`delayed()`函数是对另一个函数的包装器，旨在创建一个“延迟”的函数调用版本。这意味着它在被调用时不会立即执行函数。

然后我们多次调用延迟函数，并传递我们希望传递给它的不同参数集。例如，当我们将整数`1`传递给`cube`函数的延迟版本时，我们不会计算结果，而是为函数对象、位置参数和关键字参数分别生成一个元组`(cube, (1,), {})`。

我们使用`Parallel()`创建了引擎实例。当它像函数一样被调用并传入包含元组的列表作为参数时，它实际上会并行执行每个元组指定的任务，并在所有任务完成后将结果汇总为一个列表。在这里，我们使用`n_jobs=3`创建了`Parallel()`实例，因此会有三个进程并行运行。

我们还可以直接写出元组。因此，上面的代码可以重写为：

```py
result = Parallel(n_jobs=3)((cube, (i,), {}) for i in range(1,1000))
```

使用 `joblib` 的好处在于，我们可以通过简单地添加一个额外的参数来在多线程中运行代码：

```py
result = Parallel(n_jobs=3, prefer="threads")(delayed(cube)(i) for i in range(1,1000))
```

这隐藏了并行运行函数的所有细节。我们只需使用一种与普通列表推导式差别不大的语法。

## 进一步阅读

本节提供了更多关于该主题的资源，如果你想深入了解。

#### 书籍

+   [高性能 Python](https://www.amazon.com/dp/1718502222/)，第二版，由 Micha Gorelick 和 Ian Ozsvald 著

#### API

+   [joblib](https://joblib.readthedocs.io/en/latest/)

+   [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) 是 Python 标准库中的一部分

+   [concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html) 是 Python 标准库中的一部分

## 总结

在本教程中，你学习了如何并行运行 Python 函数以提高速度。特别是，你学到了：

+   如何使用 `multiprocessing` 模块在 Python 中创建运行函数的新进程

+   启动和完成进程的机制

+   在 `multiprocessing` 中使用进程池进行受控的多进程处理以及 `concurrent.futures` 中的对应语法

+   如何使用第三方库 `joblib` 进行多进程处理
