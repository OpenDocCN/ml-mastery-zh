# Python 中的更多特性

> 原文：[`machinelearningmastery.com/python-special-features/`](https://machinelearningmastery.com/python-special-features/)

Python 是一门非常棒的编程语言！它是开发 AI 和机器学习应用的最受欢迎的语言之一。Python 的语法非常易学，且具有一些特别的功能，使其与其他语言区分开来。在本教程中，我们将讨论 Python 编程语言的一些独特特性。

完成本教程后，你将会了解到：

+   列表和字典推导的构造

+   如何使用 zip 和 enumerate 函数

+   什么是函数上下文和装饰器

+   Python 中生成器的目的是什么

**启动你的项目**，通过我的新书 [Python for Machine Learning](https://machinelearningmastery.com/python-for-machine-learning/)，包括 *逐步教程* 和所有示例的 *Python 源代码* 文件。

让我们开始吧。![](https://machinelearningmastery.com/wp-content/uploads/2021/12/m-mani.png)

Python 特性

图片来源 M Mani，保留部分权利。

## 教程概述

本教程分为四部分，它们是：

1.  列表和字典推导

1.  Zip 和 enumerate 函数

1.  函数上下文和装饰器

1.  Python 中的生成器示例，使用 Keras 生成器

## 导入部分

本教程中使用的库在下面的代码中进行了导入。

```py
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import math
```

## 列表推导

列表推导提供了一种简短、简单的语法，用于从现有列表创建新列表。例如，假设我们需要一个新列表，其中每个新项是旧项乘以 3。一个方法是使用 `for` 循环，如下所示：

```py
original_list = [1, 2, 3, 4]
times3_list = []

for i in original_list:
        times3_list.append(i*3)
print(times3_list)
```

输出

```py
[3, 6, 9, 12]
```

使用列表推导式的简短方法只需一行代码：

```py
time3_list_awesome_method = [i*3 for i in original_list]
print(time3_list_awesome_method)
```

输出

```py
[3, 6, 9, 12]
```

你甚至可以基于特定的标准创建一个新列表。例如，如果我们只想将偶数添加到新列表中：

```py
even_list_awesome_method = [i for i in original_list if i%2==0]
print(even_list_awesome_method)
```

输出

```py
[2, 4]
```

也可以与上述代码一起使用 `else`。例如，我们可以保留所有偶数不变，并将奇数替换为零：

```py
new_list_awesome_method = [i if i%2==0 else 0 for i in original_list]
print(new_list_awesome_method)
```

输出

```py
[0, 2, 0, 4]
```

列表推导也可以用来替代嵌套循环。例如：

```py
colors = ["red", "green", "blue"]
animals = ["cat", "dog", "bird"]
newlist = []
for c in colors:
    for a in animals:
        newlist.append(c + " " + a)
print(newlist)
```

输出

```py
['red cat', 'red dog', 'red bird', 'green cat', 'green dog', 'green bird', 'blue cat', 'blue dog', 'blue bird']
```

可以如下完成，列表推导式中包含两个“for”：

```py
colors = ["red", "green", "blue"]
animals = ["cat", "dog", "bird"]

newlist = [c+" "+a for c in colors for a in animals]
print(newlist)
```

### 语法

列表推导的语法如下：

*newlist = [expression for item in iterable if condition == True]*

或者

*newList = [expression if condition == True else expression for item in iterable]*

### 想要开始使用 Python 进行机器学习吗？

立即参加我的 7 天免费电子邮件速成课程（附示例代码）。

点击报名，还可以免费获得课程的 PDF 电子书版本。

## 字典推导

字典推导类似于列表推导，不过现在我们有了 (key, value) 对。这里是一个示例；我们将通过将字符串 'number ' 连接到每个值来修改字典的每个值：

```py
original_dict = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}
new_dict = {key:'number ' + value for (key, value) in original_dict.items()}
print(new_dict)
```

输出

```py
{1: 'number one', 2: 'number two', 3: 'number three', 4: 'number four'}
```

再次，条件判断也是可能的。我们可以根据标准在新字典中选择添加(key, value)对。

```py
#Only add keys which are greater than 2
new_dict_high_keys = {key:'number ' + value for (key, value) in original_dict.items() if key>2}
print(new_dict_high_keys)

# Only change values with key>2
new_dict_2 = {key:('number ' + value if key>2 else value) for (key, value) in original_dict.items() }
print(new_dict_2)
```

输出

```py
{3: 'number three', 4: 'number four'}
{1: 'one', 2: 'two', 3: 'number three', 4: 'number four'}
```

## Python 中的枚举器和 Zip

在 Python 中，可迭代对象定义为任何可以逐个返回所有项的数据结构。这样，你可以使用`for`循环逐一处理所有项。Python 有两个附加的构造使`for`循环更易于使用，即`enumerate()`和`zip()`。

### 枚举

在传统编程语言中，你需要一个循环变量来遍历容器中的不同值。在 Python 中，这通过提供对循环变量和可迭代对象的一个值的访问来简化。`enumerate(x)`函数返回两个可迭代对象。一个可迭代对象从 0 到 len(x)-1。另一个是值等于 x 项的可迭代对象。下面显示了一个示例：

```py
name = ['Triangle', 'Square', 'Hexagon', 'Pentagon']

# enumerate returns two iterables
for i, n in enumerate(name):
    print(i, 'name: ', n)
```

输出

```py
0 name:  Triangle
1 name:  Square
2 name:  Hexagon
3 name:  Pentagon
```

默认情况下，enumerate 从 0 开始，但如果我们指定其他数字，则可以从其他数字开始。这在某些情况下非常有用，例如：

```py
data = [1,4,1,5,9,2,6,5,3,5,8,9,7,9,3]
for n, digit in enumerate(data[5:], 6):
    print("The %d-th digit is %d" % (n, digit))
```

```py
The 6-th digit is 2
The 7-th digit is 6
The 8-th digit is 5
The 9-th digit is 3
The 10-th digit is 5
The 11-th digit is 8
The 12-th digit is 9
The 13-th digit is 7
The 14-th digit is 9
The 15-th digit is 3
```

### Zip

Zip 允许你创建一个由元组组成的可迭代对象。Zip 将多个容器$(m_1, m_2, \ldots, m_n)$作为参数，并通过配对每个容器中的一个项来创建第 i 个元组。第 i 个元组是$(m_{1i}, m_{2i}, \ldots, m_{ni})$。如果传递的对象长度不同，则形成的元组总数的长度等于传递对象的最小长度。

下面是使用`zip()`和`enumerate()`的示例。

```py
sides = [3, 4, 6, 5]
colors = ['red', 'green', 'yellow', 'blue']
shapes = zip(name, sides, colors)

# Tuples are created from one item from each list
print(set(shapes))

# Easy to use enumerate and zip together for iterating through multiple lists in one go
for i, (n, s, c) in enumerate(zip(name, sides, colors)):
    print(i, 'Shape- ', n, '; Sides ', s)
```

输出

```py
{('Triangle', 3, 'red'), ('Square', 4, 'green'), ('Hexagon', 6, 'yellow'), ('Pentagon', 5, 'blue')}
0 Shape-  Triangle ; Sides  3
1 Shape-  Square ; Sides  4
2 Shape-  Hexagon ; Sides  6
3 Shape-  Pentagon ; Sides  5
```

## 函数上下文

Python 允许嵌套函数，你可以在外部函数内部定义一个内部函数。Python 中的嵌套函数有一些非常棒的特性。

+   外部函数可以返回指向内部函数的句柄。

+   内部函数保留了其环境和在其封闭函数中的所有局部变量，即使外部函数结束执行也不例外。

下面是一个示例，解释在注释中。

```py
def circle(r):
    area = 0
    def area_obj():
        nonlocal area
        area = math.pi * r * r
        print("area_obj")
    return area_obj    

def circle(r):
    area_val = math.pi * r * r
    def area():
        print(area_val)
    return area    

# returns area_obj(). The value of r passed is retained
circle_1 = circle(1)
circle_2 = circle(2)

# Calling area_obj() with radius = 1
circle_1()
# Calling area_obj() with radius = 2
circle_2()
```

输出

```py
3.141592653589793
12.566370614359172
```

## Python 中的装饰器

装饰器是 Python 的一个强大特性。你可以使用装饰器来定制类或函数的工作。可以将它们看作是应用于另一个函数的函数。使用`@`符号与函数名来定义装饰器函数。装饰器以函数作为参数，提供了很大的灵活性。

考虑以下函数`square_decorator()`，它接受一个函数作为参数，并返回一个函数。

+   内部嵌套函数`square_it()`接受一个参数`arg`。

+   `square_it()`函数将函数应用于`arg`并对结果进行平方运算。

+   我们可以将函数如`sin`传递给`square_decorator()`，它将返回$\sin²(x)$。

+   你还可以编写自定义函数，并使用特殊的@符号对其应用`square_decorator()`函数，如下所示。函数`plus_one(x)`返回`x+1`。这个函数被`square_decorator()`装饰，因此我们得到$(x+1)²$。

```py
def square_decorator(function):
    def square_it(arg):
        x = function(arg)
        return x*x
    return square_it

size_sq = square_decorator(len)
print(size_sq([1,2,3]))

sin_sq = square_decorator(math.sin)
print(sin_sq(math.pi/4))

@square_decorator
def plus_one(a):
    return a+1

a = plus_one(3)
print(a)
```

输出

```py
9
0.4999999999999999
16
```

## Python 中的生成器

Python 中的生成器允许你生成序列。生成器通过多个 `yield` 语句返回多个值，而不是编写 `return` 语句。第一次调用函数时，返回的是 `yield` 的第一个值。第二次调用返回的是 `yield` 的第二个值，以此类推。

生成器函数可以通过 `next()` 调用。每次调用 `next()` 时，都会返回下一个 `yield` 值。下面是生成 Fibonacci 序列直到给定数字 `x` 的示例。

```py
def get_fibonacci(x):
    x0 = 0
    x1 = 1
    for i in range(x):
        yield x0
        temp = x0 + x1
        x0 = x1
        x1 = temp

f = get_fibonacci(6)
for i in range(6):
    print(next(f))
```

输出

```py
0
1
1
2
3
5
```

### Keras 数据生成器示例

生成器的一个用途是 Keras 中的数据生成器。它非常有用，因为我们不想将所有数据保存在内存中，而是希望在训练循环需要时动态创建它。请记住，在 Keras 中，神经网络模型是按批训练的，因此生成器是用来发出数据批次的。下面的函数来自我们之前的帖子，“[使用 CNN 进行金融时间序列预测](https://machinelearningmastery.com/using-cnn-for-financial-time-series-prediction/)”：

```py
def datagen(data, seq_len, batch_size, targetcol, kind):
    "As a generator to produce samples for Keras model"
    batch = []
    while True:
        # Pick one dataframe from the pool
        key = random.choice(list(data.keys()))
        df = data[key]
        input_cols = [c for c in df.columns if c != targetcol]
        index = df.index[df.index < TRAIN_TEST_CUTOFF]
        split = int(len(index) * TRAIN_VALID_RATIO)
        if kind == 'train':
            index = index[:split]   # range for the training set
        elif kind == 'valid':
            index = index[split:]   # range for the validation set
        # Pick one position, then clip a sequence length
        while True:
            t = random.choice(index)      # pick one time step
            n = (df.index == t).argmax()  # find its position in the dataframe
            if n-seq_len+1 < 0:
                continue # can't get enough data for one sequence length
            frame = df.iloc[n-seq_len+1:n+1]
            batch.append([frame[input_cols].values, df.loc[t, targetcol]])
            break
        # if we get enough for a batch, dispatch
        if len(batch) == batch_size:
            X, y = zip(*batch)
            X, y = np.expand_dims(np.array(X), 3), np.array(y)
            yield X, y
            batch = []
```

上面的函数用于从 pandas 数据框中随机选择一行作为起点，并将接下来的几行剪切为一次时间间隔样本。这个过程重复几次，将许多时间间隔收集成一个批次。当我们收集到足够的间隔样本时，在上面函数的倒数第二行，使用 `yield` 命令分发批次。你可能已经注意到生成器函数没有返回语句。在这个示例中，函数将永远运行。这是有用且必要的，因为它允许我们的 Keras 训练过程运行任意多的轮次。

如果我们不使用生成器，我们将需要将数据框转换为所有可能的时间间隔，并将它们保存在内存中以供训练循环使用。这将涉及大量重复的数据（因为时间间隔是重叠的），并且占用大量内存。

由于它的实用性，Keras 库中预定义了一些生成器函数。下面是 `ImageDataGenerator()` 的示例。我们在 `x_train` 中加载了 32×32 图像的 `cifar10` 数据集。通过 `flow()` 方法将数据连接到生成器。`next()` 函数返回下一批数据。在下面的示例中，有 4 次对 `next()` 的调用。在每次调用中，返回 8 张图像，因为批量大小为 8。

以下是完整代码，也在每次调用 `next()` 后显示所有图像。

```py
(x_train, y_train), _ = keras.datasets.cifar10.load_data()
datagen = ImageDataGenerator()
data_iterator = datagen.flow(x_train, y_train, batch_size=8)

fig,ax = plt.subplots(nrows=4, ncols=8,figsize=(18,6),subplot_kw=dict(xticks=[], yticks=[]))

for i in range(4):
    # The next() function will load 8 images from CIFAR
    X, Y = data_iterator.next()
    for j, img in enumerate(X):
        ax[i, j].imshow(img.astype('int'))
```

![](https://machinelearningmastery.com/wp-content/uploads/2021/12/Untitled-1.png)

## 进一步阅读

如果你想更深入地了解这个主题，本节提供了更多资源。

### Python 文档

+   [Python 文档在 python.org](https://docs.python.org/3/contents.html)

### 书籍

+   [《Think Python: How to Think Like a Computer Scientist》](https://greenteapress.com/thinkpython/html/index.html) 由 Allen B. Downey 编写

+   [Python 3 编程：Python 语言完全介绍](https://www.amazon.com/Programming-Python-Complete-Introduction-Language-ebook-dp-B001OFK2DK/dp/B001OFK2DK/ref=mt_other?_encoding=UTF8&me=&qid=1638910263) 由 Mark Summerfield 编写

+   [Python 编程：计算机科学导论](https://www.amazon.com/Python-Programming-Introduction-Computer-Science/dp/1590282418/ref=sr_1_1?s=books&ie=UTF8&qid=1441293398&sr=1-1&keywords=Python+Zelle&pebp=1441293404611&perid=1B2BP6WM3RQHW4CY6990) 由 John Zelle 编写

### API 参考

+   [Keras ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)

## 总结

在本教程中，你发现了一些 Python 的特殊功能。

具体来说，你学习了：

+   列表和字典推导的目的

+   如何使用 zip 和 enumerate

+   嵌套函数、函数上下文和装饰器

+   Python 中的生成器和 Python 中的 ImageDataGenerator

对于本文讨论的 Python 功能，你有任何问题吗？在下方评论中提出你的问题，我会尽力回答。
