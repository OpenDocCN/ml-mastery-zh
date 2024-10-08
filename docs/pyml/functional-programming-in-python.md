# Python 中的函数式编程

> 原文：[`machinelearningmastery.com/functional-programming-in-python/`](https://machinelearningmastery.com/functional-programming-in-python/)

Python 是一个出色的编程语言。它可能是你开发机器学习或数据科学应用程序的首选。Python 有趣的地方在于，它是一种多范式编程语言，可以用于面向对象编程和命令式编程。它具有简单的语法，易于阅读和理解。

在计算机科学和数学中，许多问题的解决方案可以通过函数式编程风格更容易和自然地表达。在本教程中，我们将讨论 Python 对函数式编程范式的支持以及帮助你以这种风格编程的 Python 类和模块。

完成本教程后，你将了解：

+   函数式编程的基本概念

+   `itertools`库

+   `functools`库

+   Map-reduce 设计模式及其在 Python 中的可能实现

**通过我的新书** [《Python 与机器学习》](https://machinelearningmastery.com/python-for-machine-learning/)，**启动你的项目**，包括*逐步教程*和所有示例的*Python 源代码*文件。

让我们开始吧。![](https://machinelearningmastery.com/wp-content/uploads/2021/12/abdullahShakoortree-gdd40e365b_1920.jpg)

Python 中的函数式编程

图片由 Abdullah_Shakoor 提供，部分版权保留

## 教程概述

本教程分为五个部分；它们是：

1.  函数式编程的思想

1.  高阶函数：过滤、映射和归约

1.  Itertools

1.  Functools

1.  Map-reduce 模式

## 函数式编程的思想

如果你有编程经验，你很可能学习了命令式编程。它由语句和变量操作构成。函数式编程是一种**声明式**范式。它不同于命令式范式，程序通过应用和组合函数来构建。这里的函数应更接近数学函数的定义，其中**没有副作用**，即没有对外部变量的访问。当你用相同的参数调用它们时，它们总是给出相同的结果。

函数式编程的好处是使你的程序更少出错。没有副作用，它更可预测且更容易查看结果。我们也不需要担心程序的某个部分会干扰到另一个部分。

许多库采用了函数式编程范式。例如，以下使用 pandas 和 pandas-datareader：

```py
import pandas_datareader as pdr
import pandas_datareader.wb

df = (
    pdr.wb
    .download(indicator="SP.POP.TOTL", country="all", start=2000, end=2020)
    .reset_index()
    .filter(["country", "SP.POP.TOTL"])
    .groupby("country")
    .mean()
)
print(df)
```

这将给你以下输出：

```py
                              SP.POP.TOTL
country                                  
Afghanistan                  2.976380e+07
Africa Eastern and Southern  5.257466e+08
Africa Western and Central   3.550782e+08
Albania                      2.943192e+06
Algeria                      3.658167e+07
...                                   ...
West Bank and Gaza           3.806576e+06
World                        6.930446e+09
Yemen, Rep.                  2.334172e+07
Zambia                       1.393321e+07
Zimbabwe                     1.299188e+07
```

pandas-datareader 是一个有用的库，帮助您实时从互联网下载数据。上述示例是从世界银行下载人口数据。结果是一个带有国家和年份作为索引的 pandas dataframe，并且一个名为“SP.POP.TOTL”的列表示人口数量。然后我们逐步操作数据帧，并最终找出所有国家在多年间的平均人口数量。

我们可以这样写是因为，在 pandas 中，大多数对数据帧的函数不会改变数据帧本身，而是生成一个新的数据帧以反映函数的结果。我们称这种行为为**不可变**，因为输入数据帧从未改变。其结果是我们可以逐步链式调用函数来操作数据帧。如果我们必须使用命令式编程的风格来打破它，上面的程序等同于以下内容：

```py
import pandas_datareader as pdr
import pandas_datareader.wb

df = pdr.wb.download(indicator="SP.POP.TOTL", country="all", start=2000, end=2020)
df = df.reset_index()
df = df.filter(["country", "SP.POP.TOTL"])
groups = df.groupby("country")
df = groups.mean()

print(df)
```

## 高阶函数：过滤（Filter）、映射（Map）和减少（Reduce）

Python 不是严格的函数式编程语言。但以函数式风格编写 Python 非常简单。有三个基本的迭代函数允许我们以非常简单的方式编写一个功能强大的程序：filter、map 和 reduce。

过滤（Filter）是从可迭代对象中选择一些元素，比如一个列表。映射（Map）是逐个转换元素。最后，减少（Reducing）是将整个可迭代对象转换为不同的形式，比如所有元素的总和或将列表中的子字符串连接成更长的字符串。为了说明它们的使用，让我们考虑一个简单的任务：给定来自 Apache Web 服务器的日志文件，找到发送了最多 404 错误请求的 IP 地址。如果你不知道 Apache Web 服务器的日志文件是什么样的，以下是一个例子：

```py
89.170.74.95 - - [17/May/2015:16:05:27 +0000] "HEAD /projects/xdotool/ HTTP/1.1" 200 - "-" "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0" 
95.82.59.254 - - [19/May/2015:03:05:19 +0000] "GET /images/jordan-80.png HTTP/1.1" 200 6146 "http://www.semicomplete.com/articles/dynamic-dns-with-dhcp/" "Mozilla/5.0 (Windows NT 6.1; rv:27.0) Gecko/20100101 Firefox/27.0"
155.140.133.248 - - [19/May/2015:06:05:34 +0000] "GET /images/jordan-80.png HTTP/1.1" 200 6146 "http://www.semicomplete.com/blog/geekery/debugging-java-performance.html" "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)"
68.180.224.225 - - [20/May/2015:20:05:02 +0000] "GET /blog/tags/documentation HTTP/1.1" 200 12091 "-" "Mozilla/5.0 (compatible; Yahoo! Slurp; http://help.yahoo.com/help/us/ysearch/slurp)"
```

上述内容来自一个更大的文件，位于[这里](https://raw.githubusercontent.com/elastic/examples/master/Common%20Data%20Formats/apache_logs/apache_logs)。这些是日志中的几行内容。每行以客户端（即浏览器）的 IP 地址开头，“HTTP/1.1”后面的代码是响应状态码。通常情况下，如果请求被满足，则状态码为 200。但如果浏览器请求了服务器上不存在的内容，则代码将为 404。要找到对应于最多 404 请求的 IP 地址，我们可以简单地逐行扫描日志文件，找到其中的 404 请求，计算 IP 地址以识别出现最多次数的那一个。

在 Python 代码中，我们可以这样做。首先，我们看看如何读取日志文件并从一行中提取 IP 地址和状态码：

```py
import urllib.request
import re

# Read the log file, split into lines
logurl = "https://raw.githubusercontent.com/elastic/examples/master/Common%20Data%20Formats/apache_logs/apache_logs"
logfile = urllib.request.urlopen(logurl).read().decode("utf8")
lines = logfile.splitlines()

# using regular expression to extract IP address and status code from a line
def ip_and_code(logline):
    m = re.match(r'([\d\.]+) .*? \[.*?\] ".*?" (\d+) ', logline)
    return (m.group(1), m.group(2))

print(ip_and_code(lines[0]))
```

然后我们可以使用一些 map()和 filter()以及其他一些函数来查找 IP 地址：

```py
...

import collections

def is404(pair):
    return pair[1] == "404"
def getIP(pair):
    return pair[0]
def count_ip(count_item):
    ip, count = count_item
    return (count, ip)

# transform each line into (IP address, status code) pair
ipcodepairs = map(ip_and_code, lines)
# keep only those with status code 404
pairs404 = filter(is404, ipcodepairs)
# extract the IP address part from each pair
ip404 = map(getIP, pairs404)
# count the occurrences, the result is a dictionary of IP addresses map to the count
ipcount = collections.Counter(ip404)
# convert the (IP address, count) tuple into (count, IP address) order
countip = map(count_ip, ipcount.items())
# find the tuple with the maximum on the count
print(max(countip))
```

这里，我们没有使用 reduce()函数，因为我们有一些专门的 reduce 操作内置，比如`max()`。但事实上，我们可以使用列表推导符号来编写一个更简单的程序：

```py
...

ipcodepairs = [ip_and_code(x) for x in lines]
ip404 = [ip for ip,code in ipcodepairs if code=="404"]
ipcount = collections.Counter(ip404)
countip = [(count,ip) for ip,count in ipcount.items()]
print(max(countip))
```

或者甚至可以将它写成一个单一语句（但可读性较差）：

```py
import urllib.request
import re
import collections

logurl = "https://raw.githubusercontent.com/elastic/examples/master/Common%20Data%20Formats/apache_logs/apache_logs"
print(
    max(
        [(count,ip) for ip,count in
            collections.Counter([
                ip for ip, code in
                [ip_and_code(x) for x in
                     urllib.request.urlopen(logurl)
                     .read()
                     .decode("utf8")
                     .splitlines()
                ]
                if code=="404"
            ]).items()
        ]
    )
)
```

### 想要开始使用 Python 进行机器学习吗？

现在就参加我的免费 7 天电子邮件速成课程（包含示例代码）。

点击注册并免费获得课程的 PDF 电子书版本。

## Python 中的迭代工具

上述关于过滤器、映射器和归约器的示例说明了**迭代对象**在 Python 中的普遍性。这包括列表、元组、字典、集合，甚至生成器，所有这些都可以使用 for 循环进行**迭代**。在 Python 中，我们有一个名为`itertools`的模块，它提供了更多的函数来操作（但不改变）迭代对象。来自[Python 官方文档](https://docs.python.org/3/library/itertools.html)：

该模块标准化了一组核心的快速、内存高效的工具，这些工具本身或与其他工具结合使用都很有用。它们共同形成了一种“迭代器代数”，使得在纯 Python 中简洁高效地构造专用工具成为可能。

我们将在本教程中讨论`itertools`的一些函数。在尝试下面给出的示例时，请确保导入`itertools`和`operator`，如：

```py
import itertools
import operator
```

### 无限迭代器

无限迭代器帮助你创建无限长度的序列，如下所示。

| 构造 + 示例 | 输出 |
| --- | --- |
| `count()`  ```py
start = 0
step = 100
for i in itertools.count(start, step):
    print(i)
    if i>=1000:
        break
```  |  ```py 0
100
200
300
400
500
600
700
800
900
1000
```  |
| `cycle()`  ```py
counter = 0
cyclic_list = [1, 2, 3, 4, 5]
for i in itertools.cycle(cyclic_list):
    print(i)
    counter = counter+1
    if counter>10:
        break
```  |  ```py 1
2
3
4
5
1
2
3
4
5
1
```  |
| `repeat()`  ```py
for i in itertools.repeat(3,5):
    print(i)
```  |  ```py 3
3
3
3
3
```  |

### 组合迭代器

你可以使用这些迭代器创建排列、组合等。

| 构造 + 示例 | 输出 |
| --- | --- |
| `product()`  ```py
x = [1, 2, 3]
y = ['A', 'B']
print(list(itertools.product(x, y)))
```  |  ```py [(1, 'A'), (1, 'B'), (2, 'A'), (2, 'B'), 
 (3, 'A'), (3, 'B')]
```  |
| `permutations()`  ```py
x = [1, 2, 3]
print(list(itertools.permutations(x)))
```  |  ```py [(1, 2, 3), (1, 3, 2), (2, 1, 3), 
 (2, 3, 1), (3, 1, 2), (3, 2, 1)]
```  |
| `combinations()`  ```py
y = ['A', 'B', 'C', 'D']
print(list(itertools.combinations(y, 3)))
```  |  ```py [('A', 'B', 'C'), ('A', 'B', 'D'), 
 ('A', 'C', 'D'), ('B', 'C', 'D')]
```  |
| `combinations_with_replacement()`  ```py
z = ['A', 'B', 'C']
print(list(itertools.combinations_with_replacement(z, 2)))
```  |  ```py [('A', 'A'), ('A', 'B'), ('A', 'C'), 
 ('B', 'B'), ('B', 'C'), ('C', 'C')]
```  |

### 更多有用的迭代器

还有其他迭代器会在传入的两个列表中较短的那个结束时停止。其中一些在下面有所描述。这不是一个详尽的列表，你可以在[这里查看完整列表](https://docs.python.org/3/library/itertools.html#itertool-functions)。

#### 累积()

自动创建一个迭代器，该迭代器累计给定操作符或函数的结果并返回结果。你可以从 Python 的`operator`库中选择一个操作符，或编写自定义操作符。

```py
# Custom operator
def my_operator(a, b):
    return a+b if a>5 else a-b

x = [2, 3, 4, -6]
mul_result = itertools.accumulate(x, operator.mul)
print("After mul operator", list(mul_result))
pow_result = itertools.accumulate(x, operator.pow)
print("After pow operator", list(pow_result))
my_operator_result = itertools.accumulate(x, my_operator)
print("After customized my_operator", list(my_operator_result))
```

```py
After mul operator [2, 6, 24, -144]
After pow operator [2, 8, 4096, 2.117582368135751e-22]
After customized my_operator [2, -1, -5, 1]
```

#### Starmap()

将相同的操作符应用于项对。

```py
pair_list = [(1, 2), (4, 0.5), (5, 7), (100, 10)]

starmap_add_result = itertools.starmap(operator.add, pair_list)
print("Starmap add result: ", list(starmap_add_result))

x1 = [2, 3, 4, -6]
x2 = [4, 3, 2, 1] 

starmap_mul_result = itertools.starmap(operator.mul, zip(x1, x2))
print("Starmap mul result: ", list(starmap_mul_result))
```

```py
Starmap add result:  [3, 4.5, 12, 110]
Starmap mul result:  [8, 9, 8, -6]
```

#### filterfalse()

根据特定标准筛选数据。

```py
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_result = itertools.filterfalse(lambda x: x%2, my_list)
small_terms = itertools.filterfalse(lambda x: x>=5, my_list)                               
print('Even result:', list(even_result))
print('Less than 5:', list(small_terms))
```

```py
Even result: [2, 4, 6, 8, 10]
Less than 5: [1, 2, 3, 4]
```

## Python 中的 Functools

在大多数编程语言中，将函数作为参数传递或函数返回另一个函数可能会令人困惑或难以处理。Python 包含了`functools`库，使得处理这些函数变得容易。来自 Python 官方`functools`文档：

`functools`模块用于高阶函数：作用于或返回其他函数的函数。一般来说，任何可调用对象都可以被视为函数。

在这里，我们解释了这个库的一些有趣功能。你可以在[这里查看`functools`函数的完整列表](https://docs.python.org/3/library/functools.html)。

### 使用`lru_cache`

在命令式编程语言中，递归是非常昂贵的。每次调用函数时，它都会被评估，即使它被相同的参数集调用。在 Python 中，`lru_cache` 是一个装饰器，可以用来缓存函数评估的结果。当函数再次用相同的参数集调用时，会使用存储的结果，从而避免了与递归相关的额外开销。

我们来看以下示例。我们有相同的计算第 n 个斐波那契数的实现，有和没有 `lru_cache`。我们可以看到 `fib(30)` 有 31 次函数评估，这正如我们预期的那样，因为 `lru_cache`。`fib()` 函数仅对 n=0,1,2…30 被调用，并且结果存储在内存中，稍后使用。这明显少于 `fib_slow(30)`，它有 2692537 次评估。

```py
import functools
@functools.lru_cache
def fib(n):
    global count
    count = count + 1
    return fib(n-2) + fib(n-1) if n>1 else 1

def fib_slow(n):
    global slow_count
    slow_count = slow_count + 1
    return fib_slow(n-2) + fib_slow(n-1) if n>1 else 1

count = 0
slow_count = 0
fib(30)
fib_slow(30)

print('With lru_cache total function evaluations: ', count)
print('Without lru_cache total function evaluations: ', slow_count)
```

```py
With lru_cache total function evaluations:  31
Without lru_cache total function evaluations:  2692537
```

值得注意的是，`lru_cache` 装饰器在你在 Jupyter notebooks 中尝试机器学习问题时特别有用。如果你有一个从互联网上下载数据的函数，将其用 `lru_cache` 装饰可以将下载的内容保存在内存中，并避免即使你多次调用下载函数也重复下载相同的文件。

### 使用 `reduce()`

Reduce 类似于 `itertools.accumulate()`。它将一个函数重复应用于列表的元素，并返回结果。以下是一些带有注释的示例，以解释这些函数的工作原理。

```py
# Evaluates ((1+2)+3)+4
list_sum = functools.reduce(operator.add, [1, 2, 3, 4])
print(list_sum)

# Evaluates (2³)⁴
list_pow = functools.reduce(operator.pow, [2, 3, 4])
print(list_pow)
```

```py
10
4096
```

`reduce()` 函数可以接受任何“操作符”，并可以选择性地指定初始值。例如，前面示例中的 `collections.Counter` 函数可以如下实现：

```py
import functools

def addcount(counter, element):
    if element not in counter:
        counter[element] = 1
    else:
        counter[element] += 1
    return counter

items = ["a", "b", "a", "c", "d", "c", "b", "a"]

counts = functools.reduce(addcount, items, {})
print(counts)
```

```py
{'a': 3, 'b': 2, 'c': 2, 'd': 1}
```

### 使用 `partial()`

有时你会有一个接受多个参数的函数，其中一些参数被反复使用。`partial()` 函数返回一个具有较少参数的新版本的相同函数。

例如，如果你需要重复计算 2 的幂，你可以创建一个新的 numpy 的 `power()` 函数，如下所示：

```py
import numpy

power_2 = functools.partial(np.power, 2)
print('2⁴ =', power_2(4))
print('2⁶ =', power_2(6))
```

```py
2⁴ = 16
2⁶ = 64
```

## Map-Reduce 模式

在前面的章节中，我们提到了 filter、map 和 reduce 函数作为高阶函数。使用 map-reduce 设计模式确实是帮助我们轻松创建高可扩展性程序的一种方法。map-reduce 模式是对许多类型的计算的抽象表示，这些计算操作列表或对象集合。`map` 阶段将输入集合映射到一个中间表示。`reduce` 步骤从这个中间表示中计算出一个单一的输出。这个设计模式在函数式编程语言中非常流行。Python 也提供了构造来高效地实现这一设计模式。

### 在 Python 中的 Map-Reduce

作为 map-reduce 设计模式的一个例子，假设我们要统计列表中能被 3 整除的数字。我们将使用 `lambda` 定义一个匿名函数，并利用它来 `map()` 列表中的所有项，判断它们是否通过我们的可整除性测试，然后将它们映射为 1 或 0。`map()` 函数接受一个函数和一个可迭代对象作为参数。接下来，我们将使用 `reduce()` 来累积最终结果。

```py
# All numbers from 1 to 20
input_list = list(range(20))
# Use map to see which numbers are divisible by 3
bool_list = map(lambda x: 1 if x%3==0 else 0, input_list)
# Convert map object to list
bool_list = list(bool_list)
print('bool_list =', bool_list)

total_divisible_3 = functools.reduce(operator.add, bool_list)
print('Total items divisible by 3 = ', total_divisible_3)
```

```py
bool_list = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
Total items divisible by 3 =  7
```

尽管非常简单，但之前的示例说明了在 Python 中实现 `map-reduce` 设计模式是多么容易。您可以使用 Python 中出乎意料的简单易用的构造来解决复杂且漫长的问题。

## 进一步阅读

本节提供了更多关于该主题的资源，以便您深入了解。

### 书籍

+   [《思考 Python：如何像计算机科学家一样思考》](https://greenteapress.com/thinkpython/html/index.html) 由 Allen B. Downey 编写

+   [《Python 3 编程：Python 语言完全介绍》](https://www.amazon.com/Programming-Python-Complete-Introduction-Language-ebook-dp-B001OFK2DK/dp/B001OFK2DK/ref=mt_other?_encoding=UTF8&me=&qid=1638910263) 由 Mark Summerfield 编写

+   [《Python 编程：计算机科学导论》](https://www.amazon.com/Python-Programming-Introduction-Computer-Science/dp/1590282418/ref=sr_1_1?s=books&ie=UTF8&qid=1441293398&sr=1-1&keywords=Python+Zelle&pebp=1441293404611&perid=1B2BP6WM3RQHW4CY6990) 由 John Zelle 编写

### Python 官方文档

+   [Python 文档](https://docs.python.org/3/contents.html)

## 总结

在本教程中，您了解了支持函数式编程的 Python 特性。

具体来说，您学到了：

+   使用 `itertools` 在 Python 中返回有限或无限序列的可迭代对象

+   `functools` 支持的高阶函数

+   map-reduce 设计模式在 Python 中的实现

对于这篇文章中讨论的 Python，您有任何问题吗？请在下面的评论中提问，我会尽力回答。
