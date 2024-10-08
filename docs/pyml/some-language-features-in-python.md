# Python 中的一些语言特性

> 原文：[`machinelearningmastery.com/some-language-features-in-python/`](https://machinelearningmastery.com/some-language-features-in-python/)

Python 语言的语法非常强大且富有表现力。因此，用 Python 表达一个算法简洁明了。也许这就是它在机器学习中受欢迎的原因，因为在开发机器学习模型时，我们需要进行大量实验。

如果你对 Python 不熟悉但有其他编程语言的经验，你会发现 Python 的语法有时容易理解但又奇怪。如果你习惯于用 C++或 Java 编写代码，然后转到 Python，可能你的程序就不是**Pythonic**的。

在本教程中，我们将涵盖 Python 中的几种常见语言特性，这些特性使其与其他编程语言有所区别。

**用我的新书[Python for Machine Learning](https://machinelearningmastery.com/python-for-machine-learning/)启动你的项目**，包括*逐步教程*和所有示例的*Python 源代码*文件。

让我们开始吧。![](img/2d608b7ddfff69813b497b423179c919.png)

Python 中的一些语言特性

图片由[David Clode](https://unsplash.com/photos/QZePScKPb2Q)提供，部分权利保留。

## 教程概述

本教程分为两部分；它们是：

1.  操作符

1.  内置数据结构

1.  特殊变量

1.  内置函数

## 操作符

Python 中使用的大多数操作符与其他语言相同。优先级表如下，采用自 Python 语言参考第六章（[`docs.python.org/3/reference/expressions.html`](https://docs.python.org/3/reference/expressions.html)）：

| Operator | 描述 |
| --- | --- |
| (expressions…), [expressions…], {key: value…}, {expressions…} | 绑定或括号表达式、列表显示、字典显示、集合显示 |
| x[index], x[index:index], x(arguments…), x.attribute | 订阅、切片、调用、属性引用 |
| await x | 等待表达式 |
| ** | 幂运算 |
| +x, -x, ~x | 正数、负数、按位非 |
| *, @, /, //, % | 乘法、矩阵乘法、除法、地板除法、余数 |
| +, – | 加法和减法 |
| <<, >> | 位移 |
| & | 按位与 |
| ^ | 按位异或 |
| &#124; | 按位或 |
| in, not in, is, is not, <, <=, >, >=, !=, == | 比较，包括成员测试和身份测试 |
| not x | 布尔非 |
| and | 布尔与 |
| or | 布尔或 |
| if – else | 条件表达式 |
| lambda | Lambda 表达式 |
| := | 赋值表达式 |

与其他语言的一些关键区别：

+   布尔运算符是完整拼写的，而位运算符是字符`&`、`^`和`|`

+   幂运算使用`2**3`

+   整数除法使用`//`，而除法`/`总是返回浮点值

+   三元运算符：如果你熟悉 C 语言中的表达式`(x)?a:b`，我们在 Python 中写作`a if x else b`

+   比较两个东西是否相等可以使用 `==` 或 `is`。`==` 运算符对于相等性与其他语言相同，但 `is` 更严格，保留用于检查两个变量是否指向同一个对象。

在 Python 中，我们允许在比较操作符中进行连接。例如，要测试一个值是否在 -1 到 +1 之间，我们可以这样做：

Python

```py
if value > -1 and value < 1:
    ...
```

但我们也可以这样做：

```py
if -1 < value < 1:
    ...
```

## 内置数据结构

和许多其他语言一样，Python 中有整数和浮点数数据类型。但也有复数（例如 `3+1j`），布尔常量（`True` 和 `False`），字符串，以及一个虚拟类型 `None`。

但 Python 作为一种语言的强大之处在于它内置了容器类型：Python 数组称为“列表”，它会自动扩展。关联数组（或哈希表）称为“字典”。我们还有“元组”作为只读列表和“集合”作为存储唯一项的容器。例如，在 C++ 中，您需要 STL 来提供这些功能。

"dict" 数据结构可能是 Python 中最强大的一个，让我们在编写代码时更加方便。例如，在狗和猫的图像分类问题中，我们的机器学习模型可能只会给出 0 或 1 的值，如果想要打印名称，我们可以这样做：

```py
value = 0 # This is obtained from a model

value_to_name = {0: "cat", 1: "dog"}
print("Result is %s" % value_to_name[value])
```

```py
Result is cat
```

在这种情况下，我们使用字典 `value_to_name` 作为查找表。类似地，我们还可以利用字典来构建计数器：

```py
sentence = "Portez ce vieux whisky au juge blond qui fume"
counter = {}
for char in sentence:
    if char not in counter:
        counter[char] = 0
    counter[char] += 1

print(counter)
```

```py
{'P': 1, 'o': 2, 'r': 1, 't': 1, 'e': 5, 'z': 1, ' ': 8, 'c': 1, 'v': 1, 'i': 3, 'u': 5, 'x': 1, 'w': 1, 'h': 1, 's': 1, 'k': 1, 'y': 1, 'a': 1, 'j': 1, 'g': 1, 'b': 1, 'l': 1, 'n': 1, 'd': 1, 'q': 1, 'f': 1, 'm': 1}
```

这将构建一个名为 `counter` 的字典，将每个字符映射到句子中出现的次数。

Python 列表还具有强大的语法。与某些其他语言不同，我们可以将任何东西放入列表中：

```py
A = [1, 2, "fizz", 4, "buzz", "fizz", 7]
A += [8, "fizz", "buzz", 11, "fizz", 13, 14, "fizzbuzz"]
print(A)
```

```py
[1, 2, 'fizz', 4, 'buzz', 'fizz', 7, 8, 'fizz', 'buzz', 11, 'fizz', 13, 14, 'fizzbuzz']
```

我们可以使用 `+` 连接列表。在上例中，我们使用 `+=` 来扩展列表 `A`。

Python 列表具有切片语法。例如，在上述 `A` 中，我们可以使用 `A[1:3]` 表示第 1 和第 2 个元素，即 `[2, "fizz"]`，而 `A[1:1]` 则是一个空列表。事实上，我们可以将某些内容分配给一个切片，以插入或删除一些元素。例如：

```py
...
A[2:2] = [2.1, 2.2]
print(A)
```

```py
[1, 2, 2.1, 2.2, 'fizz', 4, 'buzz', 'fizz', 7, 8, 'fizz', 'buzz', 11, 'fizz', 13, 14, 'fizzbuzz']
```

然后，

```py
...
A[0:2] = []
print(A)
```

```py
[2.1, 2.2, 'fizz', 4, 'buzz', 'fizz', 7, 8, 'fizz', 'buzz', 11, 'fizz', 13, 14, 'fizzbuzz']
```

元组与列表具有类似的语法，只是使用圆括号来定义：

```py
A = ("foo", "bar")
```

元组是不可变的。这意味着一旦定义，就无法修改它。在 Python 中，如果用逗号分隔几个东西放在一起，它被认为是一个元组。这样做的意义在于，我们可以以非常清晰的语法交换两个变量：

```py
a = 42
b = "foo"
print("a is %s; b is %s" % (a,b))
a, b = b, a # swap
print("After swap, a is %s; b is %s" % (a,b))
```

```py
a is 42; b is foo
After swap, a is foo; b is 42
```

最后，正如您在上面的示例中看到的那样，Python 字符串支持即时替换。与 C 中的 `printf()` 函数类似的模板语法，我们可以使用 `%s` 替换字符串或 `%d` 替换整数。我们还可以使用 `%.3f` 替换带有三位小数的浮点数。以下是一个示例：

```py
template = "Square root of %d is %.3f"
n = 10
answer = template % (n, n**0.5)
print(answer)
```

```py
Square root of 10 is 3.162
```

但这只是其中的一种方法。上述内容也可以通过 f-string 和 format() 方法来实现。

## 特殊变量

Python 有几个“特殊变量”预定义。`__name__` 告诉当前命名空间，而 `__file__` 告诉脚本的文件名。更多的特殊变量存在于对象内部，但几乎所有的通常不应该被直接使用。作为一种惯例（即，仅仅是一种习惯，没有人阻止你这样做），我们以单下划线或双下划线作为前缀来命名内部变量（顺便提一下，双下划线有些人称之为“dunder”）。如果你来自 C++ 或 Java，这些相当于类的私有成员，尽管它们在技术上并不是私有的。

一个值得注意的“特殊”变量是 `_`，仅一个下划线字符。按照惯例，它表示我们不关心的变量。为什么需要一个不关心的变量？因为有时你会保存一个函数的返回值。例如，在 pandas 中，我们可以扫描数据框的每一行：

```py
import pandas as pd
A = pd.DataFrame([[1,2,3],[2,3,4],[3,4,5],[5,6,7]], columns=["x","y","z"])
print(A)

for _, row in A.iterrows():
    print(row["z"])
```

```py
x y z
0 1 2 3
1 2 3 4
2 3 4 5
3 5 6 7

3
4
5
7
```

在上述内容中，我们可以看到数据框有三列，“x”、“y”和“z”，行由 0 到 3 进行索引。如果我们调用 `A.iterrows()`，它会逐行返回索引和行，但我们不关心索引。我们可以创建一个新变量来保存它但不使用它。为了明确我们不会使用它，我们使用 `_` 作为保存索引的变量，而行则存储到变量 `row` 中。

### 想要开始学习用于机器学习的 Python 吗？

现在就领取我的免费 7 天电子邮件速成课程（附示例代码）。

点击注册，还可以获得课程的免费 PDF 电子书版本。

## 内置函数

在 Python 中，一些函数被定义为内置函数，而其他功能则通过其他包提供。所有内置函数的列表可以在 Python 标准库文档中找到（[`docs.python.org/3/library/functions.html`](https://docs.python.org/3/library/functions.html)）。以下是 Python 3.10 中定义的函数：

```py
abs()
aiter()
all()
any()
anext()
ascii()
bin()
bool()
breakpoint()
bytearray()
bytes()
callable()
chr()
classmethod()
compile()
complex()
delattr()
dict()
dir()
divmod()
enumerate()
eval()
exec()
filter()
float()
format()
frozenset()
getattr()
globals()
hasattr()
hash()
help()
hex()
id()
input()
int()
isinstance()
issubclass()
iter()
len()
list()
locals()
map()
max()
memoryview()
min()
next()
object()
oct()
open()
ord()
pow()
print()
property()
range()
repr()
reversed()
round()
set()
setattr()
slice()
sorted()
staticmethod()
str()
sum()
super()
tuple()
type()
vars()
zip()
__import__()
```

并非所有的函数每天都会用到，但有些特别值得注意：

`zip()` 允许你将多个列表组合在一起。例如，

```py
a = ["x", "y", "z"]
b = [3, 5, 7, 9]
c = [2.1, 2.5, 2.9]
for x in zip(a, b, c):
    print(x)
```

```py
('x', 3, 2.1)
('y', 5, 2.5)
('z', 7, 2.9)
```

如果你想“旋转”一个列表的列表，这很方便，例如，

```py
a = [['x', 3, 2.1], ['y', 5, 2.5], ['z', 7, 2.9]]
p,q,r = zip(*a)
print(p)
print(q)
print(r)
```

```py
('x', 'y', 'z')
(3, 5, 7)
(2.1, 2.5, 2.9)
```

`enumerate()` 非常方便，可以让你对列表项进行编号，例如：

```py
a = ["quick", "brown", "fox", "jumps", "over"]
for num, item in enumerate(a):
    print("item %d is %s" % (num, item))
```

```py
item 0 is quick
item 1 is brown
item 2 is fox
item 3 is jumps
item 4 is over
```

如果你不使用 `enumerate`，这等同于以下操作：

```py
a = ["quick", "brown", "fox", "jumps", "over"]
for num in range(len(a)):
    print("item %d is %s" % (num, a[num]))
```

与其他语言相比，Python 中的 for 循环是迭代一个预定义的范围，而不是在每次迭代中计算值。换句话说，它没有直接等同于以下的 C for 循环：

C

```py
for (i=0; i<100; ++i) {
...
}
```

在 Python 中，我们必须使用 `range()` 来完成相同的操作：

```py
for i in range(100):
    ...
```

类似地，有一些函数用于操作列表（或类似列表的数据结构，Python 称之为“可迭代对象”）：

+   `max(a)`：查找列表 `a` 中的最大值

+   `min(a)`：查找列表 `a` 中的最小值

+   `sum(a)`：查找列表 `a` 中值的总和

+   `reverse(a)`：从列表 `a` 的末尾开始迭代

+   `sorted(a)`：返回一个按排序顺序排列的列表 `a` 的副本

我们将在下一篇文章中进一步讨论这些内容。

## 进一步阅读

上述内容仅突出了 Python 中的一些关键特性。毫无疑问，没有比 Python.org 的官方文档更权威的资料了；所有初学者都应从 Python 教程开始，并查看语言参考以获取语法细节，标准库则提供了 Python 安装附带的额外库：

+   Python 教程 – [`docs.python.org/3/tutorial/index.html`](https://docs.python.org/3/tutorial/index.html)

+   Python 语言参考 – [`docs.python.org/3/reference/index.html`](https://docs.python.org/3/reference/index.html)

+   Python 标准库 – [`docs.python.org/3/library/index.html`](https://docs.python.org/3/library/index.html)

对于书籍，Lutz 的 *Learning Python* 是一个老而好的入门书籍。之后，*流畅的 Python* 可以帮助您更好地理解语言的内部结构。然而，如果您想快速入门，Al Sweigart 的书籍可以通过示例帮助您快速掌握语言。一旦熟悉 Python，您可能希望从 *Python Cookbook* 中获取某个特定任务的快速技巧。

+   *Python 学习手册* 第五版 作者 Mark Lutz, O’Reilly, 2013, [`www.amazon.com/dp/1449355730/`](https://www.amazon.com/dp/1449355730/)

+   *流畅的 Python* 作者 Luciano Ramalho, O’Reilly, 2015, [`www.amazon.com/dp/1491946008/`](https://www.amazon.com/dp/1491946008/)

+   *用 Python 自动化繁琐的工作* 第二版 作者 Al Sweigart, No Starch Press, 2019, [`www.amazon.com/dp/1593279922/`](https://www.amazon.com/dp/1593279922/)

+   *Python Cookbook* 第三版 作者 David Beazley 和 Brian K. Jones, O’Reilly, 2013, [`www.amazon.com/dp/1449340377/`](https://www.amazon.com/dp/1449340377/)

## 总结

在本教程中，您发现了 Python 的一些独特特性。具体来说，您学到了：

+   Python 提供的运算符

+   一些内置数据结构的使用

+   一些经常使用的内置函数及其实用性
