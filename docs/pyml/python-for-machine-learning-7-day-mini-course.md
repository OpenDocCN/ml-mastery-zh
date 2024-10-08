# Python 机器学习（7 天迷你课程）

> 原文：[`machinelearningmastery.com/python-for-machine-learning-7-day-mini-course/`](https://machinelearningmastery.com/python-for-machine-learning-7-day-mini-course/)

#### Python 机器学习速成课程。

在 7 天内学习核心 Python。

Python 是一种了不起的编程语言。它不仅在机器学习项目中被广泛使用，你还可以在系统工具、网页项目等许多领域中找到它的身影。拥有良好的 Python 技能可以让你的工作更加高效，因为它以简洁著称。你可以更快地尝试你的想法。你还可以用简洁的 Python 代码展示你的想法。

作为从业者，你不需要知道语言的构建方式，但你应该知道语言可以帮助你完成各种任务。你可以看到 Python 代码的简洁性，以及其库中的函数可以完成的任务。

在本速成课程中，你将发现一些常见的 Python 技巧，通过在七天内完成练习来掌握它们。

这是一个重要且内容丰富的帖子。你可能想要收藏它。

让我们开始吧。

![Python 机器学习（7 天迷你课程）](img/e18684f37e65715779290e414b076135.png)

Python 机器学习（7 天迷你课程）

图片由 [David Clode](https://unsplash.com/photos/vec5yfUvCGs) 提供，版权所有。

## 这个速成课程适合谁？

在你开始之前，让我们确保你在正确的地方。

本课程适合那些可能了解一些编程的开发者。也许你知道另一种语言，或者你可能能够用 Python 编写几行代码来完成简单的任务。

本课程的内容确实对你有一些假设，例如：

+   你对基本的 Python 知识很熟悉。

+   你理解基本的编程概念，如变量、数组、循环和函数。

+   你可以在命令行或 IDE 中使用 Python。

你不需要是：

+   一位明星程序员

+   一个 Python 专家

本速成课程可以帮助你从一个新手程序员成长为一个可以自如编写 Python 代码的专家。

本速成课程假设你已经安装了一个正常工作的 Python 3.7 环境。如果你需要环境设置方面的帮助，可以按照这里的逐步教程进行操作：

+   [如何使用 Anaconda 设置你的 Python 环境以进行机器学习](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## 速成课程概览

本速成课程分为七节课。

你可以每天完成一节课（推荐）或在一天内完成所有课程（高强度）。这完全取决于你可用的时间和你的热情程度。

以下是将帮助你开始并高效使用 Python 的七节课列表：

+   **第 01 课**：操作列表

+   **第 02 课**：字典

+   **第 03 课**：元组

+   **第 04 课**：字符串

+   **第 05 课**：列表推导式

+   **第 06 课**：枚举和压缩

+   **第 07 课**：映射、过滤和减少

每节课可能需要你花费 5 到 30 分钟。请按自己的节奏完成课程。提问，甚至可以在网上评论区发布结果。

课程可能会要求你自己去找出如何做。这份指南会给你一些提示，但每节课的部分重点是强迫你学习去哪里寻求有关算法和 Python 最佳工具的帮助。

**在评论区发布你的结果**；我会为你加油！

坚持下去，别放弃。

## 课程 01：操作列表

在本节课中，你将了解 Python 中的基础数据结构——列表。

在其他编程语言中，有数组。Python 中的对应物是**列表**。Python 列表没有限制它存储的元素数量。你可以随时向其中添加元素，它会自动扩展大小。Python 列表也不要求其元素类型相同。你可以在一个列表中混合不同的元素。

接下来，我们创建一个整数列表，然后向其中添加一个字符串：

```py
x = [1, 2, 3]
x.append("that's all")
```

Python 列表是零索引的。也就是说，要获取上面列表中的第一个元素，我们可以这样做：

```py
print(x[0])
```

这将打印`1`到屏幕上。

Python 列表允许负索引表示从后往前计数。因此，打印上述列表的最后一个元素的方式是：

```py
print(x[-1])
```

Python 还具有一个方便的语法来查找列表的切片。要打印最后两个元素，我们可以这样做：

```py
print(x[-2:])
```

通常，切片语法是`start:end`，其中 end 不包括在结果中。如果省略，默认起始元素为第一个元素，结束元素为整个列表的最后一个元素之后的元素。我们还可以使用切片语法来设置`step`。例如，这样我们可以提取偶数和奇数：

```py` ``` x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  odd = x[::2]  even = x[1::2]  print(odd)  print(even) ```py    ### Your Task    In the above example of getting odd numbers from a list of 1 to 10, you can make a step size of `-2` to ask the list go backward. How can you use the slicing syntax to print `[9,7,5,3,1]`? How about `[7,5,3]`?    Post your answer in the comments below. I would love to see what you come up with.    In the next lesson, you will discover the Python dictionary.    ## Lesson 02: Dictionaries    In this lesson, you will learn Python’s way of storing a mapping.    Similar to Perl, an associative array is also a native data structure in Python. It is called a dictionary or dict. Python uses square brackets `[]` for list and uses curly brackets `{}` for dict. A Python dict is for key-value mapping, but the key must be **hashable**, such as a number or a string. Hence we can do the following:   ``` price = {      "apple": 1.5,      "orange": 1.25,      "banana": 0.5  }  print("apple costs $", price["apple"]) ```py    Adding a key-value mapping to a dict is similar to indexing a list:   ``` price["lemon"] = 0.6  print("lemon costs $", price["lemon"]) ```py    We can check if a key is in a dict using the \codetext{in} operator, for example:   ``` if "strawberry" in price:      print("strawberry costs $", price["strawberry"])  else:      # if price is not found, assume $1      print("strawberry costs $1") ```py    But in Python dict, we can use the \codetext{get()} function to give us a default value if the key is not found:   ``` print("strawberry costs $", price.get("strawberry", 1)) ```py    But indeed, you are not required to provide a default to \codetext{get()}. If you omitted it, it will return \codetext{None}. For example:   ``` print("strawberry costs $", price.get("strawberry")) ```py    It will produce   ``` strawberry costs $ None ```py    Since the Python dict is a key-value mapping, we can extract only the keys or only the values, using:   ``` fruits = list(price.keys())  numbers = list(price.values())  print(fruits)  print(numbers) ```py    We used `list()` to convert the keys or values to a list for better printing. % The other way to manipulate a list is with the `items()` function. Its result would be key-value pairs:   ``` pairs = list(price.items())  print(pairs) ```py    This prints:   ``` [('apple', 1.5), ('orange', 1.25), ('banana', 0.5), ('lemon', 0.6)] ```py    Since they are pairs in a list, we can use list manipulation syntax to combine items from two dicts and produce a combined dict. The following is an example:   ``` price1 = {      "apple": 1.5,      "orange": 1.25,      "strawberry": 1.0  }  price2 = {      "banana": 0.5  }  pairs1 = list(price1.items())  pairs2 = list(price2.items())  price = dict(pairs1 + pairs2)  print(price) ```py    This will print:   ``` {'apple': 1.5, 'orange': 1.25, 'strawberry': 1.0, 'banana': 0.5} ```py    ### Your Task    Depending on your version of Python, the last example above can have a simplified syntax:   ``` price = price1 | price2  price = {**price1, **price2} ```py    Check in your installation if you can reproduce the same result as the last example.    In the next lesson, you will discover the tuple as a read-only list.    ## Lesson 03: Tuples    In this lesson, you will learn the tuple as a read-only data structure.    Python has a list that behaves like an array of mixed data. A Python tuple is very much like a list, but it cannot be modified after it is created. It is **immutable**. Creating a tuple is just like creating a list, except using parentheses, `()`:   ``` x = (1, 2, 3) ```py    You can refer to the first element as `x[0]` just like the case of a list. But you cannot assign a new value to `x[0]` because a tuple is immutable. If you try to do it, Python will throw a TypeError with the reason that the tuple does not support the item assignment.    A tuple is handy to represent multiple return values of a function. For example, the following function produces a value’s multiple powers as a tuple:   ``` def powers(n):      return n, n**2, n**3  x = powers(2)  print(x) ```py    This will print:   ``` (2, 4, 8) ```py    which is a tuple. But we usually use the unpacking syntax:   ``` itself, squared, cubed = powers(2) ```py    In fact, this is a powerful syntax in Python in which we can assign multiple variables in one line. For example,   ``` count, elements = 0, [] ```py    This will assign variable `count` to integer `0` and variable `elements` to an empty list. Because of the unpacking syntax, this is the **Pythonic** way of swapping the value of two variables:   ``` a, b = b, a ```py    ### Your Task    Consider a list of tuples:   ``` x = [("alpha", 0.5), ("gamma", 0.1), ("beta", 1.1), ("alpha", -3)] ```py    You can sort this list using `sorted(x)`. What is the result? From the result of comparing tuples, how does Python understand which tuple is less than or greater than another? Which is greater, the tuple `("alpha", 0.5)` or the tuple `("alpha", 0.5, 1)`?    Post your answer in the comments below. I would love to see what you come up with.    In the next lesson, you will learn about Python strings.    ## Lesson 04: Strings    In this lesson, you will learn about creating and using strings in Python.    A string is the basic way of storing text in Python. All Python strings are unicode strings, meaning you can put unicode into it. For example:   ``` x = "Hello ????"  print(x) ```py    The smiley is a unicode character of code point 0x1F600\. Python string comes with a lot of functions. For example, we can check if a string begins or ends with a substring using:   ``` if x.startswith("Hello"):      print("x starts with Hello")  if not x.endswith("World"):      print("x does not end with World") ```py    Then to check whether a string contains a substring, use the “`in`” operator:   ``` if "ll" in x:      print("x contains double-l") ```py    There is a lot more. Such as `split()` to split a string, or `upper()` to convert the entire string into uppercase.    One special property of Python strings is the **implicit concatenation**. All of the following produce the string `"hello world"`:   ``` x = "hel" \    "lo world"  x = "hello" " world"  x = ("hello "      "world") ```py    The rule is, Python will normally use `\` as a line continuation. But if Python sees two strings placed together without anything separating them, the strings will be concatenated. Hence the first example above is to concatenate `"hel"` with `"lo world"`. Likewise, the last example concatenated two strings because they are placed inside parentheses.    A Python string can also be created using a template. It is often seen in `print()` functions. For example, below all produce `"hello world"` for variable `y`:   ``` x = "world"  y = "hello %s" % x  y = "hello {}".format(x)  y = f"hello {x}" ```py    ### Your Task    Try to run this code:   ``` coord = {"lat": 51.5072, "lon": -0.1276}  print("latitude %(lat)f, longitude %(lon)f" % coord)  print("latitude {lat}, longitude {lon}".format(**coord)) ```py    This is to fill a template using a dictionary. The first uses the `%`-syntax while the second uses format syntax. Can you modify the code above to print only 2 decimal places? Hints: Check out [`docs.python.org/3/library/string.html`](https://docs.python.org/3/library/string.html)!    Post your answer in the comments below. I would love to see what you come up with.    In the next lesson, you will discover list comprehension syntax in Python.    ## Lesson 05: List Comprehension    In this lesson, you will see how list comprehension syntax can build a list on the fly.    The famous fizz-buzz problem prints 1 to 100 with all 3-multiples replaced with “fizz,” all 5-multiples replaced with “buzz,” and if a number is both a multiple of 3 and 5, print “fizzbuzz.” You can make a `for` loop and some `if` statements to do this. But we can also do it in one line:   ``` numbers = ["fizzbuzz" if n%15==0 else "fizz" if n%3==0 else "buzz" if n%5==0 else str(n)            for n in range(1,101)]  print("\n".join(numbers)) ```py    We set up the list `numbers` using list comprehension syntax. The syntax looks like a list but with a `for` inside. Before the keyword `for`, we define how each element in the list will be created.    List comprehension can be more complicated. For example, this is how to produce all multiples of 3 from 1 to 100:   ``` mul3 = [n for n in range(1,101) if n%3 == 0] ```py    And this is how we can print a $10\times 10$ multiplication table:   ``` table = [[m*n for n in range(1,11)] for m in range(1,11)]  for row in table:      print(row) ```py    And this is how we can combine strings:   ``` directions = [a+b for a in ["north", "south", ""] for b in ["east", "west", ""] if not (a=="" and b=="")]  print(directions) ```py    This prints:   ``` ['northeast', 'northwest', 'north', 'southeast', 'southwest', 'south', 'east', 'west'] ```py    ### Your Task    Python also has a dictionary comprehension. The syntax is:   ``` double = {n: 2*n for n in range(1,11)} ```py    Now try to create a dictionary `mapping` using dictionary comprehension that maps a string `x` to its length `len(x)` for these strings:   ``` keys = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]  mapping = {...} ```py    Post your answer in the comments below. I would love to see what you come up with.    In the next lesson, you will discover two very useful Python functions: `enumerate()` and `zip()`.    ## Lesson 06: Enumerate and Zip    In this lesson, you will learn an the `enumerate()` function and `zip()` function.    Very often, you will see you’re writing a for-loop like this:   ``` x = ["alpha", "beta", "gamma", "delta"]  for n in range(len(x)):      print("{}: {}".format(n, x[n])) ```py    But here we need the loop variable `n` just to use as an index to access the list `x`. In this case, we can ask Python to index the list while doing the loop, using `enumerate()`:   ``` x = ["alpha", "beta", "gamma", "delta"]  for n,string in enumerate(x):      print("{}: {}".format(n, string)) ```py    The result of `enumerate()` produces a tuple of the counter (default starts with zero) and the element of the list. We use the unpacking syntax to set it to two variables.    If we use the for-loop like this:   ``` x = ["blue", "red", "green", "yellow"]  y = ["cheese", "apple", "pea", "mustard"]  for n in range(len(x)):      print("{} {}".format(x[n], y[n])) ```py    Python has a function `zip()` to help:   ``` x = ["blue", "red", "green", "yellow"]  y = ["cheese", "apple", "pea", "mustard"]  for a, b in zip(x, y):      print("{} {}".format(a, b)) ```py    The `zip()` function is like a zipper, taking one element from each input list and putting them side by side. You may provide more than two lists to `zip()`. It will produce all matching items (i.e., stop whenever it hits the end of the shortest input list).    ### Your task    Very common in Python programs, we may do this:   ``` results = []  for n in range(1, 11):      squared, cubed = n**2, n**3      results.append([n, squared, cubed]) ```py    Then, we can get the list of 1 to 10, the square of them, and the cube of them using `zip()` (note the `*` before `results` in the argument):   ``` numbers, squares, cubes = zip(*results) ```py    Try this out. Can you recombine `numbers`, `squares`, and `cubes` back to `results`? Hints: Just use `zip()`.    In the next lesson, you will discover three more Python functions: `map()`, `filter()`, and `reduce()`.    ## Lesson 07: Map, Filter, and Reduce    In this lesson, you will learn the Python functions `map()`, `filter()`, and `reduce()`.    The name of these three functions came from the functional programming paradigm. In simple terms, `map()` is to transform elements of a list using some function, and `filter()` is to short list the elements based on certain criteria. If you learned list comprehension, they are just another method of list comprehension.    Let’s consider an example we saw previously:   ``` def fizzbuzz(n):      if n%15 == 0:          return "fizzbuzz"      if n%3 == 0:          return "fizz"      if n%5 == 0:          return "buzz"      return str(n)    numbers = map(fizzbuzz, range(1,101))  print("\n".join(numbers)) ```py    Here we have a function defined, and `map()` uses the function as the first argument and a list as the second argument. It will take each element from a list and transform it using the provided function.    Using `filter()` is likewise:   ``` def multiple3(n):      return n % 3 == 0    mul3 = filter(multiple3, range(1,101))  print(list(mul3)) ```py    If that’s appropriate, you can pass the return value from `map()` to `filter()` or vice versa.    You may consider `map()` and `filter()` as another way to write list comprehension (sometimes easier to read as the logic is modularized). The `reduce()` function is not replaceable by list comprehension. It scans the elements from a list and combines them using a function.    While Python has a `max()` function built-in, we can use `reduce()` for the same purpose. Note that `reduce()` is a function from the module `functools`:   ``` from functools import reduce  def maximum(a,b):      if a > b:          return a      else:          return b    x = [-3, 10, 2, 5, -6, 12, 0, 1]  max_x = reduce(maximum, x)  print(max_x) ```py    By default, `reduce()` will give the first two elements to the provided function, then the result will be passed to the function again with the third element, and so on until the input list is exhausted. But there is another way to invoke `reduce()`:   ``` x = [-3, 10, 2, 5, -6, 12, 0, 1]  max_x = reduce(maximum, x, -float("inf"))  print(max_x) ```py    This result is the same, but the first call to the function uses the default value (`-float("inf")` in this case, which is negative infinity) and the first element of the list. Then uses the result and the second element from the list, and so on. Providing a default value is appropriate in some cases, such as the exercise below.    ### Your Task    Let’s consider a way to convert a bitmap to an integer. If a list `[6,2,0,3]` is provided, we should consider the list as which bit to assert, and the result should be in binary, 1001101, or in decimal, 77\. In this case, bit 0 is defined to be the least significant bit or the right most bit.    We can use reduce to do this and print 77:   ``` def setbit(bitmap, bit):      return bitmap | (2**bit)    assertbits = [6, 2, 0, 3]  bitmap = reduce(setbit, assertbits, ???)  print(bitmap) ```py    What should be the `???` above? Why?    Post your answer in the comments below. I would love to see what you come up with.    This was the final lesson.    ## The End! (*Look How Far You Have Come*)    You made it. Well done!    Take a moment and look back at how far you have come.    You discovered:    *   Python list and the slicing syntax *   Python dictionary, how to use it, and how to combine two dictionaries *   Tuples, the unpacking syntax, and how to use it to swap variables *   Strings, including many ways to create a new string from a template *   List comprehension *   The use of functions `enumerate()` and `zip()` *   How to use `map()`, `filter()`, and `reduce()`    ## Summary    **How did you do with the mini-course?** Did you enjoy this crash course?    **Do you have any questions? Were there any sticking points?** Let me know. Leave a comment below. ```
