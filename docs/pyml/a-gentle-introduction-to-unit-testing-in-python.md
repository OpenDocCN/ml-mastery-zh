# Python 单元测试的温和介绍

> 原文：[`machinelearningmastery.com/a-gentle-introduction-to-unit-testing-in-python/`](https://machinelearningmastery.com/a-gentle-introduction-to-unit-testing-in-python/)

单元测试是一种测试软件的方法，关注代码中最小的可测试单元，并测试其正确性。通过单元测试，我们可以验证代码的每个部分，包括可能不对用户公开的辅助函数，是否正常工作并按预期执行。

这个理念是，我们独立检查程序中的每一个小部分，以确保它正常工作。这与回归测试和集成测试形成对比，后者测试程序的不同部分是否协同工作并按预期执行。

在这篇文章中，你将发现如何使用两个流行的单元测试框架：内置的 PyUnit 框架和 PyTest 框架来实现 Python 中的单元测试。

完成本教程后，你将知道：

+   Python 中的单元测试库，如 PyUnit 和 PyTest

+   通过使用单元测试检查预期的函数行为

**通过我的新书** [《Python 机器学习》](https://machinelearningmastery.com/python-for-machine-learning/)，**启动你的项目**，包括*逐步教程*和所有示例的*Python 源代码*文件。

让我们开始吧！！[](../Images/9387df1b101457bd8a20d9afde5e585f.png)

Python 单元测试的温和介绍

图片由[Bee Naturalles](https://unsplash.com/photos/IRM9qgZdlW0)提供。版权所有。

## 概述

本教程分为五个部分；它们是：

+   什么是单元测试，它们为什么重要？

+   什么是测试驱动开发（TDD）？

+   使用 Python 内置的 PyUnit 框架

+   使用 PyTest 库

+   单元测试的实际操作

## 什么是单元测试，它们为什么重要？

记得在学校做数学时，完成不同的算术步骤，然后将它们组合以获得正确答案吗？想象一下，你如何检查每一步的计算是否正确，确保没有粗心的错误或写错任何东西。

现在，把这个理念扩展到代码上！我们不希望不断检查代码以静态验证其正确性，那么你会如何创建测试以确保以下代码片段实际返回矩形的面积？

```py
def calculate_area_rectangle(width, height):
    return width * height
```

我们可以用一些测试示例运行代码，看看它是否返回预期的输出。

这就是单元测试的理念！单元测试是检查单一代码组件的测试，通常模块化为函数，并确保其按预期执行。

单元测试是回归测试的重要组成部分，以确保在对代码进行更改后，代码仍然按预期功能运行，并帮助确保代码的稳定性。在对代码进行更改后，我们可以运行之前创建的单元测试，以确保我们对代码库其他部分的更改没有影响到现有功能。

单元测试的另一个关键好处是它们有助于轻松隔离错误。想象一下运行整个项目并收到一连串的错误。我们该如何调试代码呢？

这就是单元测试的作用。我们可以分析单元测试的输出，查看代码中的任何组件是否出现错误，并从那里开始调试。这并不是说单元测试总能帮助我们找到错误，但它在我们开始查看集成测试中的组件集成之前提供了一个更方便的起点。

在接下来的文章中，我们将展示如何通过测试 Rectangle 类中的函数来进行单元测试：

```py
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def get_area(self):
        return self.width * self.height

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height
```

现在我们已经了解了单元测试的意义，让我们探索一下如何将单元测试作为开发流程的一部分，以及如何在 Python 中实现它们！

## 测试驱动开发

测试在良好的软件开发中如此重要，以至于甚至存在一个基于测试的软件开发过程——测试驱动开发（TDD）。Robert C. Martin 提出的 TDD 三条规则是：

+   *除非是为了让一个失败的单元测试通过，否则你不允许编写任何生产代码。*

+   *除非是为了让单元测试失败，否则你不允许编写超过必要的单元测试代码，编译失败也是失败。*

+   *除非是为了让一个失败的单元测试通过，否则你不允许编写比通过一个失败单元测试所需的更多生产代码。*

TDD 的关键理念是，我们围绕一组我们创建的单元测试来进行软件开发，这使得单元测试成为 TDD 软件开发过程的核心。这样，你可以确保你为开发的每个组件都制定了测试。

TDD 还偏向于进行更小的测试，这意味着测试更具体，并且每次测试的组件更少。这有助于追踪错误，而且小的测试更易于阅读和理解，因为每次运行中涉及的组件更少。

这并不意味着你必须在你的项目中使用 TDD。但你可以考虑将其作为同时开发代码和测试的方法。

### 想要开始使用 Python 进行机器学习吗？

立即参加我的免费 7 天邮件速成课程（附示例代码）。

点击注册，还可以获得课程的免费 PDF 电子书版本。

## 使用 Python 内置的 PyUnit 框架

你可能会想，既然 Python 和其他语言提供了 `assert` 关键字，我们为什么还需要单元测试框架？单元测试框架有助于自动化测试过程，并允许我们对同一函数运行多个测试，使用不同的参数，检查预期的异常等等。

PyUnit 是 Python 内置的单元测试框架，也是 Python 版的 JUnit 测试框架。要开始编写测试文件，我们需要导入 `unittest` 库以使用 PyUnit：

```py
import unittest
```

然后，我们可以开始编写第一个单元测试。PyUnit 中的单元测试结构为 `unittest.TestCase` 类的子类，我们可以重写 `runTest()` 方法来执行自己的单元测试，使用 `unittest.TestCase` 中的不同断言函数检查条件：

```py
class TestGetAreaRectangle(unittest.TestCase):
    def runTest(self):
        rectangle = Rectangle(2, 3)
        self.assertEqual(rectangle.get_area(), 6, "incorrect area")
```

这就是我们的第一个单元测试！它检查 `rectangle.get_area()` 方法是否返回宽度 = 2 和长度 = 3 的矩形的正确面积。我们使用 `self.assertEqual` 而不是简单使用 `assert`，以便 `unittest` 库允许测试运行器累积所有测试用例并生成报告。

使用 `unittest.TestCase` 中的不同断言函数还可以更好地测试不同的行为，例如 `self.assertRaises(exception)`。这允许我们检查某段代码是否产生了预期的异常。

要运行单元测试，我们在程序中调用 `unittest.main()`，

```py
...
unittest.main()
```

由于代码返回了预期的输出，它显示测试成功运行，输出为：

```py
.
----------------------------------------------------------------------
Ran 1 test in 0.003s

OK
```

完整的代码如下：

```py
import unittest

# Our code to be tested
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def get_area(self):
        return self.width * self.height

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

# The test based on unittest module
class TestGetAreaRectangle(unittest.TestCase):
    def runTest(self):
        rectangle = Rectangle(2, 3)
        self.assertEqual(rectangle.get_area(), 6, "incorrect area")

# run the test
unittest.main()
```

**注意：** 在上面，我们的业务逻辑 `Rectangle` 类和我们的测试代码 `TestGetAreaRectangle` 被放在一起。实际上，你可以将它们放在不同的文件中，并将业务逻辑 `import` 到你的测试代码中。这可以帮助你更好地管理代码。

我们还可以在 `unittest.TestCase` 的一个子类中嵌套多个单元测试，通过在新子类中的方法名前加上 “`test`” 前缀，例如：

```py
class TestGetAreaRectangle(unittest.TestCase):
    def test_normal_case(self):
        rectangle = Rectangle(2, 3)
        self.assertEqual(rectangle.get_area(), 6, "incorrect area")

    def test_negative_case(self): 
        """expect -1 as output to denote error when looking at negative area"""
        rectangle = Rectangle(-1, 2)
        self.assertEqual(rectangle.get_area(), -1, "incorrect negative output")
```

运行这段代码将给我们第一个错误：

```py
F.
======================================================================
FAIL: test_negative_case (__main__.TestGetAreaRectangle)
expect -1 as output to denote error when looking at negative area
----------------------------------------------------------------------
Traceback (most recent call last):
 	File "<ipython-input-96-59b1047bb08a>", line 9, in test_negative_case
 		self.assertEqual(rectangle.get_area(), -1, "incorrect negative output")
AssertionError: -2 != -1 : incorrect negative output
----------------------------------------------------------------------
Ran 2 tests in 0.003s

FAILED (failures=1)
```

我们可以看到失败的单元测试，即 `test_negative_case`，如输出中突出显示的内容和 stderr 消息，因为 `get_area()` 没有返回我们在测试中预期的 -1。

unittest 中定义了许多不同种类的断言函数。例如，我们可以使用 TestCase 类：

```py
def test_geq(self):
  """tests if value is greater than or equal to a particular target"""
  self.assertGreaterEqual(self.rectangle.get_area(), -1)
```

我们甚至可以检查在执行过程中是否抛出了特定的异常：

```py
def test_assert_raises(self): 
  """using assertRaises to detect if an expected error is raised when running a particular block of code"""
  with self.assertRaises(ZeroDivisionError):
    a = 1 / 0
```

现在，我们来看看如何建立我们的测试。如果我们有一些代码需要在每个测试运行之前执行呢？我们可以重写 `unittest.TestCase` 中的 `setUp` 方法。

```py
class TestGetAreaRectangleWithSetUp(unittest.TestCase):
  def setUp(self):
    self.rectangle = Rectangle(0, 0)

  def test_normal_case(self):
    self.rectangle.set_width(2)
    self.rectangle.set_height(3)
    self.assertEqual(self.rectangle.get_area(), 6, "incorrect area")

  def test_negative_case(self): 
    """expect -1 as output to denote error when looking at negative area"""
    self.rectangle.set_width(-1)
    self.rectangle.set_height(2)
    self.assertEqual(self.rectangle.get_area(), -1, "incorrect negative output")
```

在上述代码示例中，我们重写了来自 `unittest.TestCase` 的 `setUp()` 方法，使用了我们自己的 `setUp()` 方法来初始化一个 `Rectangle` 对象。这个 `setUp()` 方法在每个单元测试之前运行，有助于避免在多个测试依赖相同代码来设置测试时的代码重复。这类似于 JUnit 中的 `@Before` 装饰器。

同样，我们还可以重写 `tearDown()` 方法，用于在每个测试之后执行代码。

为了在每个 TestCase 类中只运行一次该方法，我们也可以使用 `setUpClass` 方法，如下所示：

```py
class TestGetAreaRectangleWithSetUp(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    self.rectangle = Rectangle(0, 0)
```

上述代码在每个 TestCase 中仅运行一次，而不是像 setUp 那样在每次测试运行时运行一次。

为了帮助我们组织测试并选择要运行的测试集，我们可以将测试用例汇总到测试套件中，这有助于将应一起执行的测试分组到一个对象中：

```py
...
# loads all unit tests from TestGetAreaRectangle into a test suite
calculate_area_suite = unittest.TestLoader() \
                       .loadTestsFromTestCase(TestGetAreaRectangleWithSetUp)
```

在这里，我们还介绍了另一种通过使用 `unittest.TextTestRunner` 类在 PyUnit 中运行测试的方法，该类允许我们运行特定的测试套件。

```py
runner = unittest.TextTestRunner()
runner.run(calculate_area_suite)
```

这与从命令行运行文件并调用 `unittest.main()` 的输出相同。

综合所有内容，这就是单元测试的完整脚本：

```py
class TestGetAreaRectangleWithSetUp(unittest.TestCase):

  @classmethod
  def setUpClass(self):
    #this method is only run once for the entire class rather than being run for each test which is done for setUp()
    self.rectangle = Rectangle(0, 0)

  def test_normal_case(self):
    self.rectangle.set_width(2)
    self.rectangle.set_height(3)
    self.assertEqual(self.rectangle.get_area(), 6, "incorrect area")

  def test_geq(self):
    """tests if value is greater than or equal to a particular target"""
    self.assertGreaterEqual(self.rectangle.get_area(), -1)

  def test_assert_raises(self): 
    """using assertRaises to detect if an expected error is raised when running a particular block of code"""
    with self.assertRaises(ZeroDivisionError):
      a = 1 / 0
```

这只是 PyUnit 能做的一部分。我们还可以编写测试，查找与正则表达式匹配的异常消息或仅运行一次的 `setUp`/`tearDown` 方法（例如 `setUpClass`）。

## 使用 PyTest

PyTest 是内置 unittest 模块的替代品。要开始使用 PyTest，您首先需要安装它，可以通过以下方式进行安装：

Shell

```py
pip install pytest
```

要编写测试，您只需编写以“`test`”为前缀的函数名，PyTest 的测试发现程序将能够找到您的测试，例如，

```py
def test_normal_case(self):
    rectangle = Rectangle(2, 3)
    assert rectangle.get_area() == 6, "incorrect area"
```

您会注意到 PyTest 使用 Python 内置的 `assert` 关键字，而不是像 PyUnit 那样的一组断言函数，这可能会更方便，因为我们可以避免查找不同的断言函数。

完整的代码如下：

```py
# Our code to be tested
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def get_area(self):
        return self.width * self.height

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

# The test function to be executed by PyTest
def test_normal_case():
    rectangle = Rectangle(2, 3)
    assert rectangle.get_area() == 6, "incorrect area"
```

将其保存到文件 `test_file.py` 中后，我们可以通过以下方式运行 PyTest 单元测试：

Shell

```py
python -m pytest test_file.py
```

这将给我们以下输出：

```py
=================== test session starts ====================
platform darwin -- Python 3.9.9, pytest-7.0.1, pluggy-1.0.0
rootdir: /Users/MLM
plugins: anyio-3.4.0, typeguard-2.13.2
collected 1 item

test_file.py .                                       [100%]

==================== 1 passed in 0.01s =====================
```

您可能会注意到，在 PyUnit 中，我们需要通过运行程序或调用 `unittest.main()` 来触发测试例程。但在 PyTest 中，我们只需将文件传递给模块。PyTest 模块将收集所有以 `test` 为前缀定义的函数并逐一调用它们。然后，它将验证 `assert` 语句是否引发了任何异常。这样可以更方便地让测试保持与业务逻辑一起。

PyTest 还支持将函数归组到类中，但类名应该以“`Test`”作为前缀（大写的 T），例如，

```py
class TestGetAreaRectangle:
    def test_normal_case(self):
        rectangle = Rectangle(2, 3)
        assert rectangle.get_area() == 6, "incorrect area"
    def test_negative_case(self): 
        """expect -1 as output to denote error when looking at negative area"""
        rectangle = Rectangle(-1, 2)
        assert rectangle.get_area() == -1, "incorrect negative output"
```

使用 PyTest 运行此测试将生成以下输出：

```py
=================== test session starts ====================
platform darwin -- Python 3.9.9, pytest-7.0.1, pluggy-1.0.0
rootdir: /Users/MLM
plugins: anyio-3.4.0, typeguard-2.13.2
collected 2 items

test_code.py .F                                      [100%]

========================= FAILURES =========================
_________ TestGetAreaRectangle.test_negative_case __________

self = <test_code.TestGetAreaRectangle object at 0x10f5b3fd0>

    def test_negative_case(self):
        """expect -1 as output to denote error when looking at negative area"""
        rectangle = Rectangle(-1, 2)
>       assert rectangle.get_area() == -1, "incorrect negative output"
E       AssertionError: incorrect negative output
E       assert -2 == -1
E        +  where -2 = <bound method Rectangle.get_area of <test_code.Rectangle object at 0x10f5b3df0>>()
E        +    where <bound method Rectangle.get_area of <test_code.Rectangle object at 0x10f5b3df0>> = <test_code.Rectangle object at 0x10f5b3df0>.get_area

unittest5.py:24: AssertionError
================= short test summary info ==================
FAILED test_code.py::TestGetAreaRectangle::test_negative_case
=============== 1 failed, 1 passed in 0.12s ================
```

完整的代码如下：

```py
# Our code to be tested
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def get_area(self):
        return self.width * self.height

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

# The test functions to be executed by PyTest
class TestGetAreaRectangle:
    def test_normal_case(self):
        rectangle = Rectangle(2, 3)
        assert rectangle.get_area() == 6, "incorrect area"
    def test_negative_case(self):
        """expect -1 as output to denote error when looking at negative area"""
        rectangle = Rectangle(-1, 2)
        assert rectangle.get_area() == -1, "incorrect negative output"
```

为我们的测试实现设置和拆解代码，PyTest 具有极其灵活的 fixture 系统，其中 fixture 是具有返回值的函数。PyTest 的 fixture 系统允许在类、模块、包或会话之间共享 fixture，并且 fixture 可以将其他 fixture 作为参数调用。

在这里，我们简单介绍了 PyTest 的 fixture 系统：

```py
@pytest.fixture
def rectangle():
    return Rectangle(0, 0)

def test_negative_case(rectangle): 
    print (rectangle.width)
    rectangle.set_width(-1)
    rectangle.set_height(2)
    assert rectangle.get_area() == -1, "incorrect negative output"
```

上述代码将 Rectangle 引入为 fixture，PyTest 在 `test_negative_case` 的参数列表中匹配这个矩形 fixture，并为 `test_negative_case` 提供来自矩形函数的输出集合。这对每个其他测试也会如此。然而，请注意，fixtures 可以在每个测试中请求多次，每个测试中 fixture 只运行一次，并且结果会被缓存。这意味着在单个测试运行期间对该 fixture 的所有引用都引用了相同的返回值（如果返回值是引用类型，这一点很重要）。

完整代码如下：

```py
import pytest

# Our code to be tested
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def get_area(self):
        return self.width * self.height

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

@pytest.fixture
def rectangle():
    return Rectangle(0, 0)

def test_negative_case(rectangle):
    print (rectangle.width)
    rectangle.set_width(-1)
    rectangle.set_height(2)
    assert rectangle.get_area() == -1, "incorrect negative output"
```

与 PyUnit 类似，PyTest 具有许多其他功能，使你能够构建更全面和高级的单元测试。

## 单元测试的实际应用

现在，我们将探讨单元测试的实际应用。在我们的示例中，我们将测试一个从 Yahoo Finance 获取股票数据的函数，使用 `pandas_datareader`，并在 PyUnit 中进行测试：

```py
import pandas_datareader.data as web

def get_stock_data(ticker):
    """pull data from stooq"""
    df = web.DataReader(ticker, "yahoo")
    return df
```

这个函数通过从 Yahoo Finance 网站爬取数据来获取特定股票代码的股票数据，并返回 pandas DataFrame。这可能会以多种方式失败。例如，数据读取器可能无法返回任何内容（如果 Yahoo Finance 出现故障）或返回一个缺少列或列中缺少数据的 DataFrame（如果来源重构了其网站）。因此，我们应该提供多个测试函数以检查多种失败模式：

```py
import datetime
import unittest

import pandas as pd
import pandas_datareader.data as web

def get_stock_data(ticker):
    """pull data from stooq"""
    df = web.DataReader(ticker, 'yahoo')
    return df

class TestGetStockData(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """We only want to pull this data once for each TestCase since it is an expensive operation"""
        self.df = get_stock_data('^DJI')

    def test_columns_present(self):
        """ensures that the expected columns are all present"""
        self.assertIn("Open", self.df.columns)
        self.assertIn("High", self.df.columns)
        self.assertIn("Low", self.df.columns)
        self.assertIn("Close", self.df.columns)
        self.assertIn("Volume", self.df.columns)

    def test_non_empty(self):
        """ensures that there is more than one row of data"""
        self.assertNotEqual(len(self.df.index), 0)

    def test_high_low(self):
        """ensure high and low are the highest and lowest in the same row"""
        ohlc = self.df[["Open","High","Low","Close"]]
        highest = ohlc.max(axis=1)
        lowest = ohlc.min(axis=1)
        self.assertTrue(ohlc.le(highest, axis=0).all(axis=None))
        self.assertTrue(ohlc.ge(lowest, axis=0).all(axis=None))

    def test_most_recent_within_week(self):
        """most recent data was collected within the last week"""
        most_recent_date = pd.to_datetime(self.df.index[-1])
        self.assertLessEqual((datetime.datetime.today() - most_recent_date).days, 7)

unittest.main()
```

我们上面的单元测试系列检查了某些列是否存在（`test_columns_present`）、数据框是否为空（`test_non_empty`）、"high" 和 "low" 列是否确实是同一行的最高和最低值（`test_high_low`），以及数据框中最最近的数据是否在过去 7 天内（`test_most_recent_within_week`）。

想象一下，你正在进行一个消耗股市数据的机器学习项目。拥有一个单元测试框架可以帮助你识别数据预处理是否按预期工作。

使用这些单元测试，我们能够识别函数输出是否发生了重大变化，这可以成为持续集成（CI）过程的一部分。我们还可以根据需要附加其他单元测试，具体取决于我们对该函数的功能的依赖。

为了完整性，这里提供一个 PyTest 的等效版本：

```py
import pytest

# scope="class" tears down the fixture only at the end of the last test in the class, so we avoid rerunning this step.
@pytest.fixture(scope="class")
def stock_df():
  # We only want to pull this data once for each TestCase since it is an expensive operation
  df = get_stock_data('^DJI')
  return df

class TestGetStockData:

  def test_columns_present(self, stock_df):
    # ensures that the expected columns are all present
    assert "Open" in stock_df.columns
    assert "High" in stock_df.columns
    assert "Low" in stock_df.columns
    assert "Close" in stock_df.columns
    assert "Volume" in stock_df.columns

  def test_non_empty(self, stock_df):
    # ensures that there is more than one row of data
    assert len(stock_df.index) != 0

  def test_most_recent_within_week(self, stock_df):
    # most recent data was collected within the last week
    most_recent_date = pd.to_datetime(stock_df.index[0])
    assert (datetime.datetime.today() - most_recent_date).days <= 7
```

构建单元测试可能看起来费时且繁琐，但它们可能是任何 CI 流水线的关键部分，并且是捕获早期 bug 的宝贵工具，以避免它们进一步传递到流水线中并变得更难处理。

> *如果你喜欢它，那么你应该对其进行测试。*

*— 谷歌的软件工程*

## 进一步阅读

本节提供了更多资源，以便你可以深入了解这个主题。

#### 库

+   unittest 模块（以及 [assert 方法列表](https://docs.python.org/3/library/unittest.html#assert-methods)），[`docs.python.org/3/library/unittest.html`](https://docs.python.org/3/library/unittest.html)

+   PyTest，[`docs.pytest.org/en/7.0.x/`](https://docs.pytest.org/en/7.0.x/)

#### 文章

+   测试驱动开发 (TDD)，[`www.ibm.com/garage/method/practices/code/practice_test_driven_development/`](https://www.ibm.com/garage/method/practices/code/practice_test_driven_development/)

+   Python 单元测试框架，[`pyunit.sourceforge.net/pyunit.html`](http://pyunit.sourceforge.net/pyunit.html)

#### 书籍

+   *谷歌的软件工程*，作者：Titus Winters, Tom Manshreck 和 Hyrum Wright [`www.amazon.com/dp/1492082791`](https://www.amazon.com/dp/1492082791)

+   *编程实践*，作者：Brian Kernighan 和 Rob Pike（第五章和第六章），[`www.amazon.com/dp/020161586X`](https://www.amazon.com/dp/020161586X)

## 摘要

在这篇文章中，你了解了什么是单元测试，以及如何使用两个流行的 Python 库（PyUnit 和 PyTest）进行单元测试。你还学会了如何配置单元测试，并看到了数据科学流程中单元测试的一个用例示例。

具体来说，你学到了：

+   什么是单元测试，及其为何有用

+   单元测试如何融入测试驱动开发流程

+   如何使用 PyUnit 和 PyTest 在 Python 中进行单元测试
