# Python 中的网页爬取

> 原文：[`machinelearningmastery.com/web-crawling-in-python/`](https://machinelearningmastery.com/web-crawling-in-python/)

以前，收集数据是一项繁琐的工作，有时非常昂贵。机器学习项目离不开数据。幸运的是，如今我们可以利用大量的网络数据来创建数据集。我们可以从网络上复制数据来构建数据集。我们可以手动下载文件并保存到磁盘。但通过自动化数据采集，我们可以更高效地完成这项工作。Python 中有几种工具可以帮助实现自动化。

完成本教程后，您将学习到：

+   如何使用 requests 库通过 HTTP 读取在线数据

+   如何使用 pandas 读取网页上的表格

+   如何使用 Selenium 模拟浏览器操作

**用我的新书** [《Python 机器学习》](https://machinelearningmastery.com/python-for-machine-learning/) **来启动您的项目**，其中包括*逐步教程*和所有示例的*Python 源代码*文件。

让我们开始吧！！[](../Images/014f73ed74ab693ef2bb8ea377287b7c.png)

Python 中的网页爬取

图片由 [Ray Bilcliff](https://www.pexels.com/photo/black-and-red-spider-on-web-in-close-up-photography-4805619/) 提供。保留所有权利。

## 概述

本教程分为三个部分；它们是：

+   使用 requests 库

+   使用 pandas 读取网页上的表格

+   使用 Selenium 读取动态内容

## 使用 Requests 库

当我们谈到编写 Python 程序来从网络读取数据时，不可避免地，我们需要使用`requests`库。您需要安装它（以及我们稍后将介绍的 BeautifulSoup 和 lxml）：

```py
pip install requests beautifulsoup4 lxml
```

它为您提供了一个界面，使您可以轻松地与网页进行交互。

一个非常简单的用例是从 URL 读取网页：

```py
import requests

# Lat-Lon of New York
URL = "https://weather.com/weather/today/l/40.75,-73.98"
resp = requests.get(URL)
print(resp.status_code)
print(resp.text)
```

```py
200
<!doctype html><html dir="ltr" lang="en-US"><head>
      <meta data-react-helmet="true" charset="utf-8"/><meta data-react-helmet="true"
name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
...
```

如果您对 HTTP 比较熟悉，您可能会记得状态码 200 表示请求成功完成。然后我们可以读取响应。在上面的例子中，我们读取了文本响应并获取了网页的 HTML。如果是 CSV 或其他文本数据，我们可以在响应对象的`text`属性中获取它们。例如，这就是如何从联邦储备经济数据中读取 CSV 文件：

```py
import io
import pandas as pd
import requests

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10YIE&cosd=2017-04-14&coed=2022-04-14"
resp = requests.get(URL)
if resp.status_code == 200:
   csvtext = resp.text
   csvbuffer = io.StringIO(csvtext)
   df = pd.read_csv(csvbuffer)
   print(df)
```

```py
            DATE T10YIE
0     2017-04-17   1.88
1     2017-04-18   1.85
2     2017-04-19   1.85
3     2017-04-20   1.85
4     2017-04-21   1.84
...          ...    ...
1299  2022-04-08   2.87
1300  2022-04-11   2.91
1301  2022-04-12   2.86
1302  2022-04-13    2.8
1303  2022-04-14   2.89

[1304 rows x 2 columns]
```

如果数据是 JSON 格式，我们可以将其作为文本读取，或者让`requests`为您解码。例如，以下是从 GitHub 拉取 JSON 格式的数据并将其转换为 Python 字典的操作：

```py
import requests

URL = "https://api.github.com/users/jbrownlee"
resp = requests.get(URL)
if resp.status_code == 200:
    data = resp.json()
    print(data)
```

```py
{'login': 'jbrownlee', 'id': 12891, 'node_id': 'MDQ6VXNlcjEyODkx',
'avatar_url': 'https://avatars.githubusercontent.com/u/12891?v=4',
'gravatar_id': '', 'url': 'https://api.github.com/users/jbrownlee',
'html_url': 'https://github.com/jbrownlee',
...
'company': 'Machine Learning Mastery', 'blog': 'https://machinelearningmastery.com',
'location': None, 'email': None, 'hireable': None,
'bio': 'Making developers awesome at machine learning.', 'twitter_username': None,
'public_repos': 5, 'public_gists': 0, 'followers': 1752, 'following': 0,
'created_at': '2008-06-07T02:20:58Z', 'updated_at': '2022-02-22T19:56:27Z'
}
```

但如果 URL 返回的是一些二进制数据，比如 ZIP 文件或 JPEG 图像，您需要从`content`属性中获取它们，因为这是二进制数据。例如，这就是如何下载一张图片（维基百科的标志）：

```py
import requests

URL = "https://en.wikipedia.org/static/images/project-logos/enwiki.png"
wikilogo = requests.get(URL)
if wikilogo.status_code == 200:
    with open("enwiki.png", "wb") as fp:
        fp.write(wikilogo.content)
```

既然我们已经获得了网页，应该如何提取数据？这超出了`requests`库的功能，但我们可以使用其他库来帮助完成。根据我们想要指定数据的方式，有两种方法可以实现。

第一种方法是将 HTML 视为一种 XML 文档，并使用 XPath 语言提取元素。在这种情况下，我们可以利用 `lxml` 库首先创建文档对象模型（DOM），然后通过 XPath 进行搜索：

```py
...
from lxml import etree

# Create DOM from HTML text
dom = etree.HTML(resp.text)
# Search for the temperature element and get the content
elements = dom.xpath("//span[@data-testid='TemperatureValue' and contains(@class,'CurrentConditions')]")
print(elements[0].text)
```

```py
61°
```

XPath 是一个字符串，用于指定如何查找一个元素。lxml 对象提供了一个 `xpath()` 函数，用于搜索匹配 XPath 字符串的 DOM 元素，这可能会有多个匹配项。上述 XPath 意味着查找任何具有 `<span>` 标签且属性 `data-testid` 匹配 “`TemperatureValue`” 和 `class` 以 “`CurrentConditions`” 开头的 HTML 元素。我们可以通过检查 HTML 源代码在浏览器的开发者工具中（例如，下面的 Chrome 截图）了解到这一点。

![](img/d516cc11a5a97395c96129cf51860901.png)

本示例旨在找到纽约市的温度，由我们从该网页获取的特定元素提供。我们知道 XPath 匹配的第一个元素就是我们需要的，我们可以读取 `<span>` 标签中的文本。

另一种方法是使用 HTML 文档上的 CSS 选择器，我们可以利用 BeautifulSoup 库：

```py
...
from bs4 import BeautifulSoup

soup = BeautifulSoup(resp.text, "lxml")
elements = soup.select('span[data-testid="TemperatureValue"][class^="CurrentConditions"]')
print(elements[0].text)
```

```py
61°
```

上述过程中，我们首先将 HTML 文本传递给 BeautifulSoup。BeautifulSoup 支持各种 HTML 解析器，每种解析器都有不同的能力。在上述过程中，我们使用 `lxml` 库作为解析器，正如 BeautifulSoup 推荐的（它也通常是最快的）。CSS 选择器是一种不同的迷你语言，与 XPath 相比有其优缺点。上面的选择器与我们在之前示例中使用的 XPath 是相同的。因此，我们可以从第一个匹配的元素中获取相同的温度。

下面是一个完整的代码示例，根据网页上的实时信息打印纽约市当前温度：

```py
import requests
from lxml import etree

# Reading temperature of New York
URL = "https://weather.com/weather/today/l/40.75,-73.98"
resp = requests.get(URL)

if resp.status_code == 200:
    # Using lxml
    dom = etree.HTML(resp.text)
    elements = dom.xpath("//span[@data-testid='TemperatureValue' and contains(@class,'CurrentConditions')]")
    print(elements[0].text)

    # Using BeautifulSoup
    soup = BeautifulSoup(resp.text, "lxml")
    elements = soup.select('span[data-testid="TemperatureValue"][class^="CurrentConditions"]')
    print(elements[0].text)
```

正如你可以想象的那样，你可以通过定期运行这个脚本来收集温度的时间序列。同样，我们可以自动从各种网站收集数据。这就是我们如何为机器学习项目获取数据的方法。

## 使用 Pandas 读取网页上的表格

很多时候，网页会使用表格来承载数据。如果页面足够简单，我们甚至可以跳过检查它以找到 XPath 或 CSS 选择器，直接使用 pandas 一次性获取页面上的所有表格。这可以用一行代码简单实现：

```py
import pandas as pd

tables = pd.read_html("https://www.federalreserve.gov/releases/h15/")
print(tables)
```

```py
[                               Instruments 2022Apr7 2022Apr8 2022Apr11 2022Apr12 2022Apr13
0          Federal funds (effective) 1 2 3     0.33     0.33      0.33      0.33      0.33
1                 Commercial Paper 3 4 5 6      NaN      NaN       NaN       NaN       NaN
2                             Nonfinancial      NaN      NaN       NaN       NaN       NaN
3                                  1-month     0.30     0.34      0.36      0.39      0.39
4                                  2-month     n.a.     0.48      n.a.      n.a.      n.a.
5                                  3-month     n.a.     n.a.      n.a.      0.78      0.78
6                                Financial      NaN      NaN       NaN       NaN       NaN
7                                  1-month     0.49     0.45      0.46      0.39      0.46
8                                  2-month     n.a.     n.a.      0.60      0.71      n.a.
9                                  3-month     0.85     0.81      0.75      n.a.      0.86
10                   Bank prime loan 2 3 7     3.50     3.50      3.50      3.50      3.50
11      Discount window primary credit 2 8     0.50     0.50      0.50      0.50      0.50
12              U.S. government securities      NaN      NaN       NaN       NaN       NaN
13   Treasury bills (secondary market) 3 4      NaN      NaN       NaN       NaN       NaN
14                                  4-week     0.21     0.20      0.21      0.19      0.23
15                                 3-month     0.68     0.69      0.78      0.74      0.75
16                                 6-month     1.12     1.16      1.22      1.18      1.17
17                                  1-year     1.69     1.72      1.75      1.67      1.67
18            Treasury constant maturities      NaN      NaN       NaN       NaN       NaN
19                               Nominal 9      NaN      NaN       NaN       NaN       NaN
20                                 1-month     0.21     0.20      0.22      0.21      0.26
21                                 3-month     0.68     0.70      0.77      0.74      0.75
22                                 6-month     1.15     1.19      1.23      1.20      1.20
23                                  1-year     1.78     1.81      1.85      1.77      1.78
24                                  2-year     2.47     2.53      2.50      2.39      2.37
25                                  3-year     2.66     2.73      2.73      2.58      2.57
26                                  5-year     2.70     2.76      2.79      2.66      2.66
27                                  7-year     2.73     2.79      2.84      2.73      2.71
28                                 10-year     2.66     2.72      2.79      2.72      2.70
29                                 20-year     2.87     2.94      3.02      2.99      2.97
30                                 30-year     2.69     2.76      2.84      2.82      2.81
31                    Inflation indexed 10      NaN      NaN       NaN       NaN       NaN
32                                  5-year    -0.56    -0.57     -0.58     -0.65     -0.59
33                                  7-year    -0.34    -0.33     -0.32     -0.36     -0.31
34                                 10-year    -0.16    -0.15     -0.12     -0.14     -0.10
35                                 20-year     0.09     0.11      0.15      0.15      0.18
36                                 30-year     0.21     0.23      0.27      0.28      0.30
37  Inflation-indexed long-term average 11     0.23     0.26      0.30      0.30      0.33,       0               1
0  n.a.  Not available.]
```

pandas 中的 `read_html()` 函数读取一个 URL，并查找页面上的所有表格。每个表格都被转换为 pandas DataFrame，然后将所有表格以列表形式返回。在此示例中，我们正在读取来自联邦储备系统的各种利率，该页面上只有一个表格。表格列由 pandas 自动识别。

可能并非所有表格都是我们感兴趣的。有时，网页仅仅使用表格作为格式化页面的一种方式，但 pandas 可能无法聪明地识别这一点。因此，我们需要测试并挑选 `read_html()` 函数返回的结果。

### 想开始使用 Python 进行机器学习吗？

现在就参加我的免费 7 天邮件速成课程（附样例代码）。

点击注册，同时获取课程的免费 PDF 电子书版本。

## 使用 Selenium 读取动态内容

现代网页中有很大一部分充满了 JavaScript。这虽然提供了更炫的体验，但却成为了提取数据时的障碍。一个例子是 Yahoo 的主页，如果我们只是加载页面并查找所有新闻标题，那么看到的新闻数量远远少于在浏览器中看到的：

```py
import requests

# Read Yahoo home page
URL = "https://www.yahoo.com/"
resp = requests.get(URL)
dom = etree.HTML(resp.text)

# Print news headlines
elements = dom.xpath("//h3/a[u[@class='StretchedBox']]")
for elem in elements:
    print(etree.tostring(elem, method="text", encoding="unicode"))
```

这是因为像这样的网站依赖 JavaScript 来填充内容。像 AngularJS 或 React 这样的著名 web 框架驱动了这一类别。Python 库，如 `requests`，无法理解 JavaScript。因此，你会看到不同的结果。如果你想从网页中获取的数据是其中之一，你可以研究 JavaScript 如何被调用，并在你的程序中模拟浏览器的行为。但这可能过于繁琐，难以实现。

另一种方法是让真实浏览器读取网页，而不是使用 `requests`。这正是 Selenium 可以做到的。在我们可以使用它之前，我们需要安装这个库：

```py
pip install selenium
```

但 Selenium 只是一个控制浏览器的框架。你需要在你的计算机上安装浏览器以及将 Selenium 连接到浏览器的驱动程序。如果你打算使用 Chrome，你还需要下载并安装[ChromeDriver](https://chromedriver.chromium.org/downloads)。你需要将驱动程序放在可执行路径中，以便 Selenium 可以像正常命令一样调用它。例如，在 Linux 中，你只需从下载的 ZIP 文件中获取 `chromedriver` 可执行文件，并将其放在 `/usr/local/bin` 中。

类似地，如果你使用的是 Firefox，你需要[GeckoDriver](https://github.com/mozilla/geckodriver/releases/)。有关设置 Selenium 的更多细节，你应该参考[其文档](https://www.selenium.dev/downloads/)。

之后，你可以使用 Python 脚本来控制浏览器行为。例如：

```py
import time
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By

# Launch Chrome browser in headless mode
options = webdriver.ChromeOptions()
options.add_argument("headless")
browser = webdriver.Chrome(options=options)

# Load web page
browser.get("https://www.yahoo.com")
# Network transport takes time. Wait until the page is fully loaded
def is_ready(browser):
    return browser.execute_script(r"""
        return document.readyState === 'complete'
    """)
WebDriverWait(browser, 30).until(is_ready)

# Scroll to bottom of the page to trigger JavaScript action
browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(1)
WebDriverWait(browser, 30).until(is_ready)

# Search for news headlines and print
elements = browser.find_elements(By.XPATH, "//h3/a[u[@class='StretchedBox']]")
for elem in elements:
    print(elem.text)

# Close the browser once finish
browser.close()
```

上述代码的工作方式如下。我们首先以无头模式启动浏览器，这意味着我们要求 Chrome 启动但不在屏幕上显示。如果我们想远程运行脚本，这一点很重要，因为可能没有图形用户界面支持。请注意，每个浏览器的开发方式不同，因此我们使用的选项语法特定于 Chrome。如果我们使用 Firefox，代码将会是这样的：

```py
options = webdriver.FirefoxOptions()
options.set_headless()
browser = webdriver.Firefox(firefox_options=options)
```

在我们启动浏览器后，我们给它一个 URL 进行加载。但由于网络传输页面需要时间，浏览器也需要时间来渲染，因此我们应该等到浏览器准备好后再进行下一步操作。我们通过使用 JavaScript 来检测浏览器是否完成渲染。我们让 Selenium 执行 JavaScript 代码，并使用`execute_script()`函数告诉我们结果。我们利用 Selenium 的`WebDriverWait`工具运行代码直到成功或直到 30 秒超时。随着页面加载，我们滚动到页面底部，以便触发 JavaScript 加载更多内容。然后我们无条件等待一秒钟，以确保浏览器触发了 JavaScript，再等到页面再次准备好。之后，我们可以使用 XPath（或使用 CSS 选择器）提取新闻标题元素。由于浏览器是一个外部程序，我们需要在脚本中负责关闭它。

使用 Selenium 与使用`requests`库在几个方面有所不同。首先，你的 Python 代码中不会直接包含网页内容。相反，每当你需要时，你都要引用浏览器的内容。因此，`find_elements()` 函数返回的网页元素指的是外部浏览器中的对象，因此我们在完成使用之前不能关闭浏览器。其次，所有操作都应基于浏览器交互，而不是网络请求。因此，你需要通过模拟键盘和鼠标操作来控制浏览器。但作为回报，你可以使用完整功能的浏览器并支持 JavaScript。例如，你可以使用 JavaScript 检查页面上元素的大小和位置，这只有在 HTML 元素渲染后才能知道。

Selenium 框架提供了很多功能，但我们在这里无法一一覆盖。它功能强大，但由于它与浏览器连接，使用起来比`requests`库更为复杂且速度较慢。通常，这是一种从网络获取信息的最后手段。

## 进一步阅读

另一个在 Python 中非常有名的网页爬取库是 Scrapy，它类似于将`requests`库和 BeautifulSoup 合并成一个库。网页协议复杂。有时我们需要管理网页 cookies 或使用 POST 方法提供额外数据。这些都可以通过 requests 库的不同函数或附加参数完成。以下是一些资源供你深入了解：

#### 文章

+   [MDN 上的 HTTP 概述](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview)

+   [MDN 上的 XPath](https://developer.mozilla.org/en-US/docs/Web/XPath)

+   [W3Schools 上的 XPath 教程](https://www.w3schools.com/xml/xpath_intro.asp)

+   [W3Schools 上的 CSS 选择器参考](https://www.w3schools.com/cssref/css_selectors.asp)

+   [Selenium Python 绑定](https://www.selenium.dev/selenium/docs/api/py/index.html)

#### API 文档

+   [Requests 库](https://docs.python-requests.org/en/latest/)

+   [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

+   [lxml.etree](https://lxml.de/api.html)

+   [Selenium Python API](https://selenium-python.readthedocs.io/)

+   [Scrapy](https://docs.scrapy.org/en/latest/)

#### 书籍

+   [Python 网页抓取](https://www.amazon.com/dp/1786462583)，第二版，作者 Katharine Jarmul 和 Richard Lawson

+   [用 Python 进行网页抓取](https://www.amazon.com/dp/1491985577/)，第二版，作者 Ryan Mitchell

+   [学习 Scrapy](https://www.amazon.com/dp/1784399787/)，作者 Dimitrios Kouzis-Loukas

+   [Python Selenium 测试](https://www.amazon.com/dp/1484262484/)，作者 Sujay Raghavendra

+   [动手实践 Python 网页抓取](https://www.amazon.com/dp/1789533392)，作者 Anish Chapagain

## **总结**

在这个教程中，你了解了我们可以用来从网络获取内容的工具。

具体来说，你学到了：

+   如何使用 requests 库发送 HTTP 请求并从响应中提取数据

+   如何从 HTML 构建文档对象模型，以便在网页上找到一些特定的信息

+   如何使用 pandas 快速轻松地读取网页上的表格

+   如何使用 Selenium 控制浏览器处理网页上的动态内容
