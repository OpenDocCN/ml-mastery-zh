# 获取时间序列数据集的指南

> 原文：[`machinelearningmastery.com/a-guide-to-obtaining-time-series-datasets-in-python/`](https://machinelearningmastery.com/a-guide-to-obtaining-time-series-datasets-in-python/)

来自真实世界场景的数据集对构建和测试机器学习模型至关重要。你可能只是想有一些数据来实验算法。你也可能想通过设置基准或使用不同的数据集来评估你的模型的弱点。有时，你可能还想创建合成数据集，通过向数据中添加噪声、相关性或冗余信息，在受控条件下测试你的算法。

在这篇文章中，我们将演示如何使用 Python 从不同来源获取一些真实的时间序列数据。我们还将使用 Python 的库创建合成时间序列数据。

完成本教程后，你将了解：

+   如何使用 `pandas_datareader`

+   如何使用 `requests` 库调用网络数据服务器的 API

+   如何生成合成时间序列数据

**开启你的项目**，请参考我新书 [《Python 机器学习》](https://machinelearningmastery.com/python-for-machine-learning/)，包括 *一步一步的教程* 和所有示例的 *Python 源代码* 文件。

让我们开始吧。![海浪和鸟的图片](https://machinelearningmastery.com/wp-content/uploads/2022/03/IMG_0628-scaled.jpg)

在 Python 中处理数据集的指南

图片来源于 Mehreen Saeed，部分权利保留

## 教程概述

本教程分为三个部分，分别是：

1.  使用 `pandas_datareader`

1.  使用 `requests` 库通过远程服务器的 API 获取数据

1.  生成合成时间序列数据

## 使用 pandas-datareader 加载数据

本文将依赖于一些库。如果你还未安装它们，可以使用 `pip` 安装：

```py
pip install pandas_datareader requests
```

`pandas_datareader` 库允许你 [从不同的数据源获取数据](https://pandas-datareader.readthedocs.io/en/latest/readers/index.html)，包括 Yahoo Finance（获取金融市场数据）、世界银行（获取全球发展数据）以及圣路易斯联邦储备银行（获取经济数据）。在本节中，我们将展示如何从不同的数据源加载数据。

在幕后，`pandas_datareader` 实时从网络中提取你所需的数据，并将其组装成 pandas DataFrame。由于网页结构差异巨大，每个数据源需要不同的读取器。因此，pandas_datareader 仅支持从有限数量的数据源读取，主要与金融和经济时间序列相关。

获取数据很简单。例如，我们知道苹果公司的股票代码是 AAPL，因此我们可以通过 Yahoo Finance 获取苹果公司股票的每日历史价格，如下所示：

```py
import pandas_datareader as pdr

# Reading Apple shares from yahoo finance server    
shares_df = pdr.DataReader('AAPL', 'yahoo', start='2021-01-01', end='2021-12-31')
# Look at the data read
print(shares_df)
```

调用`DataReader()`时，第一个参数需要指定股票代码，第二个参数指定数据来源。上述代码打印出 DataFrame：

```py
                  High         Low        Open       Close       Volume   Adj Close
Date                                                                               
2021-01-04  133.610001  126.760002  133.520004  129.410004  143301900.0  128.453461
2021-01-05  131.740005  128.429993  128.889999  131.009995   97664900.0  130.041611
2021-01-06  131.050003  126.379997  127.720001  126.599998  155088000.0  125.664215
2021-01-07  131.630005  127.860001  128.360001  130.919998  109578200.0  129.952271
2021-01-08  132.630005  130.229996  132.429993  132.050003  105158200.0  131.073914
...                ...         ...         ...         ...          ...         ...
2021-12-27  180.419998  177.070007  177.089996  180.330002   74919600.0  180.100540
2021-12-28  181.330002  178.529999  180.160004  179.289993   79144300.0  179.061859
2021-12-29  180.630005  178.139999  179.330002  179.380005   62348900.0  179.151749
2021-12-30  180.570007  178.089996  179.470001  178.199997   59773000.0  177.973251
2021-12-31  179.229996  177.259995  178.089996  177.570007   64062300.0  177.344055

[252 rows x 6 columns]
```

我们还可以从多个公司获取股票价格历史数据，方法是使用一个包含股票代码的列表：

```py
companies = ['AAPL', 'MSFT', 'GE']
shares_multiple_df = pdr.DataReader(companies, 'yahoo', start='2021-01-01', end='2021-12-31')
print(shares_multiple_df.head())
```

结果将是一个具有多层列的 DataFrame：

```py
Attributes   Adj Close                              Close              \
Symbols           AAPL        MSFT         GE        AAPL        MSFT   
Date                                                                    
2021-01-04  128.453461  215.434982  83.421600  129.410004  217.690002   
2021-01-05  130.041611  215.642776  85.811905  131.009995  217.899994   
2021-01-06  125.664223  210.051315  90.512833  126.599998  212.250000   
2021-01-07  129.952286  216.028732  89.795753  130.919998  218.289993   
2021-01-08  131.073944  217.344986  90.353485  132.050003  219.619995   

...

Attributes       Volume                          
Symbols            AAPL        MSFT          GE  
Date                                             
2021-01-04  143301900.0  37130100.0   9993688.0  
2021-01-05   97664900.0  23823000.0  10462538.0  
2021-01-06  155088000.0  35930700.0  16448075.0  
2021-01-07  109578200.0  27694500.0   9411225.0  
2021-01-08  105158200.0  22956200.0   9089963.0
```

由于 DataFrames 的结构，提取部分数据非常方便。例如，我们可以使用以下方法仅绘制某些日期的每日收盘价：

```py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# General routine for plotting time series data
def plot_timeseries_df(df, attrib, ticker_loc=1, title='Timeseries', 
                       legend=''):
    fig = plt.figure(figsize=(15,7))
    plt.plot(df[attrib], 'o-')
    _ = plt.xticks(rotation=90)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(ticker_loc))
    plt.title(title)
    plt.gca().legend(legend)
    plt.show()

plot_timeseries_df(shares_multiple_df.loc["2021-04-01":"2021-06-30"], "Close",
                   ticker_loc=3, title="Close price", legend=companies)
```

![](img/3a8ff2424899b5e433bb929c61f62270.png)

从 Yahoo Finance 获取的多个股票

完整代码如下：

```py
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

companies = ['AAPL', 'MSFT', 'GE']
shares_multiple_df = pdr.DataReader(companies, 'yahoo', start='2021-01-01', end='2021-12-31')
print(shares_multiple_df)

def plot_timeseries_df(df, attrib, ticker_loc=1, title='Timeseries', legend=''):
    "General routine for plotting time series data"
    fig = plt.figure(figsize=(15,7))
    plt.plot(df[attrib], 'o-')
    _ = plt.xticks(rotation=90)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(ticker_loc))
    plt.title(title)
    plt.gca().legend(legend)
    plt.show()

plot_timeseries_df(shares_multiple_df.loc["2021-04-01":"2021-06-30"], "Close",
                   ticker_loc=3, title="Close price", legend=companies)
```

使用 pandas-datareader 从另一个数据源读取数据的语法类似。例如，我们可以从[联邦储备经济数据（FRED）](https://fred.stlouisfed.org/)读取经济时间序列。FRED 中的每个时间序列都有一个符号。例如，所有城市消费者的消费者价格指数是[CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL)，不包括食品和能源的所有项目的消费者价格指数是[CPILFESL](https://fred.stlouisfed.org/series/CPILFESL)，个人消费支出是[PCE](https://fred.stlouisfed.org/series/PCE)。你可以在 FRED 的网页上搜索和查找这些符号。

以下是如何获取两个消费者价格指数，CPIAUCSL 和 CPILFESL，并在图中显示它们：

```py
import pandas_datareader as pdr
import matplotlib.pyplot as plt

# Read data from FRED and print
fred_df = pdr.DataReader(['CPIAUCSL','CPILFESL'], 'fred', "2010-01-01", "2021-12-31")
print(fred_df)

# Show in plot the data of 2019-2021
fig = plt.figure(figsize=(15,7))
plt.plot(fred_df.loc["2019":], 'o-')
plt.xticks(rotation=90)
plt.legend(fred_df.columns)
plt.title("Consumer Price Index")
plt.show()
```

![](img/a24165e267eec61782aae6a54ca27092.png)

消费者物价指数图

从世界银行获取数据也类似，但我们需要理解世界银行的数据更为复杂。通常，数据系列，如人口，以时间序列的形式呈现，并且还具有国家维度。因此，我们需要指定更多参数来获取数据。

使用`pandas_datareader`，我们可以使用一组特定的世界银行 API。可以从[世界银行开放数据](https://data.worldbank.org/)中查找指标的符号，或者使用以下方法进行搜索：

```py
from pandas_datareader import wb

matches = wb.search('total.*population')
print(matches[["id","name"]])
```

`search()`函数接受一个正则表达式字符串（例如，上述`.*`表示任何长度的字符串）。这将打印出：

```py
                               id                                               name
24     1.1_ACCESS.ELECTRICITY.TOT      Access to electricity (% of total population)
164            2.1_ACCESS.CFT.TOT  Access to Clean Fuels and Technologies for coo...
1999              CC.AVPB.PTPI.AI  Additional people below $1.90 as % of total po...
2000              CC.AVPB.PTPI.AR  Additional people below $1.90 as % of total po...
2001              CC.AVPB.PTPI.DI  Additional people below $1.90 as % of total po...
...                           ...                                                ...
13908           SP.POP.TOTL.FE.ZS         Population, female (% of total population)
13912           SP.POP.TOTL.MA.ZS           Population, male (% of total population)
13938              SP.RUR.TOTL.ZS           Rural population (% of total population)
13958           SP.URB.TOTL.IN.ZS           Urban population (% of total population)
13960              SP.URB.TOTL.ZS  Percentage of Population in Urban Areas (in % ...

[137 rows x 2 columns]
```

其中`id`列是时间序列的符号。

我们可以通过指定 ISO-3166-1 国家代码来读取特定国家的数据。但世界银行也包含非国家的汇总数据（例如，南亚），因此虽然`pandas_datareader`允许我们使用“`all`”字符串表示所有国家，但通常我们不希望使用它。以下是如何从世界银行获取所有国家和汇总数据列表：

```py
import pandas_datareader.wb as wb

countries = wb.get_countries()
print(countries)
```

```py
    iso3c iso2c                 name               region          adminregion          incomeLevel     lendingType capitalCity  longitude  latitude
0     ABW    AW                Aruba  Latin America & ...                               High income  Not classified  Oranjestad   -70.0167   12.5167
1     AFE    ZH  Africa Eastern a...           Aggregates                                Aggregates      Aggregates                    NaN       NaN
2     AFG    AF          Afghanistan           South Asia           South Asia           Low income             IDA       Kabul    69.1761   34.5228
3     AFR    A9               Africa           Aggregates                                Aggregates      Aggregates                    NaN       NaN
4     AFW    ZI  Africa Western a...           Aggregates                                Aggregates      Aggregates                    NaN       NaN
..    ...   ...                  ...                  ...                  ...                  ...             ...         ...        ...       ...
294   XZN    A5  Sub-Saharan Afri...           Aggregates                                Aggregates      Aggregates                    NaN       NaN
295   YEM    YE          Yemen, Rep.  Middle East & No...  Middle East & No...           Low income             IDA      Sana'a    44.2075   15.3520
296   ZAF    ZA         South Africa  Sub-Saharan Africa   Sub-Saharan Afri...  Upper middle income            IBRD    Pretoria    28.1871  -25.7460
297   ZMB    ZM               Zambia  Sub-Saharan Africa   Sub-Saharan Afri...  Lower middle income             IDA      Lusaka    28.2937  -15.3982
298   ZWE    ZW             Zimbabwe  Sub-Saharan Africa   Sub-Saharan Afri...  Lower middle income           Blend      Harare    31.0672  -17.8312
```

以下是如何获取 2020 年所有国家的人口数据，并在条形图中展示前 25 个国家的情况。当然，我们也可以通过指定不同的`start`和`end`年份来获取跨年度的人口数据：

```py
import pandas_datareader.wb as wb
import pandas as pd
import matplotlib.pyplot as plt

# Get a list of 2-letter country code excluding aggregates
countries = wb.get_countries()
countries = list(countries[countries.region != "Aggregates"]["iso2c"])

# Read countries' total population data (SP.POP.TOTL) in year 2020
population_df = wb.download(indicator="SP.POP.TOTL", country=countries, start=2020, end=2020)

# Sort by population, then take top 25 countries, and make the index (i.e., countries) as a column
population_df = (population_df.dropna()
                              .sort_values("SP.POP.TOTL")
                              .iloc[-25:]
                              .reset_index())

# Plot the population, in millions
fig = plt.figure(figsize=(15,7))
plt.bar(population_df["country"], population_df["SP.POP.TOTL"]/1e6)
plt.xticks(rotation=90)
plt.ylabel("Million Population")
plt.title("Population")
plt.show()
```

![](img/65dc3d1bc4e7adfcb14cacb657c62458.png)

不同国家总人口的条形图

### 想要开始学习用于机器学习的 Python？

现在就参加我的免费 7 天邮件速成课程（附示例代码）。

点击注册，并获取课程的免费 PDF 电子书版本。

## 使用 Web APIs 获取数据

有时，你可以选择直接从 Web 数据服务器获取数据，而无需进行任何身份验证。这可以通过使用标准库 `urllib.requests` 在 Python 中完成，或者你也可以使用 `requests` 库以获得更简单的接口。

世界银行是一个示例，其中 Web APIs 自由提供，因此我们可以轻松读取不同格式的数据，如 JSON、XML 或纯文本。页面上的 [世界银行数据存储库 API](https://datahelpdesk.worldbank.org/knowledgebase/topics/125589-developer-information) 描述了各种 API 及其相应参数。为了重复我们在之前示例中所做的，而不使用 `pandas_datareader`，我们首先构造一个 URL 以读取所有国家的列表，以便找到不是汇总的国家代码。然后，我们可以构造一个查询 URL，包含以下参数：

1.  `country` 参数值 = `all`

1.  `indicator` 参数值 = `SP.POP.TOTL`

1.  `date` 参数值 = `2020`

1.  `format` 参数值 = `json`

当然，你可以尝试不同的 [指标](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation)。默认情况下，世界银行在每页上返回 50 项，我们需要逐页查询以获取所有数据。我们可以扩大页面大小，以便一次性获取所有数据。下面是如何以 JSON 格式获取国家列表并收集国家代码：

```py
import requests

# Create query URL for list of countries, by default only 50 entries returned per page
url = "http://api.worldbank.org/v2/country/all?format=json&per_page=500"
response = requests.get(url)
# Expects HTTP status code 200 for correct query
print(response.status_code)
# Get the response in JSON
header, data = response.json()
print(header)
# Collect a list of 3-letter country code excluding aggregates
countries = [item["id"]
             for item in data
             if item["region"]["value"] != "Aggregates"]
print(countries)
```

它将打印 HTTP 状态码、页眉以及国家代码列表，如下所示：

```py
200
{'page': 1, 'pages': 1, 'per_page': '500', 'total': 299}
['ABW', 'AFG', 'AGO', 'ALB', ..., 'YEM', 'ZAF', 'ZMB', 'ZWE']
```

从页眉中，我们可以验证数据已被耗尽（第 1 页，共 1 页）。然后我们可以获取所有的人口数据，如下所示：

```py
...

# Create query URL for total population from all countries in 2020
arguments = {
    "country": "all",
    "indicator": "SP.POP.TOTL",
    "date": "2020:2020",
    "format": "json"
}
url = "http://api.worldbank.org/v2/country/{country}/" \
      "indicator/{indicator}?date={date}&format={format}&per_page=500"
query_population = url.format(**arguments)
response = requests.get(query_population)
# Get the response in JSON
header, population_data = response.json()
```

你应查看世界银行 API 文档，了解如何构造 URL。例如，`2020:2021` 的日期语法表示开始和结束年份，额外参数 `page=3` 将为你提供多页结果中的第三页。获取数据后，我们可以筛选出非汇总国家，将其转换为 pandas DataFrame 以进行排序，然后绘制条形图：

```py
...

# Filter for countries, not aggregates
population = []
for item in population_data:
    if item["countryiso3code"] in countries:
        name = item["country"]["value"]
        population.append({"country":name, "population": item["value"]})
# Create DataFrame for sorting and filtering
population = pd.DataFrame.from_dict(population)
population = population.dropna().sort_values("population").iloc[-25:]
# Plot bar chart
fig = plt.figure(figsize=(15,7))
plt.bar(population["country"], population["population"]/1e6)
plt.xticks(rotation=90)
plt.ylabel("Million Population")
plt.title("Population")
plt.show()
```

图形应与之前完全相同。但正如你所见，使用 `pandas_datareader` 有助于通过隐藏低级操作使代码更加简洁。

将所有内容整合在一起，以下是完整的代码：

```py
import pandas as pd
import matplotlib.pyplot as plt
import requests

# Create query URL for list of countries, by default only 50 entries returned per page
url = "http://api.worldbank.org/v2/country/all?format=json&per_page=500"
response = requests.get(url)
# Expects HTTP status code 200 for correct query
print(response.status_code)
# Get the response in JSON
header, data = response.json()
print(header)
# Collect a list of 3-letter country code excluding aggregates
countries = [item["id"]
             for item in data
             if item["region"]["value"] != "Aggregates"]
print(countries)

# Create query URL for total population from all countries in 2020
arguments = {
    "country": "all",
    "indicator": "SP.POP.TOTL",
    "date": 2020,
    "format": "json"
}
url = "http://api.worldbank.org/v2/country/{country}/" \
      "indicator/{indicator}?date={date}&format={format}&per_page=500"
query_population = url.format(**arguments)
response = requests.get(query_population)
print(response.status_code)
# Get the response in JSON
header, population_data = response.json()
print(header)

# Filter for countries, not aggregates
population = []
for item in population_data:
    if item["countryiso3code"] in countries:
        name = item["country"]["value"]
        population.append({"country":name, "population": item["value"]})
# Create DataFrame for sorting and filtering
population = pd.DataFrame.from_dict(population)
population = population.dropna().sort_values("population").iloc[-25:]
# Plot bar chart
fig = plt.figure(figsize=(15,7))
plt.bar(population["country"], population["population"]/1e6)
plt.xticks(rotation=90)
plt.ylabel("Million Population")
plt.title("Population")
plt.show()
```

## 使用 NumPy 创建合成数据

有时，我们可能不想使用现实世界的数据，因为我们需要特定的内容，这些内容在现实中可能不会发生。一个具体的例子是使用理想的时间序列数据测试模型。在这一部分，我们将探讨如何创建合成的自回归（AR）时间序列数据。

[numpy.random](https://numpy.org/doc/1.16/reference/routines.random.html) 库可用于从不同分布中创建随机样本。`randn()` 方法生成来自标准正态分布的数据，均值为零，方差为一。

在 AR($n$) 模型中，时间步 $t$ 的值 $x_t$ 取决于前 $n$ 个时间步的值。即，

$$

x_t = b_1 x_{t-1} + b_2 x_{t-2} + … + b_n x_{t-n} + e_t

$$

使用模型参数 $b_i$ 作为不同**滞后**的 $x_t$ 的系数，误差项 $e_t$ 预计遵循正态分布。

理解公式后，我们可以在下面的示例中生成一个 AR(3) 时间序列。我们首先使用 `randn()` 生成序列的前 3 个值，然后迭代应用上述公式生成下一个数据点。然后，再次使用 `randn()` 函数添加一个误差项，受预定义的 `noise_level` 影响：

```py
import numpy as np

# Predefined paramters
ar_n = 3                     # Order of the AR(n) data
ar_coeff = [0.7, -0.3, -0.1] # Coefficients b_3, b_2, b_1
noise_level = 0.1            # Noise added to the AR(n) data
length = 200                 # Number of data points to generate

# Random initial values
ar_data = list(np.random.randn(ar_n))

# Generate the rest of the values
for i in range(length - ar_n):
    next_val = (np.array(ar_coeff) @ np.array(ar_data[-3:])) + np.random.randn() * noise_level
    ar_data.append(next_val)

# Plot the time series
fig = plt.figure(figsize=(12,5))
plt.plot(ar_data)
plt.show()
```

上面的代码将创建以下图表：

![](img/395a5291a0d23c7080eec1631d5a21c1.png)

但我们可以通过首先将数据转换为 pandas DataFrame，然后将时间作为索引来进一步添加时间轴：

```py
...

# Convert the data into a pandas DataFrame
synthetic = pd.DataFrame({"AR(3)": ar_data})
synthetic.index = pd.date_range(start="2021-07-01", periods=len(ar_data), freq="D")

# Plot the time series
fig = plt.figure(figsize=(12,5))
plt.plot(synthetic.index, synthetic)
plt.xticks(rotation=90)
plt.title("AR(3) time series")
plt.show()
```

此后我们将得到以下图表：

![](img/ef298ff88cb9d03a3b2ce3f86c75651a.png)

合成时间序列的图表

使用类似技术，我们也可以生成纯随机噪声（即 AR(0) 系列）、ARIMA 时间序列（即带有误差项的系数）或布朗运动时间序列（即随机噪声的累计和）。

## 进一步阅读

本节提供了更多资源，如果你希望深入了解这个主题。

### 库

+   [pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest/index.html)

+   [Python requests](https://docs.python-requests.org/en/latest/)

### 数据来源

+   [Yahoo! Finance](https://finance.yahoo.com/)

+   [圣路易斯联邦储备经济数据](https://fred.stlouisfed.org/)

+   [世界银行开放数据](https://data.worldbank.org/)

+   [世界银行数据 API 文档](https://datahelpdesk.worldbank.org/knowledgebase/topics/125589-developer-information)

### 书籍

+   [Think Python: How to Think Like a Computer Scientist](https://greenteapress.com/thinkpython/html/index.html) 由 Allen B. Downey 编著

+   [Python 3 编程：Python 语言完全介绍](https://www.amazon.com/dp/B001OFK2DK/) 由 Mark Summerfield 编著

+   [Python 数据分析](https://www.amazon.com/dp/1491957662/)，由 Wes McKinney 编著，第二版

## 总结

在本教程中，你发现了在 Python 中获取数据或生成合成时间序列数据的各种选项。

具体来说，你学到了：

+   如何使用 `pandas_datareader` 从不同的数据源中获取金融数据

+   如何调用 API 从不同的 Web 服务器获取数据，使用 `requests` 库

+   如何使用 NumPy 的随机数生成器生成合成时间序列数据

对于本帖讨论的主题，你有任何问题吗？请在下面的评论中提问，我会尽力回答。
