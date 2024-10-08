# 为你的 Python 脚本添加命令行参数

> 原文：[`machinelearningmastery.com/command-line-arguments-for-your-python-script/`](https://machinelearningmastery.com/command-line-arguments-for-your-python-script/)

在机器学习项目中工作意味着我们需要进行实验。有一种简单配置脚本的方法将帮助你更快地前进。在 Python 中，我们有一种方法可以从命令行适应代码。在本教程中，我们将看到如何利用 Python 脚本的命令行参数，帮助你在机器学习项目中更有效地工作。

完成本教程后，你将学会

+   为什么我们想要在命令行中控制 Python 脚本

+   我们如何能在命令行上高效工作

**用我的新书[Python 机器学习](https://machinelearningmastery.com/python-for-machine-learning/)**快速启动你的项目，包括*逐步教程*和所有示例的*Python 源代码*文件。

让我们开始吧！![](img/c9f3ac961769a235f2a8293ea0ac8dca.png)

为你的 Python 脚本添加命令行参数。照片由[insung yoon](https://unsplash.com/photos/Uaf6XwW4n10)拍摄。部分权利保留

## 概述

本教程分为三部分；它们是：

+   在命令行中运行 Python 脚本

+   在命令行工作

+   替代命令行参数

## 在命令行中运行 Python 脚本

有许多方法可以运行 Python 脚本。有人可能在 Jupyter 笔记本中运行它。有人可能在 IDE 中运行它。但在所有平台上，始终可以在命令行中运行 Python 脚本。在 Windows 中，你可以使用命令提示符或 PowerShell（或者更好的是[Windows 终端](https://aka.ms/terminal)）。在 macOS 或 Linux 中，你可以使用终端或 xterm。在命令行中运行 Python 脚本是强大的，因为你可以向脚本传递额外的参数。

以下脚本允许我们将值从命令行传递到 Python 中：

```py
import sys

n = int(sys.argv[1])
print(n+1)
```

我们将这几行保存到一个文件中，并在命令行中运行它，带一个参数：

Shell

```py
$ python commandline.py 15
16
```

然后，你会看到它接受我们的参数，将其转换为整数，加一并打印出来。列表`sys.argv`包含我们脚本的名称和所有参数（都是字符串），在上述情况下，是`["commandline.py", "15"]`。

当你运行带有更复杂参数集的命令行时，需要一些处理列表`sys.argv`的努力。因此，Python 提供了`argparse`库来帮助。这假设 GNU 风格，可以用以下例子来解释：

```py
rsync -a -v --exclude="*.pyc" -B 1024 --ignore-existing 192.168.0.3:/tmp/ ./
```

可选参数由“`-`”或“`--`”引入，单个连字符表示单个字符的“短选项”（例如上述的 `-a`、`-B` 和 `-v`），双连字符用于多个字符的“长选项”（例如上述的 `--exclude` 和 `--ignore-existing`）。可选参数可能有附加参数，例如在 `-B 1024` 或 `--exclude="*.pyc"` 中，`1024` 和 `"*.pyc"` 分别是 `B` 和 `--exclude` 的参数。此外，我们还可能有强制性参数，我们直接将其放入命令行中。上面的 `192.168.0.3:/tmp/` 和 `./` 就是例子。强制参数的顺序很重要。例如，上面的 `rsync` 命令将文件从 `192.168.0.3:/tmp/` 复制到 `./` 而不是相反。

下面是使用 argparse 在 Python 中复制上述示例的方法：

```py
import argparse

parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-a", "--archive", action="store_true", help="archive mode")
parser.add_argument("-v", "--verbose", action="store_true", help="increase verbosity")
parser.add_argument("-B", "--block-size", help="checksum blocksize")
parser.add_argument("--ignore-existing", action="store_true", help="skip files that exist")
parser.add_argument("--exclude", help="files to exclude")
parser.add_argument("src", help="Source location")
parser.add_argument("dest", help="Destination location")
args = parser.parse_args()
config = vars(args)
print(config)
```

如果运行上述脚本，您将看到：

```py
$ python argparse_example.py
usage: argparse_example.py [-h] [-a] [-v] [-B BLOCK_SIZE] [--ignore-existing] [--exclude EXCLUDE] src dest
argparse_example.py: error: the following arguments are required: src, dest
```

这意味着您没有为 `src` 和 `dest` 提供必需的参数。也许使用 argparse 的最佳理由是，如果您提供了 `-h` 或 `--help` 作为参数，可以免费获取帮助屏幕，如下所示：

```py
$ python argparse_example.py --help
usage: argparse_example.py [-h] [-a] [-v] [-B BLOCK_SIZE] [--ignore-existing] [--exclude EXCLUDE] src dest

Just an example

positional arguments:
  src                   Source location
  dest                  Destination location

optional arguments:
  -h, --help            show this help message and exit
  -a, --archive         archive mode (default: False)
  -v, --verbose         increase verbosity (default: False)
  -B BLOCK_SIZE, --block-size BLOCK_SIZE
                        checksum blocksize (default: None)
  --ignore-existing     skip files that exist (default: False)
  --exclude EXCLUDE     files to exclude (default: None)
```

虽然脚本并未执行任何实际操作，但如果按要求提供参数，将会看到以下内容：

```py
$ python argparse_example.py -a --ignore-existing 192.168.0.1:/tmp/ /home
{'archive': True, 'verbose': False, 'block_size': None, 'ignore_existing': True, 'exclude': None, 'src': '192.168.0.1:/tmp/', 'dest': '/home'}
```

由 `ArgumentParser()` 创建的解析器对象有一个 `parse_args()` 方法，它读取 `sys.argv` 并返回一个 **namespace** 对象。这是一个携带属性的对象，我们可以使用 `args.ignore_existing` 等方式读取它们。但通常，如果它是 Python 字典，处理起来会更容易。因此，我们可以使用 `vars(args)` 将其转换为一个字典。

通常，对于所有可选参数，我们提供长选项，有时也提供短选项。然后，我们可以使用长选项作为键（将连字符替换为下划线）从命令行访问提供的值，如果没有长版本，则使用单字符短选项作为键。 “位置参数” 不是可选的，并且它们的名称在 `add_argument()` 函数中提供。

参数有多种类型。对于可选参数，有时我们将它们用作布尔标志，但有时我们期望它们带入一些数据。在上述示例中，我们使用 `action="store_true"` 来将该选项默认设置为 `False`，如果指定则切换为 `True`。对于其他选项，例如上面的 `-B`，默认情况下，它期望在其后跟随附加数据。

我们还可以进一步要求参数是特定类型。例如，对于上面的 `-B` 选项，我们可以通过添加 `type` 来使其期望整数数据，如下所示：

```py
parser.add_argument("-B", "--block-size", type=int, help="checksum blocksize")
```

如果提供了错误的类型，argparse 将帮助终止我们的程序，并显示一个信息性错误消息：

```py
python argparse_example.py -a -B hello --ignore-existing 192.168.0.1:/tmp/ /home
usage: argparse_example.py [-h] [-a] [-v] [-B BLOCK_SIZE] [--ignore-existing] [--exclude EXCLUDE] src dest
argparse_example.py: error: argument -B/--block-size: invalid int value: 'hello'
```

## 在命令行上工作

使用命令行参数增强你的 Python 脚本可以使其达到新的可重用性水平。首先，让我们看一个简单的示例，将 ARIMA 模型拟合到 GDP 时间序列上。世界银行收集了许多国家的历史 GDP 数据。我们可以利用`pandas_datareader`包来读取这些数据。如果你还没有安装它，可以使用`pip`（或者如果你安装了 Anaconda，则可以使用`conda`）来安装该包：

```py
pip install pandas_datareader
```

我们使用的 GDP 数据的代码是`NY.GDP.MKTP.CN`；我们可以通过以下方式获得国家的数据，将其转换成 pandas DataFrame：

```py
from pandas_datareader.wb import WorldBankReader

gdp = WorldBankReader("NY.GDP.MKTP.CN", "SE", start=1960, end=2020).read()
```

然后，我们可以使用 pandas 提供的工具稍微整理一下 DataFrame：

```py
import pandas as pd

# Drop country name from index
gdp = gdp.droplevel(level=0, axis=0)
# Sort data in choronological order and set data point at year-end
gdp.index = pd.to_datetime(gdp.index)
gdp = gdp.sort_index().resample("y").last()
# Convert pandas DataFrame into pandas Series
gdp = gdp["NY.GDP.MKTP.CN"]
```

拟合 ARIMA 模型并使用该模型进行预测并不困难。接下来，我们使用前 40 个数据点进行拟合，并预测未来 3 个数据点。然后，通过相对误差比较预测值和实际值：

```py
import statsmodels.api as sm

model = sm.tsa.ARIMA(endog=gdp[:40], order=(1,1,1)).fit()
forecast = model.forecast(steps=3)
compare = pd.DataFrame({"actual":gdp, "forecast":forecast}).dropna()
compare["rel error"] = (compare["forecast"] - compare["actual"])/compare["actual"]
print(compare)
```

将所有内容整合在一起，并稍加修饰后，以下是完整的代码：

```py
import warnings
warnings.simplefilter("ignore")

from pandas_datareader.wb import WorldBankReader
import statsmodels.api as sm
import pandas as pd

series = "NY.GDP.MKTP.CN"
country = "SE" # Sweden
length = 40
start = 0
steps = 3
order = (1,1,1)

# Read the GDP data from WorldBank database
gdp = WorldBankReader(series, country, start=1960, end=2020).read()
# Drop country name from index
gdp = gdp.droplevel(level=0, axis=0)
# Sort data in choronological order and set data point at year-end
gdp.index = pd.to_datetime(gdp.index)
gdp = gdp.sort_index().resample("y").last()
# Convert pandas dataframe into pandas series
gdp = gdp[series]
# Fit arima model
result = sm.tsa.ARIMA(endog=gdp[start:start+length], order=order).fit()
# Forecast, and calculate the relative error
forecast = result.forecast(steps=steps)
df = pd.DataFrame({"Actual":gdp, "Forecast":forecast}).dropna()
df["Rel Error"] = (df["Forecast"] - df["Actual"]) / df["Actual"]
# Print result
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    print(df)
```

此脚本输出以下内容：

```py
                   Actual      Forecast  Rel Error
2000-12-31  2408151000000  2.367152e+12  -0.017025
2001-12-31  2503731000000  2.449716e+12  -0.021574
2002-12-31  2598336000000  2.516118e+12  -0.031643
```

上述代码很简短，但我们通过在变量中保存一些参数使其更加灵活。我们可以将上述代码改为使用 argparse，这样我们就可以从命令行中更改一些参数，如下所示：

```py
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
warnings.simplefilter("ignore")

from pandas_datareader.wb import WorldBankReader
import statsmodels.api as sm
import pandas as pd

# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--country", default="SE", help="Two-letter country code")
parser.add_argument("-l", "--length", default=40, type=int, help="Length of time series to fit the ARIMA model")
parser.add_argument("-s", "--start", default=0, type=int, help="Starting offset to fit the ARIMA model")
args = vars(parser.parse_args())

# Set up parameters
series = "NY.GDP.MKTP.CN"
country = args["country"]
length = args["length"]
start = args["start"]
steps = 3
order = (1,1,1)

# Read the GDP data from WorldBank database
gdp = WorldBankReader(series, country, start=1960, end=2020).read()
# Drop country name from index
gdp = gdp.droplevel(level=0, axis=0)
# Sort data in choronological order and set data point at year-end
gdp.index = pd.to_datetime(gdp.index)
gdp = gdp.sort_index().resample("y").last()
# Convert pandas dataframe into pandas series
gdp = gdp[series]
# Fit arima model
result = sm.tsa.ARIMA(endog=gdp[start:start+length], order=order).fit()
# Forecast, and calculate the relative error
forecast = result.forecast(steps=steps)
df = pd.DataFrame({"Actual":gdp, "Forecast":forecast}).dropna()
df["Rel Error"] = (df["Forecast"] - df["Actual"]) / df["Actual"]
# Print result
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    print(df)
```

如果我们在命令行中运行上述代码，可以看到它现在可以接受参数：

```py
$ python gdp_arima.py --help
usage: gdp_arima.py [-h] [-c COUNTRY] [-l LENGTH] [-s START]

optional arguments:
  -h, --help            show this help message and exit
  -c COUNTRY, --country COUNTRY
                        Two-letter country code (default: SE)
  -l LENGTH, --length LENGTH
                        Length of time series to fit the ARIMA model (default: 40)
  -s START, --start START
                        Starting offset to fit the ARIMA model (default: 0)
$ python gdp_arima.py
                   Actual      Forecast  Rel Error
2000-12-31  2408151000000  2.367152e+12  -0.017025
2001-12-31  2503731000000  2.449716e+12  -0.021574
2002-12-31  2598336000000  2.516118e+12  -0.031643
$ python gdp_arima.py -c NO
                   Actual      Forecast  Rel Error
2000-12-31  1507283000000  1.337229e+12  -0.112821
2001-12-31  1564306000000  1.408769e+12  -0.099429
2002-12-31  1561026000000  1.480307e+12  -0.051709
```

在上面的最后一个命令中，我们传入`-c NO`来将相同的模型应用于挪威（NO）的 GDP 数据，而不是瑞典（SE）。因此，在不破坏代码的风险下，我们重用了我们的代码来处理不同的数据集。

引入命令行参数的强大之处在于，我们可以轻松地测试我们的代码，使用不同的参数。例如，我们想要查看 ARIMA(1,1,1)模型是否是预测 GDP 的好模型，并且我们希望使用北欧国家的不同时间窗口来验证：

+   丹麦（DK）

+   芬兰（FI）

+   冰岛（IS）

+   挪威（NO）

+   瑞典（SE）

我们希望检查 40 年的窗口，但是使用不同的起始点（从 1960 年、1965 年、1970 年、1975 年起）。根据操作系统的不同，可以在 Linux 和 mac 中使用 bash shell 语法构建 for 循环：

Shell

```py
for C in DK FI IS NO SE; do
    for S in 0 5 10 15; do
        python gdp_arima.py -c $C -s $S
    done
done
```

或者，由于 shell 语法允许，我们可以将所有内容放在一行中：

Shell

```py
for C in DK FI IS NO SE; do for S in 0 5 10 15; do python gdp_arima.py -c $C -s $S ; done ; done
```

或者更好的做法是，在循环的每次迭代中提供一些信息，然后多次运行我们的脚本：

```py
$ for C in DK FI IS NO SE; do for S in 0 5 10 15; do echo $C $S; python gdp_arima.py -c $C -s $S ; done; done
DK 0
                  Actual      Forecast  Rel Error
2000-12-31  1.326912e+12  1.290489e+12  -0.027449
2001-12-31  1.371526e+12  1.338878e+12  -0.023804
2002-12-31  1.410271e+12  1.386694e+12  -0.016718
DK 5
                  Actual      Forecast  Rel Error
2005-12-31  1.585984e+12  1.555961e+12  -0.018931
2006-12-31  1.682260e+12  1.605475e+12  -0.045644
2007-12-31  1.738845e+12  1.654548e+12  -0.048479
DK 10
                  Actual      Forecast  Rel Error
2010-12-31  1.810926e+12  1.762747e+12  -0.026605
2011-12-31  1.846854e+12  1.803335e+12  -0.023564
2012-12-31  1.895002e+12  1.843907e+12  -0.026963

...

SE 5
                   Actual      Forecast  Rel Error
2005-12-31  2931085000000  2.947563e+12   0.005622
2006-12-31  3121668000000  3.043831e+12  -0.024934
2007-12-31  3320278000000  3.122791e+12  -0.059479
SE 10
                   Actual      Forecast  Rel Error
2010-12-31  3573581000000  3.237310e+12  -0.094099
2011-12-31  3727905000000  3.163924e+12  -0.151286
2012-12-31  3743086000000  3.112069e+12  -0.168582
SE 15
                   Actual      Forecast  Rel Error
2015-12-31  4260470000000  4.086529e+12  -0.040827
2016-12-31  4415031000000  4.180213e+12  -0.053186
2017-12-31  4625094000000  4.273781e+12  -0.075958
```

如果你使用 Windows，可以在命令提示符中使用以下语法：

MS DOS

```py
for %C in (DK FI IS NO SE) do for %S in (0 5 10 15) do python gdp_arima.py -c $C -s $S
```

或者在 PowerShell 中：

PowerShell

```py
foreach ($C in "DK","FI","IS","NO","SE") { foreach ($S in 0,5,10,15) { python gdp_arima.py -c $C -s $S } }
```

两者应该产生相同的结果。

虽然我们可以将类似的循环放在 Python 脚本中，但有时如果能在命令行中完成会更容易。当我们探索不同的选项时，这可能更加方便。此外，通过将循环移到 Python 代码之外，我们可以确保每次运行脚本时都是独立的，因为我们不会在迭代之间共享任何变量。

# 命令行参数的替代方案

使用命令行参数并不是将数据传递给 Python 脚本的唯一方法。至少还有几种其他方法：

+   使用环境变量

+   使用配置文件

环境变量是操作系统提供的功能，用于在内存中保留少量数据。我们可以使用以下语法在 Python 中读取环境变量：

```py
import os
print(os.environ["MYVALUE"])
```

例如，在 Linux 中，上述两行脚本将在 shell 中如下工作：

```py
$ export MYVALUE="hello"
$ python show_env.py
hello
```

在 Windows 中，命令提示符中的语法类似：

```py
C:\MLM> set MYVALUE=hello

C:\MLM> python show_env.py
hello
```

你还可以通过控制面板中的对话框在 Windows 中添加或编辑环境变量：

![](https://machinelearningmastery.com/wp-content/uploads/2022/02/Env-Variable.jpg)

因此，我们可以将参数保存在一些环境变量中，让脚本适应其行为，例如设置命令行参数。

如果我们需要设置很多选项，最好将这些选项保存到文件中，而不是让命令行变得过于繁杂。根据我们选择的格式，我们可以使用 Python 的 `configparser` 或 `json` 模块来读取 Windows INI 格式或 JSON 格式。我们也可以使用第三方库 PyYAML 来读取 YAML 格式。

对于上述在 GDP 数据上运行 ARIMA 模型的示例，我们可以修改代码以使用 YAML 配置文件：

```py
import warnings
warnings.simplefilter("ignore")

from pandas_datareader.wb import WorldBankReader
import statsmodels.api as sm
import pandas as pd
import yaml

# Load config from YAML file
with open("config.yaml", "r") as fp:
    args = yaml.safe_load(fp)

# Set up parameters
series = "NY.GDP.MKTP.CN"
country = args["country"]
length = args["length"]
start = args["start"]
steps = 3
order = (1,1,1)

# Read the GDP data from WorldBank database
gdp = WorldBankReader(series, country, start=1960, end=2020).read()
# Drop country name from index
gdp = gdp.droplevel(level=0, axis=0)
# Sort data in choronological order and set data point at year-end
gdp.index = pd.to_datetime(gdp.index)
gdp = gdp.sort_index().resample("y").last()
# Convert pandas dataframe into pandas series
gdp = gdp[series]
# Fit arima model
result = sm.tsa.ARIMA(endog=gdp[start:start+length], order=order).fit()
# Forecast, and calculate the relative error
forecast = result.forecast(steps=steps)
df = pd.DataFrame({"Actual":gdp, "Forecast":forecast}).dropna()
df["Rel Error"] = (df["Forecast"] - df["Actual"]) / df["Actual"]
# Print result
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    print(df)
```

YAML 配置文件名为 `config.yaml`，其内容如下：

```py
country: SE
length: 40
start: 0
```

然后我们可以运行上述代码，并获得与之前相同的结果。JSON 对应的代码非常相似，我们使用 `json` 模块中的 `load()` 函数：

```py
import json
import warnings
warnings.simplefilter("ignore")

from pandas_datareader.wb import WorldBankReader
import statsmodels.api as sm
import pandas as pd

# Load config from JSON file
with open("config.json", "r") as fp:
    args = json.load(fp)

# Set up parameters
series = "NY.GDP.MKTP.CN"
country = args["country"]
length = args["length"]
start = args["start"]
steps = 3
order = (1,1,1)

# Read the GDP data from WorldBank database
gdp = WorldBankReader(series, country, start=1960, end=2020).read()
# Drop country name from index
gdp = gdp.droplevel(level=0, axis=0)
# Sort data in choronological order and set data point at year-end
gdp.index = pd.to_datetime(gdp.index)
gdp = gdp.sort_index().resample("y").last()
# Convert pandas dataframe into pandas series
gdp = gdp[series]
# Fit arima model
result = sm.tsa.ARIMA(endog=gdp[start:start+length], order=order).fit()
# Forecast, and calculate the relative error
forecast = result.forecast(steps=steps)
df = pd.DataFrame({"Actual":gdp, "Forecast":forecast}).dropna()
df["Rel Error"] = (df["Forecast"] - df["Actual"]) / df["Actual"]
# Print result
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    print(df)
```

JSON 配置文件 `config.json` 如下：

```py
{
    "country": "SE",
    "length": 40,
    "start": 0
}
```

你可以了解更多关于[JSON](https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Objects/JSON)和[YAML](https://en.wikipedia.org/wiki/YAML)的语法，以便于你的项目。但这里的核心理念是，我们可以分离数据和算法，以提高代码的可重用性。

### 想要开始使用 Python 进行机器学习？

立即领取我的免费 7 天电子邮件速成课程（包含示例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## 进一步阅读

本节提供了更多关于该主题的资源，如果你想深入了解。

#### 库

+   argparse 模块，[`docs.python.org/3/library/argparse.html`](https://docs.python.org/3/library/argparse.html)

+   Pandas Data Reader，[`pandas-datareader.readthedocs.io/en/latest/`](https://pandas-datareader.readthedocs.io/en/latest/)

+   statsmodels 中的 ARIMA，[`www.statsmodels.org/devel/generated/statsmodels.tsa.arima.model.ARIMA.html`](https://www.statsmodels.org/devel/generated/statsmodels.tsa.arima.model.ARIMA.html)

+   configparser 模块，[`docs.python.org/3/library/configparser.html`](https://docs.python.org/3/library/configparser.html)

+   json 模块，[`docs.python.org/3/library/json.html`](https://docs.python.org/3/library/json.html)

+   PyYAML，[`pyyaml.org/wiki/PyYAMLDocumentation`](https://pyyaml.org/wiki/PyYAMLDocumentation)

#### 文章

+   处理 JSON，[`developer.mozilla.org/en-US/docs/Learn/JavaScript/Objects/JSON`](https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Objects/JSON)

+   维基百科上的 YAML，[`zh.wikipedia.org/wiki/YAML`](https://zh.wikipedia.org/wiki/YAML)

#### 书籍

+   *Python Cookbook*，第三版，作者 David Beazley 和 Brian K. Jones，[`www.amazon.com/dp/1449340377/`](https://www.amazon.com/dp/1449340377/)

## 摘要

在本教程中，您已经看到如何使用命令行更有效地控制我们的 Python 脚本。具体来说，您学到了：

+   如何使用 argparse 模块向您的 Python 脚本传递参数

+   如何在不同操作系统的终端中高效控制启用 argparse 的 Python 脚本

+   我们还可以使用环境变量或配置文件来向 Python 脚本传递参数
