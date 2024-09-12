# 数据科学入门（7 天迷你课程）

> 原文：[`machinelearningmastery.com/beginning-data-science-7-day-mini-course/`](https://machinelearningmastery.com/beginning-data-science-7-day-mini-course/)

数据科学使用数学来分析数据、提炼信息并讲述故事。数据科学的结果可能只是严格验证一个假设，或从数据中发现一些有用的属性。你可以在数据科学中使用许多工具，从基本统计学到复杂的机器学习模型。即使是最常见的工具，也可以在数据科学项目中发挥出色的作用。

在这个 7 部分的速成课程中，你将通过示例学习如何进行数据科学项目。这个迷你课程专注于数据科学的核心内容。假设你已经收集了数据并做好了使用准备。编写网络爬虫和验证你收集的数据可能是一个大话题；这不在本课程的范围之内。这个迷你课程是为那些已经熟悉 Python 编程并愿意学习数据科学常用工具（如 pandas 和 matplotlib）的从业人员准备的。你将看到这些工具如何提供帮助，更重要的是，学习如何从你拥有的数据中得出定量支持的结论。让我们开始吧。

![](img/8239ac26f53fe92d9798371cb3384718.png)

数据科学初学者指南（7 天迷你课程）

图片由 [Eduardo Soares](https://unsplash.com/photos/person-holding-white-and-red-card-utWyPB8_FU8) 提供。版权所有。

## 这个迷你课程适合谁？

在我们开始之前，让我们确保你来对了地方。下面的列表提供了一些关于本课程设计对象的基本指南。如果你不完全符合这些条件也不要惊慌；你可能只需要在某个方面稍作复习以跟上进度。

+   **能够编写一些代码的开发人员**。这意味着你可以用 Python 完成任务，并且知道如何在工作站上设置生态系统（这是一个前提条件）。这并不意味着你是一个编码高手，但你不怕安装包和编写脚本。

+   **掌握一些统计学的开发人员**。这意味着你了解一些基本的统计工具，并且不怕使用它们。这并不意味着你是统计学博士，但如果遇到术语，你可以查找并学习。

+   **了解一些数据科学工具的开发人员**。在数据科学中，使用 Jupyter notebook 是很常见的。如果使用 pandas 库处理数据会更容易。这只是其中的一部分。你不需要对任何库成为专家，但你需要对调用不同的库和编写代码以操作数据感到舒适。

这个迷你课程不是一本数据科学教科书。而是一个项目指南，它将你从一个基础知识较少的开发人员，逐步带到能够自信展示如何完成数据科学项目的开发人员。

## 迷你课程概览

本迷你课程分为 7 部分。

每节课的设计时间大约是 30 分钟。你可能会更快完成某些部分，也可能会选择深入研究，花费更多时间。

你可以按照自己喜欢的速度完成每一部分。一个舒适的安排可能是每天完成一节课，持续七天。强烈推荐。

接下来 7 节课你将学习以下内容：

+   **第 1 课**：获取数据

+   **第 2 课**：缺失值

+   **第 3 课**：描述性统计

+   **第 4 课**：探索数据

+   **第 5 课**：可视化相关性

+   **第 6 课**：假设检验

+   **第 7 课**：识别异常值

这将非常有趣。

不过你得做些工作：一点阅读、研究和编程。你想学会如何完成一个数据科学项目，对吧？

**在评论中发布你的结果**；我会为你加油！

坚持住；不要放弃。

## 第 01 课：获取数据

我们将用于本迷你课程的数据集是“Kaggle 上提供的‘所有国家数据集’”：

+   [`www.kaggle.com/datasets/adityakishor1/all-countries-details`](https://www.kaggle.com/datasets/adityakishor1/all-countries-details)

这个数据集描述了几乎所有国家的人口、经济、地理、健康和政治数据。这类数据集中最著名的可能是 CIA 世界概况。抓取世界概况中的数据应该会给你更多全面和最新的数据。然而，使用这个 CSV 格式的数据集可以在构建你的网页抓取器时节省很多麻烦。

下载[Kaggle 上的这个数据集](https://www.kaggle.com/datasets/adityakishor1/all-countries-details)（你可能需要注册一个账户），你会找到 CSV 文件[`All Countries.csv`](https://github.com/Padre-Media/dataset/raw/main/All%20Countries.csv)。让我们用 pandas 检查这个数据集。

```py
import pandas as pd

df = pd.read_csv("All Countries.csv")
df.info()
```

上述代码将把一个表格打印到屏幕上，如下所示：

```py
&lt;class 'pandas.core.frame.DataFrame'&gt;
RangeIndex: 194 entries, 0 to 193
Data columns (total 64 columns):
    #   Column                                   Non-Null Count  Dtype
---  ------                                   --------------  -----
    0   country                                  194 non-null    object
    1   country_long                             194 non-null    object
    2   currency                                 194 non-null    object
    3   capital_city                             194 non-null    object
    4   region                                   194 non-null    object
    5   continent                                194 non-null    object
    6   demonym                                  194 non-null    object
    7   latitude                                 194 non-null    float64
    8   longitude                                194 non-null    float64
    9   agricultural_land                        193 non-null    float64
...
    62  political_leader                         187 non-null    object
    63  title                                    187 non-null    object
dtypes: float64(48), int64(6), object(10)
memory usage: 97.1+ KB
```

在上面，你可以看到数据集的基本信息。例如，在顶部，你知道这个 CSV 文件中有 194 条记录（行）。表格告诉你有 64 列（从 0 到 63 索引）。一些列是数值型的，比如纬度，另一些则不是，比如 capital_city。pandas 中的数据类型“object”通常意味着它是字符串类型。你还知道有一些缺失值，比如在`agricultural_land`中，194 条记录中只有 193 个非空值，这意味着这一列中有一行缺失值。

让我们更详细地查看数据集，例如以前五行作为样本：

```py
df.head(5)
```

这将以表格形式显示数据集的前五行。

### 你的任务

这是数据集的基本探索。但是，使用`head()`函数可能并不总是合适（例如，当输入数据已排序时）。还有`tail()`函数用于类似的目的。然而，运行`df.sample(5)`通常会更有帮助，因为它是随机抽取 5 行。尝试这个函数。此外，如你从上述输出中所见，列被截断到屏幕宽度。如何修改上述代码以显示样本的**所有**列？

提示：pandas 中有一个`to_string()`函数，你也可以调整通用打印选项`display.max_columns`。

在下一个课时，你将学习如何检查数据中的缺失值。

## 课时 02：缺失值

在分析数据之前，了解数据的情况非常重要。在 pandas 中，浮点型列可能会将缺失值表示为`NaN`（“不是一个数字”），并且这些值的存在会破坏许多函数。

在 pandas 中，你可以通过`isnull()`或`notnull()`来查找缺失值。这些函数用于检查值是否为 null，包括 Python 的`None`和浮点型的`NaN`。返回值为布尔值。如果应用于列，你会得到一列 True 或 False。求和结果是 True 值的数量。

在下面，你可以使用`isnull()`来查找缺失值，然后对结果进行求和以计算其数量。你可以对结果进行排序，以查看哪些列缺失值最多，哪些列缺失值最少。

```py
print(df.isnull().sum().sort_values(ascending=False))
```

你将看到上述打印内容：

```py
internally_displaced_persons           121
central_government_debt_pct_gdp         74
hiv_incidence                           61
energy_imports_pct                      56
electricty_production_renewable_pct     56
                                        ...
land_area                                0
urban_population_under_5m                0
rural_land                               0
urban_land                               0
country                                  0
Length: 64, dtype: int64
```

在上述内容中，你可以看到某些列没有缺失值，例如国家名称。缺失值最多的列是`internally_displaced_persons`，这是难民的人口统计信息。如你所见，这并不正常，合理的情况是大多数国家没有这样的群体。因此，当你处理时，可以将缺失值替换为零。这是利用领域知识进行插补的一个例子。

为了可视化缺失值，你可以使用 Python 包`missingno`。它有助于展示缺失值的分布情况：

```py
import missingno as msno
import matplotlib.pyplot as plt

msno.matrix(df, sparkline=False, fontsize=12)
plt.show()
```

上述图表显示，某些国家（行）和一些属性（列）有大量的缺失值。你可以大致猜测图表中的哪一列对应`internally_displaced_persons`。缺失值较多的国家可能是因为这些国家没有收集这些统计数据。

### 你的任务

并非所有的缺失值都应该用零来替代。另一种策略是用均值替换缺失值。你能找到这个数据集中另一个适合用均值替换缺失值的属性吗？此外，如何在 pandas DataFrame 中替换缺失值？

在下一个课时，你将学习如何使用基本统计数据来探索数据。

## 课时 03：描述性统计

给定一个 pandas DataFrame，查看描述性统计是一个重要的第一步。在代码中，你可以使用 DataFrame 的`describe()`函数：

```py
print(df.describe())
```

这显示了每个**数值**属性的均值、标准差、最小值、最大值和四分位数。非数值列不会在输出中报告。你可以通过打印列的集合并进行比较来验证这一点：

```py
print(df.columns)
print(df.describe().columns)
```

数据集中有很多列。要查看某一列的描述性统计信息，你可以将其输出过滤为 DataFrame：

```py
print(df.describe()["inflation"])
```

这会打印：

```py
count    184.000000
mean      13.046591
std       25.746553
min       -6.687320
25%        4.720087
50%        7.864485
75%       11.649325
max      254.949000
Name: inflation, dtype: float64
```

这与定义`df2=df.describe()`然后提取`df2["inflation"]`是一样的。如果列中有缺失值，描述性统计数据会通过跳过所有缺失值来计算。

### 你的任务

从之前的示例继续，你可以通过检查`df["inflation"].isnull().sum()`是否为零来判断`inflation`列中是否存在缺失值。均值可以通过`df["inflation"].mean()`来计算。你怎么验证这个均值是否跳过了所有的缺失值？

在下一节课中，你将看到如何进一步了解数据。

## 第四部分课：数据探索

数据科学的目标是从数据中讲述一个故事。让我们在这里看看一些示例。

在数据集中，有一列`life_expectancy`。什么因素影响寿命？你可以做一些假设并通过数据验证。你可以检查世界不同地区的寿命是否存在差异：

```py
print(df.groupby("region").mean(numeric_only=True)["life_expectancy"])
```

运行上述代码并观察其输出。存在一些差异，但并不十分剧烈。在 DataFrame 上应用`groupby()`函数类似于 SQL 语句中的`GROUP BY`子句。但是在 pandas 中，应用函数到**groupby**时需要注意列中的不同数据类型。如果像上面那样使用`mean()`，它会计算所有列的均值（然后你选择了`life_expectancy`），如果列不是数值型则会失败。因此，你需要添加一个参数来限制操作只针对那些列。

从上面可以看出，寿命与世界的哪个部分无关。你也可以按大洲分组，但这可能不太合适，因为某些大洲，如亚洲，既大又多样化。在这些情况下，平均值可能不具有参考价值。

你可以应用类似的操作来查找不是寿命而是人均 GDP。这是国家 GDP 除以人口，这是一种衡量国家富裕程度的指标。在代码中：

```py
df["gdp_per_capita"] = df["gdp"] / df["population"]
print(df.groupby("region").mean(numeric_only=True)["gdp_per_capita"])
```

这显示了不同地区之间的巨大差异。因此，与预期寿命不同，你所在的地区与你的财富水平有关。

除了分组，探索和总结数据的另一个有用方法是**数据透视表**。pandas DataFrame 中有一个函数可以实现这一点。让我们看看不同地区对不同类型政府的偏好：

```py
print(df.pivot_table(index="region", columns="democracy_type", aggfunc="count")["country"])
```

上表显示了**计数**，它作为汇总函数被指定。行（索引）是“区域”，列是`democracy_type`中的值。每个单元格中的数字计数了该“民主类型”在同一“区域”中的实例。一些值是 NaN，这意味着该组合没有数据可以“计数”。由于这是计数，所以你知道它意味着零。

### 你的任务

数据透视表和分组是总结数据和提炼信息的非常强大的工具。你如何使用上述透视表来查找不同地区的人均 GDP 和民主类型的平均值？你会看到缺失值。为了帮助找出不同民主类型的平均值，合理的缺失值填补是什么？

在下一课中，你将学习如何从图表中调查数据。

## 第 05 课：可视化相关性

在上一课中，我们探讨了寿命预期和人均 GDP 列。它们有相关性吗？

有很多方法可以判断两个属性是否相关。散点图是一个很好的第一步，因为它提供了视觉证明。要将人均 GDP（如第 4 课中通过除以 GDP 和人口计算）与寿命预期绘制在一起，你可以使用 Python 库 Seaborn 结合 Matplotlib：

```py
import seaborn as sns

sns.scatterplot(data=df, x="life_expectancy", y="gdp_per_capita", hue="continent")
```

上述散点图函数中的参数`hue`是可选的。它根据另一个属性的值为点着色，因此，它对于说明例如非洲在寿命预期和人均 GDP 低端的表现很有用。

然而，上述图表存在一个问题：你无法看到任何线性模式，且很难判断两个属性之间的关系。在这种情况下，你必须**转换**数据以确定关系。让我们尝试使用**半对数图**，其中 y 轴以对数刻度呈现。你可以使用 Matplotlib 来调整刻度：

```py
sns.scatterplot(data=df, x="life_expectancy", y="gdp_per_capita", hue="continent")
plt.yscale("log")  # make y axis in log scale
```

现在，看起来寿命预期与人均 GDP 对数是线性的更有可能。

从数值上，你可以计算人均 GDP 对数和寿命预期之间的相关因子。相关因子接近 +1 或 -1 意味着相关性强。无相关的属性将显示接近零的相关因子。你可以使用 pandas 查找所有数值属性中相关性最强的因子。

```py
top_features = df.corr(numeric_only=True)["life_expectancy"].abs().sort_values(ascending=False).index[:6]
print(top_features)
```

上述代码找出了与寿命预期最相关的前 6 个属性。无论是正相关还是负相关，因为排序是基于绝对值的。根据定义，寿命预期本身应该排在列表的最顶部，因为任何东西与自己具有相关性 1。

你可以使用 Seaborn 创建一个相关图，以显示任意一对属性之间的散点图：

```py
sns.pairplot(df, vars=list(top_features))
plt.show()
```

相关图帮助你快速可视化相关性。例如，自雇百分比与脆弱就业百分比高度相关。出生率与预期寿命呈负相关（也许因为年龄越大，人们生育的可能性越小）。对角线上的直方图显示了该属性的分布情况。

### 你的任务

散点图是一个强大的工具，特别是当你有计算机来帮助你制作时。上面，你已经通过视觉展示了两个属性之间的相关性，但**相关性不等于因果关系**。要建立因果关系，你需要更多的证据。在统计学中，有九项在流行病学中非常著名的“布拉德福德·希尔标准”。一个更简单且较弱的表述是格兰杰因果关系的两个原则。查看你拥有的数据，并与格兰杰因果关系原则进行比较，证明预期寿命是由人均 GDP**引起**的需要哪些额外数据？

在下一课中，你将使用统计测试来分析你的数据。

## 课程 06：假设检验

由于数据科学的任务是讲述故事，因此支持你的主张是数据科学项目中的核心工作。

让我们再次关注预期寿命：城市化是提高预期寿命的关键，因为它还与先进的医学、卫生和免疫接种相关。你如何证明这一点？

一个简单的方法是展示两个预期寿命的直方图，一个是针对更多城市化国家的，另一个是针对非城市化国家的。让我们定义一个城市国家为城市人口超过 50%的国家。你可以使用 pandas 计算人口百分比，然后将数据集分为两个部分：

```py
df["urban_pct"] = df["urban_population"]/(df["rural_population"] + df["urban_population"])
df_urban = df[df["urban_pct"] > 0.5]
df_rural = df[df["urban_pct"] <= 0.5]
```

然后，你可以创建一个重叠的直方图来显示预期寿命：

```py
plt.hist(df_urban["life_expectancy"], alpha=0.7, bins=15, color="blue", label="Urba")
plt.hist(df_rural["life_expectancy"], alpha=0.7, bins=15, color="green", label="Rural")
plt.xlabel("Life expectancy")
plt.ylabel("Number of countries")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
```

这证实了上述假设，即城市国家具有更高的预期寿命。然而，图表不是很有力的证据。更好的方法是应用一些统计测试来量化我们的主张的强度。你需要比较两个独立组之间的**均值预期寿命**，因此 t 检验是合适的。你可以使用 SciPy 包运行 t 检验，如下所示：

```py
import scipy.stats as stats

df_urban = df[(df["urban_pct"] > 0.5) & df["life_expectancy"].notnull()]
df_rural = df[(df["urban_pct"] <= 0.5) & df["life_expectancy"].notnull()]
t_stat, p_value = stats.ttest_ind(df_urban["life_expectancy"], df_rural["life_expectancy"], equal_var=False)
print("t-Statistic:", t_stat)
print("p-value", p_value)
```

与 Matplotlib 不同，Matplotlib 会忽略缺失值，而 SciPy 在提供的数据中存在任何 NaN 时不会计算统计数据。因此，上面你通过移除缺失值来清理数据，并重新创建`df_urban`和`df_rural`数据框。t 检验提供了一个 1.6×10^(-10)的 p 值，这个值非常小。因此，**零假设**被拒绝，即拒绝两个组的均值相同。但这个 t 检验并没有告诉我们`df_urban`或`df_rural`的均值更高。你可以通过分别计算均值来轻松确定。

### 你的任务

与重新创建`df_urban`和`df_rural`数据框不同，你可以通过用各自的均值填补缺失值来使 SciPy 的 t 检验正常工作。尝试一下。这会改变 p 值吗？这改变了你的结论吗？

在下一课中，你将从数据中发现异常值。

## 课程 07：识别异常值

异常值是与大多数样本非常不同的样本，这使得它很难被视为更大群体的一部分。

识别异常值最著名的方法是正态分布的 68-95-99 规则，该规则指出距离均值一个、两个和三个标准差分别覆盖 68%、95%和 99%的样本。通常，一个距离均值 2 个标准差的样本被认为是足够远的异常值。我们来看看是否有哪个国家的预期寿命是异常值。

在使用 68-95-99 规则之前，你需要将数据转换得更接近正态分布。一个方法是使用 Box-Cox 变换。你可以通过比较变换前后的**偏度**来判断变换效果是否良好。完美的正态分布偏度为零：

```py
boxcox_life, lmbda = stats.boxcox(df_rural["life_expectancy"])
boxcox_life = pd.Series(boxcox_life)
print(df_rural["life_expectancy"].skew(), boxcox_life.skew())
```

在 Box-Cox 变换后，偏度从 0.137 变化为-0.006，更接近于零。通过 Box-Cox 变换计算的 lambda 值在后续会很有用。顺便提一下，你可以验证变换后的数据大致对称：

```py
plt.hist(boxcox_life, bins=15)
plt.show()
```

假设 Box-Cox 变换后的数据符合正态分布，我们可以很容易找到均值上下 2 个标准差的位置。但这只是变换后的数据。回忆一下 Box-Cox 变换是将*y*转换为 w=(*y*^λ – 1)/λ。因此，我们可以通过 (wλ + 1)^(1/λ) 进行反变换：

```py
mean, stdev = boxcox_life.mean(), boxcox_life.std()
plus2sd = mean + 2 * stdev
minus2sd = mean - 2 * stdev
upperthreshold = (plus2sd * lmbda + 1)**(1/lmbda)
lowerthreshold = (minus2sd * lmbda + 1)**(1/lmbda)
print(lowerthreshold, upperthreshold)
```

这些是对于具有更多农村人口的国家来说，**不**是异常值的下界和上界。我们来看看是否有国家在这个范围之外：

```py
print(df_rural[df_rural["life_expectancy"] <= lowerthreshold])
print(df_rural[df_rural["life_expectancy"] >= upperthreshold])
```

所以列支敦士登在上端是异常值，而乍得和莱索托在下端。这次测试只是指出了这些异常值，没有任何解释。你需要进一步研究数据，以假设为什么会出现这些情况。这是数据科学中的典型工作流程。

### 你的任务

你可以对`df_urban`重复这个操作，找出哪些城市国家是异常值。在上下端有多少个国家是异常值？

这就是最后一课。

## 结束了！(*看看你已经走了多远*)

你做到了。做得好！

花一点时间回顾一下你已经取得的进展。

+   你发现了 pandas、missingno、scipy、seaborn 和 matplotlib 作为帮助你完成数据科学项目的 Python 库。

+   使用基本统计学，你可以从数据集中探索洞察。你也可以从数据中确认你的假设。

+   你了解了如何通过散点图等可视化工具以及统计测试来探索数据。

+   你知道数据变换如何帮助你从数据中提取信息，比如发现异常值。

不要小看这一点；你在短时间内取得了长足的进步。这只是你数据科学旅程的开始。继续练习并提升你的技能。

## 总结

**你在这个迷你课程中表现如何？**

你喜欢这个速成课程吗？

**你有任何问题吗？有没有遇到什么难点？**

告诉我。请在下方留言。
