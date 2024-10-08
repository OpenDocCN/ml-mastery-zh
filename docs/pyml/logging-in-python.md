# Python 中的日志记录

> 原文：[`machinelearningmastery.com/logging-in-python/`](https://machinelearningmastery.com/logging-in-python/)

日志记录是一种存储有关脚本的信息并跟踪发生事件的方式。在编写任何复杂的 Python 脚本时，日志记录对于在开发过程中调试软件至关重要。没有日志记录，找到代码中的问题源可能会非常耗时。

完成本教程后，你将知道：

+   为什么我们要使用日志记录模块

+   如何使用日志记录模块

+   如何自定义日志记录机制

**用我的新书启动你的项目** [《Python 机器学习》](https://machinelearningmastery.com/python-for-machine-learning/)，包括*一步步的教程*和所有示例的*Python 源代码*文件。

让我们开始吧！[](../Images/2804cdbcac2294f5b99c95ee34942d0a.png)

Python 中的日志记录

图片由[ilaria88](https://www.pexels.com/photo/brown-firewood-122588/)提供。一些权利保留。

## **教程概述**

本教程分为四个部分；它们是：

+   日志记录的好处

+   基本日志记录

+   高级日志记录配置

+   一个使用日志记录的示例

## **日志记录的好处**

你可能会问：“为什么不直接使用打印？”

当你运行一个算法并想确认它是否按预期运行时，通常会在关键位置添加一些`print()`语句以显示程序的状态。打印可以帮助调试较简单的脚本，但随着代码变得越来越复杂，打印缺乏日志记录所具有的灵活性和鲁棒性。

使用日志记录，你可以精确地找出日志调用的来源，区分消息的严重级别，并将信息写入文件，这是打印无法做到的。例如，我们可以打开或关闭来自大型程序中特定模块的消息。我们还可以增加或减少日志消息的详细程度，而无需修改大量代码。

## **基本日志记录**

Python 有一个内置库`logging`，用于此目的。创建一个“记录器”以记录你希望看到的消息或信息非常简单。

Python 中的日志系统在一个层级命名空间和不同的严重级别下运行。Python 脚本可以在命名空间下创建一个记录器，每次记录消息时，脚本必须指定其严重级别。记录的消息可以根据我们为命名空间设置的处理程序发送到不同的位置。最常见的处理程序是简单地在屏幕上打印，比如无处不在的`print()`函数。当我们启动程序时，可以注册一个新处理程序，并设置该处理程序将响应的严重级别。

有 5 种不同的日志记录级别，表示日志的严重程度，按严重程度递增排列：

1.  DEBUG

1.  INFO

1.  WARNING

1.  ERROR

1.  CRITICAL

下面是一个非常简单的日志记录示例，使用默认记录器或根记录器：

```py
import logging

logging.debug('Debug message')
logging.info('Info message')
logging.warning('Warning message')
logging.error('Error message')
logging.critical('Critical message')
```

这些将发出不同严重性的日志消息。虽然有五行日志记录，如果你运行这个脚本，你可能只会看到三行输出，如下所示：

```py
WARNING:root:This is a warning message
ERROR:root:This is an error message
CRITICAL:root:This is a critical message
```

这是因为根日志记录器默认只打印警告级别或以上的日志消息。然而，以这种方式使用根日志记录器与使用 print()函数没有太大区别。

根日志记录器的设置不是一成不变的。我们可以将根日志记录器配置为输出到特定的文件，更改其默认的严重性级别，并格式化输出。以下是一个示例：

```py
import logging

logging.basicConfig(filename = 'file.log',
                    level = logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

logging.debug('Debug message')
logging.info('Info message')
logging.warning('Warning message')
logging.error('Error message')
logging.critical('Critical message')
```

运行这个脚本不会在屏幕上产生输出，但会在新创建的文件`file.log`中产生以下内容：

```py
2022-03-22 20:41:08,151:DEBUG:root:Debug message
2022-03-22 20:41:08,152:INFO:root:Info message
2022-03-22 20:41:08,152:WARNING:root:Warning message
2022-03-22 20:41:08,152:ERROR:root:Error message
2022-03-22 20:41:08,152:CRITICAL:root:Critical message
```

对`logging.basicConfig()`的调用是为了更改根日志记录器。在我们的示例中，我们将处理程序设置为输出到文件而不是屏幕，调整日志级别，以便处理所有级别为 DEBUG 或以上的日志消息，并且还更改日志消息输出的格式以包含时间。

请注意，现在所有五条消息都已输出，因此根日志记录器现在的默认级别为“DEBUG”。可以用来格式化输出的日志记录属性（例如`%(asctime)s`）可以在[日志记录文档](https://docs.python.org/3/library/logging.html#logrecord-attributes)中找到。

尽管有一个默认日志记录器，但我们通常希望创建和使用其他可以单独配置的日志记录器。这是因为我们可能希望不同的日志记录器有不同的严重性级别或格式。可以使用以下代码创建新的日志记录器：

```py
logger = logging.getLogger("logger_name")
```

内部日志记录器是以层级方式组织的。使用以下代码创建的日志记录器：

```py
logger = logging.getLogger("parent.child")
```

将会是一个在名为“`parent`”的日志记录器下创建的子日志记录器，而“`parent`”又在根日志记录器下。字符串中的点表示子日志记录器是父日志记录器的子日志记录器。在上面的情况下，将创建一个名为“`parent.child`”的日志记录器，以及一个名为`parent`的日志记录器（隐式创建）。

创建时，子日志记录器具有其父日志记录器的所有属性，直到重新配置。我们可以通过以下示例来演示这一点：

```py
import logging

# Create `parent.child` logger
logger = logging.getLogger("parent.child")

# Emit a log message of level INFO, by default this is not print to the screen
logger.info("this is info level")

# Create `parent` logger
parentlogger = logging.getLogger("parent")

# Set parent's level to INFO and assign a new handler
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s"))
parentlogger.setLevel(logging.INFO)
parentlogger.addHandler(handler)

# Let child logger emit a log message again
logger.info("this is info level again")
```

这段代码只会输出一行：

```py
2022-03-28 19:23:29,315:parent.child:INFO:this is info level again
```

这是由 StreamHandler 对象创建的，具有自定义格式字符串。这仅在我们重新配置`parent`的日志记录器后发生，否则根日志记录器的配置会占主导地位，并且不会打印级别为 INFO 的消息。

## **高级日志记录配置**

正如我们在上一个示例中看到的，我们可以配置我们创建的日志记录器。

### 级别阈值

与根日志记录器的基本配置一样，我们也可以配置日志记录器的输出目标、严重性级别和格式。以下是如何将日志记录器的**阈值**级别设置为 INFO：

```py
parent_logger = logging.getLogger("parent")
parent_logger.setLevel(logging.INFO)
```

现在，严重性级别为 INFO 及以上的命令将由 parent_logger 记录。但如果你仅做了这些，你将不会看到来自`parent_logger.info("messages")`的任何信息，因为没有为这个 logger 分配**处理器**。实际上，根 logger 也没有处理器，除非你通过`logging.basicConfig()`设置了一个。

### 日志处理器

我们可以通过处理器配置 logger 的输出目的地。处理器负责将日志消息发送到正确的目的地。有几种类型的处理器；最常见的是`StreamHandler`和`FileHandler`。使用`StreamHandler`时，logger 会输出到终端，而使用`FileHandler`时，logger 会输出到特定的文件。

下面是使用`StreamHandler`将日志输出到终端的示例：

```py
import logging

# Set up root logger, and add a file handler to root logger
logging.basicConfig(filename = 'file.log',
                    level = logging.WARNING,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

# Create logger, set level, and add stream handler
parent_logger = logging.getLogger("parent")
parent_logger.setLevel(logging.INFO)
parent_shandler = logging.StreamHandler()
parent_logger.addHandler(parent_shandler)

# Log message of severity INFO or above will be handled
parent_logger.debug('Debug message')
parent_logger.info('Info message')
parent_logger.warning('Warning message')
parent_logger.error('Error message')
parent_logger.critical('Critical message')
```

在上面的代码中，创建了两个处理器：一个由`logging.basicConfig()`为根 logger 创建的`FileHandler`，和一个为`parent` logger 创建的`StreamHandler`。

请注意，即使存在一个将日志发送到终端的`StreamHandler`，来自`parent` logger 的日志仍然被发送到`file.log`，因为它是根 logger 的子 logger，根 logger 的处理器也对子 logger 的日志消息有效。我们可以看到，发送到终端的日志包括 INFO 级别及以上的消息：

```py
Info message
Warning message
Error message
Critical message
```

但是终端的输出没有格式化，因为我们还没有使用`Formatter`。然而，`file.log`的日志已经设置了`Formatter`，它将像下面这样：

```py
2022-03-22 23:07:12,533:INFO:parent:Info message
2022-03-22 23:07:12,533:WARNING:parent:Warning message
2022-03-22 23:07:12,533:ERROR:parent:Error message
2022-03-22 23:07:12,533:CRITICAL:parent:Critical message
```

另外，我们可以在上述`parent_logger`的示例中使用`FileHandler`：

```py
import logging

# Set up root logger, and add a file handler to root logger
logging.basicConfig(filename = 'file.log',
                    level = logging.WARNING,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

# Create logger, set level, and add stream handler
parent_logger = logging.getLogger("parent")
parent_logger.setLevel(logging.INFO)
parent_fhandler = logging.FileHandler('parent.log')
parent_fhandler.setLevel(logging.WARNING)
parent_logger.addHandler(parent_fhandler)

# Log message of severity INFO or above will be handled
parent_logger.debug('Debug message')
parent_logger.info('Info message')
parent_logger.warning('Warning message')
parent_logger.error('Error message')
parent_logger.critical('Critical message')
```

上面的例子演示了你还可以设置处理器的级别。`parent_fhandler`的级别会过滤掉非 WARNING 级别或更高级别的日志。然而，如果你将处理器的级别设置为 DEBUG，那将等同于未设置级别，因为 DEBUG 日志会被 logger 的级别（即 INFO）过滤掉。

在这种情况下，输出到`parent.log`的是：

```py
Warning message
Error message
Critical message
```

而`file.log`的输出与之前相同。总之，当 logger 记录一个日志消息时，

1.  logger 的级别将决定消息是否足够严重以被处理。如果 logger 的级别未设置，则将使用其父级别（最终是根 logger）进行考虑。

1.  如果日志消息会被处理，**所有**在 logger 层次结构中添加过的处理器都会收到该消息的副本。每个处理器的级别将决定这个特定的处理器是否应该处理这个消息。

### **格式化器**

要配置 logger 的格式，我们使用`Formatter`。它允许我们设置日志的格式，类似于我们在根 logger 的`basicConfig()`中所做的。这是我们如何将 formatter 添加到处理器中的：

```py
import logging

# Set up root logger, and add a file handler to root logger
logging.basicConfig(filename = 'file.log',
                    level = logging.WARNING,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

# Create logger, set level, and add stream handler
parent_logger = logging.getLogger("parent")
parent_logger.setLevel(logging.INFO)
parent_fhandler = logging.FileHandler('parent.log')
parent_fhandler.setLevel(logging.WARNING)
parent_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
parent_fhandler.setFormatter(parent_formatter)
parent_logger.addHandler(parent_fhandler)

# Log message of severity INFO or above will be handled
parent_logger.debug('Debug message')
parent_logger.info('Info message')
parent_logger.warning('Warning message')
parent_logger.error('Error message')
parent_logger.critical('Critical message')
```

首先，我们创建一个 formatter，然后设置我们的处理器使用那个 formatter。如果我们愿意，我们可以创建多个不同的 loggers、handlers 和 formatters，以便根据我们的喜好进行组合。

在这个示例中，`parent.log`将会有：

```py
2022-03-23 13:28:31,302:WARNING:Warning message
2022-03-23 13:28:31,302:ERROR:Error message
2022-03-23 13:28:31,303:CRITICAL:Critical message
```

与根记录器相关联的`file.log`将会有：

```py
2022-03-23 13:28:31,302:INFO:parent:Info message
2022-03-23 13:28:31,302:WARNING:parent:Warning message
2022-03-23 13:28:31,302:ERROR:parent:Error message
2022-03-23 13:28:31,303:CRITICAL:parent:Critical message
```

以下是[日志模块文档](https://docs.python.org/3/howto/logging.html#logging-flow)中日志记录器、处理程序和格式化程序的流程可视化：

![](img/c3bba5e9d7619938a0c95ee9eac93ea5.png)

日志模块中记录器和处理程序的流程图

## 使用日志记录的示例

让我们以[Nadam 算法](https://machinelearningmastery.com/gradient-descent-optimization-with-nadam-from-scratch/)为例：

```py
# gradient descent optimization with nadam for a two-dimensional test function
from math import sqrt
from numpy import asarray
from numpy.random import rand
from numpy.random import seed

# objective function
def objective(x, y):
	return x**2.0 + y**2.0

# derivative of objective function
def derivative(x, y):
	return asarray([x * 2.0, y * 2.0])

# gradient descent algorithm with nadam
def nadam(objective, derivative, bounds, n_iter, alpha, mu, nu, eps=1e-8):
	# generate an initial point
	x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	score = objective(x[0], x[1])
	# initialize decaying moving averages
	m = [0.0 for _ in range(bounds.shape[0])]
	n = [0.0 for _ in range(bounds.shape[0])]
	# run the gradient descent
	for t in range(n_iter):
		# calculate gradient g(t)
		g = derivative(x[0], x[1])
		# build a solution one variable at a time
		for i in range(bounds.shape[0]):
			# m(t) = mu * m(t-1) + (1 - mu) * g(t)
			m[i] = mu * m[i] + (1.0 - mu) * g[i]
			# n(t) = nu * n(t-1) + (1 - nu) * g(t)²
			n[i] = nu * n[i] + (1.0 - nu) * g[i]**2
			# mhat = (mu * m(t) / (1 - mu)) + ((1 - mu) * g(t) / (1 - mu))
			mhat = (mu * m[i] / (1.0 - mu)) + ((1 - mu) * g[i] / (1.0 - mu))
			# nhat = nu * n(t) / (1 - nu)
			nhat = nu * n[i] / (1.0 - nu)
			# x(t) = x(t-1) - alpha / (sqrt(nhat) + eps) * mhat
			x[i] = x[i] - alpha / (sqrt(nhat) + eps) * mhat
		# evaluate candidate point
		score = objective(x[0], x[1])
		# report progress
		print('>%d f(%s) = %.5f' % (t, x, score))
	return [x, score]

# seed the pseudo random number generator
seed(1)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 50
# steps size
alpha = 0.02
# factor for average gradient
mu = 0.8
# factor for average squared gradient
nu = 0.999
# perform the gradient descent search with nadam
best, score = nadam(objective, derivative, bounds, n_iter, alpha, mu, nu)
print('Done!')
print('f(%s) = %f' % (best, score))
```

最简单的用例是使用日志记录替换`print()`函数。我们可以进行如下更改：首先，在运行任何代码之前创建一个名为`nadam`的记录器，并分配一个`StreamHandler`：

```py
...

import logging

...

# Added: Create logger and assign handler
logger = logging.getLogger("nadam")
handler  = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s|%(levelname)s|%(name)s|%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
# seed the pseudo random number generator
seed(1)
... # rest of the code
```

我们必须分配一个处理程序，因为我们从未配置根记录器，这将是唯一创建的处理程序。然后，在函数`nadam()`中，我们重新创建了一个记录器`nadam`，但由于它已经被设置过，级别和处理程序得以保留。在`nadam()`的每个外层 for 循环结束时，我们用`logger.info()`替换了`print()`函数，这样消息将由日志系统处理：

```py
...

# gradient descent algorithm with nadam
def nadam(objective, derivative, bounds, n_iter, alpha, mu, nu, eps=1e-8):
    # Create a logger
    logger = logging.getLogger("nadam")
    # generate an initial point
    x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    score = objective(x[0], x[1])
    # initialize decaying moving averages
    m = [0.0 for _ in range(bounds.shape[0])]
    n = [0.0 for _ in range(bounds.shape[0])]
    # run the gradient descent
    for t in range(n_iter):
        # calculate gradient g(t)
        g = derivative(x[0], x[1])
        # build a solution one variable at a time
        for i in range(bounds.shape[0]):
            # m(t) = mu * m(t-1) + (1 - mu) * g(t)
            m[i] = mu * m[i] + (1.0 - mu) * g[i]
            # n(t) = nu * n(t-1) + (1 - nu) * g(t)²
            n[i] = nu * n[i] + (1.0 - nu) * g[i]**2
            # mhat = (mu * m(t) / (1 - mu)) + ((1 - mu) * g(t) / (1 - mu))
            mhat = (mu * m[i] / (1.0 - mu)) + ((1 - mu) * g[i] / (1.0 - mu))
            # nhat = nu * n(t) / (1 - nu)
            nhat = nu * n[i] / (1.0 - nu)
            # x(t) = x(t-1) - alpha / (sqrt(nhat) + eps) * mhat
            x[i] = x[i] - alpha / (sqrt(nhat) + eps) * mhat
        # evaluate candidate point
        score = objective(x[0], x[1])
        # report progress using logger
        logger.info('>%d f(%s) = %.5f' % (t, x, score))
    return [x, score]

...
```

如果我们对 Nadam 算法的更深层次机制感兴趣，我们可以添加更多日志。以下是完整的代码：

```py
# gradient descent optimization with nadam for a two-dimensional test function
import logging
from math import sqrt
from numpy import asarray
from numpy.random import rand
from numpy.random import seed

# objective function
def objective(x, y):
    return x**2.0 + y**2.0

# derivative of objective function
def derivative(x, y):
    return asarray([x * 2.0, y * 2.0])

# gradient descent algorithm with nadam
def nadam(objective, derivative, bounds, n_iter, alpha, mu, nu, eps=1e-8):
    logger = logging.getLogger("nadam")
    # generate an initial point
    x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    score = objective(x[0], x[1])
    # initialize decaying moving averages
    m = [0.0 for _ in range(bounds.shape[0])]
    n = [0.0 for _ in range(bounds.shape[0])]
    # run the gradient descent
    for t in range(n_iter):
        iterlogger = logging.getLogger("nadam.iter")
        # calculate gradient g(t)
        g = derivative(x[0], x[1])
        # build a solution one variable at a time
        for i in range(bounds.shape[0]):
            # m(t) = mu * m(t-1) + (1 - mu) * g(t)
            m[i] = mu * m[i] + (1.0 - mu) * g[i]
            # n(t) = nu * n(t-1) + (1 - nu) * g(t)²
            n[i] = nu * n[i] + (1.0 - nu) * g[i]**2
            # mhat = (mu * m(t) / (1 - mu)) + ((1 - mu) * g(t) / (1 - mu))
            mhat = (mu * m[i] / (1.0 - mu)) + ((1 - mu) * g[i] / (1.0 - mu))
            # nhat = nu * n(t) / (1 - nu)
            nhat = nu * n[i] / (1.0 - nu)
            # x(t) = x(t-1) - alpha / (sqrt(nhat) + eps) * mhat
            x[i] = x[i] - alpha / (sqrt(nhat) + eps) * mhat
            iterlogger.info("Iteration %d variable %d: mhat=%f nhat=%f", t, i, mhat, nhat)
        # evaluate candidate point
        score = objective(x[0], x[1])
        # report progress
        logger.info('>%d f(%s) = %.5f' % (t, x, score))
    return [x, score]

# Create logger and assign handler
logger = logging.getLogger("nadam")
handler  = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s|%(levelname)s|%(name)s|%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger = logging.getLogger("nadam.iter")
logger.setLevel(logging.INFO)
# seed the pseudo random number generator
seed(1)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 50
# steps size
alpha = 0.02
# factor for average gradient
mu = 0.8
# factor for average squared gradient
nu = 0.999
# perform the gradient descent search with nadam
best, score = nadam(objective, derivative, bounds, n_iter, alpha, mu, nu)
print('Done!')
print('f(%s) = %f' % (best, score))
```

我们准备了两个级别的记录器，`nadam`和`nadam.iter`，并将它们设置在不同的级别。在`nadam()`的内层循环中，我们使用子记录器来打印一些内部变量。当你运行这个脚本时，它将打印如下内容：

```py
2022-03-29 12:24:59,421|INFO|nadam.iter|Iteration 0 variable 0: mhat=-0.597442 nhat=0.110055
2022-03-29 12:24:59,421|INFO|nadam.iter|Iteration 0 variable 1: mhat=1.586336 nhat=0.775909
2022-03-29 12:24:59,421|INFO|nadam|>0 f([-0.12993798  0.40463097]) = 0.18061
2022-03-29 12:24:59,421|INFO|nadam.iter|Iteration 1 variable 0: mhat=-0.680200 nhat=0.177413
2022-03-29 12:24:59,421|INFO|nadam.iter|Iteration 1 variable 1: mhat=2.020702 nhat=1.429384
2022-03-29 12:24:59,421|INFO|nadam|>1 f([-0.09764012  0.37082777]) = 0.14705
2022-03-29 12:24:59,421|INFO|nadam.iter|Iteration 2 variable 0: mhat=-0.687764 nhat=0.215332
2022-03-29 12:24:59,421|INFO|nadam.iter|Iteration 2 variable 1: mhat=2.304132 nhat=1.977457
2022-03-29 12:24:59,421|INFO|nadam|>2 f([-0.06799761  0.33805721]) = 0.11891
...
2022-03-29 12:24:59,449|INFO|nadam.iter|Iteration 49 variable 0: mhat=-0.000482 nhat=0.246709
2022-03-29 12:24:59,449|INFO|nadam.iter|Iteration 49 variable 1: mhat=-0.018244 nhat=3.966938
2022-03-29 12:24:59,449|INFO|nadam|>49 f([-5.54299505e-05 -1.00116899e-03]) = 0.00000
Done!
f([-5.54299505e-05 -1.00116899e-03]) = 0.000001
```

设置不同的日志记录器不仅允许我们设置不同的级别或处理程序，还可以通过查看打印消息中的记录器名称来区分日志消息的来源。

事实上，一个方便的技巧是创建一个日志装饰器，并将其应用于一些函数。我们可以跟踪每次调用该函数的情况。例如，我们下面创建了一个装饰器，并将其应用于`objective()`和`derivative()`函数：

```py
...

# A Python decorator to log the function call and return value
def loggingdecorator(name):
    logger = logging.getLogger(name)
    def _decor(fn):
        function_name = fn.__name__
        def _fn(*args, **kwargs):
            ret = fn(*args, **kwargs)
            argstr = [str(x) for x in args]
            argstr += [key+"="+str(val) for key,val in kwargs.items()]
            logger.debug("%s(%s) -> %s", function_name, ", ".join(argstr), ret)
            return ret
        return _fn
    return _decor

# objective function
@loggingdecorator("nadam.function")
def objective(x, y):
    return x**2.0 + y**2.0

# derivative of objective function
@loggingdecorator("nadam.function")
def derivative(x, y):
    return asarray([x * 2.0, y * 2.0])
```

然后我们将看到日志中出现如下内容：

```py
2022-03-29 13:14:07,542|DEBUG|nadam.function|objective(-0.165955990594852, 0.4406489868843162) -> 0.22171292045649288
2022-03-29 13:14:07,542|DEBUG|nadam.function|derivative(-0.165955990594852, 0.4406489868843162) -> [-0.33191198  0.88129797]
2022-03-29 13:14:07,542|INFO|nadam.iter|Iteration 0 variable 0: mhat=-0.597442 nhat=0.110055
2022-03-29 13:14:07,542|INFO|nadam.iter|Iteration 0 variable 1: mhat=1.586336 nhat=0.775909
2022-03-29 13:14:07,542|DEBUG|nadam.function|objective(-0.12993797816930272, 0.4046309737819536) -> 0.18061010311445824
2022-03-29 13:14:07,543|INFO|nadam|>0 f([-0.12993798  0.40463097]) = 0.18061
2022-03-29 13:14:07,543|DEBUG|nadam.function|derivative(-0.12993797816930272, 0.4046309737819536) -> [-0.25987596  0.80926195]
2022-03-29 13:14:07,543|INFO|nadam.iter|Iteration 1 variable 0: mhat=-0.680200 nhat=0.177413
2022-03-29 13:14:07,543|INFO|nadam.iter|Iteration 1 variable 1: mhat=2.020702 nhat=1.429384
2022-03-29 13:14:07,543|DEBUG|nadam.function|objective(-0.09764011794760165, 0.3708277653552375) -> 0.14704682419118062
2022-03-29 13:14:07,543|INFO|nadam|>1 f([-0.09764012  0.37082777]) = 0.14705
2022-03-29 13:14:07,543|DEBUG|nadam.function|derivative(-0.09764011794760165, 0.3708277653552375) -> [-0.19528024  0.74165553]
2022-03-29 13:14:07,543|INFO|nadam.iter|Iteration 2 variable 0: mhat=-0.687764 nhat=0.215332
...
```

在日志消息中，我们可以看到每次调用这两个函数的参数和返回值，这些信息由`nadam.function`记录器记录。

### 想要开始使用 Python 进行机器学习吗？

立即参加我的免费 7 天电子邮件速成课程（附有示例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

随着日志信息越来越多，终端屏幕将变得非常繁忙。为了更容易地监视问题，可以使用`colorama`模块来高亮显示日志。你需要首先安装这个模块：

```py
pip install colorama
```

下面是一个如何使用`colorama`模块与`logging`模块结合，以改变日志颜色和文本亮度的示例：

```py
import logging
import colorama
from colorama import Fore, Back, Style

# Initialize the terminal for color
colorama.init(autoreset = True)

# Set up logger as usual
logger = logging.getLogger("color")
logger.setLevel(logging.DEBUG)
shandler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
shandler.setFormatter(formatter)
logger.addHandler(shandler)

# Emit log message with color
logger.debug('Debug message')
logger.info(Fore.GREEN + 'Info message')
logger.warning(Fore.BLUE + 'Warning message')
logger.error(Fore.YELLOW + Style.BRIGHT + 'Error message')
logger.critical(Fore.RED + Back.YELLOW + Style.BRIGHT + 'Critical message')
```

从终端中，你将看到如下内容：![](img/9f26ce61f0915bbf238b4282945e7703.png)

其中，`colorama`模块中的`Fore`、`Back`和`Style`控制文本的前景、背景和亮度样式。这利用了 ANSI 转义字符，仅在支持 ANSI 的终端上有效。因此，这不适用于日志记录到文本文件。

实际上，我们可以通过以下方式派生`Formatter`类：

```py
...
colors = {"DEBUG":Fore.BLUE, "INFO":Fore.CYAN,
          "WARNING":Fore.YELLOW, "ERROR":Fore.RED, "CRITICAL":Fore.MAGENTA}
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.levelname in colors:
            msg = colors[record.levelname] + msg + Fore.RESET
        return msg
```

并使用它来替代`logging.Formatter`。以下是我们如何进一步修改 Nadam 示例以添加颜色：

```py
# gradient descent optimization with nadam for a two-dimensional test function
import logging
import colorama
from colorama import Fore

from math import sqrt
from numpy import asarray
from numpy.random import rand
from numpy.random import seed

def loggingdecorator(name):
    logger = logging.getLogger(name)
    def _decor(fn):
        function_name = fn.__name__
        def _fn(*args, **kwargs):
            ret = fn(*args, **kwargs)
            argstr = [str(x) for x in args]
            argstr += [key+"="+str(val) for key,val in kwargs.items()]
            logger.debug("%s(%s) -> %s", function_name, ", ".join(argstr), ret)
            return ret
        return _fn
    return _decor

# objective function
@loggingdecorator("nadam.function")
def objective(x, y):
    return x**2.0 + y**2.0

# derivative of objective function
@loggingdecorator("nadam.function")
def derivative(x, y):
    return asarray([x * 2.0, y * 2.0])

# gradient descent algorithm with nadam
def nadam(objective, derivative, bounds, n_iter, alpha, mu, nu, eps=1e-8):
    logger = logging.getLogger("nadam")
    # generate an initial point
    x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    score = objective(x[0], x[1])
    # initialize decaying moving averages
    m = [0.0 for _ in range(bounds.shape[0])]
    n = [0.0 for _ in range(bounds.shape[0])]
    # run the gradient descent
    for t in range(n_iter):
        iterlogger = logging.getLogger("nadam.iter")
        # calculate gradient g(t)
        g = derivative(x[0], x[1])
        # build a solution one variable at a time
        for i in range(bounds.shape[0]):
            # m(t) = mu * m(t-1) + (1 - mu) * g(t)
            m[i] = mu * m[i] + (1.0 - mu) * g[i]
            # n(t) = nu * n(t-1) + (1 - nu) * g(t)²
            n[i] = nu * n[i] + (1.0 - nu) * g[i]**2
            # mhat = (mu * m(t) / (1 - mu)) + ((1 - mu) * g(t) / (1 - mu))
            mhat = (mu * m[i] / (1.0 - mu)) + ((1 - mu) * g[i] / (1.0 - mu))
            # nhat = nu * n(t) / (1 - nu)
            nhat = nu * n[i] / (1.0 - nu)
            # x(t) = x(t-1) - alpha / (sqrt(nhat) + eps) * mhat
            x[i] = x[i] - alpha / (sqrt(nhat) + eps) * mhat
            iterlogger.info("Iteration %d variable %d: mhat=%f nhat=%f", t, i, mhat, nhat)
        # evaluate candidate point
        score = objective(x[0], x[1])
        # report progress
        logger.warning('>%d f(%s) = %.5f' % (t, x, score))
    return [x, score]

# Prepare the colored formatter
colorama.init(autoreset = True)
colors = {"DEBUG":Fore.BLUE, "INFO":Fore.CYAN,
          "WARNING":Fore.YELLOW, "ERROR":Fore.RED, "CRITICAL":Fore.MAGENTA}
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.levelname in colors:
            msg = colors[record.levelname] + msg + Fore.RESET
        return msg

# Create logger and assign handler
logger = logging.getLogger("nadam")
handler  = logging.StreamHandler()
handler.setFormatter(ColoredFormatter("%(asctime)s|%(levelname)s|%(name)s|%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger = logging.getLogger("nadam.iter")
logger.setLevel(logging.DEBUG)
# seed the pseudo random number generator
seed(1)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 50
# steps size
alpha = 0.02
# factor for average gradient
mu = 0.8
# factor for average squared gradient
nu = 0.999
# perform the gradient descent search with nadam
best, score = nadam(objective, derivative, bounds, n_iter, alpha, mu, nu)
print('Done!')
print('f(%s) = %f' % (best, score))
```

如果我们在支持的终端上运行它，我们将看到以下输出：

![](img/71e97ab6847210d213d24abea52e7240.png)

注意，多彩的输出可以帮助我们更容易地发现任何异常行为。日志记录有助于调试，并且允许我们通过仅更改几行代码轻松控制我们想要查看的详细程度。

## 进一步阅读

本节提供了更多关于该主题的资源，如果你想深入了解。

### API

+   [日志模块](https://docs.python.org/3/library/logging.html)来自 Python 标准库

+   [Colorama](https://github.com/tartley/colorama)

### 文章

+   [Python 日志记录 HOWTO](http://,https://docs.python.org/3/howto/logging.html)

## **总结**

在本教程中，你学习了如何在脚本中实现日志记录技术。

具体来说，你学到了：

+   基本和高级日志记录技术

+   如何将日志记录应用于脚本以及这样做的好处
