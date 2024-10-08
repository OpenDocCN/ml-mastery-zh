# 管理机器学习项目的数据

> 原文：[`machinelearningmastery.com/managing-data-for-machine-learning-project/`](https://machinelearningmastery.com/managing-data-for-machine-learning-project/)

大数据、标记数据、噪声数据。机器学习项目都需要查看数据。数据是机器学习项目的一个关键方面，我们如何处理这些数据是我们项目的重要考虑因素。当数据量增长，需要管理这些数据，或让它们服务于多个项目，或者只是需要更好的数据检索方式时，考虑使用数据库系统是很自然的。这可以是关系型数据库或平面文件格式。它可以是本地的，也可以是远程的。

在这篇文章中，我们探讨了可以用于在 Python 中存储和检索数据的不同格式和库。

完成本教程后，你将学会：

+   使用 SQLite、Python dbm 库、Excel 和 Google Sheets 管理数据

+   如何使用外部存储的数据来训练你的机器学习模型

+   在机器学习项目中使用数据库的优缺点

**启动你的项目**，可以参考我新书 [《Python 机器学习》](https://machinelearningmastery.com/python-for-machine-learning/)，其中包含 *一步一步的教程* 和所有示例的 *Python 源代码* 文件。

让我们开始吧!![](img/c04dc999bff0167a744a6febecdbbee6.png)

使用 Python 管理数据

图片由 [比尔·本宗](https://www.flickr.com/photos/stc4blues/25260822078/) 提供。一些权利保留。

## 概述

本教程分为七个部分；它们是：

+   在 SQLite 中管理数据

+   SQLite 的实际应用

+   在 dbm 中管理数据

+   在机器学习流程中使用 dbm 数据库

+   在 Excel 中管理数据

+   在 Google Sheets 中管理数据

+   数据库的其他用途

## 在 SQLite 中管理数据

当我们提到数据库时，它通常指的是以表格格式存储数据的关系型数据库。

首先，让我们从 `sklearn.dataset` 获取一个表格数据集（要了解更多关于获取机器学习数据集的信息，请查看我们的 [上一篇文章](https://machinelearningmastery.com/a-guide-to-getting-datasets-for-machine-learning-in-python/)）。

```py
# Read dataset from OpenML
from sklearn.datasets import fetch_openml
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
```

上述代码读取了来自 OpenML 的“皮马印第安人糖尿病数据集”并创建了一个 pandas DataFrame。这是一个具有多个数值特征和一个二元类别标签的分类数据集。我们可以使用以下方法探索 DataFrame：

```py
print(type(dataset))
print(dataset.head())
```

这为我们提供了：

```py
<class 'pandas.core.frame.DataFrame'>
   preg   plas  pres  skin   insu  mass   pedi   age            class
0   6.0  148.0  72.0  35.0    0.0  33.6  0.627  50.0  tested_positive
1   1.0   85.0  66.0  29.0    0.0  26.6  0.351  31.0  tested_negative
2   8.0  183.0  64.0   0.0    0.0  23.3  0.672  32.0  tested_positive
3   1.0   89.0  66.0  23.0   94.0  28.1  0.167  21.0  tested_negative
4   0.0  137.0  40.0  35.0  168.0  43.1  2.288  33.0  tested_positive
```

这不是一个非常大的数据集，但如果它太大，我们可能无法将其放入内存。关系数据库是帮助我们高效管理表格数据的工具，而不必将所有内容保留在内存中。通常，关系数据库将理解 SQL 的一个方言，这是一种描述对数据进行操作的语言。SQLite 是一种无服务器数据库系统，不需要任何设置，在 Python 中有内置库支持。接下来，我们将演示如何利用 SQLite 来管理数据，但使用其他数据库如 MariaDB 或 PostgreSQL 也是非常相似的。

现在，让我们从在 SQLite 中创建一个内存数据库开始，并获取一个游标对象，以便我们可以执行对我们新数据库的查询：

```py
import sqlite3

conn = sqlite3.connect(":memory:")
cur = conn.cursor()
```

如果我们想把数据存储在硬盘上，以便稍后重复使用或与另一个程序共享，我们可以将数据库存储在数据库文件中，而不是像上面的代码片段中替换魔术字符串`:memory:`一样，可以用文件名（例如`example.db`）来替换，如下所示：

```py
conn = sqlite3.connect("example.db")
```

现在，让我们继续为我们的糖尿病数据创建一个新表。

```py
...
create_sql = """
    CREATE TABLE diabetes(
        preg NUM,
        plas NUM, 
        pres NUM,
        skin NUM,
        insu NUM,
        mass NUM,
        pedi NUM,
        age NUM,
        class TEXT
    )
"""
cur.execute(create_sql)
```

`cur.execute()`方法执行我们传递给它作为参数的 SQL 查询。在本例中，SQL 查询创建了`diabetes`表，其中包含不同的列及其各自的数据类型。SQL 的语言不在此描述，但您可以从许多数据库书籍和课程中了解更多。

接下来，我们可以继续将存储在 pandas DataFrame 中的糖尿病数据集中的数据插入到我们新创建的糖尿病表中，这个表位于我们的内存 SQL 数据库中。

```py
# Prepare a parameterized SQL for insert
insert_sql = "INSERT INTO diabetes VALUES (?,?,?,?,?,?,?,?,?)"
# execute the SQL multiple times with each element in dataset.to_numpy().tolist()
cur.executemany(insert_sql, dataset.to_numpy().tolist())
```

让我们分解上述代码：`dataset.to_numpy().tolist()`给我们提供了`dataset`中数据的行列表，我们将其作为参数传递给`cur.executemany()`。然后，`cur.executemany()`多次运行 SQL 语句，每次使用从`dataset.to_numpy().tolist()`得到的数据行，这是从`dataset`中获取的数据行。参数化的 SQL 期望每次传递一个值列表，因此我们应该将列表的列表传递给`executemany()`，这就是`dataset.to_numpy().tolist()`创建的内容。

现在，我们可以检查确认所有数据都存储在数据库中：

```py
import pandas as pd

def cursor2dataframe(cur):
    """Read the column header from the cursor and then the rows of
    data from it. Afterwards, create a DataFrame"""
    header = [x[0] for x in cur.description]
    # gets data from the last executed SQL query
    data = cur.fetchall()
    # convert the data into a pandas DataFrame
    return pd.DataFrame(data, columns=header)

# get 5 random rows from the diabetes table
select_sql = "SELECT * FROM diabetes ORDER BY random() LIMIT 5"
cur.execute(select_sql)
sample = cursor2dataframe(cur)
print(sample)
```

在上述代码中，我们使用 SQL 中的`SELECT`语句来查询`diabetes`表的 5 行随机数据。结果将作为元组列表返回（每行一个元组）。然后，我们通过为每列分配一个名称，将元组列表转换为 pandas DataFrame。运行上面的代码片段，我们会得到这个输出：

```py
   preg  plas  pres  skin  insu  mass   pedi  age            class
0     2    90    68    42     0  38.2  0.503   27  tested_positive
1     9   124    70    33   402  35.4  0.282   34  tested_negative
2     7   160    54    32   175  30.5  0.588   39  tested_positive
3     7   105     0     0     0   0.0  0.305   24  tested_negative
4     1   107    68    19     0  26.5  0.165   24  tested_negative
```

这里是使用`sqlite3`创建、插入和检索糖尿病数据集中样本的完整代码：

```py
import sqlite3

import pandas as pd
from sklearn.datasets import fetch_openml

# Read dataset from OpenML
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
print("Data from OpenML:")
print(type(dataset))
print(dataset.head())

# Create database
conn = sqlite3.connect(":memory:")
cur = conn.cursor()
create_sql = """
    CREATE TABLE diabetes(
        preg NUM,
        plas NUM,
        pres NUM,
        skin NUM,
        insu NUM,
        mass NUM,
        pedi NUM,
        age NUM,
        class TEXT
    )
"""
cur.execute(create_sql)

# Insert data into the table using a parameterized SQL
insert_sql = "INSERT INTO diabetes VALUES (?,?,?,?,?,?,?,?,?)"
rows = dataset.to_numpy().tolist()
cur.executemany(insert_sql, rows)

def cursor2dataframe(cur):
    """Read the column header from the cursor and then the rows of
    data from it. Afterwards, create a DataFrame"""
    header = [x[0] for x in cur.description]
    # gets data from the last executed SQL query
    data = cur.fetchall()
    # convert the data into a pandas DataFrame
    return pd.DataFrame(data, columns=header)

# get 5 random rows from the diabetes table
select_sql = "SELECT * FROM diabetes ORDER BY random() LIMIT 5"
cur.execute(select_sql)
sample = cursor2dataframe(cur)
print("Data from SQLite database:")
print(sample)

# close database connection
conn.commit()
conn.close()
```

使用数据库的好处在于，当数据集不是从互联网获取，而是由你随时间收集时，会更加明显。例如，你可能会在多天内从传感器收集数据。你可以通过自动化任务将每小时收集的数据写入数据库。然后，你的机器学习项目可以使用数据库中的数据集运行，你可能会看到随着数据的积累而不同的结果。

让我们看看如何将我们的关系数据库构建到机器学习流程中！

## SQLite 实战

现在我们已经探讨了如何使用 sqlite3 存储和检索数据，我们可能会对如何将其集成到我们的机器学习流程中感兴趣。

通常，在这种情况下，我们会有一个过程来收集数据并将其写入数据库（例如，从传感器读取数据多天）。这将类似于前一节中的代码，只不过我们更愿意将数据库写入磁盘以便持久存储。然后，我们将从数据库中读取数据用于机器学习过程，无论是训练还是预测。根据模型的不同，有不同的方式来使用数据。让我们考虑 Keras 中的一个二分类模型用于糖尿病数据集。我们可以构建一个生成器，从数据库中读取随机批次的数据：

```py
def datagen(batch_size):
    conn = sqlite3.connect("diabetes.db", check_same_thread=False)
    cur = conn.cursor()
    sql = f"""
        SELECT preg, plas, pres, skin, insu, mass, pedi, age, class
        FROM diabetes
        ORDER BY random()
        LIMIT {batch_size}
    """
    while True:
        cur.execute(sql)
        data = cur.fetchall()
        X = [row[:-1] for row in data]
        y = [1 if row[-1]=="tested_positive" else 0 for row in data]
        yield np.asarray(X), np.asarray(y)
```

上述代码是一个生成器函数，它从 SQLite 数据库中获取`batch_size`数量的行，并将其作为 NumPy 数组返回。我们可以使用来自这个生成器的数据在我们的分类网络中进行训练：

```py
from keras.models import Sequential
from keras.layers import Dense

# create binary classification model
model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
history = model.fit(datagen(32), epochs=5, steps_per_epoch=2000)
```

运行上述代码会给我们以下输出：

```py
Epoch 1/5
2000/2000 [==============================] - 6s 3ms/step - loss: 2.2360 - accuracy: 0.6730
Epoch 2/5
2000/2000 [==============================] - 5s 2ms/step - loss: 0.5292 - accuracy: 0.7380
Epoch 3/5
2000/2000 [==============================] - 5s 2ms/step - loss: 0.4936 - accuracy: 0.7564
Epoch 4/5
2000/2000 [==============================] - 5s 2ms/step - loss: 0.4751 - accuracy: 0.7662
Epoch 5/5
2000/2000 [==============================] - 5s 2ms/step - loss: 0.4487 - accuracy: 0.7834
```

请注意，我们在生成器函数中只读取一个批次，而不是全部数据。我们依赖数据库提供数据，而不关心数据库中数据集的大小。虽然 SQLite 不是一个客户端-服务器数据库系统，因此不适合网络扩展，但还有其他数据库系统可以做到这一点。因此，你可以想象在只提供有限内存的情况下使用异常庞大的数据集进行机器学习应用。

以下是完整的代码，从准备数据库到使用实时读取的数据训练 Keras 模型：

```py
import sqlite3

import numpy as np
from sklearn.datasets import fetch_openml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create database
conn = sqlite3.connect("diabetes.db")
cur = conn.cursor()
cur.execute("DROP TABLE IF EXISTS diabetes")
create_sql = """
    CREATE TABLE diabetes(
        preg NUM,
        plas NUM,
        pres NUM,
        skin NUM,
        insu NUM,
        mass NUM,
        pedi NUM,
        age NUM,
        class TEXT
    )
"""
cur.execute(create_sql)

# Read data from OpenML, insert data into the table using a parameterized SQL
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
insert_sql = "INSERT INTO diabetes VALUES (?,?,?,?,?,?,?,?,?)"
rows = dataset.to_numpy().tolist()
cur.executemany(insert_sql, rows)

# Commit to flush change to disk, then close connection
conn.commit()
conn.close()

# Create data generator for Keras classifier model
def datagen(batch_size):
    """A generator to produce samples from database
    """
    # Tensorflow may run in different thread, thus needs check_same_thread=False
    conn = sqlite3.connect("diabetes.db", check_same_thread=False)
    cur = conn.cursor()
    sql = f"""
        SELECT preg, plas, pres, skin, insu, mass, pedi, age, class
        FROM diabetes
        ORDER BY random()
        LIMIT {batch_size}
    """
    while True:
        # Read rows from database
        cur.execute(sql)
        data = cur.fetchall()
        # Extract features
        X = [row[:-1] for row in data]
        # Extract targets, encode into binary (0 or 1)
        y = [1 if row[-1]=="tested_positive" else 0 for row in data]
        yield np.asarray(X), np.asarray(y)

# create binary classification model
model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
history = model.fit(datagen(32), epochs=5, steps_per_epoch=2000)
```

在进入下一部分之前，我们应该强调所有数据库都有些许不同。我们使用的 SQL 语句在其他数据库实现中可能并不最优。此外，请注意，SQLite 并不非常先进，因为它的目标是成为一个不需要服务器设置的数据库。使用大型数据库及其优化使用是一个重要话题，但这里演示的概念仍然适用。

### 想要开始使用 Python 进行机器学习吗？

立即参加我的免费 7 天电子邮件速成课程（附样例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## 在 dbm 中管理数据

关系型数据库非常适合表格数据，但并非所有数据集都采用表格结构。有时，数据最适合存储在类似于 Python 字典的结构中，即键值对存储。存在许多键值对数据存储。MongoDB 可能是最著名的一种，它需要像 PostgreSQL 一样进行服务器部署。GNU dbm 是一种无服务器存储，类似于 SQLite，并且几乎每个 Linux 系统中都已安装。在 Python 的标准库中，我们有`dbm`模块来处理它。

让我们来探索一下 Python 的 `dbm` 库。该库支持两种不同的 dbm 实现：GNU dbm 和 ndbm。如果系统中未安装这两种实现，则有 Python 自己的实现作为备用。无论底层的 dbm 实现如何，我们在 Python 程序中使用相同的语法。

这次，我们将演示如何使用 scikit-learn 的数字数据集：

```py
import sklearn.datasets

# get digits dataset (8x8 images of digits)
digits = sklearn.datasets.load_digits()
```

`dbm`库使用类似字典的接口来存储和检索来自 dbm 文件的数据，将键映射到值，其中键和值都是字符串。将数字数据集存储在文件 `digits.dbm` 中的代码如下：

```py
import dbm
import pickle

# create file if not exists, otherwise open for read/write
with dbm.open("digits.dbm", "c") as db:
    for idx in range(len(digits.target)):
        db[str(idx)] = pickle.dumps((digits.images[idx], digits.target[idx]))
```

上述代码片段会在 `digits.dbm` 文件不存在时创建一个新文件。然后我们从 `digits.images` 中选择每个数字图像，从 `digits.target` 中选择标签，并创建一个元组。我们使用数据的偏移量作为键，将元组的 pickle 字符串作为值存储在数据库中。与 Python 的字典不同，dbm 只允许字符串键和序列化值。因此，我们使用 `str(idx)` 将键转换为字符串，并仅存储 pickle 数据。

你可以在我们的[上一篇文章](https://machinelearningmastery.com/a-gentle-introduction-to-serialization-for-python/)中了解更多关于序列化的内容。

以下是如何从数据库中读取数据：

```py
import random
import numpy as np

# number of images that we want in our sample
batchsize = 4
images = []
targets = []

# open the database and read a sample
with dbm.open("digits.dbm", "r") as db:
    # get all keys from the database
    keys = db.keys()
    # randomly samples n keys
    for key in random.sample(keys, batchsize):
        # go through each key in the random sample
        image, target = pickle.loads(db[key])
        images.append(image)
        targets.append(target)
    print(np.asarray(images), np.asarray(targets))
```

在上述代码片段中，我们从数据库中获取 4 个随机键，然后获取它们的对应值，并使用 `pickle.loads()` 进行反序列化。正如我们所知，反序列化的数据将是一个元组；我们将其分配给变量 `image` 和 `target`，然后将每个随机样本收集到列表 `images` 和 `targets` 中。为了方便在 scikit-learn 或 Keras 中进行训练，我们通常更喜欢将整个批次作为 NumPy 数组。

运行上述代码会得到以下输出：

```py
[[[ 0\.  0\.  1\.  9\. 14\. 11\.  1\.  0.]
  [ 0\.  0\. 10\. 15\.  9\. 13\.  5\.  0.]
  [ 0\.  3\. 16\.  7\.  0\.  0\.  0\.  0.]
  [ 0\.  5\. 16\. 16\. 16\. 10\.  0\.  0.]
  [ 0\.  7\. 16\. 11\. 10\. 16\.  5\.  0.]
  [ 0\.  2\. 16\.  5\.  0\. 12\.  8\.  0.]
  [ 0\.  0\. 10\. 15\. 13\. 16\.  5\.  0.]
  [ 0\.  0\.  0\.  9\. 12\.  7\.  0\.  0.]]
...
] [6 8 7 3]
```

综合起来，这就是检索数字数据集的代码，然后创建、插入和从 dbm 数据库中采样的代码：

```py
import dbm
import pickle
import random

import numpy as np
import sklearn.datasets

# get digits dataset (8x8 images of digits)
digits = sklearn.datasets.load_digits()

# create file if not exists, otherwise open for read/write
with dbm.open("digits.dbm", "c") as db:
    for idx in range(len(digits.target)):
        db[str(idx)] = pickle.dumps((digits.images[idx], digits.target[idx]))

# number of images that we want in our sample
batchsize = 4
images = []
targets = []

# open the database and read a sample
with dbm.open("digits.dbm", "r") as db:
    # get all keys from the database
    keys = db.keys()
    # randomly samples n keys
    for key in random.sample(keys, batchsize):
        # go through each key in the random sample
        image, target = pickle.loads(db[key])
        images.append(image)
        targets.append(target)
    print(np.array(images), np.array(targets))
```

接下来，让我们看看如何在我们的机器学习管道中使用新创建的 dbm 数据库！

## 在机器学习管道中使用 dbm 数据库

在这里，你可能意识到我们可以为数字分类创建生成器和 Keras 模型，就像在 SQLite 数据库示例中所做的那样。以下是如何修改代码的步骤。首先是我们的生成器函数。我们只需在循环中选择一个随机批次的键并从 dbm 存储中获取数据：

```py
def datagen(batch_size):
    """A generator to produce samples from database
    """
    with dbm.open("digits.dbm", "r") as db:
        keys = db.keys()
        while True:
            images = []
            targets = []
            for key in random.sample(keys, batch_size):
                image, target = pickle.loads(db[key])
                images.append(image)
                targets.append(target)
            yield np.array(images).reshape(-1,64), np.array(targets)
```

然后，我们可以为数据创建一个简单的 MLP 模型：

```py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(32, input_dim=64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["sparse_categorical_accuracy"])

history = model.fit(datagen(32), epochs=5, steps_per_epoch=1000)
```

运行上述代码会产生以下输出：

```py
Epoch 1/5
1000/1000 [==============================] - 3s 2ms/step - loss: 0.6714 - sparse_categorical_accuracy: 0.8090
Epoch 2/5
1000/1000 [==============================] - 2s 2ms/step - loss: 0.1049 - sparse_categorical_accuracy: 0.9688
Epoch 3/5
1000/1000 [==============================] - 2s 2ms/step - loss: 0.0442 - sparse_categorical_accuracy: 0.9875
Epoch 4/5
1000/1000 [==============================] - 2s 2ms/step - loss: 0.0484 - sparse_categorical_accuracy: 0.9850
Epoch 5/5
1000/1000 [==============================] - 2s 2ms/step - loss: 0.0245 - sparse_categorical_accuracy: 0.9935
```

这就是我们如何使用 dbm 数据库来训练 MLP 以处理数字数据集的。使用 dbm 训练模型的完整代码在这里：

```py
import dbm
import pickle
import random

import numpy as np
import sklearn.datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# get digits dataset (8x8 images of digits)
digits = sklearn.datasets.load_digits()

# create file if not exists, otherwise open for read/write
with dbm.open("digits.dbm", "c") as db:
    for idx in range(len(digits.target)):
        db[str(idx)] = pickle.dumps((digits.images[idx], digits.target[idx]))

# retrieving data from database for model
def datagen(batch_size):
    """A generator to produce samples from database
    """
    with dbm.open("digits.dbm", "r") as db:
        keys = db.keys()
        while True:
            images = []
            targets = []
            for key in random.sample(keys, batch_size):
                image, target = pickle.loads(db[key])
                images.append(image)
                targets.append(target)
            yield np.array(images).reshape(-1,64), np.array(targets)

# Classification model in Keras
model = Sequential()
model.add(Dense(32, input_dim=64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["sparse_categorical_accuracy"])

# Train with data from dbm store
history = model.fit(datagen(32), epochs=5, steps_per_epoch=1000)
```

在更高级的系统如 MongoDB 或 Couchbase 中，我们可以简单地要求数据库系统为我们读取随机记录，而不是从所有键的列表中选择随机样本。但思想仍然是相同的；我们可以依赖外部存储来保存我们的数据并管理数据集，而不是在我们的 Python 脚本中完成。

## 在 Excel 中管理数据

有时候，内存不是我们将数据保存在机器学习脚本之外的原因。原因是有更好的工具来处理数据。也许我们希望拥有能够在屏幕上显示所有数据并允许我们滚动、添加格式和高亮等工具。或者我们希望与不关心我们 Python 程序的其他人分享数据。在需要使用关系数据库的情况下，人们常常使用 Excel 来管理数据。虽然 Excel 可以读取和导出 CSV 文件，但我们可能希望直接处理 Excel 文件。

在 Python 中，有几个库可以处理 Excel 文件，OpenPyXL 是其中最著名的一个。在使用之前，我们需要安装这个库：

```py
pip install openpyxl
```

目前，Excel 使用的格式是“Open XML Spreadsheet”，文件名以 `.xlsx` 结尾。较旧的 Excel 文件是以二进制格式保存，文件名后缀为 `.xls`，这不被 OpenPyXL 支持（在这种情况下，你可以使用 `xlrd` 和 `xlwt` 模块进行读写）。

让我们考虑一下在 SQLite 的情况下使用的相同示例。我们可以打开一个新的 Excel 工作簿，将我们的糖尿病数据集作为工作表写入：

```py
import pandas as pd
from sklearn.datasets import fetch_openml
import openpyxl

# Read dataset from OpenML
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
header = list(dataset.columns)
data = dataset.to_numpy().tolist()

# Create Excel workbook and write data into the default worksheet
wb = openpyxl.Workbook()
sheet = wb.active # use the default worksheet
sheet.title = "Diabetes"
for n,colname in enumerate(header):
    sheet.cell(row=1, column=1+n, value=colname)
for n,row in enumerate(data):
    for m,cell in enumerate(row):
        sheet.cell(row=2+n, column=1+m, value=cell)
# Save
wb.save("MLM.xlsx")
```

上述代码用于为工作表中的每个单元格准备数据（由行和列指定）。当我们创建一个新的 Excel 文件时，默认会有一个工作表。然后，单元格通过行和列偏移来标识，从 1 开始。我们使用以下语法向单元格写入数据：

```py
sheet.cell(row=3, column=4, value="my data")
```

要从单元格中读取数据，我们使用：

```py
sheet.cell(row=3, column=4).value
```

一次一个单元格地向 Excel 写入数据是繁琐的，实际上我们可以逐行添加数据。以下是我们如何修改上述代码以逐行操作而不是逐单元格操作：

```py
import pandas as pd
from sklearn.datasets import fetch_openml
import openpyxl

# Read dataset from OpenML
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
header = list(dataset.columns)
data = dataset.to_numpy().tolist()

# Create Excel workbook and write data into the default worksheet
wb = openpyxl.Workbook()
sheet = wb.create_sheet("Diabetes")  # or wb.active for default sheet
sheet.append(header)
for row in data:
    sheet.append(row)
# Save
wb.save("MLM.xlsx")
```

一旦我们将数据写入文件后，我们可以使用 Excel 直观地浏览数据、添加格式等：![](img/d720837cd9e730ea9c4fab3e5f3b9303.png)

将其用于机器学习项目并不比使用 SQLite 数据库更难。以下是 Keras 中相同的二分类模型，但生成器从 Excel 文件中读取数据：

```py
import random

import numpy as np
import openpyxl
from sklearn.datasets import fetch_openml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Read data from OpenML
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
header = list(dataset.columns)
rows = dataset.to_numpy().tolist()

# Create Excel workbook and write data into the default worksheet
wb = openpyxl.Workbook()
sheet = wb.active
sheet.title = "Diabetes"
sheet.append(header)
for row in rows:
    sheet.append(row)
# Save
wb.save("MLM.xlsx")

# Create data generator for Keras classifier model
def datagen(batch_size):
    """A generator to produce samples from database
    """
    wb = openpyxl.load_workbook("MLM.xlsx", read_only=True)
    sheet = wb.active
    maxrow = sheet.max_row
    while True:
        # Read rows from Excel file
        X = []
        y = []
        for _ in range(batch_size):
            # data starts at row 2
            row_num = random.randint(2, maxrow)
            rowdata = [cell.value for cell in sheet[row_num]]
            X.append(rowdata[:-1])
            y.append(1 if rowdata[-1]=="tested_positive" else 0)
        yield np.asarray(X), np.asarray(y)

# create binary classification model
model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
history = model.fit(datagen(32), epochs=5, steps_per_epoch=20)
```

在上述操作中，我们故意给 `fit()` 函数传递 `steps_per_epoch=20` 参数，因为上面的代码会非常慢。这是因为 OpenPyXL 是用 Python 实现的，以最大化兼容性，但牺牲了编译模块所能提供的速度。因此，最好避免每次都逐行读取 Excel 数据。如果我们需要使用 Excel，较好的选择是一次性将所有数据读入内存，然后直接使用：

```py
import random

import numpy as np
import openpyxl
from sklearn.datasets import fetch_openml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Read data from OpenML
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
header = list(dataset.columns)
rows = dataset.to_numpy().tolist()

# Create Excel workbook and write data into the default worksheet
wb = openpyxl.Workbook()
sheet = wb.active
sheet.title = "Diabetes"
sheet.append(header)
for row in rows:
    sheet.append(row)
# Save
wb.save("MLM.xlsx")

# Read entire worksheet from the Excel file
wb = openpyxl.load_workbook("MLM.xlsx", read_only=True)
sheet = wb.active
X = []
y = []
for i, row in enumerate(sheet.rows):
    if i==0:
        continue # skip the header row
    rowdata = [cell.value for cell in row]
    X.append(rowdata[:-1])
    y.append(1 if rowdata[-1]=="tested_positive" else 0)
X, y = np.asarray(X), np.asarray(y)

# create binary classification model
model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
history = model.fit(X, y, epochs=5)
```

## 在 Google Sheets 中管理数据

除了 Excel 工作簿，有时我们会发现 Google Sheets 更方便处理数据，因为它是在“云端”中。我们也可以使用与 Excel 类似的逻辑来管理数据。但首先，我们需要安装一些模块才能在 Python 中访问它：

```py
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

假设你有一个 Gmail 账户，并且创建了一个 Google Sheet。你在地址栏看到的 URL，在 `/edit` 部分之前的部分，告诉你这个表格的 ID，我们将稍后使用这个 ID：

![](img/bb73e2dacbc3df0154c867b4e01b1c43.png)

要从 Python 程序访问这个表格，最好为你的代码创建一个**服务账户**。这是一个通过密钥进行身份验证的机器可操作账户，但由账户所有者管理。你可以控制这个服务账户的权限和过期时间。你也可以随时撤销服务账户，因为它与 Gmail 账户是分开的。

要创建服务账户，首先，你需要前往 Google 开发者控制台，[`console.developers.google.com`](https://console.developers.google.com/)，并通过点击“创建项目”按钮来创建一个项目：

![](img/57cae22114bd3a3d2ecaf15c51164e25.png)

你需要提供一个名称，然后点击“创建”：

![](img/d67a409b9978d3796ec0bed1d01b3645.png)

它会将你带回控制台，但你的项目名称会出现在搜索框旁边。下一步是通过点击搜索框下方的“启用 API 和服务”来启用 API：

![](img/840af135bd6a619b774f1b28d286b442.png)

由于我们需要创建一个服务账户来使用 Google Sheets，我们在搜索框中搜索“sheets”：

![](img/0b6c0c1cb54073588136680cea85a4a8.png)

然后点击 Google Sheets API：

![](img/401bd565b03fe212219429e95c7360fc.png)

并启用它

![](img/b82e796d273fe8c70ca3c9c0cc8e4b30.png)

之后，我们将被送回控制台主屏幕，我们可以点击右上角的“创建凭据”来创建服务账户：

![](img/debbc816737d11d651d51dc97395ad3a.png)

有不同类型的凭据，我们选择“服务账户”：

![](img/97184cb4221313f707ccfb8ae4cd98e6.png)

我们需要提供一个名称（供我们参考）、一个帐户 ID（作为项目的唯一标识符）和一个描述。显示在“服务帐户 ID”框下方的电子邮件地址是该服务帐户的电子邮件。复制它，稍后我们将将其添加到我们的 Google 表中。创建所有这些之后，我们可以跳过其余部分，然后点击“完成”：

![](img/7a1016955a665f831d1f483aaeaffe88.png)

当我们完成后，将被送回到主控制台屏幕，并且如果我们在“服务帐户”部分下看到它，就知道服务帐户已经创建：

![](img/ac270fda2a63e3a91acefd1d2f1fa770.png)

接下来，我们需要点击帐户右侧的铅笔图标，这将带我们到以下屏幕：

![](img/207ae004c0b6005a1e293c57aaeb9223.png)

我们需要为此帐户创建一个密钥，而不是密码。我们点击页面顶部的“键”，然后点击“添加键”并选择“创建新键”：

![](img/454adac64723c7ab15f9414ab97a72b3.png)

键有两种不同的格式，JSON 是首选格式。选择 JSON 并在底部点击“创建”将会下载 JSON 文件的键：

![](img/48dd0135bb8e96035149a2f2dc5c9f47.png)

JSON 文件将如下所示：

```py
{
  "type": "service_account",
  "project_id": "mlm-python",
  "private_key_id": "3863a6254774259a1249",
  "private_key": "-----BEGIN PRIVATE KEY-----\n
                  MIIEvgIBADANBgkqh...
                  -----END PRIVATE KEY-----\n",
  "client_email": "ml-access@mlm-python.iam.gserviceaccount.com",
  "client_id": "11542775381574",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/ml-access%40mlm-python.iam.gserviceaccount.com"
}
```

保存 JSON 文件后，我们可以返回到我们的 Google 表，并与我们的服务帐户共享该表。点击右上角的“共享”按钮，输入服务帐户的电子邮件地址。您可以跳过通知，直接点击“共享”。然后我们就准备好了！

![](img/2c8e3ccbf52e03e8855f791b23d889ad.png)

此时，我们已准备好使用来自我们 Python 程序的服务帐户访问此特定 Google 表格。要向 Google 表格写入内容，我们可以使用 Google 的 API。我们首先依赖于刚刚下载的服务帐户的 JSON 文件（在本示例中为 `mlm-python.json`）来创建连接：

```py
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from httplib2 import Http

cred_file = "mlm-python.json"
scopes = ['https://www.googleapis.com/auth/spreadsheets']
cred = ServiceAccountCredentials.from_json_keyfile_name(cred_file, scopes)
service = build("sheets", "v4", http=cred.authorize(Http()))
sheet = service.spreadsheets()
```

如果我们刚刚创建它，文件中应该只有一个工作表，并且其 ID 为 0。使用 Google 的 API 的所有操作都是以 JSON 格式进行的。例如，以下是我们如何使用刚刚创建的连接删除整个工作表上的所有内容：

```py
...

sheet_id = '12Pc2_pX3HOSltcRLHtqiq3RSOL9RcG72CZxRqsMeRul'
body = {
    "requests": [{
        "deleteRange": {
            "range": {
                "sheetId": 0
            },
            "shiftDimension": "ROWS"
        }
    }]
}
action = sheet.batchUpdate(spreadsheetId=sheet_id, body=body)
action.execute()
```

假设我们像上面的第一个示例那样将糖尿病数据集读入一个 DataFrame 中。然后，我们可以一次性将整个数据集写入 Google 表格中。为此，我们需要创建一个列表的列表来反映表格中单元格的二维数组结构，然后将数据放入 API 查询中：

```py
...
rows = [list(dataset.columns)]
rows += dataset.to_numpy().tolist()
maxcol = max(len(row) for row in rows)
maxcol = chr(ord("A") - 1 + maxcol)
action = sheet.values().append(
    spreadsheetId = sheet_id,
    body = {"values": rows},
    valueInputOption = "RAW",
    range = "Sheet1!A1:%s" % maxcol
)
action.execute()
```

在上述内容中，我们假设表格的名称为“Sheet1”（默认名称，您可以在屏幕底部看到）。我们将数据写入到左上角对齐，从单元格 A1（左上角）开始填充。我们使用 `dataset.to_numpy().tolist()` 将所有数据收集到一个列表的列表中，但我们还将列标题作为额外的行添加在开头。

从 Google 表格中读取数据的方式类似。以下是我们如何读取随机一行数据：

```py
...
# Check the sheets
sheet_properties = sheet.get(spreadsheetId=sheet_id).execute()["sheets"]
print(sheet_properties)
# Read it back
maxrow = sheet_properties[0]["properties"]["gridProperties"]["rowCount"]
maxcol = sheet_properties[0]["properties"]["gridProperties"]["columnCount"]
maxcol = chr(ord("A") - 1 + maxcol)
row = random.randint(1, maxrow)
readrange = f"A{row}:{maxcol}{row}"
data = sheet.values().get(spreadsheetId=sheet_id, range=readrange).execute()
```

首先，我们可以通过检查其属性来了解表中的行数。上面的`print()`语句将产生如下结果：

```py
[{'properties': {'sheetId': 0, 'title': 'Sheet1', 'index': 0,
'sheetType': 'GRID', 'gridProperties': {'rowCount': 769, 'columnCount': 9}}}]
```

由于我们只有一个表单，列表只包含一个属性字典。使用这些信息，我们可以选择一行随机行并指定读取范围。上面的变量`data`将是如下所示的字典，数据将以列表的形式存在，可以使用`data["values"]`访问：

```py
{'range': 'Sheet1!A536:I536',
 'majorDimension': 'ROWS',
 'values': [['1',
   '77',
   '56',
   '30',
   '56',
   '33.3',
   '1.251',
   '24',
   'tested_negative']]}
```

将这些内容结合起来，以下是将数据加载到 Google Sheet 并从中读取随机行的完整代码：（运行时请确保更改`sheet_id`）

```py
import random

from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.datasets import fetch_openml

# Connect to Google Sheet
cred_file = "mlm-python.json"
scopes = ['https://www.googleapis.com/auth/spreadsheets']
cred = ServiceAccountCredentials.from_json_keyfile_name(cred_file, scopes)
service = build("sheets", "v4", http=cred.authorize(Http()))
sheet = service.spreadsheets()

# Google Sheet ID, as granted access to the service account
sheet_id = '12Pc2_pX3HOSltcRLHtqiq3RSOL9RcG72CZxRqsMeRul'

# Delete everything on spreadsheet 0
body = {
    "requests": [{
        "deleteRange": {
            "range": {
                "sheetId": 0
            },
            "shiftDimension": "ROWS"
        }
    }]
}
action = sheet.batchUpdate(spreadsheetId=sheet_id, body=body)
action.execute()

# Read dataset from OpenML
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
rows = [list(dataset.columns)]       # column headers
rows += dataset.to_numpy().tolist()  # rows of data

# Write to spreadsheet 0
maxcol = max(len(row) for row in rows)
maxcol = chr(ord("A") - 1 + maxcol)
action = sheet.values().append(
    spreadsheetId = sheet_id,
    body = {"values": rows},
    valueInputOption = "RAW",
    range = "Sheet1!A1:%s" % maxcol
)
action.execute()

# Check the sheets
sheet_properties = sheet.get(spreadsheetId=sheet_id).execute()["sheets"]
print(sheet_properties)

# Read a random row of data
maxrow = sheet_properties[0]["properties"]["gridProperties"]["rowCount"]
maxcol = sheet_properties[0]["properties"]["gridProperties"]["columnCount"]
maxcol = chr(ord("A") - 1 + maxcol)
row = random.randint(1, maxrow)
readrange = f"A{row}:{maxcol}{row}"
data = sheet.values().get(spreadsheetId=sheet_id, range=readrange).execute()
print(data)
```

不可否认，以这种方式访问 Google Sheets 过于冗长。因此，我们有一个第三方模块`gspread`可用于简化操作。安装模块后，我们可以像下面这样简单地检查电子表格的大小：

```py
import gspread

cred_file = "mlm-python.json"
gc = gspread.service_account(filename=cred_file)
sheet = gc.open_by_key(sheet_id)
spreadsheet = sheet.get_worksheet(0)
print(spreadsheet.row_count, spreadsheet.col_count)
```

清除电子表格、写入行以及读取随机行可以如下进行：

```py
...
# Clear all data
spreadsheet.clear()
# Write to spreadsheet
spreadsheet.append_rows(rows)
# Read a random row of data
maxcol = chr(ord("A") - 1 + spreadsheet.col_count)
row = random.randint(2, spreadsheet.row_count)
readrange = f"A{row}:{maxcol}{row}"
data = spreadsheet.get(readrange)
print(data)
```

因此，之前的示例可以简化为以下内容，更短：

```py
import random

import gspread
from sklearn.datasets import fetch_openml

# Google Sheet ID, as granted access to the service account
sheet_id = '12Pc2_pX3HOSltcRLHtqiq3RSOL9RcG72CZxRqsMeRul'

# Connect to Google Sheet
cred_file = "mlm-python.json"
gc = gspread.service_account(filename=cred_file)
sheet = gc.open_by_key(sheet_id)
spreadsheet = sheet.get_worksheet(0)

# Clear all data
spreadsheet.clear()

# Read dataset from OpenML
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
rows = [list(dataset.columns)]       # column headers
rows += dataset.to_numpy().tolist()  # rows of data

# Write to spreadsheet
spreadsheet.append_rows(rows)

# Check the number of rows and columns in the spreadsheet
print(spreadsheet.row_count, spreadsheet.col_count)

# Read a random row of data
maxcol = chr(ord("A") - 1 + spreadsheet.col_count)
row = random.randint(2, spreadsheet.row_count)
readrange = f"A{row}:{maxcol}{row}"
data = spreadsheet.get(readrange)
print(data)
```

与读取 Excel 类似，使用存储在 Google Sheet 中的数据集，最好一次性读取，而不是在训练循环中逐行读取。这是因为每次读取时，都会发送网络请求并等待 Google 服务器的回复。这不可能很快，因此最好避免。以下是我们如何将 Google Sheet 中的数据与 Keras 代码结合起来进行训练的示例：

```py
import random

import numpy as np
import gspread
from sklearn.datasets import fetch_openml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Google Sheet ID, as granted access to the service account
sheet_id = '12Pc2_pX3HOSltcRLHtqiq3RSOL9RcG72CZxRqsMeRul'

# Connect to Google Sheet
cred_file = "mlm-python.json"
gc = gspread.service_account(filename=cred_file)
sheet = gc.open_by_key(sheet_id)
spreadsheet = sheet.get_worksheet(0)

# Clear all data
spreadsheet.clear()

# Read dataset from OpenML
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
rows = [list(dataset.columns)]       # column headers
rows += dataset.to_numpy().tolist()  # rows of data

# Write to spreadsheet
spreadsheet.append_rows(rows)

# Read the entire spreadsheet, except header
maxrow = spreadsheet.row_count
maxcol = chr(ord("A") - 1 + spreadsheet.col_count)
data = spreadsheet.get(f"A2:{maxcol}{maxrow}")
X = [row[:-1] for row in data]
y = [1 if row[-1]=="tested_positive" else 0 for row in data]
X, y = np.asarray(X).astype(float), np.asarray(y)

# create binary classification model
model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
history = model.fit(X, y, epochs=5)
```

## 数据库的其他用途

上面的示例向您展示了如何从电子表格访问数据库。我们假设数据集在训练循环中由机器学习模型存储和使用。这是一种使用外部数据存储的方法，但不是唯一的方法。数据库的其他一些使用案例包括：

+   作为存储日志的工具，以记录程序的详细信息，例如某些脚本何时被执行。这对于跟踪更改特别有用，如果脚本将要更改某些内容，例如，下载某些文件并覆盖旧版本。

+   作为收集数据的工具。就像我们可能使用来自 scikit-learn 的`GridSearchCV`一样，我们经常会用不同的超参数组合来评估模型性能。如果模型很大且复杂，我们可能希望将评估分布到不同的机器上，并收集结果。在程序末尾添加几行代码，将交叉验证结果写入数据库或电子表格，以便我们可以将结果与选择的超参数制成表格是很方便的。将这些数据以结构化格式存储允许我们稍后报告结论。

+   作为配置模型的工具。我们可以将其用作提供超参数选择的工具，以运行程序，而不是编写超参数组合和验证得分。如果我们决定更改参数，可以简单地打开一个 Google Sheet，例如，进行更改，而不是修改代码。

## 进一步阅读

以下是一些让你深入了解的资源：

**书籍**

+   [实用 SQL](https://www.amazon.com/dp/1718501064/)，第二版，作者：安东尼·德巴罗斯

+   [SQL 食谱](https://www.amazon.com/dp/1492077445/)，第二版，作者：安东尼·莫利纳罗和罗伯特·德·格拉夫

+   [用 Python 自动化无聊的事情](https://www.amazon.com/dp/1593279922/)，第二版，作者：阿尔·斯维加特

**API 和库**

+   [Python 标准库中的 sqlite3](https://docs.python.org/3/library/sqlite3.html)

+   [apsw](https://rogerbinns.github.io/apsw/) – 另一个 Python SQLite 包装器

+   [Python 标准库中的 dbm](https://docs.python.org/3/library/dbm.html)

+   [Openpyxl](https://openpyxl.readthedocs.io/en/stable/)

+   [Google Sheets API](https://developers.google.com/sheets/api)

+   [gspread](https://docs.gspread.org/en/latest/)

#### 文章

+   [Google Cloud 中的服务账户](https://cloud.google.com/iam/docs/service-accounts)

+   [创建和管理服务账户](https://cloud.google.com/iam/docs/creating-managing-service-accounts)

#### 软件

+   [SQLite](https://www.sqlite.org/index.html)

+   [GNU dbm](https://www.gnu.org.ua/software/gdbm/)

## **总结**

在本教程中，你学会了如何使用外部数据存储，包括数据库或电子表格。

具体来说，你学到了：

+   如何使你的 Python 程序通过 SQL 语句访问像 SQLite 这样的关系数据库

+   如何将 dbm 用作键值存储，并像使用 Python 字典一样使用它

+   如何从 Excel 文件中读取数据并写入数据

+   如何通过互联网访问 Google Sheet

+   我们如何利用这些来托管数据集并在机器学习项目中使用它们
