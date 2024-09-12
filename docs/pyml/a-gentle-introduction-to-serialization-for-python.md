# Python 序列化的温和介绍

> 原文：[`machinelearningmastery.com/a-gentle-introduction-to-serialization-for-python/`](https://machinelearningmastery.com/a-gentle-introduction-to-serialization-for-python/)

序列化是指将数据对象（例如，Python 对象、Tensorflow 模型）转换为一种格式，使我们可以存储或传输数据，然后在需要时使用反序列化的逆过程重新创建该对象。

数据的序列化有不同的格式，如 JSON、XML、HDF5 和 Python 的 pickle，用于不同的目的。例如，JSON 返回人类可读的字符串形式，而 Python 的 pickle 库则可以返回字节数组。

在这篇文章中，你将学习如何使用 Python 中的两个常见序列化库来序列化数据对象（即 pickle 和 HDF5），例如字典和 Tensorflow 模型，以便于存储和传输。

完成本教程后，你将了解：

+   Python 中的序列化库，如 pickle 和 h5py

+   在 Python 中序列化诸如字典和 Tensorflow 模型的对象

+   如何使用序列化进行记忆化以减少函数调用

**快速启动你的项目**，通过我的新书 [Python for Machine Learning](https://machinelearningmastery.com/python-for-machine-learning/)，包括 *逐步教程* 和所有示例的 *Python 源代码* 文件。

让我们开始吧！！[](../Images/5402df89ca73300f99566925bf47d5e0.png)

Python 序列化的温和介绍。图片来源 [little plant](https://unsplash.com/photos/TZw891-oMio)。版权所有

## 概述

本教程分为四个部分；它们是：

+   什么是序列化，为什么我们要进行序列化？

+   使用 Python 的 pickle 库

+   在 Python 中使用 HDF5

+   不同序列化方法的比较

## 什么是序列化，我们为什么要关心它？

想一想如何存储一个整数；你会如何将其存储在文件中或传输？这很简单！我们可以直接将整数写入文件中，然后存储或传输这个文件。

但是现在，如果我们考虑存储一个 Python 对象（例如，一个 Python 字典或一个 Pandas DataFrame），它有一个复杂的结构和许多属性（例如，DataFrame 的列和索引，以及每列的数据类型）呢？你会如何将它存储为一个文件或传输到另一台计算机上？

这就是序列化发挥作用的地方！

**序列化**是将对象转换为可以存储或传输的格式的过程。在传输或存储序列化数据后，我们能够稍后重建对象，并获得完全相同的结构/对象，这使得我们可以在之后继续使用存储的对象，而不必从头开始重建对象。

在 Python 中，有许多不同的序列化格式可供选择。一个跨多种语言的常见示例是 JSON 文件格式，它是可读的并允许我们存储字典并以相同的结构重新创建它。但 JSON 只能存储基本结构，如列表和字典，并且只能保留字符串和数字。我们不能要求 JSON 记住数据类型（例如，numpy float32 与 float64）。它也无法区分 Python 元组和列表。

更强大的序列化格式存在。接下来，我们将探讨两个常见的 Python 序列化库，即 `pickle` 和 `h5py`。

## 使用 Python 的 Pickle 库

`pickle` 模块是 Python 标准库的一部分，实现了序列化（pickling）和反序列化（unpickling）Python 对象的方法。

要开始使用 `pickle`，请在 Python 中导入它：

```py
import pickle
```

之后，为了序列化一个 Python 对象（如字典）并将字节流存储为文件，我们可以使用 `pickle` 的 `dump()` 方法。

```py
test_dict = {"Hello": "World!"}
with open("test.pickle", "wb") as outfile:
 	# "wb" argument opens the file in binary mode
	pickle.dump(test_dict, outfile)
```

代表`test_dict`的字节流现在存储在文件“`test.pickle`”中！

要恢复原始对象，我们使用 `pickle` 的 `load()` 方法从文件中读取序列化的字节流。

```py
with open("test.pickle", "rb") as infile:
 	test_dict_reconstructed = pickle.load(infile)
```

**警告：** 仅从您信任的来源反序列化数据，因为在反序列化过程中可能会执行任意恶意代码。

将它们结合起来，以下代码帮助您验证 `pickle` 可以恢复相同的对象：

```py
import pickle

# A test object
test_dict = {"Hello": "World!"}

# Serialization
with open("test.pickle", "wb") as outfile:
    pickle.dump(test_dict, outfile)
print("Written object", test_dict)

# Deserialization
with open("test.pickle", "rb") as infile:
    test_dict_reconstructed = pickle.load(infile)
print("Reconstructed object", test_dict_reconstructed)

if test_dict == test_dict_reconstructed:
    print("Reconstruction success")
```

除了将序列化的对象写入 `pickle` 文件外，我们还可以使用 `pickle` 的 `dumps()` 函数在 Python 中获取序列化为字节数组类型的对象：

```py
test_dict_ba = pickle.dumps(test_dict)      # b'\x80\x04\x95\x15…
```

同样，我们可以使用 `pickle` 的 load 方法将字节数组类型转换回原始对象：

```py
test_dict_reconstructed_ba = pickle.loads(test_dict_ba)
```

`pickle` 的一个有用功能是它可以序列化几乎任何 Python 对象，包括用户定义的对象，如下所示：

```py
import pickle

class NewClass:
    def __init__(self, data):
        print(data)
        self.data = data

# Create an object of NewClass
new_class = NewClass(1)

# Serialize and deserialize
pickled_data = pickle.dumps(new_class)
reconstructed = pickle.loads(pickled_data)

# Verify
print("Data from reconstructed object:", reconstructed.data)
```

上述代码将打印以下内容：

```py
1
Data from reconstructed object: 1
```

注意，在调用 `pickle.loads()` 时，类构造函数中的 print 语句没有执行。这是因为它重建了对象，而不是重新创建它。

`pickle` 甚至可以序列化 Python 函数，因为函数在 Python 中是一级对象：

```py
import pickle

def test():
    return "Hello world!"

# Serialize and deserialize
pickled_function = pickle.dumps(test)
reconstructed_function = pickle.loads(pickled_function)

# Verify
print (reconstructed_function()) #prints “Hello, world!”
```

因此，我们可以利用 `pickle` 来保存我们的工作。例如，从 Keras 或 scikit-learn 训练的模型可以通过 `pickle` 序列化并在之后加载，而不是每次使用时都重新训练模型。以下示例展示了我们如何使用 Keras 构建一个 LeNet5 模型来识别 MNIST 手写数字，然后使用 `pickle` 序列化训练好的模型。之后，我们可以在不重新训练的情况下重建模型，它应该会产生与原始模型完全相同的结果：

```py
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Load MNIST digits
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape data to (n_samples, height, wiedth, n_channel)
X_train = np.expand_dims(X_train, axis=3).astype("float32")
X_test = np.expand_dims(X_test, axis=3).astype("float32")

# One-hot encode the output
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# LeNet5 model
model = Sequential([
    Conv2D(6, (5,5), input_shape=(28,28,1), padding="same", activation="tanh"),
    AveragePooling2D((2,2), strides=2),
    Conv2D(16, (5,5), activation="tanh"),
    AveragePooling2D((2,2), strides=2),
    Conv2D(120, (5,5), activation="tanh"),
    Flatten(),
    Dense(84, activation="tanh"),
    Dense(10, activation="softmax")
])

# Train the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
earlystopping = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[earlystopping])

# Evaluate the model
print(model.evaluate(X_test, y_test, verbose=0))

# Pickle to serialize and deserialize
pickled_model = pickle.dumps(model)
reconstructed = pickle.loads(pickled_model)

# Evaluate again
print(reconstructed.evaluate(X_test, y_test, verbose=0))
```

上述代码将生成如下输出。请注意，原始模型和重建模型的评估分数在最后两行中完全一致：

```py
Epoch 1/100
1875/1875 [==============================] - 15s 7ms/step - loss: 0.1517 - accuracy: 0.9541 - val_loss: 0.0958 - val_accuracy: 0.9661
Epoch 2/100
1875/1875 [==============================] - 15s 8ms/step - loss: 0.0616 - accuracy: 0.9814 - val_loss: 0.0597 - val_accuracy: 0.9822
Epoch 3/100
1875/1875 [==============================] - 16s 8ms/step - loss: 0.0493 - accuracy: 0.9846 - val_loss: 0.0449 - val_accuracy: 0.9853
Epoch 4/100
1875/1875 [==============================] - 17s 9ms/step - loss: 0.0394 - accuracy: 0.9876 - val_loss: 0.0496 - val_accuracy: 0.9838
Epoch 5/100
1875/1875 [==============================] - 17s 9ms/step - loss: 0.0320 - accuracy: 0.9898 - val_loss: 0.0394 - val_accuracy: 0.9870
Epoch 6/100
1875/1875 [==============================] - 16s 9ms/step - loss: 0.0294 - accuracy: 0.9908 - val_loss: 0.0373 - val_accuracy: 0.9872
Epoch 7/100
1875/1875 [==============================] - 21s 11ms/step - loss: 0.0252 - accuracy: 0.9921 - val_loss: 0.0370 - val_accuracy: 0.9879
Epoch 8/100
1875/1875 [==============================] - 18s 10ms/step - loss: 0.0223 - accuracy: 0.9931 - val_loss: 0.0386 - val_accuracy: 0.9880
Epoch 9/100
1875/1875 [==============================] - 15s 8ms/step - loss: 0.0219 - accuracy: 0.9930 - val_loss: 0.0418 - val_accuracy: 0.9871
Epoch 10/100
1875/1875 [==============================] - 15s 8ms/step - loss: 0.0162 - accuracy: 0.9950 - val_loss: 0.0531 - val_accuracy: 0.9853
Epoch 11/100
1875/1875 [==============================] - 15s 8ms/step - loss: 0.0169 - accuracy: 0.9941 - val_loss: 0.0340 - val_accuracy: 0.9895
Epoch 12/100
1875/1875 [==============================] - 15s 8ms/step - loss: 0.0165 - accuracy: 0.9944 - val_loss: 0.0457 - val_accuracy: 0.9874
Epoch 13/100
1875/1875 [==============================] - 15s 8ms/step - loss: 0.0137 - accuracy: 0.9955 - val_loss: 0.0407 - val_accuracy: 0.9879
Epoch 14/100
1875/1875 [==============================] - 16s 8ms/step - loss: 0.0159 - accuracy: 0.9945 - val_loss: 0.0442 - val_accuracy: 0.9871
Epoch 15/100
1875/1875 [==============================] - 16s 8ms/step - loss: 0.0125 - accuracy: 0.9956 - val_loss: 0.0434 - val_accuracy: 0.9882
[0.0340442918241024, 0.9894999861717224]
[0.0340442918241024, 0.9894999861717224]
```

尽管 pickle 是一个强大的库，但它仍然有其自身的限制。例如，无法 pickle 包括数据库连接和已打开的文件句柄在内的活动连接。这个问题的根源在于重建这些对象需要 pickle 重新建立与数据库/文件的连接，这是 pickle 无法为你做的事情（因为它需要适当的凭证，超出了 pickle 的预期范围）。

### 想要开始使用 Python 进行机器学习吗？

现在就来参加我的免费 7 天电子邮件速成课程吧（附带示例代码）。

点击注册并获取课程的免费 PDF 电子书版本。

## 在 Python 中使用 HDF5

层次数据格式 5（HDF5）是一种二进制数据格式。`h5py` 包是一个 Python 库，提供了对 HDF5 格式的接口。根据 `h5py` 文档，HDF5 “允许你存储大量数值数据，并且可以轻松地使用 Numpy 对该数据进行操作。”

HDF5 能比其他序列化格式做得更好的是以文件系统的层次结构存储数据。你可以在 HDF5 中存储多个对象或数据集，就像在文件系统中保存多个文件一样。你也可以从 HDF5 中读取特定的数据集，就像从文件系统中读取一个文件而不需要考虑其他文件一样。如果你用 pickle 做这件事，每次加载或创建 pickle 文件时都需要读取和写入所有内容。因此，对于无法完全放入内存的大量数据，HDF5 是一个有利的选择。

要开始使用 `h5py`，你首先需要安装 `h5py` 库，可以使用以下命令进行安装：

```py
pip install h5py
```

或者，如果你正在使用 conda 环境：

```py
conda install h5py
```

接下来，我们可以开始创建我们的第一个数据集！

```py
import h5py

with h5py.File("test.hdf5", "w") as file:
    dataset = file.create_dataset("test_dataset", (100,), type="i4")
```

这将在文件 `test.hdf5` 中创建一个名为 “`test_dataset`” 的新数据集，形状为 (100, )，类型为 int32。`h5py` 的数据集遵循 Numpy 的语法，因此你可以进行切片、检索、获取形状等操作，类似于 Numpy 数组。

要检索特定索引：

```py
dataset[0]  #retrieves element at index 0 of dataset
```

要从索引 0 到索引 10 获取数据集的片段：

```py
dataset[:10]
```

如果你在 `with` 语句之外初始化了 `h5py` 文件对象，请记得关闭文件！

要从以前创建的 HDF5 文件中读取数据，你可以以 “`r`” 的方式打开文件进行读取，或者以 “`r+`” 的方式进行读写：

```py
with h5py.File("test.hdf5", "r") as file:
    print (file.keys()) #gets names of datasets that are in the file
    dataset = file["test_dataset"]
```

要组织你的 HDF5 文件，你可以使用组：

```py
with h5py.File("test.hdf5", "w") as file:
    # creates new group_1 in file
    file.create_group("group_1")
    group1 = file["group_1"]
    # creates dataset inside group1
    group1.create_dataset("dataset1", shape=(10,))
    # to access the dataset
    dataset = file["group_1"]["dataset1"]
```

另一种创建组和文件的方式是通过指定要创建的数据集的路径，`h5py` 也会在该路径上创建组（如果它们不存在）：

```py
with h5py.File("test.hdf5", "w") as file:
    # creates dataset inside group1
    file.create_dataset("group1/dataset1", shape=(10,))
```

这两段代码片段都会在未创建 `group1` 的情况下创建它，然后在 `group1` 中创建 `dataset1`。

## 在 Tensorflow 中的 HDF5

要在 Tensorflow Keras 中保存模型为 HDF5 格式，我们可以使用模型的 `save()` 函数，并将文件名指定为 `.h5` 扩展名，如下所示：

```py
from tensorflow import keras

# Create model
model = keras.models.Sequential([
 	keras.layers.Input(shape=(10,)),
 	keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")

# using the .h5 extension in the file name specifies that the model
# should be saved in HDF5 format
model.save("my_model.h5")
```

要加载存储的 HDF5 模型，我们也可以直接使用 Keras 中的函数：

```py
...
model = keras.models.load_model("my_model.h5")

# to check that the model has been successfully reconstructed
print(model.summary)
```

我们不希望为 Keras 模型使用 pickle 的一个原因是，我们需要一种更灵活的格式，不受特定版本 Keras 的限制。如果我们升级了 Tensorflow 版本，模型对象可能会改变，而 pickle 可能无法给我们一个可工作的模型。另一个原因是保留模型的必要数据。例如，如果我们检查上面创建的 HDF5 文件 `my_model.h5`，我们可以看到其中存储了以下内容：

```py
/
/model_weights
/model_weights/dense
/model_weights/dense/dense
/model_weights/dense/dense/bias:0
/model_weights/dense/dense/kernel:0
/model_weights/top_level_model_weights
```

因此，Keras 仅选择对重建模型至关重要的数据。训练好的模型将包含更多数据集，即 `/optimizer_weights/` 除了 `/model_weights/`。Keras 将恢复模型并适当地恢复权重，以给我们一个功能相同的模型。

以上面的例子为例。我们的模型保存在 `my_model.h5` 中。我们的模型是一个单层的全连接层，我们可以通过以下方式找出该层的内核：

```py
import h5py

with h5py.File("my_model.h5", "r") as infile:
    print(infile["/model_weights/dense/dense/kernel:0"][:])
```

因为我们没有为任何事情训练我们的网络，所以它会给我们初始化层的随机矩阵：

```py
[[ 0.6872471 ]
 [-0.51016176]
 [-0.5604881 ]
 [ 0.3387223 ]
 [ 0.52146655]
 [-0.6960067 ]
 [ 0.38258582]
 [-0.05564564]
 [ 0.1450575 ]
 [-0.3391946 ]]
```

并且在 HDF5 中，元数据存储在数据旁边。Keras 以 JSON 格式在元数据中存储了网络的架构。因此，我们可以按以下方式复现我们的网络架构：

```py
import json
import h5py

with h5py.File("my_model.h5", "r") as infile:
    for key in infile.attrs.keys():
        formatted = infile.attrs[key]
        if key.endswith("_config"):
            formatted = json.dumps(json.loads(formatted), indent=4)
        print(f"{key}: {formatted}")
```

这会产生：

```py
backend: tensorflow
keras_version: 2.7.0
model_config: {
    "class_name": "Sequential",
    "config": {
        "name": "sequential",
        "layers": [
            {
                "class_name": "InputLayer",
                "config": {
                    "batch_input_shape": [
                        null,
                        10
                    ],
                    "dtype": "float32",
                    "sparse": false,
                    "ragged": false,
                    "name": "input_1"
                }
            },
            {
                "class_name": "Dense",
                "config": {
                    "name": "dense",
                    "trainable": true,
                    "dtype": "float32",
                    "units": 1,
                    "activation": "linear",
                    "use_bias": true,
                    "kernel_initializer": {
                        "class_name": "GlorotUniform",
                        "config": {
                            "seed": null
                        }
                    },
                    "bias_initializer": {
                        "class_name": "Zeros",
                        "config": {}
                    },
                    "kernel_regularizer": null,
                    "bias_regularizer": null,
                    "activity_regularizer": null,
                    "kernel_constraint": null,
                    "bias_constraint": null
                }
            }
        ]
    }
}
training_config: {
    "loss": "mse",
    "metrics": null,
    "weighted_metrics": null,
    "loss_weights": null,
    "optimizer_config": {
        "class_name": "Adam",
        "config": {
            "name": "Adam",
            "learning_rate": 0.001,
            "decay": 0.0,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-07,
            "amsgrad": false
        }
    }
}
```

模型配置（即我们神经网络的架构）和训练配置（即我们传递给 `compile()` 函数的参数）存储为一个 JSON 字符串。在上面的代码中，我们使用 `json` 模块重新格式化它，以便更容易阅读。建议将您的模型保存为 HDF5，而不仅仅是您的 Python 代码，因为正如我们在上面看到的，它包含比代码更详细的网络构建信息。

## 比较不同序列化方法之间的差异

在上文中，我们看到 pickle 和 h5py 如何帮助序列化我们的 Python 数据。

我们可以使用 pickle 序列化几乎任何 Python 对象，包括用户定义的对象和函数。但 pickle 不是语言通用的。您不能在 Python 之外反序列化它。到目前为止，甚至有 6 个版本的 pickle，旧版 Python 可能无法消费新版本的 pickle 数据。

相反，HDF5 是跨平台的，并且与其他语言如 Java 和 C++ 兼容良好。在 Python 中，`h5py` 库实现了 Numpy 接口，以便更轻松地操作数据。数据可以在不同语言中访问，因为 HDF5 格式仅支持 Numpy 的数据类型，如浮点数和字符串。我们不能将任意对象（如 Python 函数）存储到 HDF5 中。

## 进一步阅读

本节提供了更多关于此主题的资源，如果您希望深入了解。

#### 文章

+   C# 编程指南中的序列化，[`docs.microsoft.com/en-us/dotnet/csharp/programming-guide/concepts/serialization/`](https://docs.microsoft.com/en-us/dotnet/csharp/programming-guide/concepts/serialization/)

+   保存和加载 Keras 模型，[`www.tensorflow.org/guide/keras/save_and_serialize`](https://www.tensorflow.org/guide/keras/save_and_serialize)

#### 库

+   pickle，[`docs.python.org/3/library/pickle.html`](https://docs.python.org/3/library/pickle.html)

+   h5py，[`docs.h5py.org/en/stable/`](https://docs.h5py.org/en/stable/)

#### API

+   Tensorflow tf.keras.layers.serialize，[`www.tensorflow.org/api_docs/python/tf/keras/layers/serialize`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/serialize)

+   Tensorflow tf.keras.models.load_model，[`www.tensorflow.org/api_docs/python/tf/keras/models/load_model`](https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model)

+   Tensorflow tf.keras.models.save_model，[`www.tensorflow.org/api_docs/python/tf/keras/models/save_model`](https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model)

## 总结

在本篇文章中，你将了解什么是序列化以及如何在 Python 中使用库来序列化 Python 对象，例如字典和 Tensorflow Keras 模型。你还学到了两个 Python 序列化库（pickle、h5py）的优缺点。

具体来说，你学到了：

+   什么是序列化，以及它的用途

+   如何在 Python 中开始使用 pickle 和 h5py 序列化库

+   不同序列化方法的优缺点
