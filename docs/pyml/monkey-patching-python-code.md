# 猴子补丁 Python 代码

> 原文：[`machinelearningmastery.com/monkey-patching-python-code/`](https://machinelearningmastery.com/monkey-patching-python-code/)

Python 是一种动态脚本语言。它不仅具有动态类型系统，允许变量首先分配为一种类型然后后续改变，而且它的对象模型也是动态的。这使得我们可以在运行时修改其行为。其结果是可以进行猴子补丁。这是一个想法，我们可以在不修改高层代码的情况下修改程序的基础层。想象一下，你可以使用 `print()` 函数将内容打印到屏幕上，我们可以修改该函数的定义，将其打印到文件中，而无需修改你的任何一行代码。

这是可能的，因为 Python 是一种解释型语言，因此我们可以在程序运行时进行更改。我们可以利用这一特性在 Python 中修改类或模块的接口。如果我们处理遗留代码或其他人的代码，我们不想对其进行广泛修改，但仍然希望在不同版本的库或环境中运行它，这就很有用。在本教程中，我们将看到如何将这一技术应用于一些 Keras 和 TensorFlow 代码。

完成本教程后，你将学到：

+   什么是猴子补丁

+   如何在运行时更改 Python 中的对象或模块

**启动你的项目**，通过我的新书 [Python for Machine Learning](https://machinelearningmastery.com/python-for-machine-learning/)，包括 *一步步的教程* 和 *Python 源代码* 文件，用于所有示例。

让我们开始吧！[](../Images/a31cd24750c6b8c4d1a22885e9693da5.png)

猴子补丁 Python 代码。照片由 [Juan Rumimpunu](https://unsplash.com/photos/nLXOatvTaLo) 提供。保留所有权利。

## 教程概述

本教程分为三部分；它们是：

+   一个模型，两种接口

+   使用猴子补丁扩展对象

+   猴子补丁以复兴遗留代码

## 一个模型，两种接口

TensorFlow 是一个庞大的库。它提供了一个高层 Keras API 来描述深度学习模型的层次结构。它还附带了很多用于训练的函数，如不同的优化器和数据生成器。仅仅因为我们需要运行**训练后的模型**，安装 TensorFlow 就显得很繁琐。因此，TensorFlow 提供了一个名为 **TensorFlow Lite** 的对等产品，体积更小，适合在诸如移动设备或嵌入式设备等小型设备上运行。

我们希望展示原始 TensorFlow Keras 模型和 TensorFlow Lite 模型的不同使用方式。因此，让我们制作一个中等大小的模型，比如 LeNet-5 模型。以下是我们如何加载 MNIST 数据集并训练一个分类模型：

```py
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape data to shape of (n_sample, height, width, n_channel)
X_train = np.expand_dims(X_train, axis=3).astype('float32')
X_test = np.expand_dims(X_test, axis=3).astype('float32')

# LeNet5 model: ReLU can be used intead of tanh
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

# Training
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["sparse_categorical_accuracy"])
earlystopping = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[earlystopping])
```

运行上述代码将使用 TensorFlow 的数据集 API 下载 MNIST 数据集并训练模型。之后，我们可以保存模型：

```py
model.save("lenet5-mnist.h5")
```

或者我们可以用测试集评估模型：

```py
print(np.argmax(model.predict(X_test), axis=1))
print(y_test)
```

然后我们应该看到：

```py
[7 2 1 ... 4 5 6]
[7 2 1 ... 4 5 6]
```

但如果我们打算在 TensorFlow Lite 中使用它，我们希望将其转换为 TensorFlow Lite 格式，如下所示：

```py
# tflite conversion with dynamic range optimization
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Optional: Save the data for testing
import numpy as np
np.savez('mnist-test.npz', X=X_test, y=y_test)

# Save the model.
with open('lenet5-mnist.tflite', 'wb') as f:
    f.write(tflite_model)
```

我们可以向转换器添加更多选项，例如将模型减少为使用 16 位浮点数。但在所有情况下，转换的输出是二进制字符串。转换不仅会将模型缩减到比从 Keras 保存的 HDF5 文件小得多的尺寸，还会允许我们使用轻量级库。有适用于 Android 和 iOS 移动设备的库。如果你使用嵌入式 Linux，可能会找到来自 PyPI 仓库的 `tflite-runtime` 模块（或从 TensorFlow 源代码编译一个）。下面是如何使用 `tflite-runtime` 运行转换后的模型：

```py
import numpy as np
import tflite_runtime.interpreter as tflite

loaded = np.load('mnist-test.npz')
X_test = loaded["X"]
y_test = loaded["y"]
interpreter = tflite.Interpreter(model_path="lenet5-mnist.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details[0]['shape'])

rows = []
for n in range(len(X_test)):
    # this model has single input and single output
    interpreter.set_tensor(input_details[0]['index'], X_test[n:n+1])
    interpreter.invoke()
    row = interpreter.get_tensor(output_details[0]['index'])
    rows.append(row)
rows = np.vstack(rows)

accuracy = np.sum(np.argmax(rows, axis=1) == y_test) / len(y_test)
print(accuracy)
```

实际上，更大的 TensorFlow 库也可以用类似的语法运行转换后的模型：

```py
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="lenet5-mnist.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

rows = []
for n in range(len(X_test)):
    # this model has single input and single output
    interpreter.set_tensor(input_details[0]['index'], X_test[n:n+1])
    interpreter.invoke()
    row = interpreter.get_tensor(output_details[0]['index'])
    rows.append(row)
rows = np.vstack(rows)

accuracy = np.sum(np.argmax(rows, axis=1) == y_test) / len(y_test)
print(accuracy)
```

注意使用模型的不同方式：在 Keras 模型中，我们有 `predict()` 函数，它以批次为输入并返回结果。然而，在 TensorFlow Lite 模型中，我们必须一次注入一个输入张量到“解释器”并调用它，然后检索结果。

将所有内容结合起来，下面的代码展示了如何构建一个 Keras 模型、训练它、将其转换为 TensorFlow Lite 格式，并用转换后的模型进行测试：

```py
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape data to shape of (n_sample, height, width, n_channel)
X_train = np.expand_dims(X_train, axis=3).astype('float32')
X_test = np.expand_dims(X_test, axis=3).astype('float32')

# LeNet5 model: ReLU can be used intead of tanh
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

# Training
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["sparse_categorical_accuracy"])
earlystopping = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[earlystopping])

# Save model
model.save("lenet5-mnist.h5")

# Compare the prediction vs test data
print(np.argmax(model.predict(X_test), axis=1))
print(y_test)

# tflite conversion with dynamic range optimization
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Optional: Save the data for testing
import numpy as np
np.savez('mnist-test.npz', X=X_test, y=y_test)

# Save the tflite model.
with open('lenet5-mnist.tflite', 'wb') as f:
    f.write(tflite_model)

# Load the tflite model and run test
interpreter = tf.lite.Interpreter(model_path="lenet5-mnist.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

rows = []
for n in range(len(X_test)):
    # this model has single input and single output
    interpreter.set_tensor(input_details[0]['index'], X_test[n:n+1])
    interpreter.invoke()
    row = interpreter.get_tensor(output_details[0]['index'])
    rows.append(row)
rows = np.vstack(rows)

accuracy = np.sum(np.argmax(rows, axis=1) == y_test) / len(y_test)
print(accuracy)
```

### 想要开始使用 Python 进行机器学习吗？

现在就领取我的免费 7 天电子邮件速成课程（包括示例代码）。

点击注册，还可以获得课程的免费 PDF 电子书版本。

## 使用猴子补丁扩展对象

我们可以在 TensorFlow Lite 解释器中使用 `predict()` 吗？

解释器对象没有这样的函数。但是，由于我们使用 Python，我们可以使用 **猴子补丁** 技术添加它。要理解我们在做什么，首先我们要注意，在之前的代码中定义的 `interpreter` 对象可能包含许多属性和函数。当我们像调用函数一样调用 `interpreter.predict()` 时，Python 会在对象内部寻找这样一个名称，然后执行它。如果没有找到这样的名称，Python 会引发 `AttributeError` 异常：

```py
...
interpreter.predict()
```

这将产生：

```py
Traceback (most recent call last):
  File "/Users/MLM/pred_error.py", line 13, in <module>
    interpreter.predict()
AttributeError: 'Interpreter' object has no attribute 'predict'
```

要使其工作，我们需要向 `interpreter` 对象添加一个名称为 `predict` 的函数，并且在调用时应表现得像一个函数。为了简单起见，我们注意到我们的模型是一个顺序模型，输入是一个数组，输出是一个 softmax 结果的数组。因此，我们可以编写一个类似于 Keras 模型中 `predict()` 函数的函数，但使用 TensorFlow Lite 解释器：

```py
...

# Monkey patching the tflite model
def predict(self, input_batch):
    batch_size = len(input_batch)
    output = []

    input_details = self.get_input_details()
    output_details = self.get_output_details()
    # Run each sample from the batch
    for sample in range(batch_size):
        self.set_tensor(input_details[0]["index"], input_batch[sample:sample+1])
        self.invoke()
        sample_output = self.get_tensor(output_details[0]["index"])
        output.append(sample_output)

    # vstack the output of each sample
    return np.vstack(output)

interpreter.predict = predict.__get__(interpreter)
```

上述最后一行将我们创建的函数分配给 `interpreter` 对象，名称为 `predict`。`__get__(interpreter)` 部分是必需的，以便将我们定义的函数变为 `interpreter` 对象的成员函数。

有了这些，我们现在可以运行一个批次：

```py
...
out_proba = interpreter.predict(X_test)
out = np.argmax(out_proba, axis=1)
print(out)

accuracy = np.sum(out == y_test) / len(y_test)
print(accuracy)
```

```py
[7 2 1 ... 4 5 6]
0.9879
```

这是可能的，因为 Python 具有动态对象模型。我们可以在运行时修改对象的属性或成员函数。实际上，这不应该让我们感到惊讶。一个 Keras 模型需要运行`model.compile()`才能运行`model.fit()`。`compile`函数的一个效果是将`loss`属性添加到模型中以保存损失函数。这是在运行时完成的。

添加了`predict()`函数到`interpreter`对象后，我们可以像使用训练好的 Keras 模型进行预测一样传递`interpreter`对象。尽管在幕后它们有所不同，但它们共享相同的接口，因此其他函数可以在不修改任何代码行的情况下使用它。

下面是完整的代码，用于加载我们保存的 TensorFlow Lite 模型，然后对`predict()`函数进行猴子补丁，使其看起来像 Keras 模型：

```py
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load MNIST data and reshape
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.expand_dims(X_train, axis=3).astype('float32')
X_test = np.expand_dims(X_test, axis=3).astype('float32')

# Monkey patching the tflite model
def predict(self, input_batch):
    batch_size = len(input_batch)
    output = []

    input_details = self.get_input_details()
    output_details = self.get_output_details()
    # Run each sample from the batch
    for sample in range(batch_size):
        self.set_tensor(input_details[0]["index"], input_batch[sample:sample+1])
        self.invoke()
        sample_output = self.get_tensor(output_details[0]["index"])
        output.append(sample_output)

    # vstack the output of each sample
    return np.vstack(output)

# Load and monkey patch
interpreter = tf.lite.Interpreter(model_path="lenet5-mnist.tflite")
interpreter.predict = predict.__get__(interpreter)
interpreter.allocate_tensors()

# test output
out_proba = interpreter.predict(X_test)
out = np.argmax(out_proba, axis=1)
print(out)
accuracy = np.sum(out == y_test) / len(y_test)
print(accuracy)
```

## 猴子补丁以恢复遗留代码

我们可以给出一个 Python 中猴子补丁的另一个示例。考虑以下代码：

```py
# https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
# Example of Dropout on the Sonar Dataset: Hidden Layer
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# load dataset
dataframe = read_csv("sonar.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# dropout in hidden layers with weight constraint
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=60, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
	model.add(Dense(30, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	sgd = SGD(lr=0.1, momentum=0.9)
	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=300, batch_size=16, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Hidden: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

这段代码编写于几年前，假设使用的是旧版本的 Keras 和 TensorFlow 1.x。数据文件`sonar.csv`可以在[另一篇文章](https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/)中找到。如果我们使用 TensorFlow 2.5 运行此代码，将会看到`SGD`行出现`ImportError`。我们需要至少在上述代码中进行两个更改以使其运行：

1.  函数和类应该从`tensorflow.keras`而不是`keras`中导入

1.  约束类`maxnorm`应该使用驼峰命名法，`MaxNorm`

以下是更新后的代码，其中我们仅修改了导入语句：

```py
# Example of Dropout on the Sonar Dataset: Hidden Layer
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.constraints import MaxNorm as maxnorm
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# load dataset
dataframe = read_csv("sonar.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# dropout in hidden layers with weight constraint
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=60, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
	model.add(Dense(30, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	sgd = SGD(lr=0.1, momentum=0.9)
	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=300, batch_size=16, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Hidden: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

如果我们有一个更大的项目，包含许多脚本，那么修改每一行导入将是繁琐的。但 Python 的模块系统实际上是一个`sys.modules`中的字典。因此，我们可以对其进行猴子补丁，使旧代码适配新库。以下是如何做的。这适用于 TensorFlow 2.5 安装（Keras 代码的向后兼容性问题在 TensorFlow 2.9 中已修复；因此在最新版本的库中不需要这种补丁）：

```py
# monkey patching
import sys
import tensorflow.keras
tensorflow.keras.constraints.maxnorm = tensorflow.keras.constraints.MaxNorm
for x in sys.modules.keys():
    if x.startswith("tensorflow.keras"):
        sys.modules[x[len("tensorflow."):]] = sys.modules[x]

# Old code below:

# Example of Dropout on the Sonar Dataset: Hidden Layer
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# load dataset
dataframe = read_csv("sonar.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# dropout in hidden layers with weight constraint
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=60, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
	model.add(Dense(30, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	sgd = SGD(lr=0.1, momentum=0.9)
	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=300, batch_size=16, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Hidden: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

这绝对不是干净整洁的代码，并且未来维护将是一个问题。因此，猴子补丁在生产代码中是不受欢迎的。然而，这是一种快速技术，利用了 Python 语言的内部机制，使事情能够轻松工作。

## 进一步阅读

本节提供了更多有关该主题的资源，如果你希望深入了解。

#### 文章

+   StackOverflow 问题 “[什么是猴子补丁？](https://stackoverflow.com/questions/5626193/what-is-monkey-patching)“

+   [Python 快速入门](https://www.tensorflow.org/lite/guide/python)，TensorFlow Lite 指南

+   [导入系统](https://docs.python.org/3/reference/import.html)，Python 语言参考

## 总结

在本教程中，我们学习了什么是猴子补丁以及如何进行猴子补丁。具体来说，

+   我们学习了如何向现有对象添加成员函数

+   如何修改 `sys.modules` 中的 Python 模块缓存以欺骗 `import` 语句
