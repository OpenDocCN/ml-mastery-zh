# Python 类及其在 Keras 中的使用

> 原文：[`machinelearningmastery.com/python-classes-and-their-use-in-keras/`](https://machinelearningmastery.com/python-classes-and-their-use-in-keras/)

类是 Python 语言的基本构建块之一，可以应用于机器学习应用的开发。正如我们将看到的，Python 的类开发语法很简单，可以用于实现 Keras 中的回调。

在本教程中，你将发现 Python 类及其功能。

完成本教程后，你将知道：

+   为什么 Python 类很重要

+   如何定义和实例化类并设置其属性

+   如何创建方法并传递参数

+   什么是类继承

+   如何使用类来实现 Keras 中的回调

**通过我的新书[Python for Machine Learning](https://machinelearningmastery.com/python-for-machine-learning/)**，*逐步教程*和*Python 源代码*文件，**启动你的项目**。

让我们开始吧。![](https://machinelearningmastery.com/wp-content/uploads/2021/12/s-migaj-Yui5vfKHuzs-unsplash-scaled.jpg)

Python 类及其在 Keras 中的使用

图片由 [S Migaj](https://unsplash.com/photos/Yui5vfKHuzs) 提供，部分权利保留。

## **教程概述**

本教程分为六个部分，它们是：

+   类的介绍

+   **定义一个类**

+   实例化和属性引用

+   创建方法并传递参数

+   类继承

+   在 Keras 中使用类

## **类的介绍**

在面向对象的语言中，如 Python，类是基本构建块之一。

> *它们可以比作对象的蓝图，因为它们定义了对象应具有的属性和方法/行为。*
> 
> – [Python Fundamentals](https://www.amazon.com/Python-Fundamentals-practical-learning-real-world-ebook/dp/B07K4CVYND/ref=sr_1_1?keywords=python+fundamentals+ebook&qid=1638986660&sr=8-1), 2018.

创建一个新类会创建一个新对象，其中每个类实例可以通过其属性来描述，以保持其状态，并通过方法来修改其状态。

## **定义一个类**

*class* 关键字允许创建新的类定义，紧接着是类名：

Python

```py
class MyClass:
    <statements>
```

这样，就创建了一个绑定到指定类名（*MyClass*，在此情况下）的新类对象。每个类对象都可以支持实例化和属性引用，我们将很快看到。

## **实例化和属性引用**

实例化是创建类的新实例。

要创建类的新实例，我们可以使用类名调用它，并将其分配给一个变量。这将创建一个新的空类对象：

Python

```py
x = MyClass()
```

创建类的新实例时，Python 调用其对象构造方法， *__init()__*，该方法通常接受用于设置实例化对象属性的参数。

> *我们可以像定义函数一样在类中定义这个构造函数方法，并指定在实例化对象时需要传递的属性。*
> 
> – [Python 基础](https://www.amazon.com/Python-Fundamentals-practical-learning-real-world-ebook/dp/B07K4CVYND/ref=sr_1_1?keywords=python+fundamentals+ebook&qid=1638986660&sr=8-1)，2018 年。

比如说，我们希望定义一个名为 *Dog* 的新类：

Python

```py
class Dog:
	family = "Canine"

	def __init__(self, name, breed):
		self.name = name
		self.breed = breed
```

在这里，构造函数方法接受两个参数，*name* 和 *breed*，这些参数在实例化对象时可以传递给它：

Python

```py
dog1 = Dog("Lassie", "Rough Collie")
```

在我们考虑的例子中，*name* 和 *breed* 被称为 *实例变量*（或属性），因为它们绑定到特定的实例。这意味着这些属性*仅*属于它们被设置的对象，而不属于从同一类实例化的其他对象。

另一方面，*family* 是一个 *类变量*（或属性），因为它由同一类的所有实例共享。

您还可以注意到，构造函数方法（或任何其他方法）的第一个参数通常被称为 *self*。这个参数指的是我们正在创建的对象。遵循将第一个参数设置为 *self* 的惯例，有助于提高代码的可读性，便于其他程序员理解。

一旦我们设置了对象的属性，可以使用点操作符来访问它们。例如，再考虑 *Dog* 类的 *dog1* 实例，它的 *name* 属性可以如下访问：

Python

```py
print(dog1.name)
```

产生如下输出：

Python

```py
Lassie
```

### 想要开始学习用于机器学习的 Python 吗？

立即参加我的免费 7 天电子邮件速成课程（附有示例代码）。

点击注册，您还可以获得课程的免费 PDF 电子书版本。

## **创建方法和传递参数**

除了拥有构造函数方法，类对象还可以有多个其他方法来修改其状态。

> *定义实例方法的语法很熟悉。我们传递参数 self … 它总是实例方法的第一个参数。*
> 
> – [Python 基础](https://www.amazon.com/Python-Fundamentals-practical-learning-real-world-ebook/dp/B07K4CVYND/ref=sr_1_1?keywords=python+fundamentals+ebook&qid=1638986660&sr=8-1)，2018 年。

类似于构造函数方法，每个实例方法可以接受多个参数，第一个参数是 *self*，它让我们能够设置和访问对象的属性：

Python

```py
class Dog:
	family = "Canine"

	def __init__(self, name, breed):
		self.name = name
		self.breed = breed

	def info(self):
		print(self.name, "is a female", self.breed)
```

相同对象的不同方法也可以使用 *self* 参数来相互调用：

Python

```py
class Dog:
	family = "Canine"

	def __init__(self, name, breed):
		self.name = name
		self.breed = breed
		self.tricks = []

	def add_tricks(self, x):
		self.tricks.append(x)

	def info(self, x):
		self.add_tricks(x)
		print(self.name, "is a female", self.breed, "that", self.tricks[0])
```

然后可以生成如下输出字符串：

Python

```py
dog1 = Dog("Lassie", "Rough Collie")
dog1.info("barks on command")
```

我们发现，在这样做时，*barks on command* 输入会在 *info()* 方法调用 *add_tricks()* 方法时附加到 *tricks* 列表中。产生如下输出：

Python

```py
Lassie is a female Rough Collie that barks on command
```

## **类继承**

Python 还支持另一个特性，即类的 *继承*。

继承是一种机制，允许*子类*（也称为*派生*或*子*类）访问*超类*（也称为*基*类或*父*类）的所有属性和方法。

使用子类的语法如下：

Python

```py
class SubClass(BaseClass):
    <statements>
```

子类也可以从多个基类继承。在这种情况下，语法如下：

Python

```py
class SubClass(BaseClass1, BaseClass2, BaseClass3):
    <statements>
```

类属性和方法在基类中以及在多重继承的情况下也会在后续的基类中进行搜索。

Python 还允许子类中的方法覆盖基类中具有相同名称的另一个方法。子类中的覆盖方法可能会替代基类方法或只是扩展其功能。当存在覆盖的子类方法时，调用时执行的是这个方法，而不是基类中具有相同名称的方法。

## **在 Keras 中使用类**

在 Keras 中使用类的一个实际用途是编写自己的回调。

回调是 Keras 中的一个强大工具，它允许我们在训练、测试和预测的不同阶段观察模型的行为。

确实，我们可以将回调列表传递给以下任意一个：

+   keras.Model.fit()

+   keras.Model.evaluate()

+   keras.Model.predict()

Keras API 提供了几个内置回调。尽管如此，我们可能希望编写自己的回调，为此，我们将看看如何构建一个*custom*回调类。为此，我们可以继承回调基类中的几个方法，这些方法可以为我们提供以下信息：

+   训练、测试和预测开始和结束

+   一个周期开始和结束

+   训练、测试和预测批次开始和结束

让我们首先考虑一个简单的自定义回调的示例，该回调每次周期开始和结束时都会报告。我们将这个自定义回调类命名为*EpochCallback*，并覆盖基类*keras.callbacks.Callback*中的周期级方法*on_epoch_begin()*和*on_epoch_end()*：

Python

```py
import tensorflow.keras as keras

class EpochCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print("Starting epoch {}".format(epoch + 1))

    def on_epoch_end(self, epoch, logs=None):
        print("Finished epoch {}".format(epoch + 1))
```

为了测试我们刚刚定义的自定义回调，我们需要一个模型进行训练。为此，让我们定义一个简单的 Keras 模型：

Python

```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

def simple_model():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])
    return model
```

我们还需要一个数据集来进行训练，为此我们将使用 MNIST 数据集：

Python

```py
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Loading the MNIST training and testing data splits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Pre-processing the training data
x_train = x_train / 255.0
x_train = x_train.reshape(60000, 28, 28, 1)
y_train_cat = to_categorical(y_train, 10)
```

现在，让我们通过将自定义回调添加到传递给*keras.Model.fit()*方法的回调列表中来尝试一下自定义回调：

Python

```py
model = simple_model()

model.fit(x_train,
          y_train_cat,
          batch_size=32,
          epochs=5,
          callbacks=[EpochCallback()],
          verbose=0)
```

我们刚刚创建的回调产生了以下输出：

Python

```py
Starting epoch 1
Finished epoch 1
Starting epoch 2
Finished epoch 2
Starting epoch 3
Finished epoch 3
Starting epoch 4
Finished epoch 4
Starting epoch 5
Finished epoch 5
```

我们可以创建另一个自定义回调，在每个周期结束时监控损失值，并仅在损失减少时存储模型权重。为此，我们将从*log*字典中读取损失值，该字典存储每个批次和周期结束时的指标。我们还将通过*self.model*访问与当前训练、测试或预测轮次对应的模型。

我们将这个自定义回调称为*CheckpointCallback*：

Python

```py
import numpy as np

class CheckpointCallback(keras.callbacks.Callback):

    def __init__(self):
        super(CheckpointCallback, self).__init__()
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.best_loss = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("loss")
        print("Current loss is {}".format(current_loss))
        if np.less(current_loss, self.best_loss):
            self.best_loss = current_loss
            self.best_weights = self.model.get_weights()
            print("Storing the model weights at epoch {} \n".format(epoch + 1))
```

我们可以再试一次，这次将*CheckpointCallback*包含到回调列表中：

Python

```py
model = simple_model()

model.fit(x_train,
          y_train_cat,
          batch_size=32,
          epochs=5,
          callbacks=[EpochCallback(), CheckpointCallback()],
          verbose=0)
```

现在产生了两个回调的以下输出：

Python

```py
Starting epoch 1
Finished epoch 1
Current loss is 0.6327750086784363
Storing the model weights at epoch 1

Starting epoch 2
Finished epoch 2
Current loss is 0.3391888439655304
Storing the model weights at epoch 2

Starting epoch 3
Finished epoch 3
Current loss is 0.29216915369033813
Storing the model weights at epoch 3

Starting epoch 4
Finished epoch 4
Current loss is 0.2625095248222351
Storing the model weights at epoch 4

Starting epoch 5
Finished epoch 5
Current loss is 0.23906977474689484
Storing the model weights at epoch 5
```

## Keras 中的其他类

除了回调之外，我们还可以在 Keras 中为自定义指标（继承自`keras.metrics.Metrics`）、自定义层（继承自`keras.layers.Layer`）、自定义正则化器（继承自`keras.regularizers.Regularizer`）或甚至自定义模型（继承自`keras.Model`，例如更改调用模型的行为）创建派生类。你需要做的就是按照指导原则更改类的成员函数。你必须在成员函数中使用完全相同的名称和参数。

以下是 Keras 文档中的一个示例：

```py
class BinaryTruePositives(tf.keras.metrics.Metric):

  def __init__(self, name='binary_true_positives', **kwargs):
    super(BinaryTruePositives, self).__init__(name=name, **kwargs)
    self.true_positives = self.add_weight(name='tp', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.bool)

    values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
    values = tf.cast(values, self.dtype)
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, self.dtype)
      values = tf.multiply(values, sample_weight)
    self.true_positives.assign_add(tf.reduce_sum(values))

  def result(self):
    return self.true_positives

  def reset_states(self):
    self.true_positives.assign(0)

m = BinaryTruePositives()
m.update_state([0, 1, 1, 1], [0, 1, 0, 0])
print('Intermediate result:', float(m.result()))

m.update_state([1, 1, 1, 1], [0, 1, 1, 0])
print('Final result:', float(m.result()))
```

这揭示了为什么我们需要一个自定义指标的类：一个指标不仅仅是一个函数，而是一个在训练周期中每批训练数据时逐步计算其值的函数。最终，结果在一个纪元结束时通过`result()`函数报告，并使用`reset_state()`函数重置其内存，以便在下一个纪元中重新开始。

有关具体需要派生的内容，请参阅 Keras 的文档。

## **进一步阅读**

如果你希望深入了解这个主题，这部分提供了更多的资源。

### **书籍**

+   [Python Fundamentals](https://www.amazon.com/Python-Fundamentals-practical-learning-real-world-ebook/dp/B07K4CVYND/ref=sr_1_1?keywords=python+fundamentals+ebook&qid=1638986660&sr=8-1)，2018。

### **网站**

+   Python 类，[`docs.python.org/3/tutorial/classes.html`](https://docs.python.org/3/tutorial/classes.html)

+   在 Keras 中创建自定义回调，[`www.tensorflow.org/guide/keras/custom_callback`](https://www.tensorflow.org/guide/keras/custom_callback)

+   在 Keras 中创建自定义指标，[`keras.io/api/metrics/#creating-custom-metrics`](https://keras.io/api/metrics/#creating-custom-metrics)

+   通过子类化创建新层和模型，[`keras.io/guides/making_new_layers_and_models_via_subclassing/`](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)

## **总结**

在本教程中，你了解了 Python 类及其功能。

具体来说，你学到了：

+   为什么 Python 类很重要

+   如何定义和实例化类并设置其属性

+   如何创建方法和传递参数

+   什么是类继承

+   如何使用类来实现 Keras 中的回调

你有任何问题吗？

在下面的评论中提问，我会尽力回答。
