# 使用检查工具开发 Python 程序

> 原文：[`machinelearningmastery.com/developing-a-python-program-using-inspection-tools/`](https://machinelearningmastery.com/developing-a-python-program-using-inspection-tools/)

Python 是一种解释型语言。这意味着有一个解释器来运行我们的程序，而不是编译代码并直接运行。在 Python 中，REPL（读-评-打印循环）可以逐行运行命令。结合 Python 提供的一些检查工具，它有助于开发代码。

在接下来的内容中，你将看到如何利用 Python 解释器来检查一个对象并开发一个程序。

完成本教程后，你将学到：

+   如何在 Python 解释器中工作

+   如何在 Python 中使用检查函数

+   如何借助检查函数一步一步开发解决方案

**快速启动你的项目**，请参阅我新的书籍 [《机器学习的 Python》](https://machinelearningmastery.com/python-for-machine-learning/)，包括*逐步教程*和所有示例的*Python 源代码*文件。

让我们开始吧！！[](../Images/6aa46dadfb52f5e53446d8edd5b2df2c.png)

使用检查工具开发 Python 程序。

图片来源：[Tekton](https://unsplash.com/photos/kzlxOJwD6i8)。保留所有权利。

## 教程概述

本教程分为四部分；它们是：

+   PyTorch 和 TensorFlow

+   寻找线索

+   从权重中学习

+   制作一个复制器

## PyTorch 和 TensorFlow

PyTorch 和 TensorFlow 是 Python 中两个最大的神经网络库。它们的代码不同，但它们能做的事情类似。

考虑经典的 MNIST 手写数字识别问题；你可以构建一个 LeNet-5 模型来对数字进行分类，如下所示：

```py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

# Load MNIST training data
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train = torchvision.datasets.MNIST('./datafiles/', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)

# LeNet5 model
torch_model = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=(5,5), stride=1, padding=2),
    nn.Tanh(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
    nn.Tanh(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),
    nn.Tanh(),
    nn.Flatten(),
    nn.Linear(120, 84),
    nn.Tanh(),
    nn.Linear(84, 10),
    nn.Softmax(dim=1)
)

# Training loop
def training_loop(model, optimizer, loss_fn, train_loader, n_epochs=100):
    model.train()
    for epoch in range(n_epochs):
        for data, target in train_loader:
            output = model(data)
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()

# Run training
optimizer = optim.Adam(torch_model.parameters())
loss_fn = nn.CrossEntropyLoss()
training_loop(torch_model, optimizer, loss_fn, train_loader, n_epochs=20)

# Save model
torch.save(torch_model, "lenet5.pt")
```

这是一个简化的代码，不需要任何验证或测试。TensorFlow 中对应的代码如下：

```py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Flatten
from tensorflow.keras.datasets import mnist

# LeNet5 model
keras_model = Sequential([
    Conv2D(6, (5,5), input_shape=(28,28,1), padding="same", activation="tanh"),
    AveragePooling2D((2,2), strides=2),
    Conv2D(16, (5,5), activation="tanh"),
    AveragePooling2D((2,2), strides=2),
    Conv2D(120, (5,5), activation="tanh"),
    Flatten(),
    Dense(84, activation="tanh"),
    Dense(10, activation="softmax")
])

# Reshape data to shape of (n_sample, height, width, n_channel)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.expand_dims(X_train, axis=3).astype('float32')

# Train
keras_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
keras_model.fit(X_train, y_train, epochs=20, batch_size=32)

# Save
keras_model.save("lenet5.h5")
```

运行这个程序会生成 PyTorch 代码中的文件 `lenet5.pt` 和 TensorFlow 代码中的 `lenet5.h5` 文件。

## 寻找线索

如果你理解了上述神经网络的工作原理，你应该能判断出每一层中只有大量的乘法和加法计算。从数学上讲，在每个全连接层中，输入与**核**之间进行矩阵乘法，然后将**偏差**加到结果上。在卷积层中，将核与输入矩阵的一部分逐元素相乘，然后对结果求和，并将偏差作为特征图的一个输出元素。

### 想开始使用 Python 进行机器学习吗？

现在就参加我的免费 7 天邮件速成课程（包含示例代码）。

点击注册并免费获取课程的 PDF 电子书版本。

在使用两个不同框架开发相同的 LeNet-5 模型时，如果它们的权重相同，应该可以使它们的工作结果相同。鉴于它们的架构相同，你如何将一个模型的权重复制到另一个模型？

你可以按如下方式加载保存的模型：

```py
import torch
import tensorflow as tf
torch_model = torch.load("lenet5.pt")
keras_model = tf.keras.models.load_model("lenet5.h5")
```

这可能不会告诉你太多。但如果你在命令行中运行 `python` 而不带任何参数，你将启动 REPL，你可以在其中输入上述代码（你可以通过 `quit()` 退出 REPL）：

```py
Python 3.9.13 (main, May 19 2022, 13:48:47)
[Clang 13.1.6 (clang-1316.0.21.2)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> import tensorflow as tf
>>> torch_model = torch.load("lenet5.pt")
>>> keras_model = tf.keras.models.load_model("lenet5.h5")
```

上述内容不会输出任何内容。但你可以使用 `type()` 内置命令检查加载的两个模型：

```py
>>> type(torch_model)
<class 'torch.nn.modules.container.Sequential'>
>>> type(keras_model)
<class 'keras.engine.sequential.Sequential'>
```

所以这里你知道它们分别是来自 PyTorch 和 Keras 的神经网络模型。由于它们是训练好的模型，权重必须被存储在其中。那么如何找到这些模型中的权重呢？由于它们是对象，最简单的方法是使用 `dir()` 内置函数来检查它们的成员：

```py
>>> dir(torch_model)
['T_destination', '__annotations__', '__call__', '__class__', '__delattr__', 
'__delitem__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', 
...
'_slow_forward', '_state_dict_hooks', '_version', 'add_module', 'append', 'apply', 
'bfloat16', 'buffers', 'children', 'cpu', 'cuda', 'double', 'dump_patches', 'eval', 
'extra_repr', 'float', 'forward', 'get_buffer', 'get_extra_state', 'get_parameter', 
'get_submodule', 'half', 'load_state_dict', 'modules', 'named_buffers', 
'named_children', 'named_modules', 'named_parameters', 'parameters', 
'register_backward_hook', 'register_buffer', 'register_forward_hook', 
'register_forward_pre_hook', 'register_full_backward_hook', 'register_module', 
'register_parameter', 'requires_grad_', 'set_extra_state', 'share_memory', 'state_dict',
'to', 'to_empty', 'train', 'training', 'type', 'xpu', 'zero_grad']
>>> dir(keras_model)
['_SCALAR_UPRANKING_ON', '_TF_MODULE_IGNORED_PROPERTIES', '__call__', '__class__', 
'__copy__', '__deepcopy__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', 
...
'activity_regularizer', 'add', 'add_loss', 'add_metric', 'add_update', 'add_variable', 
'add_weight', 'build', 'built', 'call', 'compile', 'compiled_loss', 'compiled_metrics', 
'compute_dtype', 'compute_loss', 'compute_mask', 'compute_metrics', 
'compute_output_shape', 'compute_output_signature', 'count_params', 
'distribute_strategy', 'dtype', 'dtype_policy', 'dynamic', 'evaluate', 
'evaluate_generator', 'finalize_state', 'fit', 'fit_generator', 'from_config', 
'get_config', 'get_input_at', 'get_input_mask_at', 'get_input_shape_at', 'get_layer', 
'get_output_at', 'get_output_mask_at', 'get_output_shape_at', 'get_weights', 'history', 
'inbound_nodes', 'input', 'input_mask', 'input_names', 'input_shape', 'input_spec', 
'inputs', 'layers', 'load_weights', 'loss', 'losses', 'make_predict_function', 
'make_test_function', 'make_train_function', 'metrics', 'metrics_names', 'name', 
'name_scope', 'non_trainable_variables', 'non_trainable_weights', 'optimizer', 
'outbound_nodes', 'output', 'output_mask', 'output_names', 'output_shape', 'outputs', 
'pop', 'predict', 'predict_function', 'predict_generator', 'predict_on_batch', 
'predict_step', 'reset_metrics', 'reset_states', 'run_eagerly', 'save', 'save_spec', 
'save_weights', 'set_weights', 'state_updates', 'stateful', 'stop_training', 
'submodules', 'summary', 'supports_masking', 'test_function', 'test_on_batch', 
'test_step', 'to_json', 'to_yaml', 'train_function', 'train_on_batch', 'train_step', 
'train_tf_function', 'trainable', 'trainable_variables', 'trainable_weights', 'updates',
'variable_dtype', 'variables', 'weights', 'with_name_scope']
```

每个对象中有很多成员。有些是属性，有些是类的方法。按照惯例，以下划线开头的成员是内部成员，正常情况下不应访问。如果你想查看更多的每个成员，你可以使用 `inspect` 模块中的 `getmembers()` 函数：

```py
>>> import inspect
>>> inspect(torch_model)
>>> inspect.getmembers(torch_model)
[('T_destination', ~T_destination), ('__annotations__', {'_modules': typing.Dict[str, 
torch.nn.modules.module.Module]}), ('__call__', <bound method Module._call_impl of 
Sequential(
...
```

`getmembers()` 函数的输出是一个元组列表，其中每个元组包含成员的名称和成员本身。例如，从上述内容你可以知道 `__call__` 是一个“绑定方法”，即类的成员方法。

通过仔细查看成员的名称，你可以看到在 PyTorch 模型中，“state” 应该是你的关注点，而在 Keras 模型中，你有一个名为 “weights” 的成员。要缩短它们的名称列表，你可以在解释器中执行以下操作：

```py
>>> [n for n in dir(torch_model) if 'state' in n]
['__setstate__', '_load_from_state_dict', '_load_state_dict_pre_hooks', 
'_register_load_state_dict_pre_hook', '_register_state_dict_hook', 
'_save_to_state_dict', '_state_dict_hooks', 'get_extra_state', 'load_state_dict', 
'set_extra_state', 'state_dict']
>>> [n for n in dir(keras_model) if 'weight' in n]
['_assert_weights_created', '_captured_weight_regularizer', 
'_check_sample_weight_warning', '_dedup_weights', '_handle_weight_regularization', 
'_initial_weights', '_non_trainable_weights', '_trainable_weights', 
'_undeduplicated_weights', 'add_weight', 'get_weights', 'load_weights', 
'non_trainable_weights', 'save_weights', 'set_weights', 'trainable_weights', 'weights']
```

这可能需要一些时间来试错。但并不太难，你可能会发现可以通过 `state_dict` 在 PyTorch 模型中查看权重：

```py
>>> torch_model.state_dict
<bound method Module.state_dict of Sequential(
  (0): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (1): Tanh()
  (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
  (3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (4): Tanh()
  (5): AvgPool2d(kernel_size=2, stride=2, padding=0)
  (6): Conv2d(16, 120, kernel_size=(5, 5), stride=(1, 1))
  (7): Tanh()
  (8): Flatten(start_dim=1, end_dim=-1)
  (9): Linear(in_features=120, out_features=84, bias=True)
  (10): Tanh()
  (11): Linear(in_features=84, out_features=10, bias=True)
  (12): Softmax(dim=1)
)>
>>> torch_model.state_dict()
OrderedDict([('0.weight', tensor([[[[ 0.1559,  0.1681,  0.2726,  0.3187,  0.4909],
          [ 0.1179,  0.1340, -0.0815, -0.3253,  0.0904],
          [ 0.2326, -0.2079, -0.8614, -0.8643, -0.0632],
          [ 0.3874, -0.3490, -0.7957, -0.5873, -0.0638],
          [ 0.2800,  0.0947,  0.0308,  0.4065,  0.6916]]],

        [[[ 0.5116,  0.1798, -0.1062, -0.4099, -0.3307],
          [ 0.1090,  0.0689, -0.1010, -0.9136, -0.5271],
          [ 0.2910,  0.2096, -0.2442, -1.5576, -0.0305],
...
```

对于 TensorFlow/Keras 模型，你可以通过 `get_weights()` 查找权重：

```py
>>> keras_model.get_weights
<bound method Model.get_weights of <keras.engine.sequential.Sequential object at 0x159d93eb0>>
>>> keras_model.get_weights()
[array([[[[ 0.14078194,  0.04990018, -0.06204645, -0.03128023,
          -0.22033708,  0.19721672]],

        [[-0.06618818, -0.152075  ,  0.13130261,  0.22893831,
           0.08880515,  0.01917628]],

        [[-0.28716782, -0.23207009,  0.00505603,  0.2697424 ,
          -0.1916888 , -0.25858143]],

        [[-0.41863152, -0.20710683,  0.13254236,  0.18774481,
          -0.14866787, -0.14398652]],

        [[-0.25119543, -0.14405733, -0.048533  , -0.12108403,
           0.06704573, -0.1196835 ]]],

       [[[-0.2438466 ,  0.02499897, -0.1243961 , -0.20115352,
          -0.0241346 ,  0.15888865]],

        [[-0.20548582, -0.26495507,  0.21004884,  0.32183227,
          -0.13990627, -0.02996112]],
...
```

这里也可以看到 `weights` 属性：

```py
>>> keras_model.weights
[<tf.Variable 'conv2d/kernel:0' shape=(5, 5, 1, 6) dtype=float32, numpy=
array([[[[ 0.14078194,  0.04990018, -0.06204645, -0.03128023,
          -0.22033708,  0.19721672]],

        [[-0.06618818, -0.152075  ,  0.13130261,  0.22893831,
           0.08880515,  0.01917628]],
...
         8.25365111e-02, -1.72486171e-01,  3.16280037e-01,
         4.12595004e-01]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(10,) dtype=float32, numpy=
array([-0.19007775,  0.14427921,  0.0571407 , -0.24149619, -0.03247226,
        0.18109408, -0.17159976,  0.21736498, -0.10254183,  0.02417901],
      dtype=float32)>]
```

在这里，你可以观察到以下内容：在 PyTorch 模型中，函数 `state_dict()` 返回一个 `OrderedDict`，这是一个按指定顺序排列的字典。键中有 `0.weight` 之类的键，它们映射到一个张量值。在 Keras 模型中，`get_weights()` 函数返回一个列表。列表中的每个元素都是一个 NumPy 数组。`weight` 属性也包含一个列表，但元素是 `tf.Variable` 类型。

你可以通过检查每个张量或数组的形状来了解更多：

```py
>>> [(key, val.shape) for key, val in torch_model.state_dict().items()]
[('0.weight', torch.Size([6, 1, 5, 5])), ('0.bias', torch.Size([6])), ('3.weight', 
torch.Size([16, 6, 5, 5])), ('3.bias', torch.Size([16])), ('6.weight', torch.Size([120,
16, 5, 5])), ('6.bias', torch.Size([120])), ('9.weight', torch.Size([84, 120])), 
('9.bias', torch.Size([84])), ('11.weight', torch.Size([10, 84])), ('11.bias', 
torch.Size([10]))]
>>> [arr.shape for arr in keras_model.get_weights()]
[(5, 5, 1, 6), (6,), (5, 5, 6, 16), (16,), (5, 5, 16, 120), (120,), (120, 84), (84,), 
(84, 10), (10,)]
```

尽管你在上述 Keras 模型中看不到层的名称，实际上你可以使用类似的推理来查找层并获取它们的名称：

```py
>>> keras_model.layers
[<keras.layers.convolutional.conv2d.Conv2D object at 0x159ddd850>, 
<keras.layers.pooling.average_pooling2d.AveragePooling2D object at 0x159ddd820>, 
<keras.layers.convolutional.conv2d.Conv2D object at 0x15a12b1c0>, 
<keras.layers.pooling.average_pooling2d.AveragePooling2D object at 0x15a1705e0>, 
<keras.layers.convolutional.conv2d.Conv2D object at 0x15a1812b0>, 
<keras.layers.reshaping.flatten.Flatten object at 0x15a194310>, 
<keras.layers.core.dense.Dense object at 0x15a1947c0>, <keras.layers.core.dense.Dense 
object at 0x15a194910>]
>>> [layer.name for layer in keras_model.layers]
['conv2d', 'average_pooling2d', 'conv2d_1', 'average_pooling2d_1', 'conv2d_2', 
'flatten', 'dense', 'dense_1']
>>>
```

## 从权重中学习

通过比较 PyTorch 模型的 `state_dict()` 结果和 Keras 模型的 `get_weights()` 结果，你可以看到它们都包含 10 个元素。通过 PyTorch 张量和 NumPy 数组的形状，你可以进一步注意到它们的形状相似。这可能是因为两个框架都按从输入到输出的顺序识别模型。你可以通过 `state_dict()` 输出的键与 Keras 模型的层名称进行进一步确认。

你可以通过提取一个 PyTorch 张量并检查来查看如何操作它：

```py
>>> torch_states = torch_model.state_dict()
>>> torch_states.keys()
odict_keys(['0.weight', '0.bias', '3.weight', '3.bias', '6.weight', '6.bias', '9.weight', '9.bias', '11.weight', '11.bias'])
>>> torch_states["0.weight"]
tensor([[[[ 0.1559,  0.1681,  0.2726,  0.3187,  0.4909],
          [ 0.1179,  0.1340, -0.0815, -0.3253,  0.0904],
          [ 0.2326, -0.2079, -0.8614, -0.8643, -0.0632],
          [ 0.3874, -0.3490, -0.7957, -0.5873, -0.0638],
          [ 0.2800,  0.0947,  0.0308,  0.4065,  0.6916]]],
...
        [[[ 0.0980,  0.0240,  0.3295,  0.4507,  0.4539],
          [-0.1530, -0.3991, -0.3834, -0.2716,  0.0809],
          [-0.4639, -0.5537, -1.0207, -0.8049, -0.4977],
          [ 0.1825, -0.1284, -0.0669, -0.4652, -0.2961],
          [ 0.3402,  0.4256,  0.4329,  0.1503,  0.4207]]]])
>>> dir(torch_states["0.weight"])
['H', 'T', '__abs__', '__add__', '__and__', '__array__', '__array_priority__', 
'__array_wrap__', '__bool__', '__class__', '__complex__', '__contains__', 
...
'trunc', 'trunc_', 'type', 'type_as', 'unbind', 'unflatten', 'unfold', 'uniform_', 
'unique', 'unique_consecutive', 'unsafe_chunk', 'unsafe_split', 
'unsafe_split_with_sizes', 'unsqueeze', 'unsqueeze_', 'values', 'var', 'vdot', 'view', 
'view_as', 'vsplit', 'where', 'xlogy', 'xlogy_', 'xpu', 'zero_']
>>> torch_states["0.weight"].numpy()
array([[[[ 0.15587455,  0.16805592,  0.27259687,  0.31871665,
           0.49091515],
         [ 0.11791296,  0.13400094, -0.08148099, -0.32530317,
           0.09039831],
...
         [ 0.18252987, -0.12838107, -0.0669101 , -0.4652463 ,
          -0.2960882 ],
         [ 0.34022188,  0.4256311 ,  0.4328527 ,  0.15025541,
           0.4207182 ]]]], dtype=float32)
>>> torch_states["0.weight"].shape
torch.Size([6, 1, 5, 5])
>>> torch_states["0.weight"].numpy().shape
(6, 1, 5, 5)
```

从 PyTorch 张量的 `dir()` 输出中，你发现了一个名为 `numpy` 的成员，通过调用这个函数，它似乎将张量转换为 NumPy 数组。你可以相当有信心，因为你看到数字匹配，形状也匹配。实际上，通过查看文档，你可以更有信心：

```py
>>> help(torch_states["0.weight"].numpy)
```

`help()` 函数将显示函数的文档字符串，这通常是其文档。

由于这是第一个卷积层的内核，通过将该内核的形状与 Keras 模型的形状进行比较，你会发现它们的形状不同：

```py
>>> keras_weights = keras_model.get_weights()
>>> keras_weights[0].shape
(5, 5, 1, 6)
```

知道第一个层的输入是一个 28×28×1 的图像数组，而输出是 6 个特征图。将内核形状中的 1 和 6 对应为输入和输出中的通道数是很自然的。此外，根据我们对卷积层机制的理解，内核应该是一个 5×5 的矩阵。

此时，你可能已经猜到在 PyTorch 卷积层中，内核表示为 (output × input × height × width)，而在 Keras 中，它表示为 (height × width × input × output)。

类似地，你也看到在全连接层中，PyTorch 将内核表示为 (output × input)，而 Keras 中是 (input × output)：

```py
>>> keras_weights[6].shape
(120, 84)
>>> list(torch_states.values())[6].shape
torch.Size([84, 120])
```

匹配权重和张量并将它们的形状并排显示应该可以让这些更清楚：

```py
>>> for k,t in zip(keras_weights, torch_states.values()):
...     print(f"Keras: {k.shape}, Torch: {t.shape}")
...
Keras: (5, 5, 1, 6), Torch: torch.Size([6, 1, 5, 5])
Keras: (6,), Torch: torch.Size([6])
Keras: (5, 5, 6, 16), Torch: torch.Size([16, 6, 5, 5])
Keras: (16,), Torch: torch.Size([16])
Keras: (5, 5, 16, 120), Torch: torch.Size([120, 16, 5, 5])
Keras: (120,), Torch: torch.Size([120])
Keras: (120, 84), Torch: torch.Size([84, 120])
Keras: (84,), Torch: torch.Size([84])
Keras: (84, 10), Torch: torch.Size([10, 84])
Keras: (10,), Torch: torch.Size([10])
```

我们还可以匹配 Keras 权重和 PyTorch 张量的名称：

```py
>>> for k, t in zip(keras_model.weights, torch_states.keys()):
...     print(f"Keras: {k.name}, Torch: {t}")
...
Keras: conv2d/kernel:0, Torch: 0.weight
Keras: conv2d/bias:0, Torch: 0.bias
Keras: conv2d_1/kernel:0, Torch: 3.weight
Keras: conv2d_1/bias:0, Torch: 3.bias
Keras: conv2d_2/kernel:0, Torch: 6.weight
Keras: conv2d_2/bias:0, Torch: 6.bias
Keras: dense/kernel:0, Torch: 9.weight
Keras: dense/bias:0, Torch: 9.bias
Keras: dense_1/kernel:0, Torch: 11.weight
Keras: dense_1/bias:0, Torch: 11.bias
```

## 制作一个复制器

既然你已经了解了每个模型中的权重是什么样的，那么创建一个程序来将权重从一个模型复制到另一个模型似乎并不困难。关键是要回答：

1.  如何在每个模型中设置权重

1.  每个模型中权重的形状和数据类型应该是什么样的

第一个问题可以从之前使用 `dir()` 内置函数的检查中回答。你在 PyTorch 模型中看到了 `load_state_dict` 成员，它似乎是工具。同样，在 Keras 模型中，你看到了一个名为 `set_weight` 的成员，它正是 `get_weight` 的对应名称。你可以通过查看它们的文档或使用 `help()` 函数进一步确认这一点：

```py
>>> keras_model.set_weights
<bound method Layer.set_weights of <keras.engine.sequential.Sequential object at 0x159d93eb0>>
>>> torch_model.load_state_dict
<bound method Module.load_state_dict of Sequential(
  (0): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (1): Tanh()
  (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
  (3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (4): Tanh()
  (5): AvgPool2d(kernel_size=2, stride=2, padding=0)
  (6): Conv2d(16, 120, kernel_size=(5, 5), stride=(1, 1))
  (7): Tanh()
  (8): Flatten(start_dim=1, end_dim=-1)
  (9): Linear(in_features=120, out_features=84, bias=True)
  (10): Tanh()
  (11): Linear(in_features=84, out_features=10, bias=True)
  (12): Softmax(dim=1)
)>
>>> help(torch_model.load_state_dict)

>>> help(keras_model.set_weights)
```

你确认这些都是函数，它们的文档解释了它们的功能。根据文档，你进一步了解到 PyTorch 模型的 `load_state_dict()` 函数期望参数与 `state_dict()` 函数返回的格式相同；Keras 模型的 `set_weights()` 函数期望的格式与 `get_weights()` 函数返回的格式相同。

现在你已经完成了在 Python REPL 中的冒险（你可以输入 `quit()` 退出）。

通过研究如何 **重塑** 权重和 **转换** 数据类型，你得出了以下程序：

```py
import torch
import tensorflow as tf

# Load the models
torch_model = torch.load("lenet5.pt")
keras_model = tf.keras.models.load_model("lenet5.h5")

# Extract weights from Keras model
keras_weights = keras_model.get_weights()

# Transform shape from Keras to PyTorch
for idx in [0, 2, 4]:
    # conv layers: (out, in, height, width)
    keras_weights[idx] = keras_weights[idx].transpose([3, 2, 0, 1])
for idx in [6, 8]:
    # dense layers: (out, in)
    keras_weights[idx] = keras_weights[idx].transpose()

# Set weights
torch_states = torch_model.state_dict()
for key, weight in zip(torch_states.keys(), keras_weights):
    torch_states[key] = torch.tensor(weight)
torch_model.load_state_dict(torch_states)

# Save new model
torch.save(torch_model, "lenet5-keras.pt")
```

反之，从 PyTorch 模型到 Keras 模型复制权重也可以类似地完成，

```py
import torch
import tensorflow as tf

# Load the models
torch_model = torch.load("lenet5.pt")
keras_model = tf.keras.models.load_model("lenet5.h5")

# Extract weights from PyTorch model
torch_states = torch_model.state_dict()
weights = list(torch_states.values())

# Transform tensor to numpy array
weights = [w.numpy() for w in weights]

# Transform shape from PyTorch to Keras
for idx in [0, 2, 4]:
    # conv layers: (height, width, in, out)
    weights[idx] = weights[idx].transpose([2, 3, 1, 0])
for idx in [6, 8]:
    # dense layers: (in, out)
    weights[idx] = weights[idx].transpose()

# Set weights
keras_model.set_weights(weights)

# Save new model
keras_model.save("lenet5-torch.h5")
```

然后，你可以通过传入一个随机数组来验证它们是否工作相同，你可以期待输出完全一致：

```py
import numpy as np
import torch
import tensorflow as tf

# Load the models
torch_orig_model = torch.load("lenet5.pt")
keras_orig_model = tf.keras.models.load_model("lenet5.h5")
torch_converted_model = torch.load("lenet5-keras.pt")
keras_converted_model = tf.keras.models.load_model("lenet5-torch.h5")

# Create a random input
sample = np.random.random((28,28))

# Convert sample to torch input shape
torch_sample = torch.Tensor(sample.reshape(1,1,28,28))

# Convert sample to keras input shape
keras_sample = sample.reshape(1,28,28,1)

# Check output
keras_converted_output = keras_converted_model.predict(keras_sample, verbose=0)
keras_orig_output = keras_orig_model.predict(keras_sample, verbose=0)
torch_converted_output = torch_converted_model(torch_sample).detach().numpy()
torch_orig_output = torch_orig_model(torch_sample).detach().numpy()

np.set_printoptions(precision=4)
print(keras_orig_output)
print(torch_converted_output)
print()
print(torch_orig_output)
print(keras_converted_output)
```

在我们的例子中，输出是：

```py
[[9.8908e-06 2.4246e-07 3.1996e-04 8.2742e-01 1.6853e-10 1.7212e-01
  3.6018e-10 1.5521e-06 1.3128e-04 2.2083e-06]]
[[9.8908e-06 2.4245e-07 3.1996e-04 8.2742e-01 1.6853e-10 1.7212e-01
  3.6018e-10 1.5521e-06 1.3128e-04 2.2083e-06]]

[[4.1505e-10 1.9959e-17 1.7399e-08 4.0302e-11 9.5790e-14 3.7395e-12
  1.0634e-10 1.7682e-16 1.0000e+00 8.8126e-10]]
[[4.1506e-10 1.9959e-17 1.7399e-08 4.0302e-11 9.5791e-14 3.7395e-12
  1.0634e-10 1.7682e-16 1.0000e+00 8.8127e-10]]
```

这在足够的精度下彼此一致。请注意，由于训练的随机性，你的结果可能不会完全相同。此外，由于浮点计算的特性，即使权重相同，PyTorch 和 TensorFlow/Keras 模型也不会产生完全相同的输出。

然而，目标是展示你如何利用 Python 的检查工具来理解你不熟悉的东西并开发解决方案。

## 进一步阅读

本节提供了更多关于该主题的资源，如果你希望深入了解。

#### 文章

+   [inspect](https://docs.python.org/3/library/inspect.html) 模块在 Python 标准库中

+   [dir](https://docs.python.org/3/library/functions.html#dir) 内置函数

+   [PyTorch 中的 `state_dict` 是什么](https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html)

+   [TensorFlow `get_weights` 方法](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#get_weights)

## 摘要

在本教程中，你学习了如何在 Python REPL 中工作并使用检查函数来开发解决方案。具体而言，

+   你学习了如何在 REPL 中使用检查函数来了解对象的内部成员

+   你学习了如何使用 REPL 来实验 Python 代码

+   结果是，你开发了一个将 PyTorch 和 Keras 模型转换的程序
