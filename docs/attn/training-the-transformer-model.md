# 训练 Transformer 模型

> 原文：[`machinelearningmastery.com/training-the-transformer-model/`](https://machinelearningmastery.com/training-the-transformer-model/)

我们已经整合了 [完整的 Transformer 模型](https://machinelearningmastery.com/joining-the-transformer-encoder-and-decoder-and-masking)，现在我们准备为神经机器翻译训练它。为此，我们将使用一个包含短英语和德语句子对的训练数据集。在训练过程中，我们还将重新审视掩码在计算准确度和损失指标中的作用。

在本教程中，您将了解如何为神经机器翻译训练 Transformer 模型。

完成本教程后，您将了解：

+   如何准备训练数据集

+   如何将填充蒙版应用于损失和准确度计算

+   如何训练 Transformer 模型

**用我的书 [使用注意力构建 Transformer 模型](https://machinelearningmastery.com/transformer-models-with-attention/) 快速启动您的项目**。它提供了具有 **工作代码** 的 **自学教程**，指导您构建一个完全可用的 Transformer 模型，可以...

*将句子从一种语言翻译为另一种语言*...

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2022/05/training_cover-scaled.jpg)

训练 Transformer 模型

图片由 [v2osk](https://unsplash.com/photos/PGExULGintM) 拍摄，部分权利保留。

## **教程概览**

本教程分为四部分；它们是：

+   Transformer 架构回顾

+   准备训练数据集

+   将填充蒙版应用于损失和准确度计算

+   训练 Transformer 模型

## **先决条件**

对于本教程，我们假设您已经熟悉：

+   [Transformer 模型背后的理论](https://machinelearningmastery.com/the-transformer-model/)

+   [Transformer 模型的实现](https://machinelearningmastery.com/joining-the-transformer-encoder-and-decoder-and-masking)

## **Transformer 架构回顾**

[回忆](https://machinelearningmastery.com/the-transformer-model/) 曾见过 Transformer 架构遵循编码器-解码器结构。编码器位于左侧，负责将输入序列映射为连续表示序列；解码器位于右侧，接收编码器的输出以及前一时间步的解码器输出，生成输出序列。

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png)

Transformer 架构的编码器-解码器结构

摘自“[Attention Is All You Need](https://arxiv.org/abs/1706.03762)”

在生成输出序列时，Transformer 不依赖于循环和卷积。

你已经了解了如何实现完整的 Transformer 模型，现在可以开始训练它进行神经机器翻译。

首先准备数据集以进行训练。

### 想要开始构建带有注意力机制的 Transformer 模型吗？

立即参加我的免费 12 天邮件速成课程（附示例代码）。

点击注册，还可以获得课程的免费 PDF 电子书版。

## **准备训练数据集**

为此，你可以参考之前的教程，了解如何[准备文本数据](https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/)以用于训练。

你还将使用一个包含短的英语和德语句子对的数据集，你可以在[这里](https://github.com/Rishav09/Neural-Machine-Translation-System/blob/master/english-german-both.pkl)下载。这个数据集已经过清理，移除了不可打印的、非字母的字符和标点符号，进一步将所有 Unicode 字符归一化为 ASCII，并将所有大写字母转换为小写字母。因此，你可以跳过清理步骤，这通常是数据准备过程的一部分。然而，如果你使用的数据集没有经过预处理，你可以参考[这个教程](https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/)学习如何处理。

让我们通过创建 `PrepareDataset` 类来实施以下步骤：

+   从指定的文件名加载数据集。

Python

```py
clean_dataset = load(open(filename, 'rb'))
```

+   从数据集中选择要使用的句子数量。由于数据集很大，你将减少其大小以限制训练时间。然而，你可以考虑使用完整的数据集作为本教程的扩展。

Python

```py
dataset = clean_dataset[:self.n_sentences, :]
```

+   在每个句子中附加开始（<START>）和结束（<EOS>）标记。例如，英语句子 `i like to run` 现在变为 `<START> i like to run <EOS>`。这也适用于其对应的德语翻译 `ich gehe gerne joggen`，现在变为 `<START> ich gehe gerne joggen <EOS>`。

Python

```py
for i in range(dataset[:, 0].size):
	dataset[i, 0] = "<START> " + dataset[i, 0] + " <EOS>"
	dataset[i, 1] = "<START> " + dataset[i, 1] + " <EOS>"
```

+   随机打乱数据集。

Python

```py
shuffle(dataset)
```

+   根据预定义的比例拆分打乱的数据集。

Python

```py
train = dataset[:int(self.n_sentences * self.train_split)]
```

+   创建并训练一个分词器，用于处理将输入编码器的文本序列，并找到最长序列的长度及词汇表大小。

Python

```py
enc_tokenizer = self.create_tokenizer(train[:, 0])
enc_seq_length = self.find_seq_length(train[:, 0])
enc_vocab_size = self.find_vocab_size(enc_tokenizer, train[:, 0])
```

+   对将输入编码器的文本序列进行分词，通过创建一个词汇表并用相应的词汇索引替换每个词。<START> 和 <EOS> 标记也将成为词汇表的一部分。每个序列也会填充到最大短语长度。

Python

```py
trainX = enc_tokenizer.texts_to_sequences(train[:, 0])
trainX = pad_sequences(trainX, maxlen=enc_seq_length, padding='post')
trainX = convert_to_tensor(trainX, dtype=int64)
```

+   创建并训练一个分词器，用于处理将输入解码器的文本序列，并找到最长序列的长度及词汇表大小。

Python

```py
dec_tokenizer = self.create_tokenizer(train[:, 1])
dec_seq_length = self.find_seq_length(train[:, 1])
dec_vocab_size = self.find_vocab_size(dec_tokenizer, train[:, 1])
```

+   对将输入解码器的文本序列进行类似的分词和填充处理。

Python

```py
trainY = dec_tokenizer.texts_to_sequences(train[:, 1])
trainY = pad_sequences(trainY, maxlen=dec_seq_length, padding='post')
trainY = convert_to_tensor(trainY, dtype=int64)
```

完整的代码清单如下（有关详细信息，请参阅 [这个之前的教程](https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/)）：

Python

```py
from pickle import load
from numpy.random import shuffle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, int64

class PrepareDataset:
	def __init__(self, **kwargs):
		super(PrepareDataset, self).__init__(**kwargs)
		self.n_sentences = 10000  # Number of sentences to include in the dataset
		self.train_split = 0.9  # Ratio of the training data split

	# Fit a tokenizer
	def create_tokenizer(self, dataset):
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(dataset)

		return tokenizer

	def find_seq_length(self, dataset):
		return max(len(seq.split()) for seq in dataset)

	def find_vocab_size(self, tokenizer, dataset):
		tokenizer.fit_on_texts(dataset)

		return len(tokenizer.word_index) + 1

	def __call__(self, filename, **kwargs):
		# Load a clean dataset
		clean_dataset = load(open(filename, 'rb'))

		# Reduce dataset size
		dataset = clean_dataset[:self.n_sentences, :]

		# Include start and end of string tokens
		for i in range(dataset[:, 0].size):
			dataset[i, 0] = "<START> " + dataset[i, 0] + " <EOS>"
			dataset[i, 1] = "<START> " + dataset[i, 1] + " <EOS>"

		# Random shuffle the dataset
		shuffle(dataset)

		# Split the dataset
		train = dataset[:int(self.n_sentences * self.train_split)]

		# Prepare tokenizer for the encoder input
		enc_tokenizer = self.create_tokenizer(train[:, 0])
		enc_seq_length = self.find_seq_length(train[:, 0])
		enc_vocab_size = self.find_vocab_size(enc_tokenizer, train[:, 0])

		# Encode and pad the input sequences
		trainX = enc_tokenizer.texts_to_sequences(train[:, 0])
		trainX = pad_sequences(trainX, maxlen=enc_seq_length, padding='post')
		trainX = convert_to_tensor(trainX, dtype=int64)

		# Prepare tokenizer for the decoder input
		dec_tokenizer = self.create_tokenizer(train[:, 1])
		dec_seq_length = self.find_seq_length(train[:, 1])
		dec_vocab_size = self.find_vocab_size(dec_tokenizer, train[:, 1])

		# Encode and pad the input sequences
		trainY = dec_tokenizer.texts_to_sequences(train[:, 1])
		trainY = pad_sequences(trainY, maxlen=dec_seq_length, padding='post')
		trainY = convert_to_tensor(trainY, dtype=int64)

		return trainX, trainY, train, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size
```

在开始训练 Transformer 模型之前，我们首先来看一下 `PrepareDataset` 类对应于训练数据集中第一句话的输出：

Python

```py
# Prepare the training data
dataset = PrepareDataset()
trainX, trainY, train_orig, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = dataset('english-german-both.pkl')

print(train_orig[0, 0], '\n', trainX[0, :])
```

Python

```py
<START> did tom tell you <EOS> 
 tf.Tensor([ 1 25  4 97  5  2  0], shape=(7,), dtype=int64)
```

（注意：由于数据集已被随机打乱，你可能会看到不同的输出。）

你可以看到，最初，你有一个三词句子（*did tom tell you*），然后你添加了开始和结束字符串的标记。接着你对其进行了向量化（你可能会注意到 <START> 和 <EOS> 标记分别被分配了词汇表索引 1 和 2）。向量化文本还用零进行了填充，使得最终结果的长度与编码器的最大序列长度匹配：

Python

```py
print('Encoder sequence length:', enc_seq_length)
```

Python

```py
Encoder sequence length: 7
```

你可以类似地检查输入到解码器的目标数据：

Python

```py
print(train_orig[0, 1], '\n', trainY[0, :])
```

Python

```py
<START> hat tom es dir gesagt <EOS> 
 tf.Tensor([  1  14   5   7  42 162   2   0   0   0   0   0], shape=(12,), dtype=int64)
```

在这里，最终结果的长度与解码器的最大序列长度相匹配：

Python

```py
print('Decoder sequence length:', dec_seq_length)
```

Python

```py
Decoder sequence length: 12
```

## **应用填充掩码到损失和准确度计算**

[回顾](https://machinelearningmastery.com/how-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras) 看到在编码器和解码器中使用填充掩码的重要性是为了确保我们刚刚添加到向量化输入中的零值不会与实际输入值一起处理。

这对于训练过程也是适用的，其中需要填充掩码，以确保在计算损失和准确度时，目标数据中的零填充值不被考虑。

让我们首先来看一下损失的计算。

这将使用目标值和预测值之间的稀疏分类交叉熵损失函数进行计算，然后乘以一个填充掩码，以确保只考虑有效的非零值。返回的损失是未掩码值的均值：

Python

```py
def loss_fcn(target, prediction):
    # Create mask so that the zero padding values are not included in the computation of loss
    padding_mask = math.logical_not(equal(target, 0))
    padding_mask = cast(padding_mask, float32)

    # Compute a sparse categorical cross-entropy loss on the unmasked values
    loss = sparse_categorical_crossentropy(target, prediction, from_logits=True) * padding_mask

    # Compute the mean loss over the unmasked values
    return reduce_sum(loss) / reduce_sum(padding_mask)
```

计算准确度时，首先比较预测值和目标值。预测输出是一个大小为 (*batch_size*, *dec_seq_length*, *dec_vocab_size*) 的张量，包含输出中令牌的概率值（由解码器端的 softmax 函数生成）。为了能够与目标值进行比较，只考虑每个具有最高概率值的令牌，并通过操作 `argmax(prediction, axis=2)` 检索其字典索引。在应用填充掩码后，返回的准确度是未掩码值的均值：

Python

```py
def accuracy_fcn(target, prediction):
    # Create mask so that the zero padding values are not included in the computation of accuracy
    padding_mask = math.logical_not(math.equal(target, 0))

    # Find equal prediction and target values, and apply the padding mask
    accuracy = equal(target, argmax(prediction, axis=2))
    accuracy = math.logical_and(padding_mask, accuracy)

    # Cast the True/False values to 32-bit-precision floating-point numbers
    padding_mask = cast(padding_mask, float32)
    accuracy = cast(accuracy, float32)

    # Compute the mean accuracy over the unmasked values
    return reduce_sum(accuracy) / reduce_sum(padding_mask)
```

## **训练 Transformer 模型**

首先定义模型和训练参数，按照 [Vaswani 等人（2017）](https://arxiv.org/abs/1706.03762) 的规范：

Python

```py
# Define the model parameters
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of model layers' outputs
d_ff = 2048  # Dimensionality of the inner fully connected layer
n = 6  # Number of layers in the encoder stack

# Define the training parameters
epochs = 2
batch_size = 64
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-9
dropout_rate = 0.1
```

（注意：只考虑两个时代以限制训练时间。然而，您可以将模型训练更多作为本教程的延伸部分。）

您还需要实现一个学习率调度器，该调度器最初会线性增加前`warmup_steps`的学习率，然后按步骤数的倒数平方根比例减少它。Vaswani 等人通过以下公式表示这一点：

$$\text{learning_rate} = \text{d_model}^{−0.5} \cdot \text{min}(\text{step}^{−0.5}, \text{step} \cdot \text{warmup_steps}^{−1.5})$$

Python

```py
class LRScheduler(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super(LRScheduler, self).__init__(**kwargs)

        self.d_model = cast(d_model, float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step_num):

        # Linearly increasing the learning rate for the first warmup_steps, and decreasing it thereafter
        arg1 = step_num ** -0.5
        arg2 = step_num * (self.warmup_steps ** -1.5)

        return (self.d_model ** -0.5) * math.minimum(arg1, arg2)
```

随后将`LRScheduler`类的一个实例作为 Adam 优化器的`learning_rate`参数传递：

Python

```py
optimizer = Adam(LRScheduler(d_model), beta_1, beta_2, epsilon)
```

接下来，将数据集分割成批次，以准备进行训练：

Python

```py
train_dataset = data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset = train_dataset.batch(batch_size)
```

这之后是创建一个模型实例：

Python

```py
training_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
```

在训练 Transformer 模型时，您将编写自己的训练循环，该循环包含先前实现的损失和精度函数。

在 Tensorflow 2.0 中，默认的运行时是急切执行，这意味着操作立即执行。急切执行简单直观，使得调试更容易。然而，它的缺点是不能利用在*图执行*中运行代码的全局性能优化。在图执行中，首先构建一个图形，然后才能执行张量计算，这会导致计算开销。因此，对于大模型训练，通常建议使用图执行，而不是对小模型训练使用急切执行更合适。由于 Transformer 模型足够大，建议应用图执行来进行训练。

为了这样做，您将如下使用`@function`装饰器：

Python

```py
@function
def train_step(encoder_input, decoder_input, decoder_output):
    with GradientTape() as tape:

        # Run the forward pass of the model to generate a prediction
        prediction = training_model(encoder_input, decoder_input, training=True)

        # Compute the training loss
        loss = loss_fcn(decoder_output, prediction)

        # Compute the training accuracy
        accuracy = accuracy_fcn(decoder_output, prediction)

    # Retrieve gradients of the trainable variables with respect to the training loss
    gradients = tape.gradient(loss, training_model.trainable_weights)

    # Update the values of the trainable variables by gradient descent
    optimizer.apply_gradients(zip(gradients, training_model.trainable_weights))

    train_loss(loss)
    train_accuracy(accuracy)
```

添加了`@function`装饰器后，接受张量作为输入的函数将被编译为图形。如果`@function`装饰器被注释掉，则该函数将通过急切执行运行。

下一步是实现训练循环，该循环将调用上述的`train_step`函数。训练循环将遍历指定数量的时代和数据集批次。对于每个批次，`train_step`函数计算训练损失和准确度度量，并应用优化器来更新可训练的模型参数。还包括一个检查点管理器，以便每五个时代保存一个检查点：

Python

```py
train_loss = Mean(name='train_loss')
train_accuracy = Mean(name='train_accuracy')

# Create a checkpoint object and manager to manage multiple checkpoints
ckpt = train.Checkpoint(model=training_model, optimizer=optimizer)
ckpt_manager = train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=3)

for epoch in range(epochs):

    train_loss.reset_states()
    train_accuracy.reset_states()

    print("\nStart of epoch %d" % (epoch + 1))

    # Iterate over the dataset batches
    for step, (train_batchX, train_batchY) in enumerate(train_dataset):

        # Define the encoder and decoder inputs, and the decoder output
        encoder_input = train_batchX[:, 1:]
        decoder_input = train_batchY[:, :-1]
        decoder_output = train_batchY[:, 1:]

        train_step(encoder_input, decoder_input, decoder_output)

        if step % 50 == 0:
            print(f'Epoch {epoch + 1} Step {step} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    # Print epoch number and loss value at the end of every epoch
    print("Epoch %d: Training Loss %.4f, Training Accuracy %.4f" % (epoch + 1, train_loss.result(), train_accuracy.result()))

    # Save a checkpoint after every five epochs
    if (epoch + 1) % 5 == 0:
        save_path = ckpt_manager.save()
        print("Saved checkpoint at epoch %d" % (epoch + 1))
```

需要记住的一个重要点是，解码器的输入相对于编码器输入向右偏移一个位置。这种偏移的背后思想，与解码器的第一个多头注意力块中的前瞻遮罩结合使用，是为了确保当前令牌的预测仅依赖于先前的令牌。

> *这种掩码，结合输出嵌入偏移一个位置的事实，确保了位置 i 的预测只能依赖于位置小于 i 的已知输出。*
> 
> *–* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，2017。

正因如此，编码器和解码器输入是以以下方式输入到 Transformer 模型中的：

`encoder_input = train_batchX[:, 1:]`

`decoder_input = train_batchY[:, :-1]`

汇总完整的代码列表如下：

Python

```py
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.metrics import Mean
from tensorflow import data, train, math, reduce_sum, cast, equal, argmax, float32, GradientTape, TensorSpec, function, int64
from keras.losses import sparse_categorical_crossentropy
from model import TransformerModel
from prepare_dataset import PrepareDataset
from time import time

# Define the model parameters
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of model layers' outputs
d_ff = 2048  # Dimensionality of the inner fully connected layer
n = 6  # Number of layers in the encoder stack

# Define the training parameters
epochs = 2
batch_size = 64
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-9
dropout_rate = 0.1

# Implementing a learning rate scheduler
class LRScheduler(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super(LRScheduler, self).__init__(**kwargs)

        self.d_model = cast(d_model, float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step_num):

        # Linearly increasing the learning rate for the first warmup_steps, and decreasing it thereafter
        arg1 = step_num ** -0.5
        arg2 = step_num * (self.warmup_steps ** -1.5)

        return (self.d_model ** -0.5) * math.minimum(arg1, arg2)

# Instantiate an Adam optimizer
optimizer = Adam(LRScheduler(d_model), beta_1, beta_2, epsilon)

# Prepare the training and test splits of the dataset
dataset = PrepareDataset()
trainX, trainY, train_orig, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = dataset('english-german-both.pkl')

# Prepare the dataset batches
train_dataset = data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset = train_dataset.batch(batch_size)

# Create model
training_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)

# Defining the loss function
def loss_fcn(target, prediction):
    # Create mask so that the zero padding values are not included in the computation of loss
    padding_mask = math.logical_not(equal(target, 0))
    padding_mask = cast(padding_mask, float32)

    # Compute a sparse categorical cross-entropy loss on the unmasked values
    loss = sparse_categorical_crossentropy(target, prediction, from_logits=True) * padding_mask

    # Compute the mean loss over the unmasked values
    return reduce_sum(loss) / reduce_sum(padding_mask)

# Defining the accuracy function
def accuracy_fcn(target, prediction):
    # Create mask so that the zero padding values are not included in the computation of accuracy
    padding_mask = math.logical_not(equal(target, 0))

    # Find equal prediction and target values, and apply the padding mask
    accuracy = equal(target, argmax(prediction, axis=2))
    accuracy = math.logical_and(padding_mask, accuracy)

    # Cast the True/False values to 32-bit-precision floating-point numbers
    padding_mask = cast(padding_mask, float32)
    accuracy = cast(accuracy, float32)

    # Compute the mean accuracy over the unmasked values
    return reduce_sum(accuracy) / reduce_sum(padding_mask)

# Include metrics monitoring
train_loss = Mean(name='train_loss')
train_accuracy = Mean(name='train_accuracy')

# Create a checkpoint object and manager to manage multiple checkpoints
ckpt = train.Checkpoint(model=training_model, optimizer=optimizer)
ckpt_manager = train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=3)

# Speeding up the training process
@function
def train_step(encoder_input, decoder_input, decoder_output):
    with GradientTape() as tape:

        # Run the forward pass of the model to generate a prediction
        prediction = training_model(encoder_input, decoder_input, training=True)

        # Compute the training loss
        loss = loss_fcn(decoder_output, prediction)

        # Compute the training accuracy
        accuracy = accuracy_fcn(decoder_output, prediction)

    # Retrieve gradients of the trainable variables with respect to the training loss
    gradients = tape.gradient(loss, training_model.trainable_weights)

    # Update the values of the trainable variables by gradient descent
    optimizer.apply_gradients(zip(gradients, training_model.trainable_weights))

    train_loss(loss)
    train_accuracy(accuracy)

for epoch in range(epochs):

    train_loss.reset_states()
    train_accuracy.reset_states()

    print("\nStart of epoch %d" % (epoch + 1))

    start_time = time()

    # Iterate over the dataset batches
    for step, (train_batchX, train_batchY) in enumerate(train_dataset):

        # Define the encoder and decoder inputs, and the decoder output
        encoder_input = train_batchX[:, 1:]
        decoder_input = train_batchY[:, :-1]
        decoder_output = train_batchY[:, 1:]

        train_step(encoder_input, decoder_input, decoder_output)

        if step % 50 == 0:
            print(f'Epoch {epoch + 1} Step {step} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
            # print("Samples so far: %s" % ((step + 1) * batch_size))

    # Print epoch number and loss value at the end of every epoch
    print("Epoch %d: Training Loss %.4f, Training Accuracy %.4f" % (epoch + 1, train_loss.result(), train_accuracy.result()))

    # Save a checkpoint after every five epochs
    if (epoch + 1) % 5 == 0:
        save_path = ckpt_manager.save()
        print("Saved checkpoint at epoch %d" % (epoch + 1))

print("Total time taken: %.2fs" % (time() - start_time))
```

运行代码会产生类似于以下的输出（你可能会看到不同的损失和准确率值，因为训练是从头开始的，而训练时间取决于你用于训练的计算资源）：

Python

```py
Start of epoch 1
Epoch 1 Step 0 Loss 8.4525 Accuracy 0.0000
Epoch 1 Step 50 Loss 7.6768 Accuracy 0.1234
Epoch 1 Step 100 Loss 7.0360 Accuracy 0.1713
Epoch 1: Training Loss 6.7109, Training Accuracy 0.1924

Start of epoch 2
Epoch 2 Step 0 Loss 5.7323 Accuracy 0.2628
Epoch 2 Step 50 Loss 5.4360 Accuracy 0.2756
Epoch 2 Step 100 Loss 5.2638 Accuracy 0.2839
Epoch 2: Training Loss 5.1468, Training Accuracy 0.2908
Total time taken: 87.98s
```

在仅使用 CPU 的相同平台上，仅使用即时执行需要 155.13 秒来运行代码，这显示了使用图执行的好处。

## **进一步阅读**

本节提供了更多关于此主题的资源，如果你希望更深入地了解。

### **书籍**

+   [Advanced Deep Learning with Python](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X)，2019

+   [Transformers for Natural Language Processing](https://www.amazon.com/Transformers-Natural-Language-Processing-architectures/dp/1800565798)，2021

### **论文**

+   [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，2017

### **网站**

+   从头开始在 Keras 中编写训练循环：[`keras.io/guides/writing_a_training_loop_from_scratch/`](https://keras.io/guides/writing_a_training_loop_from_scratch/)

## **总结**

在本教程中，你了解了如何训练 Transformer 模型进行神经机器翻译。

具体来说，你学到了：

+   如何准备训练数据集

+   如何将填充掩码应用于损失和准确率计算

+   如何训练 Transformer 模型

你有任何问题吗？

在下方评论中提出你的问题，我将尽力回答。
