# 绘制 Transformer 模型的训练和验证损失曲线

> 原文：[`machinelearningmastery.com/plotting-the-training-and-validation-loss-curves-for-the-transformer-model/`](https://machinelearningmastery.com/plotting-the-training-and-validation-loss-curves-for-the-transformer-model/)

我们之前已经看到如何训练 Transformer 模型用于神经机器翻译。在进行训练模型的推断之前，让我们首先探索如何稍微修改训练代码，以便能够绘制在学习过程中生成的训练和验证损失曲线。

训练和验证损失值提供了重要信息，因为它们让我们更好地了解学习性能如何随轮次的变化而变化，并帮助我们诊断任何可能导致模型欠拟合或过拟合的问题。它们还将告诉我们在推断阶段使用训练好的模型权重的轮次。

在本教程中，你将学习如何绘制 Transformer 模型的训练和验证损失曲线。

完成本教程后，你将了解：

+   如何修改训练代码以包括验证和测试划分，除了数据集的训练划分

+   如何修改训练代码以存储计算出的训练和验证损失值，以及训练好的模型权重

+   如何绘制保存的训练和验证损失曲线

**通过我的书** [《构建具有注意力机制的 Transformer 模型》](https://machinelearningmastery.com/transformer-models-with-attention/) **启动你的项目**。它提供了 **自学教程** 和 **可用代码** 来指导你构建一个完全可用的 Transformer 模型。

*将句子从一种语言翻译成另一种语言*...

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2022/10/training_validation_loss_cover.jpg)

绘制 Transformer 模型的训练和验证损失曲线

照片由 [Jack Anstey](https://unsplash.com/photos/zS4lUqLEiNA) 提供，部分权利保留。

## **教程概述**

本教程分为四部分，它们是：

+   Transformer 架构回顾

+   准备数据集的训练、验证和测试划分

+   训练 Transformer 模型

+   绘制训练和验证损失曲线

## **先决条件**

对于本教程，我们假设你已经熟悉：

+   [Transformer 模型背后的理论](https://machinelearningmastery.com/the-transformer-model/)

+   [Transformer 模型的实现](https://machinelearningmastery.com/joining-the-transformer-encoder-and-decoder-and-masking/)

+   [训练 Transformer 模型](https://machinelearningmastery.com/training-the-transformer-model/)

## **Transformer 架构回顾**

[回忆](https://machinelearningmastery.com/the-transformer-model/)你已经看到 Transformer 架构遵循编码器-解码器结构。左侧的编码器负责将输入序列映射到一系列连续表示；右侧的解码器接收编码器的输出以及前一个时间步的解码器输出，以生成输出序列。

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png)

Transformer 架构的编码器-解码器结构

取自“[Attention Is All You Need](https://arxiv.org/abs/1706.03762)“

在生成输出序列时，Transformer 不依赖于递归和卷积。

你已经看到如何训练完整的 Transformer 模型，现在你将看到如何生成和绘制训练和验证损失值，这将帮助你诊断模型的学习性能。

### 想要开始构建带有注意力机制的 Transformer 模型吗？

现在就参加我的免费 12 天电子邮件速成课程（包含示例代码）。

点击注册，还能获得课程的免费 PDF 电子书版本。

## **准备数据集的训练、验证和测试拆分**

为了能够包括数据的验证和测试拆分，你将通过引入以下代码行来修改[准备数据集的代码](https://machinelearningmastery.com/?p=13585&preview=true)，这些代码行：

+   指定验证数据拆分的大小。这反过来决定了训练数据和测试数据的大小，我们将把数据分成 80:10:10 的比例，分别用于训练集、验证集和测试集：

Python

```py
self.val_split = 0.1  # Ratio of the validation data split
```

+   除了训练集外，将数据集拆分为验证集和测试集：

Python

```py
val = dataset[int(self.n_sentences * self.train_split):int(self.n_sentences * (1-self.val_split))]
test = dataset[int(self.n_sentences * (1 - self.val_split)):]
```

+   通过标记化、填充和转换为张量来准备验证数据。为此，你将把这些操作收集到一个名为`encode_pad`的函数中，如下面的完整代码列表所示。这将避免在对训练数据进行这些操作时代码的过度重复：

Python

```py
valX = self.encode_pad(val[:, 0], enc_tokenizer, enc_seq_length)
valY = self.encode_pad(val[:, 1], dec_tokenizer, dec_seq_length)
```

+   将编码器和解码器的标记化器保存到 pickle 文件中，并将测试数据集保存到一个文本文件中，以便在推断阶段使用：

Python

```py
self.save_tokenizer(enc_tokenizer, 'enc')
self.save_tokenizer(dec_tokenizer, 'dec')
savetxt('test_dataset.txt', test, fmt='%s')
```

完整的代码列表现已更新如下：

Python

```py
from pickle import load, dump, HIGHEST_PROTOCOL
from numpy.random import shuffle
from numpy import savetxt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, int64

class PrepareDataset:
    def __init__(self, **kwargs):
        super(PrepareDataset, self).__init__(**kwargs)
        self.n_sentences = 15000  # Number of sentences to include in the dataset
        self.train_split = 0.8  # Ratio of the training data split
        self.val_split = 0.1  # Ratio of the validation data split

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

    # Encode and pad the input sequences
    def encode_pad(self, dataset, tokenizer, seq_length):
        x = tokenizer.texts_to_sequences(dataset)
        x = pad_sequences(x, maxlen=seq_length, padding='post')
        x = convert_to_tensor(x, dtype=int64)

        return x

    def save_tokenizer(self, tokenizer, name):
        with open(name + '_tokenizer.pkl', 'wb') as handle:
            dump(tokenizer, handle, protocol=HIGHEST_PROTOCOL)

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

        # Split the dataset in training, validation and test sets
        train = dataset[:int(self.n_sentences * self.train_split)]
        val = dataset[int(self.n_sentences * self.train_split):int(self.n_sentences * (1-self.val_split))]
        test = dataset[int(self.n_sentences * (1 - self.val_split)):]

        # Prepare tokenizer for the encoder input
        enc_tokenizer = self.create_tokenizer(dataset[:, 0])
        enc_seq_length = self.find_seq_length(dataset[:, 0])
        enc_vocab_size = self.find_vocab_size(enc_tokenizer, train[:, 0])

        # Prepare tokenizer for the decoder input
        dec_tokenizer = self.create_tokenizer(dataset[:, 1])
        dec_seq_length = self.find_seq_length(dataset[:, 1])
        dec_vocab_size = self.find_vocab_size(dec_tokenizer, train[:, 1])

        # Encode and pad the training input
        trainX = self.encode_pad(train[:, 0], enc_tokenizer, enc_seq_length)
        trainY = self.encode_pad(train[:, 1], dec_tokenizer, dec_seq_length)

        # Encode and pad the validation input
        valX = self.encode_pad(val[:, 0], enc_tokenizer, enc_seq_length)
        valY = self.encode_pad(val[:, 1], dec_tokenizer, dec_seq_length)

        # Save the encoder tokenizer
        self.save_tokenizer(enc_tokenizer, 'enc')

        # Save the decoder tokenizer
        self.save_tokenizer(dec_tokenizer, 'dec')

        # Save the testing dataset into a text file
        savetxt('test_dataset.txt', test, fmt='%s')

        return trainX, trainY, valX, valY, train, val, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size
```

## **训练 Transformer 模型**

我们将对[训练 Transformer 模型的代码](https://machinelearningmastery.com/?p=13585&preview=true)进行类似的修改，以：

+   准备验证数据集的批次：

Python

```py
val_dataset = data.Dataset.from_tensor_slices((valX, valY))
val_dataset = val_dataset.batch(batch_size)
```

+   监控验证损失指标：

Python

```py
val_loss = Mean(name='val_loss')
```

+   初始化字典以存储训练和验证的损失，并最终将损失值存储在相应的字典中：

Python

```py
train_loss_dict = {}
val_loss_dict = {}

train_loss_dict[epoch] = train_loss.result()
val_loss_dict[epoch] = val_loss.result()
```

+   计算验证损失：

Python

```py
loss = loss_fcn(decoder_output, prediction)
val_loss(loss)
```

+   在每个周期保存训练的模型权重。你将在推理阶段使用这些权重来调查模型在不同周期产生的结果差异。在实践中，更高效的做法是包含一个回调方法，该方法根据训练过程中监控的指标停止训练过程，并在此时保存模型权重：

Python

```py
# Save the trained model weights
training_model.save_weights("weights/wghts" + str(epoch + 1) + ".ckpt")
```

+   最后，将训练和验证损失值保存到 pickle 文件中：

Python

```py
with open('./train_loss.pkl', 'wb') as file:
    dump(train_loss_dict, file)

with open('./val_loss.pkl', 'wb') as file:
    dump(val_loss_dict, file)
```

修改后的代码列表现在变为：

Python

```py
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.metrics import Mean
from tensorflow import data, train, math, reduce_sum, cast, equal, argmax, float32, GradientTape, function
from keras.losses import sparse_categorical_crossentropy
from model import TransformerModel
from prepare_dataset import PrepareDataset
from time import time
from pickle import dump

# Define the model parameters
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of model layers' outputs
d_ff = 2048  # Dimensionality of the inner fully connected layer
n = 6  # Number of layers in the encoder stack

# Define the training parameters
epochs = 20
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

# Prepare the training dataset
dataset = PrepareDataset()
trainX, trainY, valX, valY, train_orig, val_orig, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = dataset('english-german.pkl')

print(enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size)

# Prepare the training dataset batches
train_dataset = data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset = train_dataset.batch(batch_size)

# Prepare the validation dataset batches
val_dataset = data.Dataset.from_tensor_slices((valX, valY))
val_dataset = val_dataset.batch(batch_size)

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
val_loss = Mean(name='val_loss')

# Create a checkpoint object and manager to manage multiple checkpoints
ckpt = train.Checkpoint(model=training_model, optimizer=optimizer)
ckpt_manager = train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=None)

# Initialise dictionaries to store the training and validation losses
train_loss_dict = {}
val_loss_dict = {}

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
    val_loss.reset_states()

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

    # Run a validation step after every epoch of training
    for val_batchX, val_batchY in val_dataset:

        # Define the encoder and decoder inputs, and the decoder output
        encoder_input = val_batchX[:, 1:]
        decoder_input = val_batchY[:, :-1]
        decoder_output = val_batchY[:, 1:]

        # Generate a prediction
        prediction = training_model(encoder_input, decoder_input, training=False)

        # Compute the validation loss
        loss = loss_fcn(decoder_output, prediction)
        val_loss(loss)

    # Print epoch number and accuracy and loss values at the end of every epoch
    print("Epoch %d: Training Loss %.4f, Training Accuracy %.4f, Validation Loss %.4f" % (epoch + 1, train_loss.result(), train_accuracy.result(), val_loss.result()))

    # Save a checkpoint after every epoch
    if (epoch + 1) % 1 == 0:

        save_path = ckpt_manager.save()
        print("Saved checkpoint at epoch %d" % (epoch + 1))

        # Save the trained model weights
        training_model.save_weights("weights/wghts" + str(epoch + 1) + ".ckpt")

        train_loss_dict[epoch] = train_loss.result()
        val_loss_dict[epoch] = val_loss.result()

# Save the training loss values
with open('./train_loss.pkl', 'wb') as file:
    dump(train_loss_dict, file)

# Save the validation loss values
with open('./val_loss.pkl', 'wb') as file:
    dump(val_loss_dict, file)

print("Total time taken: %.2fs" % (time() - start_time))
```

## **绘制训练和验证损失曲线**

为了能够绘制训练和验证损失曲线，你首先需要加载包含训练和验证损失字典的 pickle 文件，这些文件是你在早期训练 Transformer 模型时保存的。

然后你将从各自的字典中检索训练和验证损失值，并在同一图上绘制它们。

代码列表如下，你应该将其保存到一个单独的 Python 脚本中：

Python

```py
from pickle import load
from matplotlib.pylab import plt
from numpy import arange

# Load the training and validation loss dictionaries
train_loss = load(open('train_loss.pkl', 'rb'))
val_loss = load(open('val_loss.pkl', 'rb'))

# Retrieve each dictionary's values
train_values = train_loss.values()
val_values = val_loss.values()

# Generate a sequence of integers to represent the epoch numbers
epochs = range(1, 21)

# Plot and label the training and validation loss values
plt.plot(epochs, train_values, label='Training Loss')
plt.plot(epochs, val_values, label='Validation Loss')

# Add in a title and axes labels
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Set the tick locations
plt.xticks(arange(0, 21, 2))

# Display the plot
plt.legend(loc='best')
plt.show()
```

运行上述代码会生成类似下面的训练和验证损失曲线图：

![](https://machinelearningmastery.com/wp-content/uploads/2022/10/training_validation_loss_1.png)

训练和验证损失值在多个训练周期上的折线图

注意，尽管你可能会看到类似的损失曲线，但它们可能不一定与上面的一模一样。这是因为你从头开始训练 Transformer 模型，结果的训练和验证损失值取决于模型权重的随机初始化。

尽管如此，这些损失曲线为我们提供了更好的洞察力，了解学习性能如何随训练周期数变化，并帮助我们诊断可能导致欠拟合或过拟合模型的学习问题。

关于如何使用训练和验证损失曲线来诊断模型的学习表现，您可以参考 Jason Brownlee 的[这篇教程](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/)。

## **进一步阅读**

本节提供了更多资源，如果你希望深入了解这个话题。

### **书籍**

+   [Python 深度学习进阶](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X)，2019 年

+   [用于自然语言处理的 Transformers](https://www.amazon.com/Transformers-Natural-Language-Processing-architectures/dp/1800565798)，2021 年

### **文献**

+   [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，2017 年

### **网站**

+   如何使用学习曲线诊断机器学习模型性能，[`machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/`](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/)

## **总结**

在本教程中，您学习了如何绘制 Transformer 模型的训练和验证损失曲线。

具体来说，您学到了：

+   如何修改训练代码以包括验证集和测试集分割，除了数据集的训练分割。

+   如何修改训练代码以存储计算的训练和验证损失值，以及训练好的模型权重。

+   如何绘制保存的训练和验证损失曲线。

你有任何问题吗？

在下面的评论中提出你的问题，我会尽力回答。
