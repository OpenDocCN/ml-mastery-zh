# 推断 Transformer 模型

> 原文：[`machinelearningmastery.com/inferencing-the-transformer-model/`](https://machinelearningmastery.com/inferencing-the-transformer-model/)

我们已经了解了如何在英语和德语句子对的数据集上[训练 Transformer 模型](https://machinelearningmastery.com/training-the-transformer-model/)，以及如何[绘制训练和验证损失曲线](https://machinelearningmastery.com/?p=13879&preview=true)来诊断模型的学习性能，并决定在第几个 epoch 上对训练好的模型进行推断。我们现在准备对训练好的 Transformer 模型进行推断，以翻译输入句子。

在本教程中，你将发现如何对训练好的 Transformer 模型进行推断，以实现神经机器翻译。

完成本教程后，你将了解到：

+   如何对训练好的 Transformer 模型进行推断

+   如何生成文本翻译

**用我的书籍** [《使用注意力构建 Transformer 模型》](https://machinelearningmastery.com/transformer-models-with-attention/) **启动你的项目**。它提供了带有**可操作代码**的**自学教程**，指导你构建一个完全可用的 Transformer 模型，该模型可以

*将句子从一种语言翻译成另一种语言*...

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2022/10/karsten-wurth-algc0FKHeMA-unsplash-scaled.jpg)

推断 Transformer 模型

图片由 [Karsten Würth](https://unsplash.com/photos/algc0FKHeMA) 提供，版权所有。

## **教程概述**

本教程分为三个部分；它们是：

+   Transformer 架构的回顾

+   推断 Transformer 模型

+   测试代码

## **先决条件**

对于本教程，我们假设你已经熟悉：

+   [Transformer 模型背后的理论](https://machinelearningmastery.com/the-transformer-model/)

+   [Transformer 模型的实现](https://machinelearningmastery.com/joining-the-transformer-encoder-and-decoder-and-masking/)

+   [训练 Transformer 模型](https://machinelearningmastery.com/training-the-transformer-model/)

+   [绘制 Transformer 模型的训练和验证损失曲线](https://machinelearningmastery.com/?p=13879&preview=true)

## **Transformer 架构的回顾**

[回忆](https://machinelearningmastery.com/the-transformer-model/) Transformer 架构遵循编码器-解码器结构。左侧的编码器负责将输入序列映射到一系列连续表示；右侧的解码器接收编码器的输出以及前一步的解码器输出，以生成输出序列。

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png)

Transformer 架构的编码器-解码器结构

摘自“[Attention Is All You Need](https://arxiv.org/abs/1706.03762)”

在生成输出序列时，Transformer 不依赖于递归和卷积。

你已经了解了如何实现完整的 Transformer 模型，并随后在英语和德语句子对的数据集上训练它。现在让我们继续对训练好的模型进行神经机器翻译推理。

## **推理 Transformer 模型**

让我们从创建一个新的 `TransformerModel` 类实例开始，该类之前在[这个教程](https://machinelearningmastery.com/joining-the-transformer-encoder-and-decoder-and-masking/)中实现过。

你将向其中输入论文中[Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)所指定的相关输入参数以及有关使用的数据集的信息：

Python

```py
# Define the model parameters
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of model layers' outputs
d_ff = 2048  # Dimensionality of the inner fully connected layer
n = 6  # Number of layers in the encoder stack

# Define the dataset parameters
enc_seq_length = 7  # Encoder sequence length
dec_seq_length = 12  # Decoder sequence length
enc_vocab_size = 2405  # Encoder vocabulary size
dec_vocab_size = 3858  # Decoder vocabulary size

# Create model
inferencing_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, 0)
```

在这里，请注意，最后输入到 `TransformerModel` 中的输入对应于 Transformer 模型中每个 `Dropout` 层的丢弃率。这些 `Dropout` 层在模型推理过程中将不会被使用（你最终会将 `training` 参数设置为 `False`），所以你可以安全地将丢弃率设置为 0。

此外，`TransformerModel` 类已经保存到一个名为 `model.py` 的单独脚本中。因此，为了能够使用 `TransformerModel` 类，你需要包含 `from model import TransformerModel`。

接下来，让我们创建一个类 `Translate`，该类继承自 Keras 的 `Module` 基类，并将初始化的推理模型分配给变量 `transformer`：

Python

```py
class Translate(Module):
    def __init__(self, inferencing_model, **kwargs):
        super(Translate, self).__init__(**kwargs)
        self.transformer = inferencing_model
        ...
```

当你[训练 Transformer 模型](https://machinelearningmastery.com/training-the-transformer-model/)时，你看到你首先需要对要输入到编码器和解码器的文本序列进行分词。你通过创建一个词汇表来实现这一点，并用相应的词汇表索引替换每个单词。

在将待翻译的文本序列输入到 Transformer 模型之前，你需要在推理阶段实现类似的过程。

为此，你将在类中包含以下 `load_tokenizer` 方法，该方法将用于加载在[训练阶段生成并保存的编码器和解码器分词器](https://machinelearningmastery.com/?p=13879&preview=true)：

Python

```py
def load_tokenizer(self, name):
    with open(name, 'rb') as handle:
        return load(handle)
```

在推理阶段使用与 Transformer 模型训练阶段生成的相同分词器对输入文本进行分词是非常重要的，因为这些分词器已经在与你的测试数据类似的文本序列上进行了训练。

下一步是创建 `call()` 类方法，该方法将负责：

+   将开始（<START>）和结束符号（<EOS>）令牌添加到输入句子中：

Python

```py
def __call__(self, sentence):
    sentence[0] = "<START> " + sentence[0] + " <EOS>"
```

+   加载编码器和解码器分词器（在本例中，分别保存在 `enc_tokenizer.pkl` 和 `dec_tokenizer.pkl` pickle 文件中）：

Python

```py
enc_tokenizer = self.load_tokenizer('enc_tokenizer.pkl')
dec_tokenizer = self.load_tokenizer('dec_tokenizer.pkl')
```

+   准备输入句子，首先进行标记化，然后填充到最大短语长度，最后转换为张量：

Python

```py
encoder_input = enc_tokenizer.texts_to_sequences(sentence)
encoder_input = pad_sequences(encoder_input, maxlen=enc_seq_length, padding='post')
encoder_input = convert_to_tensor(encoder_input, dtype=int64)
```

+   对输出中的<START>和<EOS>标记重复类似的标记化和张量转换过程：

Python

```py
output_start = dec_tokenizer.texts_to_sequences(["<START>"])
output_start = convert_to_tensor(output_start[0], dtype=int64)

output_end = dec_tokenizer.texts_to_sequences(["<EOS>"])
output_end = convert_to_tensor(output_end[0], dtype=int64)
```

+   准备一个输出数组来包含翻译后的文本。由于你事先不知道翻译句子的长度，因此你将输出数组的大小初始化为 0，但将其`dynamic_size`参数设置为`True`，以便它可以超过初始大小。然后你将把这个输出数组中的第一个值设置为<START>标记：

Python

```py
decoder_output = TensorArray(dtype=int64, size=0, dynamic_size=True)
decoder_output = decoder_output.write(0, output_start)
```

+   迭代直到解码器序列长度，每次调用 Transformer 模型来预测一个输出标记。在这里，`training`输入被设置为`False`，然后传递到每个 Transformer 的`Dropout`层，以便在推断期间不丢弃任何值。然后选择得分最高的预测，并写入输出数组的下一个可用索引。当预测到<EOS>标记时，`for`循环将通过`break`语句终止：

Python

```py
for i in range(dec_seq_length):

    prediction = self.transformer(encoder_input, transpose(decoder_output.stack()), training=False)

    prediction = prediction[:, -1, :]

    predicted_id = argmax(prediction, axis=-1)
    predicted_id = predicted_id[0][newaxis]

    decoder_output = decoder_output.write(i + 1, predicted_id)

    if predicted_id == output_end:
        break
```

+   将预测的标记解码成输出列表并返回：

Python

```py
output = transpose(decoder_output.stack())[0]
output = output.numpy()

output_str = []

# Decode the predicted tokens into an output list
for i in range(output.shape[0]):

   key = output[i]
   translation = dec_tokenizer.index_word[key]
   output_str.append(translation)

return output_str
```

迄今为止的完整代码清单如下：

Python

```py
from pickle import load
from tensorflow import Module
from keras.preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, int64, TensorArray, argmax, newaxis, transpose
from model import TransformerModel

# Define the model parameters
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of model layers' outputs
d_ff = 2048  # Dimensionality of the inner fully connected layer
n = 6  # Number of layers in the encoder stack

# Define the dataset parameters
enc_seq_length = 7  # Encoder sequence length
dec_seq_length = 12  # Decoder sequence length
enc_vocab_size = 2405  # Encoder vocabulary size
dec_vocab_size = 3858  # Decoder vocabulary size

# Create model
inferencing_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, 0)

class Translate(Module):
    def __init__(self, inferencing_model, **kwargs):
        super(Translate, self).__init__(**kwargs)
        self.transformer = inferencing_model

    def load_tokenizer(self, name):
        with open(name, 'rb') as handle:
            return load(handle)

    def __call__(self, sentence):
        # Append start and end of string tokens to the input sentence
        sentence[0] = "<START> " + sentence[0] + " <EOS>"

        # Load encoder and decoder tokenizers
        enc_tokenizer = self.load_tokenizer('enc_tokenizer.pkl')
        dec_tokenizer = self.load_tokenizer('dec_tokenizer.pkl')

        # Prepare the input sentence by tokenizing, padding and converting to tensor
        encoder_input = enc_tokenizer.texts_to_sequences(sentence)
        encoder_input = pad_sequences(encoder_input, maxlen=enc_seq_length, padding='post')
        encoder_input = convert_to_tensor(encoder_input, dtype=int64)

        # Prepare the output <START> token by tokenizing, and converting to tensor
        output_start = dec_tokenizer.texts_to_sequences(["<START>"])
        output_start = convert_to_tensor(output_start[0], dtype=int64)

        # Prepare the output <EOS> token by tokenizing, and converting to tensor
        output_end = dec_tokenizer.texts_to_sequences(["<EOS>"])
        output_end = convert_to_tensor(output_end[0], dtype=int64)

        # Prepare the output array of dynamic size
        decoder_output = TensorArray(dtype=int64, size=0, dynamic_size=True)
        decoder_output = decoder_output.write(0, output_start)

        for i in range(dec_seq_length):

            # Predict an output token
            prediction = self.transformer(encoder_input, transpose(decoder_output.stack()), training=False)

            prediction = prediction[:, -1, :]

            # Select the prediction with the highest score
            predicted_id = argmax(prediction, axis=-1)
            predicted_id = predicted_id[0][newaxis]

            # Write the selected prediction to the output array at the next available index
            decoder_output = decoder_output.write(i + 1, predicted_id)

            # Break if an <EOS> token is predicted
            if predicted_id == output_end:
                break

        output = transpose(decoder_output.stack())[0]
        output = output.numpy()

        output_str = []

        # Decode the predicted tokens into an output string
        for i in range(output.shape[0]):

            key = output[i]
            print(dec_tokenizer.index_word[key])

        return output_str
```

### 想开始构建带有注意力机制的 Transformer 模型吗？

立即参加我的免费 12 天电子邮件速成课程（包含示例代码）。

点击注册并获取课程的免费 PDF 电子书版本。

## **测试代码**

为了测试代码，让我们查看你在[准备训练数据集](https://machinelearningmastery.com/?p=13879&preview=true)时保存的`test_dataset.txt`文件。这个文本文件包含了一组英语-德语句子对，已保留用于测试，你可以从中选择几句进行测试。

让我们从第一句开始：

Python

```py
# Sentence to translate
sentence = ['im thirsty']
```

对于这一句的对应德语原文翻译，包括<START>和<EOS>解码器标记，应为：`<START> ich bin durstig <EOS>`。

如果你查看这个模型的[绘制训练和验证损失曲线](https://machinelearningmastery.com/?p=13879&preview=true)（在这里你正在训练 20 轮），你可能会注意到验证损失曲线显著减缓，并在第 16 轮左右开始趋于平稳。

现在让我们加载第 16 轮的保存模型权重，并查看模型生成的预测：

Python

```py
# Load the trained model's weights at the specified epoch
inferencing_model.load_weights('weights/wghts16.ckpt')

# Create a new instance of the 'Translate' class
translator = Translate(inferencing_model)

# Translate the input sentence
print(translator(sentence))
```

运行上面的代码行会生成以下翻译后的单词列表：

Python

```py
['start', 'ich', 'bin', 'durstig', ‘eos']
```

这等同于期望的德语原文句子（请始终记住，由于你是从头开始训练 Transformer 模型，结果可能会因为模型权重的随机初始化而有所不同）。

让我们看看如果您加载了一个对应于较早 epoch（如第 4 个 epoch）的权重集会发生什么。在这种情况下，生成的翻译如下：

Python

```py
['start', 'ich', 'bin', 'nicht', 'nicht', 'eos']
```

英文中的翻译为：*我不是不*，这显然与输入的英文句子相去甚远，但这是预期的，因为在这个 epoch 中，Transformer 模型的学习过程仍处于非常早期的阶段。

让我们再试试测试数据集中的第二个句子：

Python

```py
# Sentence to translate
sentence = ['are we done']
```

这句话的德语对应的地面真相翻译，包括<START>和<EOS>解码器标记，应为：<START> sind wir dann durch <EOS>。

使用保存在第 16 个 epoch 的权重的模型翻译此句子为：

Python

```py
['start', 'ich', 'war', 'fertig', 'eos']
```

相反，这句话的翻译是：*我已准备好*。尽管这也不等同于真相，但它*接近*其意思。

然而，最后的测试表明，Transformer 模型可能需要更多的数据样本来有效训练。这也得到了验证损失曲线在验证损失平稳期间保持相对较高的支持。

的确，Transformer 模型以需求大量数据而闻名。例如，[Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)在训练其英语到德语翻译模型时，使用了包含大约 450 万个句对的数据集。

> *我们在标准的 WMT 2014 英德数据集上进行了训练，该数据集包含约 450 万个句对…对于英法，我们使用了数量显著更多的 WMT 2014 英法数据集，其中包含了 3600 万个句子…*
> 
> *–* [全神关注](https://arxiv.org/abs/1706.03762), 2017.

他们报告称，他们花费了 8 个 P100 GPU、3.5 天的时间来训练英语到德语的翻译模型。

相比之下，您只在此处的数据集上进行了训练，其中包括 10,000 个数据样本，分为训练、验证和测试集。

所以下一个任务实际上是给你。如果您有可用的计算资源，请尝试在更大的句子对集上训练 Transformer 模型，并查看是否可以获得比在有限数据量下获得的翻译结果更好的结果。

## **进一步阅读**

本节提供了更多关于这一主题的资源，如果您希望深入了解。

### **书籍**

+   [Python 深度学习进阶](https://www.amazon.com/Advanced-Deep-Learning-Python-next-generation/dp/178995617X), 2019

+   [自然语言处理中的 Transformer](https://www.amazon.com/Transformers-Natural-Language-Processing-architectures/dp/1800565798), 2021

### **论文**

+   [全神关注](https://arxiv.org/abs/1706.03762), 2017

## **总结**

在本教程中，您学会了如何对训练过的 Transformer 模型进行神经机器翻译推理。

具体来说，您学到了：

+   如何对训练过的 Transformer 模型进行推理

+   如何生成文本翻译

您有任何问题吗？

在下方的评论中提出你的问题，我会尽力回答。
