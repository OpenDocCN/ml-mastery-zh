# ChatGPT 作为你的专家助手

> 原文：[`machinelearningmastery.com/chatgpt-as-your-expert-helper/`](https://machinelearningmastery.com/chatgpt-as-your-expert-helper/)

ChatGPT 可以帮助我们学习新的编程语言、课程、技术和技能。它已经成为许多专业人士提升工作流程或学习新事物的首选工具。ChatGPT 专家助手提示可以减少我们对 Google 的依赖，并提供详细的目标实现计划。

在这篇文章中，你将学习如何利用简单的 ChatGPT 提示来学习技术或执行专门的任务。具体来说，你将学习如何：

+   生成在 TPU（张量处理单元）上训练模型的指令。

+   生成详细的安全砍伐树木的指令。

+   编写专家助手提示。

**开始使用 ChatGPT**，请参考我的书籍 [《利用 ChatGPT 最大化生产力》](https://machinelearningmastery.com/productivity-with-chatgpt/)。这本书提供了**真实的使用案例**和**提示示例**，旨在让你快速上手 ChatGPT。

让我们开始吧！[](../Images/fc0bd1a026a8bba433846cb8f60ccd59.png)

ChatGPT 作为你的专家助手

图片由 Adrian Tam 使用 Stable Diffusion 生成。保留部分版权。

## 概述

本文分为三个部分；它们是

+   使用 TPU 加速机器学习模型

+   安全砍伐树木指南

+   专家助手提示的写作技巧

## 使用 TPU 加速机器学习模型

在第一个示例中，你将处理许多数据科学家可能不知晓的专门问题。为了加速机器学习模型训练，大多数工程师目前使用 GPU（图形处理单元）或多个 GPU。然而，TPU（张量处理单元）可以提供更快的训练时间。你可能找不到很多关于 TPU 使用的指令，但你可以通过编写正确的提示向 ChatGPT 寻求帮助。

在你的提示中，你应该定义：

+   **角色**：ChatGPT 应表现得像一位专业的机器学习工程师。

+   **请求**：编写用于在 TPU 上训练图像分类器的代码。

+   **环境**：Google Colab。

例如，提示可以是以下内容：

> 作为专业的机器学习工程师，编写用于在 TPU 上训练 TensorFlow 图像分类模型的 Python 代码。计算环境为 Google Colab。

这可能是 ChatGPT 响应中看到的输出：

```py
import tensorflow as tf

# Check if TPU is available
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print("Running on TPU:", tpu.master())
except ValueError:
    strategy = tf.distribute.OneDeviceStrategy("GPU")
    print("Running on GPU or CPU")

# Load and preprocess your dataset
# Replace `your_dataset` with your own dataset loading and preprocessing code

# Define the model architecture
# Replace `your_model` with your own model architecture

# Compile the model
# Replace `your_loss`, `your_metrics`, and `your_optimizer` with appropriate values
with strategy.scope():
    model = your_model
    model.compile(loss=your_loss, metrics=your_metrics, optimizer=your_optimizer)

# Define training parameters
# Replace `your_epochs` and `your_batch_size` with appropriate values
epochs = your_epochs
batch_size = your_batch_size

# Train the model
# Replace `your_training_data` with your own training data
model.fit(your_training_data, epochs=epochs, batch_size=batch_size)

# Evaluate the model
# Replace `your_test_data` with your own test data
loss, accuracy = model.evaluate(your_test_data)
print("Test loss:", loss)
print("Test accuracy:", accuracy)
```

ChatGPT 示例展示了如何通过生成带有占位符的 Python 代码在 TPU 上训练模型。为了正确执行代码，你必须用你的数据集加载和预处理代码、模型架构、损失函数、评估指标、优化器、训练参数、训练数据和测试数据替换占位符。

替换代码后，你可以将其复制并粘贴到新的 Colab 笔记本中，并使用 TPU 硬件加速运行代码。你获得的代码可能无法完全执行，并且可能包含一些错误。然而，你应该能够轻松识别和纠正这些错误。

**注意：** ChatGPT 的输出不总是正确的。你必须使用其他方法验证其准确性。

## 安全砍伐树木指南

在第二个示例中，你将专注于安全砍伐树木所需的非技术性技巧。为了获得适当的指导，你必须在提示中提供足够的情况，例如：

+   **角色：** ChatGPT 应表现为一名经验丰富的专业树艺师。

+   **请求**：列出安全砍伐树木所需的步骤。

+   **环境**：树木为枫树，高 50 英尺，位于两栋房屋之间。

+   **位置**：郊区，山景城，加利福尼亚州。

例如，ChatGPT 的提示可以是：

> 作为一名经验丰富的专业树艺师，列出在加利福尼亚州山景城的一个郊区邻里中，位于两栋房屋之间的 50 英尺高的枫树的移除的关键安全程序。

这可能是你在输出中看到的内容：

> 作为一名经验丰富的专业树艺师，我可以为你提供有关在加利福尼亚州山景城的一个郊区邻里中，位于两栋房屋之间的 50 英尺高的枫树移除的关键安全程序的清单。值得注意的是，树木移除可能是危险的，我强烈建议在进行此类任务时咨询认证的树艺师或树木移除专业人员。然而，以下是一些需要考虑的一般安全程序：
> 
> 1.  **进行全面评估**：在开始树木移除过程之前，评估树木的状况、潜在危险和周围环境。检查是否有腐烂、疾病或结构弱点的迹象，这些可能会影响树木的稳定性。
> 1.  
> 1.  **获取必要的许可证**：请与当地当局联系，并获得在加利福尼亚州山景城进行树木移除所需的任何许可证或许可。遵守当地法规对于确保安全和避免法律问题至关重要。
> 1.  
> 1.  **使用适当的个人防护装备（PPE）**：佩戴适当的 PPE，包括安全帽、安全眼镜、耳罩、手套和结实的防滑靴。PPE 有助于保护你免受掉落的碎片和其他潜在危险的伤害。
> 1.  
> 1.  **建立安全区**：清晰标记并保护树木周围的区域，以防止未经授权的进入。使用警戒带或路障将行人和车辆隔离在工作区外。
> 1.  
> 1.  **规划设备和人员**：确定安全树木移除所需的设备、工具和人员。这可能包括链锯、绳索、索具设备和经验丰富的队伍。
> 1.  
> 1.  **采取适当的修剪技术**：如有必要，进行树木修剪以去除枯死的树枝或减少树木的重量，以便在主要移除过程中。适当的修剪技术有助于防止事故发生并控制树木的倒落方向。
> 1.  
> 1.  **进行受控的树木砍伐**：在砍伐树木时，采用适当的切割技术以确保控制倒落。考虑树木的倾斜、周围障碍物、风向以及可能影响其倒落路径的其他因素。
> 1.  
> 1.  **使用吊装和降落设备**：如果树木的位置对附近的建筑物或障碍物构成风险，使用吊装和降落设备安全控制树木的下降。这种技术有助于减少财产损坏和受伤的潜在风险。
> 1.  
> 1.  **分段进行树木移除**：处理较大的树木时，通常更安全的方法是分段移除。这涉及到小心地切割和降落树木的较小部分，而不是尝试一次性砍倒整棵树。
> 1.  
> 1.  **使用正确的树木移除技术**：遵循行业最佳实践进行树木移除，包括进行精确的切割、避免过度用力，并在整个过程中保持控制。注意潜在的危险，如树枝反弹和木材开裂。
> 1.  
> 1.  **考虑雇佣专业人士**：鉴于在两栋房子之间移除一棵 50 英尺高的树木的复杂性和潜在风险，强烈建议聘请认证的树艺师或经验丰富的树木移除公司。他们的专业知识和专用设备将有助于确保安全高效的移除过程。
> 1.  
> 请记住，树木移除可能极其危险，尤其是处理大型树木或靠近建筑物的树木时。始终将安全放在首位对于保护自己、团队和周围财产至关重要。

生成的列表涵盖了许多重要的安全步骤。然而，移除像描述的那样的大树是一项复杂的工作，可能需要专用设备、训练有素的人员、必要的许可证、正确的技术、关键信息的获取和应急计划。即使你聘请了专业人员，在批准之前，你需要访问树木的位置、区域和分布情况。

总之，尽管 ChatGPT 提供了很多关于如何进行此类工作的细节，但跟随其指示可能不是一个好主意。雇佣认证的树艺师和有保险的树木移除公司来完成这项工作可能是更明智的选择。

## 编写专家帮助提示的技巧

想象一下咨询一位真正的律师或医生——你不会走进来模糊地说“我需要帮助”。你会详细解释你的具体请求的背景，包括谁、什么、哪里、为什么和如何。使用 ChatGPT 专家助手时也是相同的原则。

要生成有效的专家帮助提示，请提供尽可能多的相关背景信息。解释：

+   你希望获得帮助的专家的角色或职业。要具体。例如：“认证的树艺师”

+   任务或请求的性质。要明确、直接，并提供关键细节。例如：“安全地移除这棵两栋房子之间的 50 英尺高的树。”

+   任何影响情况的变量、限制或条件。根据需要提供测量、位置和规格。

    引入复杂性的环境因素、情况或障碍。

+   你希望实现的预期结果或目标。

描述场景越准确、具体和全面，ChatGPT 的回应可能越精准和有用。ChatGPT 模拟“专家”的能力受到你提供的信息的清晰度和完整性的限制。

## 总结

在这篇文章中，我们讨论了如何设计 ChatGPT 专家助手提示以生成有用的回应。具体来说，我们涵盖了：

+   如何撰写一个详细解释的提示，以加速使用 TPU 的机器学习模型训练。

+   如何生成树木移除的关键安全程序列表。这比你想象的要复杂。

+   制作更有针对性的提示的技巧包括从相关专家的角度提供清晰的任务描述、必要的变量和明确的目标。
