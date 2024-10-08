# 使用稳定扩散的 OpenPose

> 原文：[`machinelearningmastery.com/openpose-with-stable-diffusion/`](https://machinelearningmastery.com/openpose-with-stable-diffusion/)

我们刚刚了解了 ControlNet。现在，让我们探索基于人体姿势控制角色的最有效方式。OpenPose 是一个强大的工具，可以在图像和视频中检测身体关键点的位置。通过将 OpenPose 与稳定扩散集成，我们可以引导 AI 生成与特定姿势匹配的图像。

在这篇文章中，您将了解 ControlNet 的 OpenPose 及如何使用它生成类似的姿势角色。具体来说，我们将涵盖：

+   Openpose 是什么，它是如何工作的？

+   如何使用 ControlNet Hugging Face Spaces 使用参考图像生成精确图像。

+   如何在稳定扩散的 WebUI 中设置 OpenPose 并使用它创建高质量的图像。

+   不同的 OpenPose 处理器专注于身体的特定部位。

**使用我的书[《稳定扩散数字艺术精通》](https://machinelearningmastery.com/mastering-digital-art-with-stable-diffusion/)**来**启动您的项目**。它提供了**自学教程**和**工作代码**。

让我们开始吧。

![](img/3606e61ba6df95141b1126b5f32b458e.png)

使用稳定扩散的 OpenPose

照片由[engin akyurt](https://unsplash.com/photos/brown-wooden-figurine-on-red-wooden-surface-udh1F6tuOr8)拍摄。部分权利保留。

## 概述

这篇文章分为四个部分，它们是：

+   什么是 ControlNet OpenPose？

+   在 Hugging Face Space 中的 ControlNet

+   在稳定扩散的 Web UI 中的 OpenPose 编辑器

+   图像生成

## 什么是 ControlNet OpenPose？

OpenPose 是一种深度学习模型，用于从图像中检测人体姿势。它的输出是图片中人物的多个**关键点**（如肘部、手腕和膝盖）的位置。ControlNet 中的 OpenPose 模型将这些关键点作为额外的条件输入到扩散模型中，并生成与这些关键点对齐的人物图像。一旦您能指定关键点的精确位置，就能够基于骨架图生成真实的人体姿势图像。您可以使用它来创建不同姿势的艺术照片、动画或插图。

## 在 Hugging Face Spaces 中的 ControlNet

要尝试 ControlNet OpenPose 模型的能力，您可以在 Hugging Face Spaces 的免费在线演示中使用：

+   [`hf.co/spaces/hysts/ControlNet-v1-1`](https://hf.co/spaces/hysts/ControlNet-v1-1)

开始时，您需要创建姿势关键点。这可以通过上传图像并让 OpenPose 模型检测它们来轻松完成。首先，您可以下载[Yogendra Singh](https://www.pexels.com/photo/dancing-man-wearing-pants-and-long-sleeved-shirt-1701194/)的照片，然后将其上传到 ControlNet Spaces。这个 ControlNet 帮助您锚定姿势，但您仍然需要提供文本提示以生成图片。让我们写一个简单的提示：“一个女人在雨中跳舞。”，然后点击运行按钮。

![](img/8410b1e5c81d06b59d90f605a984a48c.png)

在 Hugging Face Spaces 上使用 OpenPose ControlNet 模型

由于图像生成的随机性，你可能需要进行多次尝试。你还可以优化提示，以提供更多细节，例如光照、场景和女性穿着的服装。你甚至可以展开底部的“高级选项”面板，提供更多设置，例如负面提示。

![](img/69642440b5561e212027d88f4db558a3.png)

“高级选项”面板中的设置

在上述示例中，你可以看到从骨架图像生成的高质量女性在雨中跳舞的图像，与上传的图像姿势类似。以下是三个在相同提示下生成的其他图像，所有图像都非常出色，并准确地遵循了参考图像的姿势。

![](img/ca2b3dbe3be2f5e251bfa3c185ac4fa6.png)

使用相同提示生成的其他图像

## Stable Diffusion Web UI 中的 OpenPose 编辑器

你还可以使用 Stable Diffusion Web UI 中的 OpenPose ControlNet 模型。实际上，你不仅可以上传图像以获取姿势，还可以在应用到扩散模型之前编辑姿势。在本节中，你将学习如何在本地设置 OpenPose 并使用 OpenPose 编辑器生成图像。

在开始使用 OpenPose 编辑器之前，你需要先安装它并下载模型文件。

1.  确保你已安装 ControlNet 扩展，如果没有，请查看之前的帖子。

1.  安装 OpenPose 编辑器扩展：在 WebUI 的“Extensions”标签中，点击“Install from URL”并输入以下网址进行安装：

    +   https://github.com/fkunn1326/openpose-editor

1.  前往 Hugging Face 仓库：[`hf.co/lllyasviel/ControlNet-v1-1/tree/main`](https://hf.co/lllyasviel/ControlNet-v1-1/tree/main)

1.  下载 OpenPose 模型 “[control_v11p_sd15_openpose.pth](https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_openpose.pth)”

1.  将模型文件放入 SD WebUI 目录中的 stable-diffusion-webui/extensions/sd-webui-controlnet/models 或 stable-diffusion-webui/models/ControlNet 文件夹中

现在你已经完成了所有设置，并且 Web UI 中添加了一个名为“OpenPose Editor”的新标签。导航到“OpenPose Editor”标签，并根据你的喜好调整画布的宽度和高度。接下来，你可以开始使用鼠标修改右侧的骨架图像。这是一个简单的过程。

让我们尝试创建一张男人拿着大枪的图片。你可以对骨架图像进行修改，使其看起来像下面的样子：

![](img/f457dd5e4e12f3425c110256c1693677.png)

使用 OpenPose 编辑器创建姿势

然后，点击“Send to text2img”按钮。这将带你进入 text2img 界面，并将骨架图像添加到 ControlNet 面板中。

![](img/2c06ba12b0b0c0354ef90c988b663830.png)

ControlNet 面板上创建的姿势

接下来，为该 ControlNet 模型选择“启用”，确保选中“OpenPose”选项。您还可以选中“低 VRAM”和“像素完美”。前者适用于 GPU 内存不足的计算机，后者是为了使 ControlNet 模型使用最佳分辨率以匹配输出。

然后，设置正面和负面提示，调整输出图像大小、采样方法和采样步骤。例如，正面提示可以是

> 详细，杰作，最佳质量，令人惊叹，迷人，引人注目，汤姆克兰西的分裂，持枪的人，美国海军陆战队员，海滩背景

负面提示可以是

> 最差质量，低质量，低分辨率，单色，灰度，多视角，漫画，素描，解剖不良，畸形，变形，水印，多视角，变异手部，水印，面部不佳

下面的图像，使用尺寸为 912×512 和 DDIM 采样器进行 30 步，结果完全匹配相似的姿势，并且有很好的细节。

![](img/18199b4bcb421afb2ac7c46dedaa3345.png)

使用 OpenPose ControlNet 模型生成的输出

## 图像生成

如果您在 Web UI 中尝试了 ControlNet 模型，您应该注意到有多个 OpenPose 预处理器。接下来，让我们探索其中一些，重点放在面部和上半身。

我们将使用 [Andrea Piacquadio 的照片](https://www.pexels.com/photo/woman-in-white-blazer-holding-tablet-computer-789822/) 作为参考图像。在 Web UI 中，切换到“img2img”选项卡并上传参考图像。然后在 ControlNet 面板中，启用并选择“OpenPose”作为控制类型。在 img2img 中，默认情况下将与 ControlNet 共享参考图像。接下来，在 ControNet 面板中将预处理器改为“openpose_face”，如下所示：

![](img/c3c92ab6adea486406ccc31de9e89e94.png)

使用“openpose_face”作为预处理器

然后，将正面提示设置为与参考图像风格相匹配，并生成图像。不再是拿着平板电脑的图片，让我们让这位女士拿着手机：

> 详细，最佳质量，令人惊叹，迷人，引人注目，纽约，建筑，城市，手机放在耳边

以下是可能获得的结果：

![](img/0cea84197b9246661d5f40ff6962ee8a.png)

使用 img2img 生成的图像

我们通过类似的姿势获得了高质量的结果。您需要调整提示来匹配这个姿势。这里使用的预处理器是“openpose_face”，意味着不仅匹配姿势还包括面部表情。因此，生成的图片不仅在肢体位置上与参考图像匹配，而且在面部表情上也是如此。

让我们将预处理器更改为“openpose_faceonly”，只专注于面部特征。这样，只有面部关键点被识别，不会从 ControlNet 模型应用有关身体姿势的信息。现在，将提示设置为

> 详细，最佳质量，令人惊叹，迷人，引人注目，纽约，建筑，城市

按照提示中的每个关键词生成了一个更准确的结果，但身体姿势与之前的姿势有很大不同：

![](img/c8d056c474f104ed7845ca78c61509c7.png)

使用 ControlNet 仅提供面部关键点生成的图像

要了解为什么会这样，你可以检查**预处理器**的输出图像。上面的图像是使用“openpose_face”预处理器生成的，而下面的图像是使用“openpose_faceonly”生成的。同样，你可以通过分析这两种骨架结构来了解各种预处理器的输出。

![](img/6a8ce4fcfff7bd56d07814d7370cbbc3.png)

从不同 OpenPose 预处理器生成的关键点

## 进一步阅读

本节提供了更多资源，如果你希望深入了解这一主题。

+   [OpenPose: 实时多人 2D 姿态估计使用部件关联场](https://arxiv.org/abs/1812.08008) 作者 Cao 等（2019）

+   [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) 在 GitHub 上

+   [Controlnet – 人体姿态版本](https://huggingface.co/lllyasviel/sd-controlnet-openpose) 在 Hugging Face 上

+   [Openpose Controlnets (V1.1): 使用姿势和生成新姿势](https://civitai.com/articles/157/openpose-controlnets-v11-using-poses-and-generating-new-ones)

## 总结

在这篇文章中，我们深入探讨了 ControlNet OpenPose 的世界以及如何利用它获得精准的结果。具体来说，我们讨论了：

+   什么是 OpenPose，它如何在不设置任何东西的情况下立即生成图像？

+   如何使用 Stable Diffusion WebUI 和 OpenPose 编辑器通过修改提示和骨架图像生成自定义姿势的图像。

+   多种 OpenPose 预处理器，用于在 Stable Diffusion WebUI 中生成全脸和仅脸部的图像。
