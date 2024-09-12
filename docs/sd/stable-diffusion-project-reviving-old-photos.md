# 稳定扩散项目：复兴旧照片

> 原文：[`machinelearningmastery.com/stable-diffusion-project-reviving-old-photos/`](https://machinelearningmastery.com/stable-diffusion-project-reviving-old-photos/)

摄影已有一个多世纪的历史。周围有许多旧照片，可能您的家庭也有一些。受当时相机和胶片的限制，您可能有分辨率低、模糊或有折痕或划痕的照片。恢复这些旧照片，使它们像今天用相机拍摄的新照片一样，是一个具有挑战性的任务，但是即使您也可以使用诸如 Photoshop 等图片编辑软件来完成。

在这篇文章中，您将看到如何使用稳定扩散修复旧照片，并使它们焕发新生。完成本文后，您将学到：

+   如何清除扫描照片中的瑕疵

+   如何给黑白照片上色

使用我的书籍[《稳定扩散数字艺术大师》](https://machinelearningmastery.com/mastering-digital-art-with-stable-diffusion/)，**启动您的项目**。它提供**自学教程**和**可运行的代码**。

让我们开始吧。

![](img/e735625e075e47e33f278b9696c92248.png)

稳定扩散项目：复兴旧照片

照片由[Antonio Scant](https://unsplash.com/photos/black-and-silver-camera-on-brown-wooden-table-NrcF80ZC8EI)拍摄。部分权利保留。

## 概览

本教程分为三部分；它们是

+   项目构想

+   超分辨率

+   重建

## 项目构想

让我们来看一个来自纽约公共图书馆的[旧照片](https://digitalcollections.nypl.org/items/510d47d9-a975-a3d9-e040-e00a18064a99)：

![](img/e2d99de9bc5a97f153f0b984b8b67135.png)

“年轻的开蚝者”照片。来自 NYPL 数字收藏。

如果您下载它，您会注意到照片分辨率低。从胶片颗粒中有少许噪声（不太明显）。照片是黑白的。这个项目的目标是将这个照片制作成高分辨率彩色照片，描绘相同的人物。

## 超分辨率

原始图片分辨率低。将分辨率放大称为**超分辨率**，已开发了多个机器学习模型用于此目的。

处理旧照片的第一步是进行超分辨率处理。通常，旧照片分辨率较低，但这并不是问题所在。即使分辨率很高（例如，因为您以高分辨率扫描了旧照片），您可能仍希望对照片进行降采样，并运行超分辨率以去除噪声和胶片颗粒。

在稳定扩散 Web 界面中，您可以将照片上传到“额外”选项卡。该选项卡允许您执行许多操作，但与扩散过程无关。相反，它是应用于各种现有机器学习模型的图像。在这里，您可以启用“放大”并设置“按比例缩放”的合理因子。对于这张特定的照片，您可以将因子设置为 2。然后，您应该选择一个放大器，例如“R-ESRGAN 4x+”。

在“Extra”标签页上，您可以做的下一件事是使用 CodeFormer。这是一个用于修复面部的模型。启用它并设置权重。低权重可以使 CodeFormer 更自由地改变面部，而高权重则倾向于保留原始的面部表情。最佳权重应取决于原始照片的质量。如果面部上有更多划痕或标记，您希望使用较低的权重来更轻松地进行重建。

![](img/2954cd2775a760309601a1d230f60305.png)

超分辨率是处理旧照片的第一步。

您应该下载提升后的输出以进行下一步操作。

## 重建

要重建一张旧照片，您将使用 txt2img。您不应该使用 img2img，因为提供旧照片作为起点会对输出施加太多影响，您无法看到期望的修正。

但首先，您应该在 img2img 标签页中上传提升后的输出，并点击“CLIP 询问器”旁边的纸夹图标。这将根据上传的图像自动填充积极提示。您将根据 CLIP 询问器的结果在 txt2img 标签页构建您的提示。

![](img/72964ad1603a238aba499d764501d447.png)

您可以在 img2img 标签页中使用 CLIP 询问器。

现在前往 txt2img 标签页。让我们使用 SD1.5 模型进行逼真生成，例如 Realistic Vision v6。设置积极提示，如

> 一群孩子站在一起，手持水桶，穿着帽子和礼服，背景是一座建筑物，奥古斯特·桑德尔（August Sander）拍摄的彩色照片，WPA，美国巴比松学派，最佳质量，8K 原始照片，详细的面部

最后几个关键词是用来控制输出风格的。您可以使用一个常规的负面提示，例如

> 绘画，绘画，蜡笔，素描，石墨，印象派，嘈杂，模糊，柔和，畸形，丑陋，低分辨率，坏解剖，坏手，裁剪，最差质量，低质量，普通质量，JPEG 伪影，签名，水印，单色，灰度，旧照片

旧照片重建的关键是使用 ControlNet。您需要两个 ControlNet 单元以获得最佳结果。

首先将提升后的图像上传到第一个 ControlNet 单元，并设置类型为 Canny。记得启用此单元并勾选“Pixel Perfect”。这有助于 ControlNet 预处理器使用最佳分辨率。将第一个单元的控制权重设置为 0.8。

然后启用第二个 ControlNet 单元。上传相同的图像，打开 Pixel Perfect，并选择控制类型为“重新着色”。这是一个用于给黑白照片上色的 ControlNet 模型。您应该使用“recolor_luminance”模型作为预处理器。将第二单元的控制权重设置为 0.2。可选地，您可以调整伽马校正，如果需要微调输出的亮度。

记得在 txt2img 中设置输出尺寸，以便与原始图像的宽高比类似，并接近你的 Stable Diffusion 模型的原生分辨率。在此示例中，我们使用 760×600 像素。点击生成，你将看到以下内容：

![](img/5e133f015176cfe571055e79ae594851.png)

一张使用 Stable Diffusion 上色的旧照片

你可以下载结果。看看你得到的是什么：

![](img/4ec1c0f7cea482a217f3a72a1683bc4f.png)

重建的旧照片。

这张照片有些过曝，但你可以看到一张旧照片被复原。所有细节都被保留：人物的面部表情、他们穿的衣物上的污点等等。

那么它是如何工作的？这要求 Stable Diffusion 重新绘制照片。因此，你需要一个提示来指导绘制的扩散过程。但为了精准控制形状和人物，你使用了 Canny 类型的 ControlNet 来勾勒图像，并要求扩散过程符合轮廓。然而，这个轮廓并不完美，因为 Canny 边缘检测算法无法将照片转换为线条图。为了降低失真，你使用第二个 ControlNet 单元基于亮度重新上色输入照片。所有原始颜色被忽略（而且没有），颜色是基于机器学习模型填充的。然而，你不希望照片中有这些缺陷。因此，你为 Canny 设置了更高的权重，为 Recolor 设置了更低的权重。

如果你再次点击生成按钮，你可能会看到人物穿着不同颜色的衣物。这是因为模型对他们应该穿什么颜色没有把握。你可以在提示中描述他们的颜色来控制这一点。你还可以尝试关闭一个 ControlNet 单元并观察结果。只有当两个单元一起工作时，才能获得最佳结果。

关于带有面孔的照片的注意事项：如果你的原始照片状况很差，以至于人物面孔不太可辨认，你可能需要启用 ADetailer 来重建面孔。但仅在必要时进行！否则，你可能会发现你的照片描绘了完全不同的人。

在上述设置中，输出分辨率设置为 760×600。但是，你可能希望获得比 Stable Diffusion 模型支持的更高分辨率。你可以在 txt2img 中使用“高分辨率修复”功能，在图像生成后运行放大器。选项与 Extra 标签中的非常相似。但记得将放大器中的去噪强度设置为低值（如 0.2），因为你不希望施加额外的失真。

## 进一步阅读

本节提供了更多关于该主题的资源，如果你想深入了解。

+   [Lewis Wickes Hine 纪录片摄影，1905-1938](https://digitalcollections.nypl.org/collections/lewis-wickes-hine-documentary-photographs-1905-1938#/) 在纽约公共图书馆数字藏品

+   在[Civitai](https://civitai.com/models/4201/realistic-vision-v60-b1)和[Hugging Face Hub](https://huggingface.co/SG161222)上的**Realistic Vision**模型。

+   [OpenModelDB 用于提升模型](https://openmodeldb.info/)

+   ControlNet 的[重新上色模型](https://civitai.com/models/272562/controlnet-recolor)

+   ControlNet 的[canny 模型](https://huggingface.co/lllyasviel/sd-controlnet-canny)

## 摘要

在这篇文章中，你清理了一张旧照片。你去除了缺陷并给一张黑白照片上色，将其带入了现代。在这个过程中，你使用文本提示来驱动扩散过程，在粗糙层级生成图片。然后你使用 ControlNet 来微调输出。你在粗略层面控制了氛围，并在精细层面保留了细节。稳定扩散用于填补空白并重建了照片。