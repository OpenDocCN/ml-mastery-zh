# 使用 Python 运行 Stable Diffusion

> 原文：[`machinelearningmastery.com/running-stable-diffusion-with-python/`](https://machinelearningmastery.com/running-stable-diffusion-with-python/)

Stable Diffusion 是一个能够生成图片的深度学习模型。实际上，它是一个程序，你可以提供输入（例如文本提示），并获得一个表示像素数组的张量，这个张量可以保存为图像文件。没有规定必须使用特定的用户界面。在任何用户界面可用之前，你需要通过代码运行 Stable Diffusion。

在本教程中，我们将学习如何使用来自 Hugging Face 的 `diffusers` 库来运行 Stable Diffusion。

完成本教程后，你将学到

+   如何安装 `diffusers` 库及其依赖项

+   如何在 `diffusers` 中创建一个 pipeline

+   如何微调你的图像生成过程

**通过我的书籍** [《掌握数字艺术与稳定扩散》](https://machinelearningmastery.com/mastering-digital-art-with-stable-diffusion/) **来启动你的项目**。它提供了**自学教程**和**可运行的代码**。

让我们开始吧。

![](img/b30ccb65f63b9a59926a30d8a657f565.png)

在 Python 中运行 Stable Diffusion

图片由 [Himanshu Choudhary](https://unsplash.com/photos/orange-tabby-cat-lying-on-ground-RLo7QtKLyAY) 提供。保留所有权利。

## 概述

本教程分为三个部分；它们是

+   Diffusers 库简介

+   自定义 Stable Diffusion 管道

+   Diffusers 库中的其他模块

## Diffusers 库简介

Stable Diffusion 在文本到图像生成领域引起了轰动。它能够从文本描述中生成高质量、详细的图像，使其成为艺术家、设计师以及任何具有创造力的人士的强大工具。借助 Stable Diffusion 模型文件，你可以使用 PyTorch 重建深度学习模型，但因为涉及很多步骤，你需要编写大量代码。Hugging Face 的 Diffusers 库可以利用 Stable Diffusion 的潜力，让你创作出梦幻般的作品。

在使用之前，你应该在你的 Python 环境中安装 diffusers 库：

Shell

```py
pip install diffusers transformers accelerate
```

这些 Python 包有很多依赖项，包括 PyTorch。

在这篇文章中，你将使用 diffusers 库中的 pipeline 函数。之所以称为 pipeline，是因为不是单个深度学习模型能够从输入生成图片，而是许多较小的模型协同工作来实现这一点。我们来看一个例子：

```py
from diffusers import StableDiffusionPipeline, DDPMScheduler
import torch

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
                                               variant="fp16", torch_dtype=torch.float16)
pipe.to("cuda")
prompt = "A cat took a fish and running in a market"
scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear")
image = pipe(
    prompt,
    scheduler=scheduler,
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]
image.save("cat.png")
```

以下是生成一张图片并将其保存为 PNG 格式到 `cat.png` 的几行代码。这是生成的图片示例：

![](img/331984db9fff0d71aeed19f863bbd244.png)

使用 Stable Diffusion 管道生成的图片。

然而，后台正在进行大量工作。你传递了一个文本提示。这个提示已经通过预训练的嵌入模型转换成了数值张量。张量然后被传递给从 Hugging Face 仓库“CompVis/stable-diffusion-v1-4”（官方 Stable Diffusion v1.4 模型）下载的 Stable Diffusion 模型。这个模型将运行 30 步和 DDPM 调度器。Stable Diffusion 模型的输出将是一个浮点张量，必须转换成像素值后才能保存。所有这些都是通过将组件链式连接到`pipe`对象中来完成的。

## 自定义 Stable Diffusion 管道

在之前的代码中，你从 Hugging Face 仓库下载了一个预训练模型。即使是同一仓库，不同的“变体”模型也是可用的。通常，默认变体使用 32 位浮点数，适合在 CPU 和 GPU 上运行。你在上述代码中使用的变体是`fp16`，即使用 16 位浮点数。它并不总是可用的，也不总是以这种名称存在。你应该查看相应的仓库以了解更多详细信息。

由于使用的变体是 16 位浮点数，你还指定了`torch_dtype`使用`torch.float16`。注意，大多数 CPU 不能处理 16 位浮点数（也称为半精度浮点数），但它在 GPU 上是可用的。因此，你会看到创建的管道通过语句`pipe.to("cuda")`传递给了 GPU。

你可以尝试以下修改，你应该会观察到生成速度大幅变慢，因为它是在 CPU 上运行的：

```py
from diffusers import StableDiffusionPipeline, DDPMScheduler

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
prompt = "A cat took a fish and running in a market"
scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear")
image = pipe(
    prompt,
    scheduler=scheduler,
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]
image.save("cat.png")
```

然而，假设你一直在使用 Stable Diffusion Web UI 并下载了第三方 Stable Diffusion 模型，那么你应该对以 SafeTensors 格式保存的模型文件比较熟悉。这种格式不同于上述 Hugging Face 仓库的格式。特别是，仓库通常会包含一个`config.json`文件来描述如何使用模型，但这些信息应该从 SafeTensor 模型文件中推断出来。

你仍然可以使用你下载的模型文件。例如，使用以下代码：

```py
from diffusers import StableDiffusionPipeline, DDPMScheduler

model = "./path/realisticVisionV60B1_v60B1VAE.safetensors"
pipe = StableDiffusionPipeline.from_single_file(model)
pipe.to("cuda")
prompt = "A cat took a fish and running away from the market"
scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear")
image = pipe(
    prompt,
    scheduler=scheduler,
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]
image.save("cat.png")
```

这段代码使用了`StableDiffusionPipeline.from_single_file()`，而不是`StableDiffusionPipeline.from_pretrained()`。这个函数的参数被假定为模型文件的路径。它会识别文件是 SafeTensors 格式。`diffusers`库的整洁性在于，在你更改了创建管道的方法之后，不需要进行其他任何更改。

请注意，每个 Pipeline 假设了特定的架构。例如，`diffusers`库中的`StableDiffusionXLPipeline`专用于 Stable Diffusion XL。你不能使用不匹配的 Pipeline 构建器的模型文件。

当触发过程时，您可以在`pipe()`函数调用中看到稳定扩散图像生成过程的最重要参数。例如，您可以指定调度器、步长和 CFG 比例。调度器确实有另一组配置参数。您可以从`diffusers`库支持的众多调度器中进行选择，详情可以在 diffusers API 手册中找到。

例如，以下是使用更快的替代方案，即欧拉调度器，并保持其它一切不变：

```py
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

model = "./path/realisticVisionV60B1_v60B1VAE.safetensors"
pipe = StableDiffusionPipeline.from_single_file(model)
pipe.to("cuda")
prompt = "A cat took a fish and running away from the market"
scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012,
                                   beta_schedule="scaled_linear")
image = pipe(
    prompt,
    scheduler=scheduler,
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]
image.save("cat.png")
```

## Diffusers 库中的其他模块

`StableDiffusionPipeline`不是`diffusers`库中唯一的管道。如上所述，您有`StableDiffusionXLPipeline`适用于 XL 模型，但还有更多。例如，如果您不仅仅提供文本提示而是使用 img2img 调用稳定扩散模型，您必须使用`StableDiffusionImg2ImgPipeline`。您可以将 PIL 对象的图像作为管道的参数。您可以从`diffusers`文档中查看可用的管道：

+   [`huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/overview`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/overview)

即使有许多不同的管道，您应该发现它们都工作类似。工作流程与上面的示例代码非常相似。您会发现它易于使用，无需了解幕后的详细机制。

## 进一步阅读

如果您想更深入地了解这个主题，本节提供了更多资源。

+   [diffusers API 手册](https://huggingface.co/docs/diffusers/main/en/index)

+   [稳定扩散管道概述](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/overview)

+   [稳定扩散管道 API](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/text2img)

+   [欧拉调度器 API](https://huggingface.co/docs/diffusers/main/en/api/schedulers/euler)

+   [DDPM 调度器 API](https://huggingface.co/docs/diffusers/main/en/api/schedulers/ddpm)

## 总结

在本文中，您发现了如何从 Hugging Face 的`diffusers`库中使用。特别是，您了解到：

+   如何创建一个从提示创建图像的管道

+   如何重复使用本地模型文件而不是动态从在线仓库下载？

+   从扩散器库中可用的其他管道模型有哪些？
