# 更深入的 Stable Diffusion 管道与 Diffusers

> 原文：[`machinelearningmastery.com/further-stable-diffusion-pipeline-with-diffusers/`](https://machinelearningmastery.com/further-stable-diffusion-pipeline-with-diffusers/)

有许多方法可以访问 Stable Diffusion 模型并生成高质量图像。一种流行的方法是使用 Diffusers Python 库。它提供了一个简单的接口来使用 Stable Diffusion，使得利用这些强大的 AI 图像生成模型变得更加容易。`diffusers` 降低了使用前沿生成 AI 的门槛，使得快速实验和开发成为可能。这个库非常强大。不仅可以用来根据文本提示生成图片，还可以利用 LoRA 和 ControlNet 创建更好的图像。

在这篇文章中，你将学习关于 Hugging Face 的 Diffusers、如何生成图像以及如何应用类似于 Stable Diffusion WebUI 的各种图像生成技术。具体来说，你将学习如何：

+   构建一个 Diffusers 管道，并通过提示生成一个简单的图像。

+   加载微调模型的 LoRA 权重并生成 IKEA 风格的图像。

+   构建 ControlNet OpenPose 管道，使用参考图像生成图像。

**通过我的书** [《掌握稳定扩散的数字艺术》](https://machinelearningmastery.com/mastering-digital-art-with-stable-diffusion/) **来启动你的项目**。它提供了**自学教程**和**可运行的代码**。

让我们开始吧。

![](img/34c4912f68b0bec21520237b0a1fc7ef.png)

更深入的 Stable Diffusion 管道与 Diffusers

照片由 [Felicia Buitenwerf](https://unsplash.com/photos/white-and-black-light-bulb-8xFgmFnOnAg) 提供。保留所有权利。

## 概览

本文分为三个部分；它们是：

+   在 Google Colab 上使用 Diffusers

+   加载 LoRA 权重

+   ControlNet OpenPose

## 在 Google Colab 上使用 Diffusers

Hugging Face 的 diffusers 是一个 Python 库，允许你访问用于生成真实图像、音频和 3D 分子结构的预训练扩散模型。你可以用它进行简单推断或训练自己的扩散模型。这个库的特别之处在于，只需几行代码，你就可以从 Hugging Face Hub 下载模型并用其生成图像，类似于 Stable Diffusion WebUI。

不需要在本地设置，你可以使用 Google Colab 的免费 GPU 基础笔记本。为此，请访问 [`colab.research.google.com/`](https://colab.research.google.com/) 并创建一个新的笔记本。要访问 GPU，你必须前往“运行时” → “更改运行时类型”并选择“T4 GPU”选项。

![](img/d453ae11b11152c6017994290692e9a0.png)

在 Google Colab 上选择 GPU

使用 Colab 可以让你避免拥有 GPU 设备以高效运行 Stable Diffusion 的负担。由于 Jupyter 笔记本的特性，你只需要将所有以下代码保存在各自的单元格中运行。这将方便你进行实验。

之后，安装所有必要的 Python 库来运行`diffusers`管道。您需要创建一个笔记本单元，包含以下行：

```py
!pip install diffusers transformers scipy ftfy peft accelerate -q
```

在 colab 笔记本中，行首的`!`表示这是一个系统命令，而不是 Python 代码。

要使用提示生成图像，必须首先创建一个 Diffusion 管道。接下来，您将下载并使用“float 16”类型的 Stable Diffusion XL 以节省内存。然后，您将设置一个使用 GPU 作为加速器的管道。

```py
from diffusers import DiffusionPipeline
import torch

pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16)
pipe.to("cuda");
```

要生成一个年轻女性的图像，您将为管道提供相同的通用提示。

```py
prompt = "photo of young woman, sitting outside restaurant, color, wearing dress, " \
    "rim lighting, studio lighting, looking at the camera, up close, perfect eyes"

image = pipe(prompt).images[0]

image
```

如您所见，您用几行代码就获得了异常的结果：

![](img/c1e6045c73d34ce4f01d60520c127531.png)

使用 Stable Diffusion XL 管道生成的 diffusers 库图像

类似于 Stable Diffusion WebUI，您可以提供正向提示、负向提示、推理步骤、设置随机种子、更改大小和指导比例来生成您想要的图像：

```py
prompt = "Astronaut in space, realistic, detailed, 8k"
neg_prompt = "ugly, deformed, disfigured, poor details, bad anatomy"
generator = torch.Generator("cuda").manual_seed(127)

image = pipe(
    prompt,
    num_inference_steps=50,
    generator=generator,
    negative_prompt=neg_prompt,
    height=512,
    width=912,
    guidance_scale=6,
).images[0]

image
```

图片完美无缺，看起来像是一位数字艺术家花费了将近 200 小时创作：

![](img/73fd244c562d96bb4fd921f7f89e18f7.png)

使用 Stable Diffusion XL 管道生成的另一张图片

## 加载 LoRA 权重

您不仅可以直接调用管道，还可以将 LoRA 权重加载到您的管道中。LoRA 权重是针对特定类型图像进行微调的模型适配器。它们可以附加到基础模型以生成定制结果。接下来，您将使用 LoRA 权重生成 IKEA 说明图像风格的图像。

您将通过提供 Hugging Face 链接、适配器在存储库中的位置和适配器名称来下载和加载 LoRA 适配器`ostris/ikea-instructions-lora-sdxl`。

```py
pipe.load_lora_weights(
    "ostris/ikea-instructions-lora-sdxl",
    weight_name="ikea_instructions_xl_v1_5.safetensors",
    adapter_name="ikea",
)
```

要生成 IKEA 风格的图像，您将为管道提供一个简单的提示、推理步骤、规模参数和手动种子。

```py
prompt = "super villan"

image = pipe(
    prompt,
    num_inference_steps=30,
    cross_attention_kwargs={"scale": 0.9},
    generator=torch.manual_seed(125),
).images[0]

image
```

您创建了一个带有说明书的超级反派。虽然不完美，但可以用来为您的工作生成定制图像：

![](img/6bcc1d5a4d8f670dd886abcfe055d8ef.png)

使用 LoRA 生成的 IKEA 风格图片

## ControlNet OpenPose

让我们再看一个扩展。现在，您将使用 ControlNet OpenPose 模型来生成一个控制图像，使用参考图像。ControlNet 是一种神经网络架构，通过添加额外条件来控制扩散模型。

您将安装`controlnet_aux`来检测图像中身体的姿势。

```py
!pip install controlnet_aux -q
```

然后，您将通过从 Hugging Face Hub 加载 fp16 类型的模型来构建 ControlNet 管道。之后，您将使用来自 Pexels.com 的免费图像链接到我们的环境中。

```py
from diffusers import ControlNetModel, AutoPipelineForText2Image
from diffusers.utils import load_image
import torch

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_OpenPose",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

original_image = load_image(
"https://images.pexels.com/photos/1701194/pexels-photo-1701194.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
)
```

要显示图像网格，您将创建一个 Python 函数，该函数接受图像列表并在 Colab 笔记本中以网格形式显示它们。

```py
from PIL import Image

def image_grid(imgs, rows, cols, resize=256):
    assert len(imgs) == rows * cols

    if resize is not None:
        imgs = [img.resize((resize, resize)) for img in imgs]
    w, h = imgs[0].size
    grid_w, grid_h = cols * w, rows * h
    grid = Image.new("RGB", size=(grid_w, grid_h))

    for i, img in enumerate(imgs):
        x = i % cols * w
        y = i // cols * h
        grid.paste(img, box=(x, y))
    return grid
```

在下一步中，您将构建 OpenPose 检测器管道并将其馈送到加载的图像中。为了将原始图像和 OpenPose 图像并排显示，您将使用 `image_grid` 函数。

```py
from controlnet_aux import OpenPoseDetector

model = OpenPoseDetector.from_pretrained("lllyasviel/ControlNet")
pose_image = model(original_image)

image_grid([original_image,pose_image], 1, 2)
```

检测器成功生成了人体姿势的结构。

![](img/e944bb08888a21f5c184b10ef1dfbbd8.png)

原始图像和检测到的姿势。请注意，这两张图片都是 1:1 的长宽比，以匹配 Stable Diffusion 的默认设置。

现在，您将把所有内容结合起来。您将创建 Stable Diffusion 1.5 文本到图像管道，并提供 ControlNet OpenPose 模型。您正在使用 fp16 变体进行内存优化。

```py
controlnet_pipe = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")
```

您将使用相同的正面和负面提示生成四张图像，并在网格中显示它们。请注意，您提供的是姿势图像而不是原始图像。

```py
prompt = "a woman dancing in the rain, masterpiece, best quality, enchanting, " \
         "striking, beach background"
neg_prompt = "worst quality, low quality, lowres, monochrome, greyscale, " \
             "multiple views, comic, sketch, bad anatomy, deformed, disfigured, " \
             "watermark, multiple_views, mutation hands, watermark, bad facial"

image = controlnet_pipe(
    prompt,
    negative_prompt=neg_prompt,
    num_images_per_prompt = 4,
    image=pose_image,
).images
image_grid(image, 1, 4)
```

结果非常出色。所有女性都在同一个姿势中跳舞。有一些畸变，但从 Stable Diffusion 1.5 中不能期望太多。

![](img/a6cf788480411ef491e669204b9fcd00.png)

使用 ControlNet 管道生成了四张图像

## 进一步阅读

如果你想深入了解这个主题，本节提供了更多资源。

+   [diffusers API 手册](https://huggingface.co/docs/diffusers/main/en/index)

+   [DiffusionPipeline API](https://huggingface.co/docs/diffusers/main/en/api/diffusion_pipeline)

+   [AutoPipelineForText2Image API](https://huggingface.co/docs/diffusers/en/api/pipelines/auto_pipeline)

+   [controlnet_aux Python 包](https://github.com/huggingface/controlnet_aux)

## 总结

在本文中，您了解了 Hugging Face Diffuser 库及其如何生成高质量和定制图像的用法。具体来说，您涵盖了：

+   Diffusers 是什么，它是如何工作的？

+   如何应用高级设置和负面提示来生成一致的图像。

+   如何加载 LoRA 权重以生成 IKEA 风格的图像。

+   如何使用 ControlNet OpenPose 模型控制 Stable Diffusion 的输出。
