# 使用 Dreambooth 训练稳定扩散

> 原文：[`machinelearningmastery.com/training-stable-diffusion-with-dreambooth/`](https://machinelearningmastery.com/training-stable-diffusion-with-dreambooth/)

Stable Diffusion 是在 LAION-5B 上训练的，这是一个包含数十亿通用图像文本对的大规模数据集。然而，在理解特定主题及其生成在不同语境中方面存在不足（通常是模糊的、模糊的或无意义的）。为了解决这个问题，对稳定扩散进行特定用例的微调变得至关重要。对于稳定扩散有两种重要的微调技术：

+   文本反转：这种技术专注于重新训练模型的文本嵌入，将一个词注入为主题。

+   DreamBooth：与文本反转不同，DreamBooth 涉及重新训练整个模型，专门针对主题进行优化，从而实现更好的个性化。

在这篇文章中，您将探索以下概念：

+   DreamBooth 中的微调挑战和推荐设置

+   使用 DreamBooth 进行稳定扩散微调的示例

+   有效使用 Dreambooth 在稳定扩散中的提示

**启动你的项目**，使用我的书籍 [掌握稳定扩散数字艺术](https://machinelearningmastery.com/mastering-digital-art-with-stable-diffusion/)。它提供了 **自学教程** 和 **工作代码**。

让我们开始吧

![](img/f22bad9034f63a4c6f3fe080e6b0b1fe.png)

使用 DreamBooth 训练稳定扩散

摄影师 [Sasha Freemind](https://unsplash.com/photos/woman-standing-on-grass-field-frq5Q6Ne9k4)。部分权利保留。

## 概述

本文分为五部分；它们是：

+   什么是 DreamBooth？

+   微调挑战

+   使用 DreamBooth 进行微调的工作流程

+   使用您训练过的模型

+   使用 DreamBooth 效果的技巧

## 什么是 DreamBooth？

DreamBooth 是生成 AI 中的重大飞跃，特别是 Text2Img 模型。它是谷歌研究人员引入的一种专门技术，用于微调预训练的大型 Text2Img 模型，例如 Stable Diffusion，以特定的主题、字符或对象。现在你可以向模型注入自定义对象或概念，以获得更个性化和多样化的生成结果。这是谷歌研究人员的表述方式

> 它就像一个照相亭，但一旦捕捉到主题，它可以在你的梦中合成。

DreamBooth 在各个领域提供了一系列令人兴奋的用例，主要集中在增强图像生成方面。这包括

+   **个性化：** 用户可以创建所爱之人、宠物或特定对象的图像，适合作为礼物、社交媒体和个性化商品。

+   **艺术和商业用途：** 艺术家和设计师可以用他们的艺术作品训练模型，生成多样化的艺术图像和视觉化效果。这对于商业用途也很有益，可以根据品牌和营销需求进行定制图像生成。

+   **研究与实验：** DreamBooth 是一个强大的研究工具。它使探索深度学习模型、领域特定应用和受控实验成为可能。通过微调模型，研究人员可以推动生成式 AI 的边界，并获得对其潜力的新见解。

只需提供几张你主题的图像，加上训练期间的类别名称和推理期间特别定制的提示，你就可以生成个性化的输出。让我们深入了解 DreamBooth 微调过程：

1.  创建微调图像数据集

    包含几张（3-5 张）高质量和具有代表性的主题图像。模型将根据训练数据学习如何适应主题，因此需要仔细设计。有关更多详细信息，请参见微调部分。

1.  绑定唯一标识符和类别名称

    主题必须与文本模型词汇表中不常用的稀有标记关联。模型将通过这个唯一标识符识别主题。为了保持主题的原始本质，你还需要在训练期间提供一个类别名称（模型已经知道）。例如，你可以将个人宠物狗与标识符 “[V]” 和类别名称 “dog” 关联，以便当提示 “a [V] dog sitting” 时，模型在生成狗图像时会识别出宠物狗的标识符。

![](img/392d2351304ef91063b51ed21e8936a2.png)

微调模型。图像来自 [DreamBooth](https://dreambooth.github.io/)

上述第一个标准在图像生成中因显而易见的原因而重要。第二个标准也很重要，因为你不是从头开始训练 Stable Diffusion，而是对现有模型进行微调以适应你的数据集。模型已经学会了什么是狗。让模型学习你的标记是狗的变种比让模型忘记一个词（即将一个词的意义重新分配给一个完全不同的概念）或从头开始学习一个新词（例如，从一个从未见过任何动物的模型中学习狗是什么）要容易。

## 微调挑战

微调一个 Stable Diffusion 模型意味着从一个已知有效的现有模型开始重新训练。这种特定类型的训练有一些挑战：

1.  过拟合

    过拟合发生在模型过于紧密地记忆训练数据时，从而忘记如何进行泛化。它开始只对训练数据表现良好，但对新数据表现不佳。由于你只提供了少量图像给 Dreambooth，它可能会迅速发生过拟合。

1.  语言漂移

    语言漂移是在微调语言模型时常见的现象，并且这种影响扩展到 Txt2Img 模型。在微调期间，模型可能会丧失有关类别内不同表现形式的重要信息。这种漂移导致模型在生成同一类别中的多样化主题时出现困难，从而影响输出的丰富性和多样性。

这里有几个 DreamBooth 设置，通过仔细调整这些设置，您可以使模型更适应生成多样化的输出，同时还减少过拟合和语言漂移的风险：

1.  优化学习率和训练步骤

    调整学习率、训练步骤和批量大小是克服过拟合的关键。高学习率和许多训练步骤会导致过拟合（影响多样性）。小学习率和较少的训练步骤将导致模型拟合不足（未能捕捉主题）。因此，建议从较低的学习率开始，并逐步增加训练步骤，直到生成看起来令人满意为止。

1.  先前保留损失

    通过生成相同类别的新样本（大约 200 到 300 个）以及主题图像，然后将这些添加到我们的训练图像集中来完成。这些额外的图像是通过一个类提示由稳定扩散模型本身生成的。

1.  GPU 高效技术

    诸如 8bit-Adam（支持量化）、fp16 混合精度训练（将梯度计算的精度降低到 16 位浮点数）和梯度累积（分步计算梯度而不是整个批量）等技术可以帮助优化内存利用和加快训练速度。

## DreamBooth 微调工作流程

我们已经准备好开始微调过程，并使用基于扩散器的 DreamBooth 训练脚本的简化版本，如下所示。借助上述 GPU 高效技术，您可以在 Google Colab 笔记本中的 Tesla T4 GPU 上运行此脚本。

在开始之前，您应该设置环境。在笔记本单元格中，您可以使用以下内容来运行 shell 命令：

```py
!wget -q https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py
!wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py
%pip install -qq git+https://github.com/ShivamShrirao/diffusers
%pip install -q -U --pre triton
%pip install -q accelerate transformers ftfy bitsandbytes natsort safetensors xformers
```

在上述中，下载的训练脚本名为`train_dreambooth.py`。还下载了一个转换脚本来处理训练输出。一些包已安装在 Python 环境中。为了验证其是否正常工作，您可以在新的单元格中运行这些导入语句，以确保没有触发任何错误：

```py
import json
import os

from google.colab import files
import shutil

from natsort import natsorted
from glob import glob

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
from IPython.display import display
```

训练稳定扩散模型需要大量资源。最好利用一些在线资源来存储模型。脚本假定您已注册了 Hugging Face 账户并获得了 API 令牌。请在标记为`HUGGINGFACE_TOKEN`的指定字段中提供您的访问令牌，以使脚本正常工作，即，

```py
!mkdir -p ~/.huggingface
HUGGINGFACE_TOKEN = "put your token here"
!echo -n "{HUGGINGFACE_TOKEN}" > ~/.huggingface/token
```

让我们指定我们的基础模型和输出目录，模型将在其中保存。我们将在启动脚本时传递这些变量。

```py
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "/content/stable_diffusion_weights/zwx"

# Create output directory
!mkdir -p $OUTPUT_DIR
```

现在是从 DreamBooth 中获得满意和一致结果的最关键部分。重要的是使用代表您主题的高质量图像。请注意，模型将从训练集中学习到低分辨率或运动模糊等工件。在创建数据集时应考虑以下几点：

+   数据集大小：由于这些模型可能很快过拟合，最好包括 10 到 120 个主题样本。将它们裁剪并调整大小为 512×512 像素

+   图像多样性：选择一致的样本，并尝试包括不同角度的图像。为了多样性，你可能需要包含背景场景，如人物、风景或动物/物体。此外，移除内部的多余物体（例如，水印、被边缘剪裁的人物）。避免使用背景单一或透明的图像。

在你将图像上传到 Colab 笔记本之前，让我们运行以下单元格：

```py
# The concepts_list is a list of concepts/subject, each represented as a dictionary
concepts_list = [
    {
        "instance_prompt":   "photo of zwx dog",
        "class_prompt":      "photo of a dog",
        "instance_data_dir": "/content/data/zwx",
        "class_data_dir":    "/content/data/dog"
    },
]

# Create a directory for each concept according to its instance_data_dir
for c in concepts_list:
    os.makedirs(c["instance_data_dir"], exist_ok=True)

#Dump the concepts_list to a JSON file
with open("concepts_list.json", "w") as f:
    json.dump(concepts_list, f, indent=4)
```

运行此单元格后，映射到`instance_data_dir`的路径会被创建为一个目录。你准备的图像应通过 Colab 笔记本侧边面板上的文件图标上传到上述目录。

`instance_prompt` 是一个示例。你应根据你希望为图像命名的方式进行更新。建议使用新的令牌（例如上述“zwx”）作为唯一标识符。但`class_prompt`应仅使用易于理解的词语来突出图像。

在这个示例中，我们正在对 Husky 犬进行 Stable Diffusion 微调，我们提供了 7 张实例图像。你可以尝试类似的操作。在此单元格之后，你的目录结构可能如下所示：

![](img/1c4886b2f339915d81fa10b0637e0f3b.png)

从 Colab 笔记本侧边面板中查看的目录，其中包含训练图像

现在你准备好开始训练了。以下表格列出了根据计算要求的最佳标志。限制因素通常是 GPU 上的内存。

![](img/a3541f181cb248ef0c76432b87dc34ec.png)

训练的推荐标志

训练是执行`diffusers`提供的训练脚本。这里是一些你可能考虑的参数：

+   `--use_8bit_adam` 启用完全精度和量化支持

+   `--train_text_encoder` 启用文本编码器微调

+   `--with_prior_preservation` 启用先验保持损失

+   `--prior_loss_weight` 控制先验保持的强度

在 Colab 笔记本上创建并运行以下单元格将完成训练：

```py
!python3 train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --output_dir=$OUTPUT_DIR \
  --revision="fp16" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=1337 \
  --resolution=512 \
  --train_batch_size=1 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=50 \
  --sample_batch_size=4 \
  --max_train_steps=800 \
  --save_interval=10000 \
  --save_sample_prompt="photo of zwx dog" \
  --concepts_list="concepts_list.json"
```

耐心等待此脚本完成。然后你可以运行以下命令，将其转换为我们可以使用的格式：

```py
WEIGHTS_DIR = natsorted(glob(OUTPUT_DIR + os.sep + "*"))[-1]
ckpt_path = WEIGHTS_DIR + "/model.ckpt"

half_arg = ""
fp16 = True
if fp16:
    half_arg = "--half"
!python convert_diffusers_to_original_stable_diffusion.py --model_path $WEIGHTS_DIR  --checkpoint_path $ckpt_path $half_arg
print(f"[*] Converted ckpt saved at {ckpt_path}")
```

上述操作将在指定给`WEIGHTS_DIR`（即最新检查点目录）的目录下创建文件`model.ckpt`。该输出文件将与 Automatic1111 提供的 Stable Diffusion Web UI 兼容。如果`fp16`设置为`True`，则只占用一半的空间（即 2GB），通常推荐使用。

## 使用你训练好的模型

创建的模型就像任何其他 Stable Diffusion 模型权重文件一样。你可以将其加载到 WebUI 中。你也可以使用 Python 代码加载它，如下单元格所示：

```py
model_path = WEIGHTS_DIR

pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None,
                                               torch_dtype=torch.float16
                                              ).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()

g_cuda = torch.Generator(device='cuda')
seed = 52362
g_cuda.manual_seed(seed)
```

让我们尝试四个样本：

```py
prompt = "photo of zwx dog in a bucket wearing 3d glasses"
negative_prompt = ""
num_samples = 4
guidance_scale = 7.5
num_inference_steps = 24
height = 512
width = 512

with autocast("cuda"), torch.inference_mode():
    images = pipe(
        prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=g_cuda
    ).images

for img in images:
    display(img)

# Free runtime memory
exit()
```

这就是生成图像的样子：

![](img/84f5ac64212dbe75595e50f22b7b6700.png)![](img/853a1e2140c4f69eb9fd3ed6e3c42313.png)![](img/549b8899412a17161470a4c37d8ffc76.png)![](img/003094259564e256edc06f67c50ca8c2.png)

就这样！

## 有效使用 DreamBooth 的提示

如果您正在为面部微调模型，先前的保留至关重要。面部需要更严格的训练，因为梦幻亭在面部微调时需要更多的训练步骤和更低的学习率。

像 DDIM（首选）、PNDM 和 LMS Discrete 这样的调度器可以帮助减轻模型的过拟合。当输出看起来嘈杂或缺乏锐度或细节时，您应该尝试使用调度器。

除了 U-Net 训练外，训练文本编码器可以显著提升输出质量（特别是在面部），但会消耗内存；至少需要一个 24GB 的 GPU。您也可以使用上述讨论的高效 GPU 技术进行优化。

## 进一步阅读

如果您想深入了解此主题，本节提供了更多资源。

+   LAION-5B 数据集：[`laion.ai/blog/laion-5b/`](https://laion.ai/blog/laion-5b/)

+   [通过视觉基础对抗语言漂移](https://arxiv.org/abs/1909.04499)，Lee 等人（2019）

+   [梦幻亭](https://dreambooth.github.io/)

+   [来自扩散器的梦幻亭训练示例](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README.md)

## 总结

现在您已经探索了梦幻亭，它是用于个性化内容的稳定扩散模型的强大工具。然而，由于图像较少和语言漂移，它面临着过拟合等挑战。为了充分利用梦幻亭，您还看到了一些优化方法。请记住，成功使用梦幻亭取决于仔细的数据集准备和精确的参数调整。有关更深入的见解，请参考详细的梦幻亭训练指南。
