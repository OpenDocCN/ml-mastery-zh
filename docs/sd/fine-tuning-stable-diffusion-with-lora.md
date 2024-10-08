# 使用 LoRA 微调稳定扩散

> 原文：[`machinelearningmastery.com/fine-tuning-stable-diffusion-with-lora/`](https://machinelearningmastery.com/fine-tuning-stable-diffusion-with-lora/)

稳定扩散可以根据你的输入生成图像。有许多模型在架构和流程上类似，但它们的输出可能差别很大。有许多方法可以调整它们的行为，例如，当你给出一个提示时，输出默认会呈现某种风格。LoRA 是一种不需要重新创建大型模型的技术。在这篇文章中，你将看到如何自己创建一个 LoRA。

完成本文后，你将学习到

+   如何准备和训练 LoRA 模型

+   如何在 Python 中使用训练好的 LoRA

**启动你的项目**，可以参考我的书籍 [《掌握稳定扩散数字艺术》](https://machinelearningmastery.com/mastering-digital-art-with-stable-diffusion/)。它提供了 **自学教程** 和 **可运行的代码**。

让我们开始吧。

![](img/995f329f4e69f8321a2ac32530d3a5bd.png)

使用 LoRA 微调稳定扩散

图片来源 [Thimo Pedersen](https://unsplash.com/photos/red-and-white-ladybug-toy-on-white-and-yellow-book-dip9IIwUK6w)。部分权利保留。

## 概述

本文分为三个部分；它们是

+   训练 LoRA 的准备工作

+   使用 Diffusers 库训练 LoRA

+   使用你训练好的 LoRA

## 训练 LoRA 的准备工作

我们在 [上一篇文章](https://machinelearningmastery.com/using-lora-in-stable-diffusion/) 中介绍了在 Web UI 中使用 LoRA 的概念。如果你想创建自己的 LoRA，Web UI 插件允许你这样做，或者你可以使用自己的程序来创建一个。由于所有训练将是计算密集型的，请确保你有一台配备 GPU 的机器继续进行。

我们将使用来自 diffusers 库示例目录的训练脚本。在开始之前，你需要通过安装所需的 Python 库来设置环境，使用以下命令：

```py
pip install git+https://github.com/huggingface/diffusers
pip install accelerate wand
pip install -r https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/requirements.txt

accelerate config default
# accelerate configuration saved at $HOME/.cache/huggingface/accelerate/default_config.yaml
```

第一个命令是从 GitHub 安装 `diffusers` 库，这是开发版本。这样做是因为你将使用来自 GitHub 的训练脚本，因此需要使用匹配的版本。

上述最后一个命令确认你已经安装了 `accelerate` 库，并检测了你计算机上的 GPU。你已经下载并安装了许多库。你可以尝试运行下面的 Python 语句来确认所有库已正确安装，并且没有导入错误：

```py
import wandb
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, AutoPipelineForText2Image
from huggingface_hub import model_info
```

你将使用来自 diffusers 示例的 LoRA 训练脚本。首先下载脚本：

```py
wget -q https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora.py
```

## 使用 Diffusers 库训练 LoRA

对于微调，你将使用 [Pokémon BLIP captions with English and Chinese dataset](https://huggingface.co/datasets/svjack/pokemon-blip-captions-en-zh) 在基础模型 `runwayml/stable-diffusion-v1-5`（官方 Stable Diffusion v1.5 模型）上。你可以调整超参数以适应你的具体用例，但你可以从以下 Linux shell 命令开始。

```py
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./finetune_lora/pokemon"
export HUB_MODEL_ID="pokemon-lora"
export DATASET_NAME="svjack/pokemon-blip-captions-en-zh"

mkdir -p $OUTPUT_DIR

accelerate launch --mixed_precision="bf16"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=500 \
  --caption_column="en_text" \
  --validation_prompt="A pokemon with blue eyes." \
  --seed=1337
```

运行此命令将需要几个小时才能完成，即使使用高端 GPU。但让我们更详细地了解一下它的作用。

accelerate 命令帮助你在多个 GPU 上启动训练。如果你只有一个 GPU，也不会有害。许多现代 GPU 支持 Google Brain 项目引入的“Brain Float 16”浮点数。如果支持，选项 `--mixed_precision="bf16"` 将节省内存并运行更快。

该命令脚本从 Hugging Face Hub 下载数据集，并使用它来训练 LoRA 模型。批量大小、训练步骤、学习率等是训练的超参数。训练好的模型会在每 500 步时将检查点保存到输出目录。

训练 LoRA 需要一个包含图像（像素）和对应标题（文本）的数据集。标题文本描述了图像，而训练好的 LoRA 将理解这些标题应代表那些图像。如果你查看 Hugging Face Hub 上的数据集，你会看到标题名称是 `en_text`，如上所述设置为 `--caption_column`。

如果你提供自己的数据集（例如，手动为你收集的图像创建标题），你应该创建一个 CSV 文件 `metadata.csv`，第一列命名为 `file_name`，第二列为你的文本标题，如下所示：

```py
file_name,caption
image_0.png,a drawing of a green pokemon with red eyes
image_1.png,a green and yellow toy with a red nose
image_2.png,a red and white ball with an angry look on its face
...
```

并将此 CSV 文件与所有图像（匹配 `file_name` 列）放在同一目录中，并使用目录名作为你的数据集名称。

在上述脚本中指定为 `OUTPUT_DIR` 的目录下将创建许多子目录和文件。每个检查点将包含完整的稳定扩散模型权重和提取的 LoRA safetensors。一旦你完成训练，你可以删除除了最终 LoRA 文件 `pytorch_lora_weights.safetensors` 之外的所有文件。

## 使用你的训练好的 LoRA

运行一个稳定扩散（Stable Diffusion）管道与 LoRA 只需要对你的 Python 代码做一个小修改。一个例子如下：

```py
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from huggingface_hub import model_info
import torch

# LoRA weights ~3 MB
model_path = "pcuenq/pokemon-lora"

info = model_info(model_path)
model_base = info.cardData["base_model"]
pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

image = pipe("Green pokemon with menacing face", num_inference_steps=25).images[0]
image.save("green_pokemon.png")
```

上述代码从 Hugging Face Hub 存储库 `pcuenq/pokemon-lora` 下载了一个 LoRA 并使用 `pipe.unet.load_attn_procs(model_path)` 将其附加到管道中。其余的与平常一样。生成的图像可能如下所示：

![](img/9bdee499d4377ef93bbb00b2885ed4f6.png)

生成的绿色宝可梦

这是使用 LoRA 的一种更详细的方法，因为你需要知道这个特定的 LoRA 应该被加载到管道的 `unet` 部分的注意力过程（attention process）中。这些细节应该在存储库的模型卡中找到。

使用 LoRA 的更简单方法是使用自动流水线，从中推断出模型架构从模型文件中：

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                     torch_dtype=torch.float16
                                                    ).to("cuda")
pipeline.load_lora_weights("finetune_lora/pokemon",
                           weight_name="pytorch_lora_weights.safetensors")
image = pipeline("A pokemon with blue eyes").images[0]
```

`load_lora_weights()`函数的参数是您训练的 LoRA 文件的目录名称和文件名称。这也适用于其他 LoRA 文件，比如您从 Civitai 下载的文件。

## 进一步阅读

如果您希望更深入地了解该主题，本节提供了更多资源。

+   LoRA 训练：[`huggingface.co/docs/diffusers/en/training/lora`](https://huggingface.co/docs/diffusers/en/training/lora)

+   稳定扩散文本转图像流水线：[`huggingface.co/docs/diffusers/v0.29.0/en/api/pipelines/stable_diffusion/text2img`](https://huggingface.co/docs/diffusers/v0.29.0/en/api/pipelines/stable_diffusion/text2img)

## 总结

在这篇文章中，您看到如何根据一组图像和描述文本创建自己的 LoRA 模型。这是一个耗时的过程，但结果是您拥有一个可以修改扩散模型行为的小权重文件。您学会了如何使用`diffusers`库运行 LoRA 的训练。您还看到了如何在您的稳定扩散流水线代码中使用 LoRA 权重。
