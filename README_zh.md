![FlagEval](logo.png)

[English](README.md)

--------------------------------------------------------------------------------

**FlagEval**是一个面向AI大模型的开源评测工具包，同时也是一个开放的[评测平台](https://flageval.baai.ac.cn/#/home)。

FlagEval 评测平台的目标是覆盖三个主要的评估对象——基础模型、预训练算法以及微调/压缩算法，以及四个关键领域下丰富的下游任务，包括自然语言处理（NLP）、计算机视觉（CV）、语音（Audio）和多模态（Multimodal）。您可以在我们的官方网站 [flageval.baai.ac.cn](https://flageval.baai.ac.cn/#/home) 上找到更多信息。我们致力于建立科学、公正、开放的评测基准、方法、工具集，协助研究人员全方位评估基础模型及训练算法的性能，同时探索利用AI方法实现对主观评测的辅助，大幅提升评测的效率和客观性。

FlagEval 开源评测工具包现在包含以下子项目。

## 1. mCLIPEval

[**mCLIPEval**](https://github.com/FlagOpen/FlagEval/tree/master/mCLIPEval) 是一个多语言 CLIP(Contrastive Language–Image Pre-training)系列模型的评测工具包，特点如下：

* 支持多语言（12种）评测数据和单语言（英文/中文）评测数据；
* 支持多种任务评测，包括 Zero-shot classification、Zero-shot retrieval 以及 zeroshot composition等；
* 支持已**适配的基础模型**及**用户自定义基础模型**的评测，目前已适配的基础模型包括[FlagAI](https://github.com/FlagAI-Open/FlagAI) 中的([AltCLIP](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP)， [EVA-CLIP](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/EVA_CLIP))，[OpenCLIP](https://github.com/mlfoundations/open_clip) ，[Chinese CLIP](https://github.com/OFA-Sys/Chinese-CLIP)，[Multilingual CLIP](https://github.com/FreddeFrallan/Multilingual-CLIP) ，[Taiyi](https://fengshenbang-doc.readthedocs.io/zh/latest/docs/%E5%A4%AA%E4%B9%99%E7%B3%BB%E5%88%97/index.html)系列等；
* 支持多种数据来源，如[torchvision](https://pytorch.org/vision/stable/datasets.html)，[huggingface](https://huggingface.co/datasets)，[kaggle](https://www.kaggle.com/datasets)等；
* 通过多种可视化的方式展示评测结果，方便对多个模型进行比较。

### 如何使用

环境建议： 

* Pytorch version >= 1.8.0
* Python version >= 3.8
* For evaluating models on GPUs, you'll also need install CUDA and NCCL

使用方式：

```shell
git clone https://github.com/FlagOpen/FlagEval.git
cd FlagEval/mCLIPEval/
pip install -r requirements.txt
```
更多的细节请参考 [mCLIPEval/README.md](https://github.com/FlagOpen/FlagEval/tree/master/mCLIPEval/README.md) 。


## 2. ImageEval-prompt

[ImageEval-prompt](https://github.com/FlagOpen/FlagEval/tree/master/imageEval) 是一组在实体、风格和细节等细粒度上进行评估的Prompt。通过在细粒度上的综合评估，研究者可以更好地理解文生图（Text-to-Image，T2I）模型的优点和局限性，以便进一步改进模型的性能。

* 英文数据集来自 PartiPrompts benchmark 的 1624 条英文 Prompt，其中 339 条 Prompt 翻译成中文作为中文数据集。
* 每条 Prompt 采取“双人背靠背标注+第三人仲裁”方式进行标注，分为实体、风格和细节三个维度：
    * 实体包括 5 个子维度：主体对象、状态、颜色、数量与位置；
    * 风格包括 2 个子维度：绘画风格与文化风格；
    * 细节包括 4 个子维度：手部、五官、性别与反常识。

更多细节请参考 [imageEVAL/README_zh.md](https://github.com/FlagOpen/FlagEval/blob/master/imageEval/README_zh.md) 

## 3. C-SEM

C-SEM 针对当前大模型的可能存在的缺陷和不足，创新地构造多种层次、多种难度的评测数据， 并参考人类的语言思维习惯，考察模型在理解语义时的“思考”过程。当前开源的 C-SEM v1.0版本共包括四个子评测项，分别从词汇、句子两个级别评测模型的语义理解能力，通用性较强，适用于研究对照。

当前 C-SEM 的子评测项分别为

* 词汇级的语义关系判断（LLSRC）
* 句子级别的语义关系判断（SLSRC）
* 词汇的多义理解问题（SLPWC）
* 基础修饰知识检测（SLRFC）

后续 C-SEM 评测基准将会持续迭代，覆盖更多语义理解相关知识，形成多层次的语义理解评测体系。同时，[FlagEval 大模型评测平台](https://flageval.baai.ac.cn/#/trending) 将在第一时间集成最新版本，加强对大语言模型的中文能力评测的全面性。

更多细节请参考 [csem/README-zh.md](https://github.com/FlagOpen/FlagEval/blob/master/csem/README-zh.md) 


## 联系我们

* 如果有关于 FlagEval的意见，建议或错误报告，请提交[GitHub Issue](https://github.com/FlagOpen/FlagEval/issues) 或者邮件至 flageval@baai.ac.cn，让我们共同建设更好的FlagEval。
* <font color="Red">**诚聘行业精英加入FlagEval团队。** </font>如果您有兴趣加入我们一起推进基础模型评测的工具，请联系 flageval@baai.ac.cn，期待您的加入！
* <font color="Red">**欢迎共同建设FlagEval。** </font>如果您有新的任务或者新的数据或者新的工具希望加入FlagEval，请联系flageval@baai.ac.cn，期待与您合作，共同建设基础模型评测体系！


## [许可证](/LICENSE)
本项目大部分是基于协议[Apache 2.0 license](LICENSE), 但是部分的代码是基于其他的协议:

* CLIP_benchmark 是基于协议[MIT license](https://github.com/LAION-AI/CLIP_benchmark/blob/main/LICENSE)
* ImageNet1k数据集是基于协议[huggingface datasets license and ImageNet licenese](https://huggingface.co/datasets/imagenet-1k/blob/main/README.md#licensing-information)


## 其他
#### &#8627; Stargazers, 谢谢支持!
[![Stargazers repo roster for @FlagOpen/FlagEval](https://reporoster.com/stars/FlagOpen/FlagEval)](https://github.com/FlagOpen/FlagEval/stargazers)

#### &#8627; Forkers, 谢谢支持!
[![Forkers repo roster for @FlagOpen/FlagEval](https://reporoster.com/forks/FlagOpen/FlagEval)](https://github.com/FlagOpen/FlagEval/network/members)

#### 如果您觉得我们的工作对您有价值有帮助，请给我们鼓励的**星星🌟**，谢谢您的支持！
