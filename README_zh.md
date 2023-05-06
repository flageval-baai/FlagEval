![FlagEval](logo.png)

[English](README.md)

--------------------------------------------------------------------------------

**FlagEval**是一个面向AI基础模型的评测工具包。我们的目标是探索和集合**科学**、**公正**、**开放**的基础模型评测基准、方法及工具，对**多领域**（如语言、语音、视觉及多模态）的基础模型进行**多维度**（如准确性、效率、鲁棒性等）的评测。我们希望通过对基础模型的评测，加深对基础模型的理解，促进相关的技术创新及产业应用。

目前开放**多模态领域**的评测工具，更多领域、更多维度的评测工具正在持续开发中，欢迎加入共同建设。

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
    * 实体包括五个考点：主体对象、状态、颜色、数量与位置；
    * 风格包括两个考点：绘画风格与文化风格；
    * 细节包括四个考点：手部、五官、性别与反常识。

更多细节请参考 [imageEVAL/README_zh.md](https://github.com/FlagOpen/FlagEval/blob/master/imageEval/README_zh.md) 


## 联系我们

* 如果有关于FlagEval的意见，建议或错误报告，请提交[GitHub Issue](https://github.com/FlagOpen/FlagEval/issues) 或者邮件至 flageval@baai.ac.cn，让我们共同建设更好的FlagEval。
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