![FlagEval](logo.png)
[简体中文](README_zh.md)

--------------------------------------------------------------------------------

### Overview

**FlagEval** is an evaluation platform for AI large foundation models. Our goal is to explore and integrate **scientific**, **fair** and **open** foundation model evaluation benchmarks, methods and tools. FlagEval will support **multi-dimensional** evaluation (such as accuracy, efficiency, robustness, etc.) of foundation models in/cross different modalities (such as NLP, audio, CV and multimodal) in the future. We hope that through the evaluation of the foundation models, we can deepen the understanding of the foundation models and promote related technological innovation and industrial application.


* A evaluation toolkit [**mCLIPEval**](https://github.com/FlagOpen/FlagEval/tree/master/mCLIPEval) for vision-language models (such as CLIP, Contrastive Language–Image Pre-training).
	- Multilingual (12 languages) datasets and monolingual (English/Chinese) datasets.
	- Support for zeroshot classification, zeroshot retrieval and zeroshot composition tasks.
	- Adapted to [FlagAI](https://github.com/FlagAI-Open/FlagAI) pretrained models ([AltCLIP](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP), [EVA-CLIP](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/EVA_CLIP)), [OpenCLIP](https://github.com/mlfoundations/open_clip) pretrained models, [Chinese CLIP](https://github.com/OFA-Sys/Chinese-CLIP) models, [Multilingual CLIP](https://github.com/FreddeFrallan/Multilingual-CLIP) models, [Taiyi Series](https://fengshenbang-doc.readthedocs.io/zh/latest/docs/%E5%A4%AA%E4%B9%99%E7%B3%BB%E5%88%97/index.html) pretrained models, or customized models.
	- Data preparation from various resources, like [torchvision](https://pytorch.org/vision/stable/datasets.html), [huggingface](https://huggingface.co/datasets), [kaggle](https://www.kaggle.com/datasets), etc.
	- Visualization of evaluation results through leaderboard figures or tables, and detailed comparsions between two specific models.
* A evaluation toolkit for Chinese text-to-image models（TODO）.

### News
- [24 Dec 2022] made our first contribution in **


### Environment Preparation
* Pytorch version >= 1.8.0
* Python version >= 3.8
* For evaluating models on GPUs, you'll also need install CUDA and NCCL

### How to use [mCLIPEval](https://github.com/FlagOpen/FlagEval/tree/master/mCLIPEval)
```shell
git clone https://github.com/FlagOpen/FlagEval.git
cd FlagEval/mCLIPEval/
pip install -r requirements.txt
```
Please refer to [mCLIPEval/README.md](https://github.com/FlagOpen/FlagEval/tree/master/mCLIPEval/README.md) for more details.

### Citation
If you find our work helpful, please **star🌟** this repo and **cite📑** our paper. Thanks for your support!
```
@article{https://doi.org/10.48550/arxiv.2211.06679,
  doi = {10.48550/ARXIV.2211.06679},
  url = {https://arxiv.org/abs/2211.06679},
  author = {Chen, Zhongzhi and Liu, Guang and Zhang, Bo-Wen and Ye, Fulong and Yang, Qinghong and Wu, Ledell},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences},
  title = {AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

### Contact us

* For help and issues associated with FlagEval, or reporting a bug, please open a [GitHub Issue](https://github.com/FlagOpen/FlagEval/issues) or e-mail to flageval@baai.ac.cn. Let's build a better & stronger FlagEval together :)
* If you are interested in working with us on foundation model evaluation, please contact flageval@baai.ac.cn.


### [License](/LICENSE)
The majority of FlagEval is licensed under the [Apache 2.0 license](LICENSE), however portions of the project are available under separate license terms:

* he usage of CLIP_benchmark is licensed under the [MIT license](https://github.com/LAION-AI/CLIP_benchmark/blob/main/LICENSE)
* The usage of ImageNet1k datasets in under the [huggingface datasets license and ImageNet licenese](https://huggingface.co/datasets/imagenet-1k/blob/main/README.md#licensing-information)


### Misc
#### &#8627; Stargazers, thank you for your support!
[![Stargazers repo roster for @FlagOpen/FlagEval](https://reporoster.com/stars/FlagOpen/FlagEval)](https://github.com/FlagOpen/FlagEval/stargazers)

#### &#8627; Forkers, thank you for your support!
[![Forkers repo roster for @FlagOpen/FlagEval](https://reporoster.com/forks/FlagOpen/FlagEval)](https://github.com/FlagOpen/FlagEval/network/members)
