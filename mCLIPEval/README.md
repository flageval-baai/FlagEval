<h1 align="center">
    <p>𝕞CLIP𝔼val</p>
</h1>
<h3 align="center">
    <p>An easy-to-use and easily-extendible evaluation tookit for vision-language models.</p>
</h3>


--------------------------------------------------------------------------------

The mCLIPEval tookit provides generic evaluation for pretrained vision-language models (such as CLIP, Contrastive Language–Image Pre-training). More precisely, mCLIPEval provides evaluations:

- On multilingual (12 languages) datasets and monolingual (English/Chinese) datasets, for tasks like zeroshot image/video classification, zeroshot image-to-text and text-to image retrieval and zeroshot visio-linguistic compositionality.

- Adapted to various open-source pretrained models, like [FlagAI](https://github.com/FlagAI-Open/FlagAI) pretrained models ([AltCLIP](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP), [EVA-CLIP](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/EVA_CLIP)), [OpenCLIP](https://github.com/mlfoundations/open_clip) pretrained models, [Chinese CLIP](https://github.com/OFA-Sys/Chinese-CLIP) models, [Multilingual CLIP](https://github.com/FreddeFrallan/Multilingual-CLIP) models, [Taiyi Series](https://fengshenbang-doc.readthedocs.io/zh/latest/docs/%E5%A4%AA%E4%B9%99%E7%B3%BB%E5%88%97/index.html) pretrained models. Customized models can be also supported with model scripts.

mCLIPEval provides APIs to quickly download and preparation of public datasets from [torchvision](https://pytorch.org/vision/stable/datasets.html), [huggingface](https://huggingface.co/datasets), [kaggle](https://www.kaggle.com/datasets), as well as  download and use those open-source pretrained models on these datasets.

mCLIPEval provides visualization of evaluation results through [streamlit](https://streamlit.io/) web app, to see the comparsions of performances on specific languages, tasks, or model parameters.

Below is the line chart to show the performances of some open-source models.

![snapshot0.png](snapshot0.png)

--------------------------------------------------------------------------------
## Online Demo

You can use the online demo to experience mCLIPEval (will be online soon).

## with mCLIPEval, I can ...

1. Easy-to-compare capabilities of open-source models
    - on **specific** datasets, languages, tasks, model parameters
    - through **interactive inferfaces** to switch different configuration settings
    - **without computational resources needs** to support inference and evaluation.

2. Easy-to-evaluate pretrained models using checkpoint files
    - trained through **various frameworks**, like [FlagAI](https://github.com/FlagAI-Open/FlagAI), [OpenCLIP](https://github.com/mlfoundations/open_clip), [Transformers](https://github.com/huggingface/transformers)
    - or **any customized frameworks** with model scripts
    - on **specific** datasets, languages, tasks.

3. Easy-to-build a evaluation framework **from scratch**
    - including download and preparation of datasets and models,
    - evaluation with various configuration settings
    - and visualization of the evaluation results.

--------------------------------------------------------------------------------

<!-- toc -->
## Guide 
- [Requiremnets Installation](#requiremnets-installation)
- [How to Use](#how-to-use)
    - [Structure](#structure)
    - [Quick Tour](#quick-tour)
    - [Advanced Usage Examples](#advanced-usage-examples)
- [Datasets and Models](#datasets-and-models)
    - [Supported Datasets](#supported-datasets)
    - [Built-in Models](#built-in-models)
- [Reference](#reference)
- [Credits](#credits)
- [License](#license)

<!-- tocstop -->

## Requiremnets Installation
To use mCLIPEval, you need:
* Pytorch version >= 1.8.0
* Python version >= 3.8
* For evaluating models on GPUs, you'll also need install CUDA and NCCL

**[Recommended]** For complete usage, you are recommended to install the required packages through:

```shell
pip install -r requirements.txt
```

* [Optional] To use datasets from huggingface (`imagenet1k` in any languages, `winoground`), you need to:
    * 1. generate huggingface API TOKEN (select the role "read") from [huggingface](https://huggingface.co/settings/tokens) following the [instructions](https://huggingface.co/docs/hub/security-tokens);
    * 2. run the command and add the generated token as git credential: `huggingface-cli login` or modify the [download/constants.py](download/constants.py) file with generated token `>>> _HUGGINGFACE_AUTH_TOKEN = "hf_..."`
    * 3. click the `Agree and access repository` button on dataset pages ([imagenet-1k]("https://huggingface.co/datasets/imagenet-1k") and [winoground]("https://huggingface.co/datasets/facebook/winoground")) to accept the license agreements of the datasets.
        <p align="center">
        <img src="agreement.png" width=40%>
        </p>


* [Optional] To use datasets from kaggle (`fer2013`, `flickr30k`, `flickr30k_cn`, `multi30k`), you need to:
    * 1. generate API token from [kaggle](https://www.kaggle.com/) following the [instructions](https://github.com/Kaggle/kaggle-api#api-credentials).
    * 2. install `unzip` command, for Debian/Ubuntu Linux, use ```sudo apt-get install unzip```; for CentOS/RHEL Linux, use
```yum install unzip```; for macOS, use ```brew install unzip```.

**[Partial]** For partial usage, you need to install required packages: 

* **[Necessary]** for basic evaluation: `pip install torch,torchvision,sklearn`

* [Optional] for data preparation:   `pip install -r requirements_dataset.txt`

* [Optional] for usage of pretrained models: `pip install -r requirements_models.txt`

* [Optional] for visualization: `pip install -r requirements_visual.txt`

## How to use?
### Structure
The complete use of mCLIPEval contains three standalone modules: data preparation, evaluation and visualization.

| Module | Entry | Function | Documentation|
|:------:|:-----:|----------|:-----:|
|Data Preparation| [download.py](download.py) |  Download datasets and organize the data directories properly for evaluation|[Data Doc](download.md)|
|Evaluation|[evaluate.py](evaluate.py)|Evaluate a model on selected datasets and output results in a json file|[Evaluation Doc](evaluate.md)|
|Visualization|[visual.py](visual.py)|Visualize the evaluation results through an interactive web app|[Visualization Doc](visual.md)|

### Quick tour
* To immediately see the comparison results of built-in open-source models, we provide `outputs` as early-run evaluation results. You just need to run:

    ```
    streamlit run visual.py -- --json="outputs/*.json"
    ```

* To evaluate a pretrained model with checkpoint files, you need to:
    * specify the model script, for example [models/altclip.py](models/altclip.py)
    * choose the evaluation datasets (for example `cifar10` is a classical image classification dataset)
    * download and prepare the datasets with:
        ```
        python download.py --datasets=cifar10
        ```
    * evaluate the pretrained model in the directory `[MODEL_DIR]`
        ```
        python evaluate.py --model_name=[MODEL_NAME] --model_dir=[MODEL_DIR] --model_script=models.altclip --datasets=cifar10 --output=[MODEL_NAME].json
        ```
    * the evaluation results are saved in `[MODEL_NAME].json` file
    * [Tips] if the parameter `--datasets` is not specified, all supported datasets are chosen (the process of data preparation and evaluation would take a long time).

### Advanced usage examples

| Function | Description |
|:------:|-------|
|[Multi-datasets Preparation]()|Download and prepare the datasets specified by names|
|[Full-datasets Preparation]()|Download and prepare all supported datasets|
|[Specified-datasets Evaluation]()|Evaluation on specified datasets|
|[Specified-languages Evaluation]()|Evaluation on specified languages|
|[Specified-tasks Evaluation]()|Evaluation on specified tasks|
|[Built-in Model Evaluation]()|Evaluate a built-in model specified by name|
|[Pretrained checkpoint Evaluation]()|Evaluate a pretrained model with supported pretraining framework and checkpoint directories|
|[Customized-model Evaluation]()|Evaluate a customized model with customized pretraining framework|
|[Visualization]()|Visualize the evaluation result json/jsonl files|

## Datasets and Models

### Supported Datasets

|Dataset Names| Languages | Task |Instructions|
|:------:|:-----:|:-----:|-------|
|[imagenet1k](https://huggingface.co/datasets/imagenet-1k)|EN/[CN](https://github.com/ningbonb/imagenet_classes_chinese)/[JP](https://github.com/rinnakk/japanese-clip/blob/master/src/japanese_clip/utils/imagenet_zeroshot_data.py#L1)/[IT](https://github.com/clip-italian/clip-italian/tree/imagenet_templates/evaluation)|Image Classification|multilinual classnames and prompts, including `imagenet1k, imagenet1k_cn, imagenet1k_jp, imagenet1k_it`|
|[imagenet-a](https://github.com/hendrycks/natural-adv-examples)|EN/CN/JP/IT|Image Classification|multilinual classnames and prompts, including `imagenet-a, imagenet-a_cn, imagenet-a_jp, imagenet-a_it`|
|[imagenet-r](https://github.com/hendrycks/imagenet-r)|EN/CN/JP/IT|Image Classification|multilinual classnames and prompts, including `imagenet-r, imagenet-r_cn, imagenet-r_jp, imagenet-r_it`|
|[imagenet-sketch](https://github.com/HaohanWang/ImageNet-Sketch)|EN/CN/JP/IT|Image Classification|multilinual classnames and prompts, including `imagenet-sketch, imagenet-sketch_cn, imagenet-sketch_jp, imagenet-sketch_it`|
|[imagenetv2](https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-matched-frequency.tar.gz)|EN/CN/JP/IT|Image Classification|multilinual classnames and prompts, including `imagenetv2, imagenetv2_cn, imagenetv2_jp, imagenetv2_it`|
|[objectnet](https://objectnet.dev)|EN|Image Classification|`objectnet`|
|[torchvision datasets](https://github.com/pytorch/vision/tree/main/torchvision/datasets)|EN|Image Classification, <br>OCR, <br>Geo-Localization|including: `caltech101, cars, cifar10, cifar100, country211, dtd, eurosat, fer2013, fgvc-aircraft, flowers, food101, gtsrb, mnist, objectnet, pcam, pets, renderedsst2, resisc45, stl10, sun397, voc2007, voc2007_multilabel`|
|[winoground](https://huggingface.co/datasets/facebook/winoground)|EN|Image-Text Compositionality|visio-linguistic compositional reasoning, `winoground`|
|[mscoco_captions](https://cocodataset.org/#captions-2015)|EN/[CN](https://github.com/li-xirong/coco-cn)|Image-Text Retrieval|apart from English version, there are also 1k/5k Chinese translation version with different splits, including `mscoco_captions, mscoco_captions_cn_1k, mscoco_captions_cn_5k`|
|[xtd](https://github.com/adobe-research/Cross-lingual-Test-Dataset-XTD10)|EN/CN/DE/ES/FR/IT/JP/KO/PL/RU/TR|Image-Text Retrieval|multilingual translation of MSCOCO captions, with the same image test splits, including `xtd_en, xtd_de, xtd_es, xtd_fr, xtd_it, xtd_jp, xtd_ko, xtd_pl, xtd_ru, xtd_tr, xtd_zh`|
|[flickr_30k](https://www.kaggle.com/datasets/adityajn105/flickr30k)|EN/[CN](https://github.com/li-xirong/cross-lingual-cap)|Image-Text Retrieval|apart from English version, there is also Chinese translation version with different splits, including `flickr30k, flickr30k_cn`|
|[multi30k](https://github.com/multi30k/dataset)|EN/FR/DE/CS|Image-Text Retrieval|multilingual translation of Flickr30k captions, with the same image test splits, including `multi30k_en, multi30k_fr, multi30k_de, multi30k_cs`|
|[birdsnap](http://thomasberg.org/datasets/birdsnap/1.1/birdsnap.tgz)|EN|Image Classification|download not supported, `birdsnap`|
|[kinetics](https://www.deepmind.com/open-source/kinetics)|EN|Video Act Recognition|download not supported, including `kinetics400, kinetics600, kinetics700`|
|[ucf101](https://www.crcv.ucf.edu/data/UCF101.php)|EN|Video Act Recognition|download not supported, including `ucf101`|




### Built-in Models

|Model Name|Text Encoder|Vision Encoder|Description|
|:---:|:----------:|:------------:|-----------|
|[openai-clip-L](https://huggingface.co/openai/clip-vit-large-patch14)|CLIP-L|VIT-L| openai's CLIP with `Vit-L-14` as image encoder and default tranformer as text encoder|
|[openai-clip-L-336](https://huggingface.co/openai/clip-vit-large-patch14-336)|CLIP-L|VIT-L|the same to `openai-clip-L` while the input image size is 336*336|
|[openclip-L](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K)|CLIP-L|VIT-L|openclip's implementation with `openai-clip-L` trained with `laion 2B` data|
|[openclip-L-v0](https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_l_14-laion400m_e32-3d133497.pt)|CLIP-L|VIT-L|openclip's implementation with `openai-clip-L` trained with `laion 400m` data|
|[openclip-H](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)|CLIP-H|VIT-H|openclip's pretrained model with `Vit-H-14` as vision encoder|
|[openclip-H-XLMR-L](https://huggingface.co/laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k)|XLMR-L|VIT-H|openclip's pretrained model with `Vit-H-14` as vision encoder and `XLMR-Large` as text encoder|
|[openclip-B-XLMR-B](https://huggingface.co/laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k)|XLMR-B|VIT-B|openclip's pretrained model with `Vit-B-32` as vision encoder and `XLMR-Base` as text encoder|
|[cn-clip-L](https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14)|RoBERTa-wwm-L|VIT-L|damo's Chinese CLIP model with `Vit-L-14` as image encoder and `RoBERTa-wwm-Large` as text encoder|
|[cn-clip-L-336](https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14-336px)|RoBERTa-wwm-L|VIT-L|the same to `cn-clip-L` while the input image size is 336*336| 
|[M-CLIP](https://huggingface.co/gzomer/clip-multilingual)|XLMR-L|VIT-L|RISE's multilingual clip model with `Vit-L-14` as image encoder and `XLMR-Large` as text encoder|
|[AltCLIP-XLMR-L](https://huggingface.co/BAAI/AltCLIP)|XLMR-L|VIT-L|BAAI's bilingual clip model with `Vit-L-14` as image encoder and `XLMR-Large` as text encoder|
|[AltCLIP-XLMR-L-m9](https://huggingface.co/BAAI/AltCLIP-m9)|XLMR-L|VIT-L|BAAI's multilingual clip model with `Vit-L-14` as image encoder and `XLMR-Large` as text encoder|
|[eva-clip](https://huggingface.co/BAAI/EVA#eva-clip)|CLIP-L|eva VIT-g|BAAI's clip model with the pretrained `eva`(size `Vit-g-14`) as image encoder and and default tranformer as text encoder|
|[Taiyi-CLIP-L](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese)|RoBERTa-wwm-L|VIT-L|Noah's Chinese CLIP model with `Vit-L-14` as image encoder and `RoBERTa-wwm-Large` as text encoder|





## Reference

If you find this work helpful, please consider to ⭐️ this repo and cite
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

## Contributing 
Thanks for your interest in contributing! Apart from regular commits, we also welcome contributions to resources (datasets, models, tasks).

## Credits

* Thanks to [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark) authors, the zeroshot classification and retrieval code, and the dataset building code are adpated from there. 

* Thanks to [CLIP](https://github.com/openai/CLIP/blob/main/data/prompts.md), [Lit](https://arxiv.org/pdf/2111.07991.pdf), [SLIP](https://github.com/facebookresearch/SLIP/blob/main/templates.json), [imagenet_classes_chinese](https://github.com/ningbonb/imagenet_classes_chinese), [japanese_clip](https://github.com/rinnakk/japanese-clip/blob/master/src/japanese_clip/utils/imagenet_zeroshot_data.py#L1), [clip-italian](https://github.com/clip-italian/clip-italian/tree/imagenet_templates/evaluation) authors, as the orginal zeroshot templates and multilingual classnames.

* Thanks to [Winoground](https://arxiv.org/abs/2204.03162), [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch), [ImageNet-A](https://github.com/hendrycks/natural-adv-examples), [ImageNet-R](https://github.com/hendrycks/imagenet-r) authors to provide original datasets. 

* Thanks to [COCO-CN](https://github.com/li-xirong/coco-cn) and [Flickr30k CN](https://github.com/li-xirong/cross-lingual-cap) authors to provide the original datasets.



## License
The majority of mCLIPEval is licensed under the [Apache 2.0 license](LICENSE), however portions of the project are available under separate license terms:

* The usage of CLIP_benchmark is licensed under the [MIT license](https://github.com/LAION-AI/CLIP_benchmark/blob/main/LICENSE)
* The usage of ImageNet1k datasets in under the [huggingface datasets license and ImageNet licenese](https://huggingface.co/datasets/imagenet-1k/blob/main/README.md#licensing-information)
