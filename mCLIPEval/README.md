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

## Environment Preparation
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

The complete use of mCLIPEval contains three standalone modules: data preparation, evaluation and visualization.

| Module | Entry | Function | Documentation|
|--------|-------|----------|-------|
|Data Preparation| [download.py](download.py) |  Download datasets and organize the data directories properly for evaluation.|[download.md](download.md)|
|Evaluation|[evaluate.py](evaluate.py)|Evaluate a model on selected datasets and output results in a json file.|[evaluate.md](evaluate.md)|
|Visualization|[visual.py](visual.py)|Visualize the evaluation results through an interactive web app.|[visual.md](visual.md)|

### Quick tour
* To immediately see the comparison results of built-in open-source models, we provide `outputs` as early-run evaluation results. You just need to run:

    ```
    streamlit run visual.py -- --json="outputs/*.json"
    ```

* To evaluate a pretrained model with checkpoint files, you need to:
    * specify the model script, for example [models/altclip.py](models/altclip.py)
    * choose the evaluation datasets (for example `cifar10` is an image classification dataset)
    * download and prepare the datasets with:
        ```
        python download.py --datasets=cifar10
        ```
    * evaluate the pretrained model in the directory `[MODEL_DIR]`
        ```
        python evaluate.py --model_name=[MODEL_NAME] --model_dir=[MODEL_DIR] --datasets=cifar10 --output=[MODEL_NAME].json
        ```
    




## Datasets and Models



--------------------------------------------------------------------------------

<!-- toc -->
- [Environment Preparation](#environment-preparation)
- [How to Use](#how-to-use)
    - [Quick Start](#quick-start)
    - [Data Preparation](#data-preparation)
    - [Evaluation](#evaluation)
        - [CIFAR-10 and CIFAR-100 example](#cifar-10-and-cifar-100-example)
        - [Chinese Image Classification example](#chinese-image-classification-example)
        - [Built-In Model example](#built-in-model-example)
        - [Pretrained Checkpoint example](#pretrained-checkpoint-example)
        - [Customized Model example](#customized-model-example)
    - [Visualization](#visualization)
- [Reference](#reference)
- [Credits](#credits)
- [License](#license)

<!-- tocstop -->

## How to Use?
The complete use of mCLIPEval contains data preparation, evaluation and visualization. 

### Quick Start

* First, download and prepare datasets (imagenet1k as example):

```
python download.py --datasets=imagenet1k 
```

The downloaded datasets would be saved in `eval_benchmarks/imagenet1k`

* Second, evaluate built-in models (`AltCLIP-XLMR-L-m9`, `M-CLIP`) on downloaded datasets (`imagenet1k`, `imagenet1k_cn`, `imagenet1k_jp`,  `imagenet1k_it`):

```shell
python evaluate.py --model_name=AltCLIP-XLMR-L --datasets=imagenet1k --output=altclip.json
python evaluate.py --model_name=M-CLIP --datasets=imagenet1k --output=mclip.json
```
The evaluation results would be saved in `altclip.json` and `mclip.json`

* Finally, visualize the evaluation results:

```shell
streamlit run visual.py -- --json=altclip.json,mclip.json
```

For advanced usages of each modules, please follow to instructions below:

### Data Preparation

Please refer to [download.md](download.md) for instructions on how to download and prepare the supported datasets.

### Evaluation 

You can evaluate a model on multiple datasets once a time with `evaluate.py` with different parameters. 
* [Tips] The evaluation process would take some time. You might use `screen` or `nohup` in case interuption of the process.
* [Tips] The default data directory is `eval_benchmarks`. You can use `--root=[YOUR_DATA_DIRECTORY]` to use customize data directory.
* [Tips] You can use `--verbose` to save temporary evaluation results and resume from these results. 


#### CIFAR-10 and CIFAR-100 example
`--datasets` specify the datasets which are used for evaluation, default to choose all datasets.

Here is an example for multiple datasets (CIFAR-10 and CIFAR-100) using a bilingual pretrained model [AltCLIP](https://arxiv.org/abs/2211.06679):

`python evaluate.py --datasets=cifar10,cifar100 --model_name=AltCLIP-XLMR-L`

Here is the content of `output.json` after the evaluation is done:

```json
{
    "model_info": {
        "model_name": "AltCLIP-XLMR-L",
        "vision_encoder": "VIT-L", 
        "text_encoder": "XLMR", 
        "agency": "BAAI"
    }, 
    "cifar10": {
        "acc1": 0.9545, 
        "acc5": 0.9962, 
        "mean_per_class_recall": 0.9544
    },
    "cifar100": {
        "acc1": 0.7891, 
        "acc5": 0.9443, 
        "mean_per_class_recall": 0.7891000000000001
    }
}
```

To have the list of datasets, you can use:

`python -c 'from dataset.constants import _SUPPORTED_DATASETS; print("\n".join(_SUPPORTED_DATASETS))'`


### Chinese Image Classification example
`--groups` to specify the task groups and `--languages` to specify the dataset languages. The datasets which satisfy the task groups and languages are chosen automatically.

Here is an example for evaluating the model `AltCLIP-XLMR-L` on all Chinese image classification datasets.

`python evaluate.py --groups=CLASSIFICATION --languages=CN --model_name=AltCLIP-XLMR-L`

Here is the content of `output.json` after the evaluation is done:
```json
{
    "model_info":{
        "model_name":"AltCLIP-XLMR-L",
        "vision_encoder":"VIT-L",
        "text_encoder":"XLMR",
        "agency":"BAAI"
    },
    "imagenet1k_cn":{
        "acc1":0.59568,
        "acc5":0.84918,
        "mean_per_class_recall":0.5948599999999999
    },
    "imagenet-a_cn":{
        "acc1":0.6124,
        "acc5":0.8713333333333333,
        "mean_per_class_recall":0.5921668034633918
    },
    "imagenet-r_cn":{
        "acc1":0.8243333333333334,
        "acc5":0.9463,
        "mean_per_class_recall":0.810219060166447
    },
    "imagenet-sketch_cn":{
        "acc1":0.4841714319401049,
        "acc5":0.7541708424217415,
        "mean_per_class_recall":0.4831835294117647
    },
    "imagenetv2_cn":{
        "acc1":0.5401,
        "acc5":0.8121,
        "mean_per_class_recall":0.5401
    }
}
```


To have the list of languages, you can use:

```
python -c 'from dataset.constants import _SUPPORTED_LANGUAGES; 
print("\n".join(_SUPPORTED_LANGUAGES))'
```

To have the list of task groups, you can use:

`python -c 'from dataset.constants import _SUPPORTED_GROUPS; print("\n".join(_SUPPORTED_GROUPS))'`


### Built-In Model example
`--model_name` to specify the name of models, some are built-in models. 

Here is an example of evaluating `eva-clip` model on all datasets.

```shell
python evaluate.py --model_name=eva-clip --output=eva-clip.json
```

To have the list of built-in models, you can use:

`python -c 'from models.constants import _SUPPORTED_MODELS; print("\n".join(_SUPPORTED_MODELS))'`


### Pretrained Checkpoint example

`--model_dir` to specify the directory to load a model.

Here is an example of evaluating `eva-clip` model from [MODEL_DIRECTORY] on all datasets.

```shell
python evaluate.py --model_name=eva-clip --model_dir=[MODEL_DIRECTORY] --output=eva-clip.json
```

### Customized Model example

`--model_script` to specify the customized model script to load the model.

Here is an example of evaluating a customized model.

First, you need to select from `altclip`, `cnclip`, `evaclip`, `mclip`, `openclip` or `taiyi` as your model script file, or implement a customized model script. Within a model script file, you need to:

* See `models/altclip.py` as an example:
* Define an "XCLIP" class based on `model.base.TemplateModel` class

```python
from .base import TemplateModel
import torch
class AltClip(TemplateModel):
    def __init__(self, name='AltCLIP-XLMR-Lmodel_dir=None, agency='BAAI', vision_encoder_name='VIT-L', text_encoder_name='XLMR', **kwargs): 
        super().__init__(name, model_dir, agency, vision_encoder_name, text_encoder_name, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

* Implement a `create_model_and_processors(self, model_dir, **kwargs)` function to load and initialize model, as well as text and image processors. 

```python
    def create_model_and_processors(self, model_dir, **kwargs):
        from flagai.auto_model.auto_loader import AutoLoader
        if not model_dir:
            model_dir = './'
        loader = AutoLoader(
            task_name="txt_img_matching",
            model_dir=model_dir,
            model_name=self.name
        )
        model = loader.get_model()
        model = model.eval()
        tokenizer = loader.get_tokenizer()
        transform = loader.get_transform()
        def text_processor(texts):
            return tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors='pt')
        def image_processor(images):
            return transform(images)['pixel_values']
        return model, text_processor, image_processor
```

* Implement a `get_image_features(self, images)` function and a `get_text_features(self, texts)` function to take the tensor input of images and texts, and output the embeddings.

```python
    def get_image_features(self, images):
        self.model = self.model.to(self.device)
        images = torch.cat(images).to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(images)
            return image_features
    
    def get_text_features(self, texts):
        self.model = self.model.to(self.device)
        texts = texts.to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**texts)
            return text_features
```



Then, use `--model_script=model.altclip` options to load and initialize the customized model from directory `'models/altclip.py'`. 

`python evaluate.py --datasets=cifar10 --model_name=AltCLIP-XLMR-L-m9 --model_script=models.altclip`

Here is the content of `output.json` after the evaluation is done:

```json
{
    "model_info":{
        "model_name":"AltCLIP-XLMR-L-m9",
        "vision_encoder":"VIT-L",
        "text_encoder":"XLMR",
        "agency":"BAAI"
    },
    "cifar10":{
        "acc1":0.9565,
        "acc5":0.9982,
        "mean_per_class_recall":0.9563
    }
}
```

### Visualization

Here is an example of visualization.

First, you need to need to install the packages to support visualization:

`pip install -r requirements_visual.txt`

Second, you should prepare evaluation result files. In this example, we take two evaluation result files named `altclip.json` and `evaclip.json`.

Then, you can run the visualization module by providing the evaluation file names:

`streamlit run visual.py -- --json=altclip.json,evaclip.json`

[Tips] wildcard variables is supported, you can use "*", "?":

`streamlit run visual.py -- --json="outputs/*.json"`

[Tips] jsonl file input is also supported, you can use `--jsonl=[JSONL_FILE]` to initialize.

The default url of web application is: `http://localhost:8501/`

Here are the snapshots of the visualization webpages.

![snapshot1.jpg](snapshot1.jpg)

![snapshot2.jpg](snapshot2.jpg)
 

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

## Credits

* Thanks to [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark) authors, the zeroshot classification and retrieval code, and the dataset building code are adpated from there. 

* Thanks to [CLIP](https://github.com/openai/CLIP/blob/main/data/prompts.md), [Lit](https://arxiv.org/pdf/2111.07991.pdf), [SLIP](https://github.com/facebookresearch/SLIP/blob/main/templates.json), [imagenet_classes_chinese](https://github.com/ningbonb/imagenet_classes_chinese), [japanese_clip](https://github.com/rinnakk/japanese-clip/blob/master/src/japanese_clip/utils/imagenet_zeroshot_data.py#L1), [clip-italian](https://github.com/clip-italian/clip-italian/tree/imagenet_templates/evaluation) authors, as the orginal zeroshot templates and multilingual classnames.

* Thanks to [Winoground](https://arxiv.org/abs/2204.03162), [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch), [ImageNet-A](https://github.com/hendrycks/natural-adv-examples), [ImageNet-R](https://github.com/hendrycks/imagenet-r) authors to provide original datasets. 


## License
The majority of mCLIPEval is licensed under the [Apache 2.0 license](LICENSE), however portions of the project are available under separate license terms:

* The usage of CLIP_benchmark is licensed under the [MIT license](https://github.com/LAION-AI/CLIP_benchmark/blob/main/LICENSE)
* The usage of ImageNet1k datasets in under the [huggingface datasets license and ImageNet licenese](https://huggingface.co/datasets/imagenet-1k/blob/main/README.md#licensing-information)
