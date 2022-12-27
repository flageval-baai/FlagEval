# Evaluation
The tutorial provides you the guidance to evaluate on various setups, which is the core function of mCLIPEval. 

## Parameters

The entry file of evaluation is [evaluate.py](evaluate.py) which accepts various params.

|Param|Instruction|Values|Optional|Default|
|:-:|---|:-:|:-:|:-:|
datasets|to specify the evaluation datasets by names|[The Dataset List](#dataset-list)|Yes|Select All|
languages|to specify the evaluation datasets by languages|EN/CN/DE/CS/<br>FR/ES/IT/JP/<br>KO/PL/RU/TR|Yes|Select All|
groups|to specify the evaluation datasets by groups|CLASSIFICATION<br>RETRIEVAL<br>COMPOSITIONALITY|Yes|Select All|
tasks|to specify the evaluation datasets by tasks|Image Classification,<br>Optical Character Recognition,<br>Geo-Localization,<br>Video Act Recognition,<br>Image-Text Retrieval,<br>Image-Text Compositionality|Yes|Select All|
model_name|the name of model| [List of Built-in models](#built-in-model-list) or any| No | - |
model_dir|the model checkpoint folder or the directory to save the downloaded model| - |Yes for built-in models| - |
agency | the agency to provide the model|-|Yes|-|
output | the result json file ([format example](outputs/AltCLIP-XLMR-L.json))| - | Yes |    `output.json`|
verbose| the verbose mode| True/False| Yes | False|
restore| the restore mode, save temporary evaluation results and resume from these results| True/False |Yes|False| 
root| to customize the evaluation data directory| - | Yes| `eval_benchmarks`|
batch_size| the batch size of image encoders during evaluation | - | Yes|128|
num_workers| the number of workers during evaluation | - | Yes| 4|

* [Tips] The evaluation process would take some time. You might use `screen` or `nohup` in case interuption of the process caused by network or other reasons.

* [Tips] Use `--restore`, the temporary evaluation results would be saved in `eval.[MODEL_NAME]/[DATASET_NAME].json`, delete the files when not needed.

### Examples 



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

#### Dataset List
* To have the list of datasets, you can use in Python environment:

    ```python
    >>> from download.constants import _SUPPORTED_DOWNLOAD_DATASETS
    >>> print("\n".join(_SUPPORTED_DOWNLOAD_DATASETS))
    ```


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

#### Language List
* To have the list of languages, you can use:

    ```python
    from dataset.constants import _SUPPORTED_LANGUAGES
    print("\n".join(_SUPPORTED_LANGUAGES))
    ```

#### Group List
* To have the list of task groups, you can use:

    ```python
    from dataset.constants import _SUPPORTED_GROUPS
    print("\n".join(_SUPPORTED_GROUPS))
    ```


### Built-In Model example
`--model_name` to specify the name of models, some are built-in models. 

Here is an example of evaluating `eva-clip` model on all datasets.

```shell
python evaluate.py --model_name=eva-clip --output=eva-clip.json
```

#### Built-in Model List 

* To have the list of built-in models, you can use:

    ```python
    from models.constants import _SUPPORTED_MODELS
    print("\n".join(_SUPPORTED_MODELS))
    ```


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

 
