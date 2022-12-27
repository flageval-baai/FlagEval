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
 
