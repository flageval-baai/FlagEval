_DATASET_ROOT_DIR = 'eval_benchmarks'
_HUGGINGFACE_AUTH_TOKEN = True


_SUPPORTED_DOWNLOAD_DATASETS = [
    "imagenet1k", "imagenet1k_cn", "imagenet1k_jp",  "imagenet1k_it",
    'imagenet-a', 'imagenet-a_cn', "imagenet-a_jp",  "imagenet-a_it",
    'imagenet-r', 'imagenet-r_cn', "imagenet-r_jp",  "imagenet-r_it",
    'imagenet-sketch', 'imagenet-sketch_cn', "imagenet-sketch_jp",  "imagenet-sketch_it",
    'imagenetv2', 'imagenetv2_cn', "imagenetv2_jp",  "imagenetv2_it",
    # "birdsnap", NOT SUPPORTED SINCE THE DATASET IS PRIVATE AND THE DATASET IS NOT COMPLETED
    "caltech101",
    "cars",
    "cifar10", "cifar100",
    "country211",
    "dtd",
    "eurosat",
    "fer2013", 
    "fgvc-aircraft",
    "flowers",
    "food101",
    "gtsrb",
    "kinetics400", "kinetics600", "kinetics700", # NOT SUPPORTED SINCE VIDEO PROCESSING NEEDED
    "mnist",
    "objectnet",
    "pcam", 
    "pets",
    "renderedsst2",
    "resisc45",
    "stl10",
    "sun397",
    # "ucf101",  NOT SUPPORTED SINCE VIDEO PROCESSING NEEDED
    "voc2007", "voc2007_multilabel", 
    "flickr30k", "flickr30k_cn",
    "multi30k_en", "multi30k_fr", "multi30k_de", "multi30k_cs",  
    "mscoco_captions", "mscoco_captions_cn_1k", "mscoco_captions_cn_5k",
    "xtd_en", "xtd_de", "xtd_es", "xtd_fr", "xtd_it", "xtd_jp", "xtd_ko", "xtd_pl", "xtd_ru", "xtd_tr", "xtd_zh",
    "winoground"
]