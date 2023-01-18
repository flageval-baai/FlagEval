from abc import ABC, abstractmethod
from .constants import (
    _DEFAULT_DATA_ROOT_DIR, _DEFAULT_DATASET_INFO,
    _CLASSIFICATION_TEMPLATES, _DEFAULT_CLASSIFICATION_TEMPLATES,
    _CLASSNAMES, _DEFAULT_CLASSNAMES,
    _SUPPORTED_GROUPS, _SUPPORTED_LANGUAGES, _SUPPORTED_METRICS, 
    _SUPPORTED_MODALS, _SUPPORTED_TASKS, _SUPPORTED_DATASETS
)
import torch
from torch.utils.data import default_collate
import os

class TemplateDataSet(ABC):
    def __init__(self, name='__template__', **kwargs) -> None:
        self.name = name
        self.initialize_with_dict(_DEFAULT_DATASET_INFO)
        self.initialize_with_dict(config_dict=kwargs)
        self.initialize_root_dir(**kwargs)
        self.__check__()

    def __check__(self):
        def issublist(list1, list2):
            set1 = set(list1)
            set2 = set(list2)
            return set1.issubset(set2)
        assert os.path.exists(self.root_dir)
        assert self.name in _SUPPORTED_DATASETS
        assert hasattr(self, 'group') and self.group in _SUPPORTED_GROUPS
        assert hasattr(self, 'language') and self.language in _SUPPORTED_LANGUAGES
        assert hasattr(self, 'metrics') and issublist(self.metrics, _SUPPORTED_METRICS)
        assert hasattr(self, 'modal') and self.modal in _SUPPORTED_MODALS
        assert hasattr(self, 'task') and self.task in _SUPPORTED_TASKS
        assert hasattr(self, 'root_dir') and self.root_dir
        if self.group == 'CLASSIFICATION':
            self.initialize_classes_and_templates()
            assert hasattr(self, 'classes') and isinstance(self.classes, list)
            assert hasattr(self, 'templates') and isinstance(self.templates, list)
        if self.group == 'RETRIEVAL':
            assert hasattr(self, 'ann_file')

    
    def initialize_with_dict(self, config_dict):
        valid_config_dict = {key:value for key, value in config_dict.items() if value}
        self.__dict__.update(valid_config_dict)
    
    def initialize_root_dir(self, **kwargs):
        _dataset_root = kwargs.get("root", None)
        if not _dataset_root:
            _dataset_root = _DEFAULT_DATA_ROOT_DIR
        _dir_name = kwargs.get("dir_name", self.name)
        self.root_dir = os.path.join(_dataset_root, _dir_name)
    
    def initialize_classes_and_templates(self):
        self.classes = _CLASSNAMES.get(self.name, _DEFAULT_CLASSNAMES)
        self.templates = _CLASSIFICATION_TEMPLATES.get(self.name, _DEFAULT_CLASSIFICATION_TEMPLATES)

    @abstractmethod
    def build(self, transform=None, verbose=False):
        pass

    def get_collate_fn(self):
        def image_captions_collate_fn(batch):
            # img, text
            transposed = list(zip(*batch))
            if isinstance(transposed[1][0],list):
                imgs = default_collate(transposed[0])
                texts = transposed[1]
            elif isinstance(transposed[0][0],list):
                imgs = transposed[0]
                texts = default_collate(transposed[1])
            return imgs, texts
        if self.group in ["CLASSIFICATION"]:
            return default_collate
        elif self.group in ["RETRIEVAL"]:
            return image_captions_collate_fn
        elif self.group in ["COMPOSITIONALITY"]:
            return default_collate

    def get_dataloader(self, batch_size, num_workers, image_processor=None):
        collate_fn = self.get_collate_fn()
        dataset = self.build(transform=image_processor)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, collate_fn=collate_fn
        )
    
    def __info__(self):
        return {
            "group": self.group,
            "task": self.task,
            "language": self.language,
            "modal": self.modal,
            "metrics": self.metrics
        }

    
class EvalDataset(object):
    def __init__(self, root=None, dataset_names=None, task_names=None, group_names=None, languages=None, verbose=False) -> None:
        from .constants import _SUPPORTED_DATASETS
        datasets = []
        if dataset_names and isinstance(dataset_names, list):
            _full_dataset_names = [ds_name for ds_name in dataset_names if ds_name in _SUPPORTED_DATASETS]
        else:
            _full_dataset_names = _SUPPORTED_DATASETS
        for dataset in _full_dataset_names:
            try:
                datasets.append(self.get_dataset(dataset, root=root))
            except:
                print(f'Dataset {dataset} data not exits, skip.')
                continue
        datasets = [dataset for dataset in datasets if dataset]
        if task_names and isinstance(task_names, list):
            datasets = [dataset for dataset in datasets if dataset.task in task_names]
        if group_names and isinstance(group_names, list):
            datasets = [dataset for dataset in datasets if dataset.group in group_names]
        if languages and isinstance(languages, list):
            datasets = [dataset for dataset in datasets if dataset.language in languages]
        self.datasets = datasets
        dataset_names = [dataset.name for dataset in self.datasets]
        if verbose:
            print(f'Datasets: {dataset_names}')
            self.build_dataset(verbose=verbose)
    
    def get_dataset(self, dataset_name, root=None):
        from .imagenet import (
            Imagenet1k, ImagenetA, ImagenetR, ImagenetSketch, ImagenetV2, 
            Imagenet1kCN, ImagenetACN, ImagenetRCN, ImagenetSketchCN, ImagenetV2CN,
            Imagenet1kJp, ImagenetAJp, ImagenetRJp, ImagenetSketchJp, ImagenetV2Jp,
            Imagenet1kIt, ImagenetAIt, ImagenetRIt, ImagenetSketchIt, ImagenetV2It,
            Imagenet1kOther, ImagenetAKo, ImagenetRKo, ImagenetSketchKo, ImagenetV2Ko,
            ImagenetAAr, ImagenetRAr, ImagenetSketchAr, ImagenetV2Ar,
            ImagenetAEs, ImagenetREs, ImagenetSketchEs, ImagenetV2Es,
            ImagenetAFr, ImagenetRFr, ImagenetSketchFr, ImagenetV2Fr,
            ImagenetARu, ImagenetRRu, ImagenetSketchRu, ImagenetV2Ru
        )
        from .flickr import Flickr30k, Flickr30kCN
        from .coco import Mscoco, MscocoCN1k, MscocoCN5k
        from .birdsnap import Birdsnap
        from .caltech import Caltech101
        from .cars import Cars
        from .cifar import Cifar10, Cifar100
        from .country211 import Country211
        from .dtd import Dtd
        from .eurosat import Eurosat
        from .fer2013 import Fer2013
        from .fgvc import Fgvcaircraft
        from .flowers import Flowers
        from .food101 import Food101
        from .gtsrb import Gtsrb
        from .kinetics import Kinetics400, Kinetics600, Kinetics700
        from .mnist import Mnist
        from .objectnet import Objectnet
        from .pcam import Pcam
        from .pets import Pets
        from .renderedsst2 import Renderedsst2
        from .resisc45 import Resisc45
        from .stl10 import Stl10
        from .sun397 import Sun397
        from .ucf101 import Ucf101
        from .voc2007 import Voc2007, Voc2007multi
        from .winoground import Winoground
        from .multi30k import Multi30kEn, Multi30kCs, Multi30kDe, Multi30kFr
        from .xtd import XtdEn, XtdDe, XtdEs, XtdFr, XtdIt, XtdJp, XtdKo, XtdPl, XtdRu, XtdTr, XtdZh
        from .aic import Aicicc
        from .muge import Muge
        if dataset_name == 'imagenet1k':
            return Imagenet1k(root=root)
        elif dataset_name == 'imagenet1k_cn':
            return Imagenet1kCN(root=root)
        elif dataset_name == 'imagenet1k_jp':
            return Imagenet1kJp(root=root)
        elif dataset_name == 'imagenet1k_it':
            return Imagenet1kIt(root=root)
        elif dataset_name == 'imagenet1k_ko':
            return Imagenet1kOther(root=root, language='KO')
        elif dataset_name == 'imagenet1k_ar':
            return Imagenet1kOther(root=root, language='AR')
        elif dataset_name == 'imagenet1k_es':
            return Imagenet1kOther(root=root, language='ES')
        elif dataset_name == 'imagenet1k_fr':
            return Imagenet1kOther(root=root, language='FR')
        elif dataset_name == 'imagenet1k_ru':
            return Imagenet1kOther(root=root, language='RU')
        elif dataset_name == 'imagenet-a':
            return ImagenetA(root=root)
        elif dataset_name == 'imagenet-a_cn':
            return ImagenetACN(root=root)
        elif dataset_name == 'imagenet-a_jp':
            return ImagenetAJp(root=root)
        elif dataset_name == 'imagenet-a_it':
            return ImagenetAIt(root=root)
        elif dataset_name == 'imagenet-a_ko':
            return ImagenetAKo(root=root)
        elif dataset_name == 'imagenet-a_ar':
            return ImagenetAAr(root=root)
        elif dataset_name == 'imagenet-a_es':
            return ImagenetAEs(root=root)
        elif dataset_name == 'imagenet-a_fr':
            return ImagenetAFr(root=root)
        elif dataset_name == 'imagenet-a_ru':
            return ImagenetARu(root=root)
        elif dataset_name == 'imagenet-r':
            return ImagenetR(root=root)
        elif dataset_name == 'imagenet-r_cn':
            return ImagenetRCN(root=root)
        elif dataset_name == 'imagenet-r_jp':
            return ImagenetRJp(root=root)
        elif dataset_name == 'imagenet-r_it':
            return ImagenetRIt(root=root)
        elif dataset_name == 'imagenet-r_ko':
            return ImagenetRKo(root=root)
        elif dataset_name == 'imagenet-r_ar':
            return ImagenetRAr(root=root)
        elif dataset_name == 'imagenet-r_es':
            return ImagenetREs(root=root)
        elif dataset_name == 'imagenet-r_fr':
            return ImagenetRFr(root=root)
        elif dataset_name == 'imagenet-r_ru':
            return ImagenetRRu(root=root)
        elif dataset_name == 'imagenet-sketch':
            return ImagenetSketch(root=root)
        elif dataset_name == 'imagenet-sketch_cn':
            return ImagenetSketchCN(root=root)
        elif dataset_name == 'imagenet-sketch_jp':
            return ImagenetSketchJp(root=root)
        elif dataset_name == 'imagenet-sketch_it':
            return ImagenetSketchIt(root=root)
        elif dataset_name == 'imagenet-sketch_ko':
            return ImagenetSketchKo(root=root)
        elif dataset_name == 'imagenet-sketch_ar':
            return ImagenetSketchAr(root=root)
        elif dataset_name == 'imagenet-sketch_es':
            return ImagenetSketchEs(root=root)
        elif dataset_name == 'imagenet-sketch_fr':
            return ImagenetSketchFr(root=root)
        elif dataset_name == 'imagenet-sketch_ru':
            return ImagenetSketchRu(root=root)
        elif dataset_name == 'imagenetv2':
            return ImagenetV2(root=root)
        elif dataset_name == 'imagenetv2_cn':
            return ImagenetV2CN(root=root)
        elif dataset_name == 'imagenetv2_jp':
            return ImagenetV2Jp(root=root)
        elif dataset_name == 'imagenetv2_it':
            return ImagenetV2It(root=root)
        elif dataset_name == 'imagenetv2_ko':
            return ImagenetV2Ko(root=root)
        elif dataset_name == 'imagenetv2_ar':
            return ImagenetV2Ar(root=root)
        elif dataset_name == 'imagenetv2_es':
            return ImagenetV2Es(root=root)
        elif dataset_name == 'imagenetv2_fr':
            return ImagenetV2Fr(root=root)
        elif dataset_name == 'imagenetv2_ru':
            return ImagenetV2Ru(root=root)
        elif dataset_name == 'birdsnap':
            return Birdsnap(root=root)
        elif dataset_name == 'caltech101':
            return Caltech101(root=root)
        elif dataset_name == 'cars':
            return Cars(root=root)
        elif dataset_name == 'cifar10':
            return Cifar10(root=root)
        elif dataset_name == 'cifar100':
            return Cifar100(root=root)
        elif dataset_name == 'country211':
            return Country211(root=root)
        elif dataset_name == 'dtd':
            return Dtd(root=root)
        elif dataset_name == 'eurosat':
            return Eurosat(root=root)
        elif dataset_name == 'fer2013':
            return Fer2013(root=root)
        elif dataset_name == 'fgvc-aircraft':
            return Fgvcaircraft(root=root)
        elif dataset_name == 'flowers':
            return Flowers(root=root)
        elif dataset_name == 'flickr30k':
            return Flickr30k(root=root)
        elif dataset_name == 'flickr30k_cn':
            return Flickr30kCN(root=root)
        elif dataset_name == 'food101':
            return Food101(root=root)
        elif dataset_name == 'gtsrb':
            return Gtsrb(root=root)
        elif dataset_name == 'kinetics400':
            return Kinetics400(root=root)
        elif dataset_name == 'kinetics600':
            return Kinetics600(root=root)
        elif dataset_name == 'kinetics700':
            return Kinetics700(root=root)
        elif dataset_name == 'mnist':
            return Mnist(root=root)
        elif dataset_name == 'mscoco_captions':
            return Mscoco(root=root)
        elif dataset_name == 'mscoco_captions_cn_1k':
            return MscocoCN1k(root=root)
        elif dataset_name == 'mscoco_captions_cn_5k':
            return MscocoCN5k(root=root)
        elif dataset_name == 'objectnet':
            return Objectnet(root=root)
        elif dataset_name == 'pcam':
            return Pcam(root=root)
        elif dataset_name == 'pets':
            return Pets(root=root)
        elif dataset_name == 'renderedsst2':
            return Renderedsst2(root=root)
        elif dataset_name == 'resisc45':
            return Resisc45(root=root)
        elif dataset_name == 'stl10':
            return Stl10(root=root)
        elif dataset_name == 'sun397':
            return Sun397(root=root)
        elif dataset_name == 'ucf101':
            return Ucf101(root=root)
        elif dataset_name == 'voc2007':
            return Voc2007(root=root)
        elif dataset_name == 'voc2007_multilabel':
            return Voc2007multi(root=root)
        elif dataset_name == 'winoground':
            return Winoground(root=root)
        elif dataset_name == 'multi30k_en':
            return Multi30kEn(root=root)
        elif dataset_name == 'multi30k_fr':
            return Multi30kFr(root=root)
        elif dataset_name == 'multi30k_cs':
            return Multi30kCs(root=root)
        elif dataset_name == 'multi30k_de':
            return Multi30kDe(root=root)
        elif dataset_name == 'xtd_en':
            return XtdEn(root=root)
        elif dataset_name == 'xtd_de':
            return XtdDe(root=root)
        elif dataset_name == 'xtd_es':
            return XtdEs(root=root)
        elif dataset_name == 'xtd_fr':
            return XtdFr(root=root)
        elif dataset_name == 'xtd_it':
            return XtdIt(root=root)
        elif dataset_name == 'xtd_jp':
            return XtdJp(root=root)
        elif dataset_name == 'xtd_ko':
            return XtdKo(root=root)
        elif dataset_name == 'xtd_pl':
            return XtdPl(root=root)
        elif dataset_name == 'xtd_ru':
            return XtdRu(root=root)
        elif dataset_name == 'xtd_tr':
            return XtdTr(root=root)
        elif dataset_name == 'xtd_zh':
            return XtdZh(root=root)
        elif dataset_name == 'aic-icc':
            return Aicicc(root=root)
        elif dataset_name == 'muge':
            return Muge(root=root)
        else:
            return None
    
    def build_dataset(self, transform=None, verbose=False):
        eval_datasets = []
        for eval_dataset in self.datasets:
            try:
                eval_dataset.build(transform=transform, verbose=verbose)
                eval_datasets.append(eval_dataset)
            except:
                print(f'Dataset {eval_dataset.name} intialization failure.')
                continue
        self.datasets = eval_datasets

    def __info__(self):
        return {ds.name: ds.__info__() for ds in self.datasets}
