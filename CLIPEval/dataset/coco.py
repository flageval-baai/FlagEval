from .base import TemplateDataSet
from .flickr import Flickr30k
from .constants import _CONST_TASK_RETRIEVAL
from torchvision.datasets import CocoCaptions
import os

class Mscoco(TemplateDataSet):
    def __init__(self, ann_file='coco_test_karpathy.json', **kwargs) -> None:
        self.name = 'mscoco_captions'
        self.modal = 'VL'
        self.group = 'RETRIEVAL'
        self.task = _CONST_TASK_RETRIEVAL
        self.metrics = ["TR@1", "TR@5", "TR@10", "IR@1", "IR@5", "IR@10", "MR"]
        self.ann_file = ann_file
        super().__init__(name=self.name, group=self.group, modal=self.modal, task=self.task, metrics=self.metrics, **kwargs)

    def build(self, transform=None, verbose=False):
        root_dir = os.path.join(self.root_dir, 'val2014')
        ann_file = os.path.join(self.root_dir, self.ann_file)
        ds = CocoCaptions(root=root_dir, annFile=ann_file, transform=transform)
        if verbose:
            print(f'Creating Dataset: {self.name}')
            print(f'Dataset Number of Images: {len(ds.ids)}')
        return ds

class MscocoCN1k(Flickr30k):
    def __init__(self, ann_file='coco_captions_cn_test_1k.txt', **kwargs) -> None:
        self.name = 'mscoco_captions_cn_1k'
        super().__init__(ann_file, language='CN', dir_name='mscoco_captions_cn', **kwargs)

class MscocoCN5k(Flickr30k):
    def __init__(self, ann_file='coco_captions_cn_test_5k.txt', **kwargs) -> None:
        self.name = 'mscoco_captions_cn_5k'
        super().__init__(ann_file, language='CN', dir_name='mscoco_captions_cn', **kwargs)

