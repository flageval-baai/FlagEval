from .base import TemplateDataSet
from .flickr import Flickr
from .constants import _CONST_TASK_RETRIEVAL

class Aicicc(TemplateDataSet):
    def __init__(self, ann_file='aic-icc_valid.txt', **kwargs) -> None:
        if not hasattr(self, 'name'):
            self.name = 'aic-icc'
        self.modal = 'VL'
        self.group = 'RETRIEVAL'
        self.task = _CONST_TASK_RETRIEVAL
        self.language = 'CN'
        self.metrics = ["TR@1", "TR@5", "TR@10", "IR@1", "IR@5", "IR@10", "MR"]
        self.ann_file = ann_file
        super().__init__(name=self.name, dir_name='aic-icc', language=self.language ,group=self.group, modal=self.modal, task=self.task, metrics=self.metrics, **kwargs)

    def build(self, transform=None, verbose=False):
        ds = Flickr(root=self.root_dir, ann_file=self.ann_file, transform=transform)
        if verbose:
            print(f'Creating Dataset: {self.name}')
            print(f'Dataset Size: {sum([len(targets) for (_, targets) in ds.data])}')
            print(f'Dataset Number of Images: {len(ds.data)}')
        return ds