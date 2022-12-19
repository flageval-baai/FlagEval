from .base import TemplateDataSet
from .flickr import Flickr
from .constants import _CONST_TASK_RETRIEVAL

class Multi30kEn(TemplateDataSet):
    def __init__(self, ann_file='multi30k-en_test.txt', **kwargs) -> None:
        if not hasattr(self, 'name'):
            self.name = 'multi30k_en'
        self.modal = 'VL'
        self.group = 'RETRIEVAL'
        self.task = _CONST_TASK_RETRIEVAL
        self.metrics = ["TR@1", "TR@5", "TR@10", "IR@1", "IR@5", "IR@10", "MR"]
        self.ann_file = ann_file
        super().__init__(name=self.name, dir_name='multi30k', group=self.group, modal=self.modal, task=self.task, metrics=self.metrics, **kwargs)

    def build(self, transform=None, verbose=False):
        ds = Flickr(root=self.root_dir, ann_file=self.ann_file, transform=transform)
        if verbose:
            print(f'Creating Dataset: {self.name}')
            print(f'Dataset Size: {sum([len(targets) for (_, targets) in ds.data])}')
            print(f'Dataset Number of Images: {len(ds.data)}')
        return ds


class Multi30kDe(Multi30kEn):
    def __init__(self, ann_file='multi30k-de_test.txt', **kwargs) -> None:
        self.name = 'multi30k_de'
        super().__init__(ann_file, language='DE', **kwargs)

class Multi30kFr(Multi30kEn):
    def __init__(self, ann_file='multi30k-fr_test.txt', **kwargs) -> None:
        self.name = 'multi30k_fr'
        super().__init__(ann_file, language='FR', **kwargs)

class Multi30kCs(Multi30kEn):
    def __init__(self, ann_file='multi30k-cs_test.txt', **kwargs) -> None:
        self.name = 'multi30k_cs'
        super().__init__(ann_file, language='CS', **kwargs)