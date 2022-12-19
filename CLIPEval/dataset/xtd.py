from .base import TemplateDataSet
from .constants import _CONST_TASK_RETRIEVAL
from .flickr import Flickr

class XtdEn(TemplateDataSet):
    def __init__(self, ann_file='xtd10_en_pair.txt', **kwargs) -> None:
        if not hasattr(self, 'name'):
            self.name = 'xtd_en'
        self.modal = 'VL'
        self.group = 'RETRIEVAL'
        self.task = _CONST_TASK_RETRIEVAL
        self.metrics = ["TR@1", "TR@5", "TR@10", "IR@1", "IR@5", "IR@10", "MR"]
        self.ann_file = ann_file
        super().__init__(name=self.name, dir_name='xtd', group=self.group, modal=self.modal, task=self.task, metrics=self.metrics, **kwargs)

    def build(self, transform=None, verbose=False):
        ds = Flickr(root=self.root_dir, ann_file=self.ann_file, transform=transform)
        if verbose:
            print(f'Creating Dataset: {self.name}')
            print(f'Dataset Size: {sum([len(targets) for (_, targets) in ds.data])}')
            print(f'Dataset Number of Images: {len(ds.data)}')
        return ds

class XtdDe(XtdEn):
    def __init__(self, ann_file='xtd10_de_pair.txt', **kwargs) -> None:
        self.name = 'xtd_de'
        super().__init__(ann_file, language='DE', **kwargs)

class XtdFr(XtdEn):
    def __init__(self, ann_file='xtd10_fr_pair.txt', **kwargs) -> None:
        self.name = 'xtd_fr'
        super().__init__(ann_file, language='FR', **kwargs)

class XtdEs(XtdEn):
    def __init__(self, ann_file='xtd10_es_pair.txt', **kwargs) -> None:
        self.name = 'xtd_es'
        super().__init__(ann_file, language='ES', **kwargs)

class XtdIt(XtdEn):
    def __init__(self, ann_file='xtd10_it_pair.txt', **kwargs) -> None:
        self.name = 'xtd_it'
        super().__init__(ann_file, language='IT', **kwargs)

class XtdJp(XtdEn):
    def __init__(self, ann_file='xtd10_jp_pair.txt', **kwargs) -> None:
        self.name = 'xtd_jp'
        super().__init__(ann_file, language='JP', **kwargs)

class XtdKo(XtdEn):
    def __init__(self, ann_file='xtd10_ko_pair.txt', **kwargs) -> None:
        self.name = 'xtd_ko'
        super().__init__(ann_file, language='KO', **kwargs)

class XtdPl(XtdEn):
    def __init__(self, ann_file='xtd10_pl_pair.txt', **kwargs) -> None:
        self.name = 'xtd_pl'
        super().__init__(ann_file, language='PL', **kwargs)

class XtdRu(XtdEn):
    def __init__(self, ann_file='xtd10_ru_pair.txt', **kwargs) -> None:
        self.name = 'xtd_ru'
        super().__init__(ann_file, language='RU', **kwargs)

class XtdTr(XtdEn):
    def __init__(self, ann_file='xtd10_tr_pair.txt', **kwargs) -> None:
        self.name = 'xtd_tr'
        super().__init__(ann_file, language='TR', **kwargs)

class XtdZh(XtdEn):
    def __init__(self, ann_file='xtd10_zh_pair.txt', **kwargs) -> None:
        self.name = 'xtd_zh'
        super().__init__(ann_file, language='CN', **kwargs)