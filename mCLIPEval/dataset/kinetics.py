from .base import TemplateDataSet
from torchvision.datasets import ImageFolder
import os
from .constants import _CONST_TASK_VIDEO
class Kinetics400(TemplateDataSet):
    def __init__(self, version='400', **kwargs) -> None:
        if not hasattr(self, 'name'):
            self.name = f'kinetics{version}'
        self.task = _CONST_TASK_VIDEO
        super().__init__(name=self.name, task=self.task, dir_name=self.name+'/val_images', **kwargs)
    
    def build(self, transform=None, verbose=False):
        ds = ImageFolder(root=self.root_dir, transform=transform)
        self.classes = ds.classes
        if verbose:
            print(f'Creating Dataset: {self.name}')
            print(f"Dataset size: {len(ds)}")
            print(f"Dataset classes: {ds.classes}")
            print(f"Dataset number of classes: {len(ds.classes)}")
        return ds

class Kinetics600(Kinetics400):
    def __init__(self, version='600', **kwargs) -> None:
        super().__init__(version, **kwargs)

class Kinetics700(Kinetics400):
    def __init__(self, version='700', **kwargs) -> None:
        super().__init__(version, **kwargs)