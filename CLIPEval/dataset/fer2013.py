from .base import TemplateDataSet
from torchvision.datasets import ImageFolder
import os

class Fer2013(TemplateDataSet):
    def __init__(self, **kwargs) -> None:
        if not hasattr(self, 'name'):
            self.name = 'fer2013' 
        super().__init__(name=self.name, dir_name=self.name+'/test', **kwargs)
    
    def build(self, transform=None, verbose=False):
        ds = ImageFolder(root=self.root_dir, transform=transform)
        self.classes = ds.classes
        if verbose:
            print(f'Creating Dataset: {self.name}')
            print(f"Dataset size: {len(ds)}")
            print(f"Dataset classes: {ds.classes}")
            print(f"Dataset number of classes: {len(ds.classes)}")
        return ds