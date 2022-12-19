from .base import TemplateDataSet
from torchvision.datasets import CIFAR10, CIFAR100

class Cifar10(TemplateDataSet):
    def __init__(self, **kwargs) -> None:
        if not hasattr(self, 'name'):
            self.name = 'cifar10' 
        super().__init__(name=self.name, **kwargs)
    
    def build(self, transform=None, verbose=False):
        ds = CIFAR10(root=self.root_dir, train=False, transform=transform, download=False)
        self.classes = ds.classes
        if verbose:
            print(f'Creating Dataset: {self.name}')
            print(f"Dataset size: {len(ds)}")
            print(f"Dataset classes: {ds.classes}")
            print(f"Dataset number of classes: {len(ds.classes)}")
        return ds

class Cifar100(TemplateDataSet):
    def __init__(self, **kwargs) -> None:
        if not hasattr(self, 'name'):
            self.name = 'cifar100' 
        super().__init__(name=self.name, **kwargs)
    
    def build(self, transform=None, verbose=False):
        ds = CIFAR100(root=self.root_dir, train=False, transform=transform, download=False)
        self.classes = ds.classes
        if verbose:
            print(f'Creating Dataset: {self.name}')
            print(f"Dataset size: {len(ds)}")
            print(f"Dataset classes: {ds.classes}")
            print(f"Dataset number of classes: {len(ds.classes)}")
        return ds