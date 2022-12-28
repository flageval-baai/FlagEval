from .base import TemplateDataSet
from torchvision.datasets import StanfordCars
from .constants import _CONST_TASK_CLASSIFICATION

class Cars(TemplateDataSet):
    def __init__(self, **kwargs) -> None:
        if not hasattr(self, 'name'):
            self.name = 'cars' 
        self.task = _CONST_TASK_CLASSIFICATION
        super().__init__(name=self.name, task=self.task, **kwargs)
    
    def build(self, transform=None, verbose=False):
        ds = StanfordCars(root=self.root_dir, split='test', transform=transform, download=False)
        self.classes = ds.classes
        if verbose:
            print(f'Creating Dataset: {self.name}')
            print(f"Dataset size: {len(ds)}")
            print(f"Dataset classes: {ds.classes}")
            print(f"Dataset number of classes: {len(ds.classes)}")
        return ds