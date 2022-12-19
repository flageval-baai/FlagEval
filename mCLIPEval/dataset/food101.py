from .base import TemplateDataSet
from torchvision.datasets import Food101 as Food101Dataset

class Food101(TemplateDataSet):
    def __init__(self, **kwargs) -> None:
        if not hasattr(self, 'name'):
            self.name = 'food101' 
        super().__init__(name=self.name, **kwargs)
    
    def build(self, transform=None, verbose=False):
        ds = Food101Dataset(root=self.root_dir, split='test', transform=transform, download=False)
        ds.classes = [cl.replace("_", " ") for cl in ds.classes]
        self.classes = ds.classes
        if verbose:
            print(f'Creating Dataset: {self.name}')
            print(f"Dataset size: {len(ds)}")
            print(f"Dataset classes: {ds.classes}")
            print(f"Dataset number of classes: {len(ds.classes)}")
        return ds