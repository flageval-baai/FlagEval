from .base import TemplateDataSet
from .constants import _CONST_TASK_COMPOSITIONALITY
from torchvision.datasets import VisionDataset

import os
from typing import Any, Callable, Optional, Tuple

from PIL import Image
import json

class WinogroundDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        ann_file: str = "examples.jsonl",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        ann_file = os.path.join(self.root, ann_file)
        self.ann_file = os.path.expanduser(ann_file)
        data = []
        with open(ann_file) as fd:
            for line in fd:
                line = line.strip()
                if line:
                    try:
                        example = json.loads(line)
                        caption_0 = example["caption_0"]
                        caption_1 = example["caption_1"]
                        image_0 = example["image_0"] + '.png'
                        image_1 = example["image_1"] + '.png'
                        data.append(
                            (caption_0, caption_1, image_0, image_1)
                        )
                    except:
                        print(line)
        self.data = data

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        cap_0, cap_1, img_0, img_1 = self.data[index]
        img_0 = Image.open(os.path.join(self.root, 'images', img_0)).convert('RGB')
        img_1 = Image.open(os.path.join(self.root, 'images', img_1)).convert('RGB')
        if self.transform is not None:
            img_0 = self.transform(img_0)
            img_1 = self.transform(img_1)
        
        if self.target_transform is not None:
            cap_0 = self.target_transform(cap_0)
            cap_1 = self.target_transform(cap_1)
        
        image = [img_0, img_1]
        target = [cap_0, cap_1]
        return image, target

    def __len__(self) -> int:
        return len(self.data)

class Winoground(TemplateDataSet):
    def __init__(self, **kwargs) -> None:
        self.name = 'winoground'
        self.modal = 'LANGUAGE'
        self.group = 'COMPOSITIONALITY'
        self.task = _CONST_TASK_COMPOSITIONALITY
        self.metrics = ["image_score", "text_scrore", "group_score"]
        super().__init__(name=self.name, group=self.group, modal=self.modal, task=self.task, metrics=self.metrics, **kwargs)
    
    def build(self, transform=None, verbose=False):
        ds = WinogroundDataset(root=self.root_dir+'/data', transform=transform)
        if verbose:
            print(f'Creating Dataset: {self.name}')
            print(f'Dataset Number of Examples: {len(ds.data)}')
        return ds
    