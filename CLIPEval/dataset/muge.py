from .base import TemplateDataSet
from .constants import _CONST_TASK_RETRIEVAL

"""
Adapted from https://github.com/pytorch/vision/blob/main/torchvision/datasets/flickr.py
Thanks to the authors of torchvision
"""
from collections import defaultdict
import glob
import os
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image
from torchvision.datasets import VisionDataset

class MugeDataset(VisionDataset):

    def __init__(
        self,
        root: str,
        ann_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(os.path.join(root, ann_file))
        data = defaultdict(list)
        with open(self.ann_file) as fd:
            fd.readline()
            for line in fd:
                line = line.strip()
                if line:
                    # some lines have comma in the caption, se we make sure we do the split correctly
                    img, caption = line.strip().split(".jpg,")
                    img = img + ".jpg"
                    data[caption].append(img)
        self.data = list(data.items())
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        caption, imgs = self.data[index]

        # Image
        imgs = [Image.open(os.path.join(self.root, 'Images', img)).convert("RGB") for img in imgs]
        if self.transform is not None:
            imgs = [self.transform(img) for img in imgs]

        # Captions
        target =  caption
        if self.target_transform is not None:
            target = self.target_transform(target)

        return imgs, target


    def __len__(self) -> int:
        return len(self.data)

class Muge(TemplateDataSet):
    def __init__(self, ann_file='muge_valid.txt', **kwargs) -> None:
        if not hasattr(self, 'name'):
            self.name = 'muge'
        self.modal = 'VL'
        self.group = 'RETRIEVAL'
        self.task = _CONST_TASK_RETRIEVAL
        self.language = 'CN'
        self.metrics = ["TR@1", "TR@5", "TR@10", "IR@1", "IR@5", "IR@10", "MR"]
        self.ann_file = ann_file
        super().__init__(name=self.name, dir_name='muge', language=self.language ,group=self.group, modal=self.modal, task=self.task, metrics=self.metrics, **kwargs)

    def build(self, transform=None, verbose=False):
        ds = MugeDataset(root=self.root_dir, ann_file=self.ann_file, transform=transform)
        if verbose:
            print(f'Creating Dataset: {self.name}')
            print(f'Dataset Size: {sum([len(targets) for (_, targets) in ds.data])}')
            print(f'Dataset Number of Images: {len(ds.data)}')
        return ds