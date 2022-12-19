from .base import TemplateDataSet
from .constants import _CONST_TASK_RETRIEVAL
from torchvision.datasets import VisionDataset
"""
Adapted from https://github.com/pytorch/vision/blob/main/torchvision/datasets/flickr.py and 
https://github.com/LAION-AI/CLIP_benchmark/blob/main/clip_benchmark/datasets/flickr.py

Thanks to the authors of torchvision and clip_benchmark
"""
from collections import defaultdict
import os
from typing import Any, Callable, Optional, Tuple

from PIL import Image

class Flickr(VisionDataset):
    def __init__(
        self,
        root: str,
        ann_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        ann_file = os.path.join(self.root, ann_file)
        self.ann_file = os.path.expanduser(ann_file)
        data = defaultdict(list)
        with open(ann_file) as fd:
            fd.readline()
            for line in fd:
                line = line.strip()
                if line:
                    # some lines have comma in the caption, se we make sure we do the split correctly
                    try:
                        img, caption = line.strip().split(".jpg,")
                    except Exception as e:
                        print(line)
                    img = img + ".jpg"
                    data[img].append(caption)
        self.data = list(data.items())
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img, captions = self.data[index]

        # Image
        img = Image.open(os.path.join(self.root, 'Images', img)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        target =  captions
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self) -> int:
        return len(self.data)


class Flickr30k(TemplateDataSet):
    def __init__(self, ann_file='flickr30k_test_karpathy.txt', **kwargs) -> None:
        if not hasattr(self, 'name'):
            self.name = 'flickr30k'
        self.modal = 'VL'
        self.group = 'RETRIEVAL'
        self.task = _CONST_TASK_RETRIEVAL
        self.metrics = ["TR@1", "TR@5", "TR@10", "IR@1", "IR@5", "IR@10", "MR"]
        self.ann_file = ann_file
        super().__init__(name=self.name, group=self.group, modal=self.modal, task=self.task, metrics=self.metrics, **kwargs)
     
    def build(self, transform=None, verbose=False):
        ds = Flickr(root=self.root_dir, ann_file=self.ann_file, transform=transform)
        if verbose:
            print(f'Creating Dataset: {self.name}')
            print(f'Dataset Size: {sum([len(targets) for (_, targets) in ds.data])}')
            print(f'Dataset Number of Images: {len(ds.data)}')
        return ds

class Flickr30kCN(Flickr30k):
    def __init__(self, ann_file='flickr30k_test_CNA.txt', **kwargs) -> None:
        self.name = 'flickr30k_cn'
        super().__init__(ann_file, language='CN', dir_name='flickr30k-cn', **kwargs)