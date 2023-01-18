from .base import TemplateDataSet
from .constants import _CONST_TASK_MATCHING
from torchvision.datasets import VisionDataset

from collections import defaultdict
import os
from typing import Any, Callable, Optional, Tuple

from PIL import Image
class ADVDataset(VisionDataset):
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
            for line in fd:
                line = line.strip()
                if line:
                    # some lines have comma in the caption, se we make sure we do the split correctly
                    try:
                        img, caption = line.strip().split(".jpg,")
                    except Exception as e:
                        print(line)
                    is_true = int(img[0])
                    img = img[2:] + ".jpg"
                    data[img].append((caption, is_true))
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
        target =  captions[0]
        label = captions[1]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, label


    def __len__(self) -> int:
        return len(self.data)

class Flickr30kAdv(TemplateDataSet):
    def __init__(self, ann_file='adv.noun.person', **kwargs) -> None:
        self.name = f'flickr30k_{ann_file}'
        self.modal = 'VL'
        self.group = 'MATCH'
        self.task = _CONST_TASK_MATCHING
        self.ann_file = ann_file
        super.__init__(name=self.name, dir_name='flickr30k', group=self.group )