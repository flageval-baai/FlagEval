import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from model.user_model import get_model  # noqa E402


class ModelAdapter:

    def model_init(self, checkpoint_path):
        model = get_model()
        checkpoint = torch.load(checkpoint_path)
        # change checkpoint key name from x to model.x
        checkpoint = {f"model.{k}": v for k, v in checkpoint.items()}
        print(f"load checkpoint from {checkpoint_path}")
        model.load_state_dict(checkpoint, strict=True)
        return model
