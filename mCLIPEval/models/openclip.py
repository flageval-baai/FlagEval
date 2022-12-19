from .base import TemplateModel
import torch
import numpy as np
from .constants import _OPEN_CLIP_MODEL_CONFIGS, _SUPPORTED_OPENCLIP_MODELS

class OpenCLIPModel(TemplateModel):
    def __init__(self, name='openclip-L', model_config=None, **kwargs):
        assert model_config or name in _SUPPORTED_OPENCLIP_MODELS
        self.model_dir = None
        if name in _SUPPORTED_OPENCLIP_MODELS:
            self.name = name
            self.agency = _OPEN_CLIP_MODEL_CONFIGS[name]['agency']
            self.text_encoder_name = _OPEN_CLIP_MODEL_CONFIGS[name]['text_encoder_name']
            self.vision_encoder_name= _OPEN_CLIP_MODEL_CONFIGS[name]['vision_encoder_name']
            self.model_arch = _OPEN_CLIP_MODEL_CONFIGS[name]['model']
            self.pretrained = _OPEN_CLIP_MODEL_CONFIGS[name]['pretrained']
        else:
            self.name = name
            self.agency = model_config['agency']
            self.text_encoder_name = model_config['text_encoder_name']
            self.vision_encoder_name = model_config['vision_encoder_name']
            self.model_arch = model_config['model']
            self.pretrained = model_config['pretrained']
        if model_config:
            self.model_dir = model_config.get('model_dir', None)

        super().__init__(name, self.model_dir, self.agency, self.vision_encoder_name, self.text_encoder_name, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def create_model_and_processors(self, model_dir, **kwargs):
        import open_clip
        model, _, transform = open_clip.create_model_and_transforms(self.model_arch, pretrained=self.pretrained, cache_dir=model_dir)
        tokenizer = open_clip.get_tokenizer(self.model_arch)
        return model, tokenizer, transform
    
    def get_image_features(self, images):
        self.model = self.model.to(self.device)
        images = images.to(self.device)
        return self.model.encode_image(images)
    
    def get_text_features(self, texts):
        self.model = self.model.to(self.device)
        texts = texts.to(self.device)
        return self.model.encode_text(texts)