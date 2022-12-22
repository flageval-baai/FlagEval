from .base import TemplateModel
import torch
import numpy as np
from .constants import _CN_CLIP_MODEL_CONFIGS, _SUPPORTED_CNCLIP_MODELS

class CnCLIP(TemplateModel):
    def __init__(self, name='cn-clip-L', model_config=None, **kwargs):
        assert model_config or name in _SUPPORTED_CNCLIP_MODELS
        self.model_dir = None
        if name in _SUPPORTED_CNCLIP_MODELS:
            self.name = name
            self.agency = _CN_CLIP_MODEL_CONFIGS[name]['agency']
            self.text_encoder_name = _CN_CLIP_MODEL_CONFIGS[name]['text_encoder_name']
            self.vision_encoder_name = _CN_CLIP_MODEL_CONFIGS[name]['vision_encoder_name']
            self.model_arch = _CN_CLIP_MODEL_CONFIGS[name]['model']
        else:
            self.name = name
            self.agency = model_config['agency']
            self.text_encoder_name = model_config['text_encoder_name']
            self.vision_encoder_name = model_config['vision_encoder_name']
            self.model_arch = model_config['model']
        if model_config:
            self.model_dir = model_config.get('model_dir', None)

        super().__init__(name, self.model_dir, self.agency, self.vision_encoder_name, self.text_encoder_name, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def create_model_and_processors(self, model_dir, **kwargs):
        import cn_clip.clip as clip
        from cn_clip.clip import load_from_name, available_models      
        model, preprocess = load_from_name(self.model_arch, device=self.device, download_root=model_dir)
        def image_processor(images):
            return preprocess(images)
        def text_processor(texts):
            tokenizer = clip.tokenize(texts)
            return tokenizer
        return model, text_processor, image_processor
    
    def get_image_features(self, images):
        images = images.to(self.device)
        return self.model.encode_image(images)
    
    def get_text_features(self, texts):
        texts = texts.to(self.device)
        return self.model.encode_text(texts)