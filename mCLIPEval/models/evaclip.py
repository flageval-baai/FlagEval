from .base import TemplateModel
import torch
import numpy as np

class EvaClip(TemplateModel):
    def __init__(self, name='eva-clip', model_dir=None, agency='BAAI', vision_encoder_name='eva VIT-g', text_encoder_name='CLIP-L', **kwargs):
        super().__init__(name, model_dir, agency, vision_encoder_name, text_encoder_name, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def create_model_and_processors(self, model_dir, **kwargs):
        from flagai.auto_model.auto_loader import AutoLoader
        from flagai.data.dataset.mm.clip_dataset import clip_transform      
        if not model_dir:
            import os
            model_dir = os.path.join(os.path.expanduser('~'), '.cache', 'flagai')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
        loader = AutoLoader(
            task_name="txt_img_matching",
            model_dir=model_dir,
            model_name=self.name
        )
        model = loader.get_model()
        model = model.eval()
        tokenizer = loader.get_tokenizer()
        transform = clip_transform(img_size=model.visual.image_size)
        def text_processor(texts):
            return tokenizer.tokenize_as_tensor(texts)
        def image_processor(images):
            return transform(images)
        return model, text_processor, image_processor
    
    def get_image_features(self, images):
        """
        images: [Tensor Size: (batch, 3, 224, 224)]
        """
        self.model = self.model.to(self.device)
        images = images.to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            return image_features
    
    def get_text_features(self, texts):
        self.model = self.model.to(self.device)
        texts = texts.to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(texts)
            return text_features
