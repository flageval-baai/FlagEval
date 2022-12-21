from .base import TemplateModel
import torch
import numpy as np

class AltClip(TemplateModel):
    def __init__(self, name='AltCLIP-XLMR-L', model_dir=None, agency='BAAI', vision_encoder_name='VIT-L', text_encoder_name='XLMR', **kwargs):
        super().__init__(name, model_dir, agency, vision_encoder_name, text_encoder_name, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def create_model_and_processors(self, model_dir, **kwargs):
        from flagai.auto_model.auto_loader import AutoLoader
        loader = AutoLoader(
            task_name="txt_img_matching",
            model_dir=model_dir,
            model_name=self.name
        )
        model = loader.get_model()
        model = model.eval()
        tokenizer = loader.get_tokenizer()
        transform = loader.get_transform()
        def text_processor(texts):
            return tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors='pt')
        def image_processor(images):
            return transform(images)['pixel_values']
        return model, text_processor, image_processor
    
    def get_image_features(self, images):
        """
        images: [Tensor Size: (batch, 3, 224, 224)]
        """
        
        self.model = self.model.to(self.device)
        images = torch.cat(images).to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(images)
            return image_features
    
    def get_text_features(self, texts):
        self.model = self.model.to(self.device)
        texts = texts.to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**texts)
            return text_features
