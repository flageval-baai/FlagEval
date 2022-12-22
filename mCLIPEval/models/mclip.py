from .base import TemplateModel
import torch
import numpy as np

class MCLIP(TemplateModel):
    def __init__(self, name='M-CLIP', model_dir=None, agency='RISE', vision_encoder_name='VIT-L', text_encoder_name='XLMR', **kwargs):
        super().__init__(name, model_dir, agency, vision_encoder_name, text_encoder_name, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def create_model_and_processors(self, model_dir=None, **kwargs):
        from multilingual_clip import pt_multilingual_clip
        from transformers import AutoTokenizer, CLIPProcessor, CLIPModel      
        model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'
        text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name, cache_dir=model_dir).eval().to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
        # 加载CLIP的image encoder
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=model_dir)
        # transform = processor.feature_extractor
        image_model =  CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=model_dir).eval().to(self.device)
        def text_processor(texts):
            return tokenizer(texts, return_tensors='pt', padding=True)
        def image_processor(images):
            images = processor(images=images, return_tensors='pt')
            images["pixel_values"] = images["pixel_values"].squeeze(0)
            return images["pixel_values"]
        return (image_model, text_model), text_processor, image_processor
    
    def get_image_features(self, images):
        images = images.to(self.device)
        return self.model[0].get_image_features(pixel_values=images)
    
    def get_text_features(self, texts):
        texts = texts.to(self.device)
        embs = self.model[1].transformer(**texts)[0]
        att = texts['attention_mask']
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        return self.model[1].LinearTransformation(embs)