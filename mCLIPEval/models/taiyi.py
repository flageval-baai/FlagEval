from .base import TemplateModel
import torch
import numpy as np

class TaiyiCLIP(TemplateModel):
    def __init__(self, name='Taiyi-CLIP-L', model_dir=None, agency='Noah', vision_encoder_name='VIT-L', text_encoder_name='RoBERTa-wwm-L', **kwargs):
        super().__init__(name, model_dir, agency, vision_encoder_name, text_encoder_name, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def create_model_and_processors(self, model_dir=None, **kwargs):
        from transformers import BertTokenizer,BertForSequenceClassification,CLIPModel,CLIPProcessor
        tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese", cache_dir=model_dir)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=model_dir)
        # transform = processor.feature_extractor
        text_model = BertForSequenceClassification.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese", cache_dir=model_dir).eval().to(self.device)
        image_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=model_dir).to(self.device)
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
        return self.model[1](**texts).logits