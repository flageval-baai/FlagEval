_DEFAULT_AGENCY = 'NULL'
_DEFAULT_VISION_ENCODER = 'Unknown'
_DEFAULT_TEXT_ENCODER = 'Unknown'

_OPEN_CLIP_MODEL_CONFIGS = {
    'openai-clip-L': {'agency': 'openai', 'text_encoder_name': 'CLIP-L', 'vision_encoder_name': 'VIT-L', 'model': 'ViT-L-14', 'pretrained': 'openai'},
    'openai-clip-L-336': {'agency': 'openai', 'text_encoder_name': 'CLIP-L', 'vision_encoder_name': 'VIT-L', 'model': 'ViT-L-14-336', 'pretrained': 'openai'},
    'openclip-L': {'agency': 'openclip', 'text_encoder_name': 'CLIP-L', 'vision_encoder_name': 'VIT-L', 'model': 'ViT-L-14', 'pretrained': 'laion2b_s32b_b82k'},
    'openclip-L-v0': {'agency': 'openclip', 'text_encoder_name': 'CLIP-L', 'vision_encoder_name': 'VIT-L', 'model': 'ViT-L-14', 'pretrained': 'laion400m_e32'},
    'openclip-H': {'agency': 'openclip', 'text_encoder_name': 'CLIP-H', 'vision_encoder_name': 'VIT-H', 'model': 'ViT-H-14', 'pretrained': 'laion2b_s32b_b79k'},
    'openclip-H-XLMR-L': {'agency': 'openclip', 'text_encoder_name': 'XLMR-L', 'vision_encoder_name': 'VIT-H', 'model': 'xlm-roberta-large-ViT-H-14', 'pretrained': 'frozen_laion5b_s13b_b90k'},
    'openclip-B-XLMR-B': {'agency': 'openclip', 'text_encoder_name': 'XLMR-B', 'vision_encoder_name': 'VIT-B', 'model': 'xlm-roberta-base-ViT-B-32', 'pretrained': 'laion5b_s13b_b90k'},
}
_SUPPORTED_OPENCLIP_MODELS = list(_OPEN_CLIP_MODEL_CONFIGS.keys())

_CN_CLIP_MODEL_CONFIGS = {
    'cn-clip-L': {'agency': 'damo', 'text_encoder_name': 'RoBERTa-wwm-L', 'vision_encoder_name': 'VIT-L', 'model': 'ViT-L-14'},
    'cn-clip-L-336': {'agency': 'damo', 'text_encoder_name': 'RoBERTa-wwm-L', 'vision_encoder_name': 'VIT-L', 'model': 'ViT-L-14-336'}
}
_SUPPORTED_CNCLIP_MODELS = list(_CN_CLIP_MODEL_CONFIGS.keys())

_SUPPORTED_MODELS = ['M-CLIP', 'AltCLIP-XLMR-L', 'AltCLIP-XLMR-L-m9', 'eva-clip', 'Taiyi-CLIP-L'] + _SUPPORTED_OPENCLIP_MODELS + _SUPPORTED_CNCLIP_MODELS