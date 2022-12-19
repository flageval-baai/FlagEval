
from abc import ABC, abstractmethod
from .constants import _DEFAULT_AGENCY, _DEFAULT_TEXT_ENCODER, _DEFAULT_VISION_ENCODER

class TemplateModel(ABC):
    def __init__(self, name=None, model_dir=None, agency=None, vision_encoder_name=None, text_encoder_name=None, **kwargs):
        self.name = name
        self.model_dir = model_dir
        self.agency = agency if agency else _DEFAULT_AGENCY
        self.vision_encoder = vision_encoder_name if vision_encoder_name else _DEFAULT_VISION_ENCODER
        self.text_encoder = text_encoder_name if text_encoder_name else _DEFAULT_TEXT_ENCODER
        self.kwargs = kwargs
    
    def initialize(self):
        self.model, self.text_processor, self.image_processor = self.create_model_and_processors(self.model_dir, **self.kwargs)
    
    @abstractmethod
    def create_model_and_processors(self, model_dir, **kwargs):
        pass

    @abstractmethod
    def get_text_features(self, texts):
        pass

    @abstractmethod
    def get_image_features(self, images):
        pass

    def __info__(self):
        return {
            'model_name': self.name,
            'vision_encoder': self.vision_encoder,
            'text_encoder': self.text_encoder,
            'agency': self.agency
        }

class EvalModel(object):
    def __init__(self, model_config=None) -> None:
        from .constants import _SUPPORTED_MODELS, _SUPPORTED_CNCLIP_MODELS, _SUPPORTED_OPENCLIP_MODELS
        from .altclip import AltClip
        from .evaclip import EvaClip
        from .openclip import OpenCLIPModel
        from .cnclip import CnCLIP
        from .mclip import MCLIP
        from .taiyi import TaiyiCLIP
        for model_name, config_dict in model_config.items():
            model_script = config_dict.get('model_script', None)
            if model_name in ['AltCLIP-XLMR-L', 'AltCLIP-XLMR-L-m9']:
                model = AltClip(name=model_name, model_dir=config_dict.get('model_dir', None))
            elif model_name in ['eva-clip']:
                model = EvaClip(model_dir=config_dict.get('model_dir', None))
            elif model_name in ['M-CLIP']:
                model = MCLIP(model_dir=config_dict.get('model_dir', None))
            elif model_name in ['Taiyi-CLIP-L']:
                model = TaiyiCLIP(model_dir=config_dict.get('model_dir', None))
            elif model_name in _SUPPORTED_OPENCLIP_MODELS or model_script=='openclip':
                model = OpenCLIPModel(name=model_name, model_config=config_dict)
            elif model_name in _SUPPORTED_CNCLIP_MODELS or model_script=='cnclip':
                model = CnCLIP(name=model_name, model_config=config_dict)
            elif model_script:
                def get_main_class(script):
                    import importlib, inspect
                    module = importlib.import_module(script)
                    module_main_cls = None
                    for name, obj in module.__dict__.items():
                        if name == 'TemplateModel':
                            continue
                        if inspect.isclass(obj) and issubclass(obj, TemplateModel):
                            module_main_cls = obj
                            break
                    return module_main_cls
                main_cls = get_main_class(model_script)
                assert main_cls
                config_dict['name'] = config_dict.get('name', model_name)
                model = main_cls(**config_dict)
        self.model = model
    
    def __info__(self):
        return self.model.__info__()