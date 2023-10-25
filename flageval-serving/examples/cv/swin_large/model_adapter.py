import os
import sys
from mmengine.config import Config

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from model.user_model import SwinTransformer  # noqa E402


class ModelAdapter:

    def model_init(self, checkpoint_path):
        config = Config.fromfile(os.path.join(current_dir, 'model', 'config.py'))

        model = SwinTransformer(pretrain_img_size=224,
                                patch_size=config.backbone.patch_size,
                                in_channels=config.backbone.in_chans,
                                embed_dims=config.backbone.embed_dim,
                                depths=config.backbone.depths,
                                num_heads=config.backbone.num_heads,
                                window_size=config.backbone.window_size,
                                mlp_ratio=config.backbone.mlp_ratio,
                                qkv_bias=config.backbone.qkv_bias,
                                qk_scale=config.backbone.qk_scale,
                                drop_rate=config.backbone.drop_rate,
                                drop_path_rate=config.backbone.drop_path_rate,
                                use_abs_pos_embed=config.backbone.ape,
                                norm_cfg={'type': 'layernorm'},
                                act_cfg={'type': 'gelu'},
                                patch_norm=config.backbone.patch_norm,
                                use_checkpoint=False,
                                noFPN=config.backbone.noFPN,
                                strides=config.backbone.strides,
                                out_indices=config.backbone.out_indices)
        model.load_pretrained(model, checkpoint_path)
        return model
