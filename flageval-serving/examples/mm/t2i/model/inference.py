import os
import json

import torch
import numpy as np
from PIL import Image
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler
import requests


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()

    return model


def get_one_prompt(url_template, index):
    url = url_template.format(index)
    response = requests.get(url, timeout=500).json()
    return response['prompt'], response['id']


def run(model, meta_info, url_template, output_dir):
    seed_everything(42)

    sampler = PLMSSampler(model)
    batch_size = 1

    ddim_steps = 50
    scale = 7.5

    text_num = meta_info['length']

    shape = [4, 512 // 8, 512 // 8]
    output_info = []
    with model.ema_scope():
        for i in range(text_num):
            prompt, question_id = get_one_prompt(url_template, i)
            uc = model.get_learned_conditioning(batch_size * [""])
            c = model.get_learned_conditioning(prompt)
            samples_ddim, _ = sampler.sample(
                S=ddim_steps,
                conditioning=c,
                batch_size=batch_size,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
                eta=0.0,
                x_T=None
            )

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp(
                (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
            )
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3,
                                                          1).numpy() * 255

            x_sample = x_samples_ddim[0]
            img = Image.fromarray(x_sample.astype(np.uint8))
            image_out_name = f"{question_id}.png"
            img.save(os.path.join(output_dir, image_out_name))
            output_info.append(
                {
                    "prompt": prompt,
                    "id": question_id,
                    "image": image_out_name
                }
            )

    json.dump(
        output_info,
        open(f"{output_dir}/output_info.json", "w"),
        indent=2,
        ensure_ascii=False
    )
    print(
        f"Your samples are ready and waiting for you here: \n{output_dir} \n"
    )
