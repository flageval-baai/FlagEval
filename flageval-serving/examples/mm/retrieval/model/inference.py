import argparse
import torch
import clip
from PIL import Image
import numpy as np
import os
import requests


def get_model(checkpoint):
    model, preprocess = clip.load(checkpoint)
    return model, preprocess


def get_image(image_id, url_template, preprocess):
    url = url_template.format(image_id, 'img')
    response = requests.get(url, timeout=500).json()
    image = Image.open(response['img_path']).convert('RGB')
    return preprocess(image).cuda()


def get_caption(caption_id, url_template):
    url = url_template.format(caption_id, 'text')
    response = requests.get(url, timeout=500).json()
    return response['caption']


def run(
    model,
    preprocess,
    meta_info,
    output_dir,
    url_template,
    itd=50,
):
    max_cap_len = 77

    N = meta_info['image_number']
    for i in range(0, N, itd):
        if i % itd == 0:
            print('{}/{}=={}%'.format(i, N, 100. * i / N))
        _s, _e = i, i + itd
        _e = min(_e, N)
        images = [
            get_image(image_id, url_template, preprocess)
            for image_id in range(_s, _e)
        ]
        # images = [image.cuda() for image in images]
        images = torch.stack(images, 0).squeeze()

        caption = [
            get_caption(caption_id, url_template)
            for caption_id in range(_s * 5, _e * 5)
        ]
        texts = [
            clip.tokenize([cap], context_length=max_cap_len,
                          truncate=True).cuda() for cap in caption
        ]
        texts = torch.stack(texts, 0).squeeze()
        with torch.no_grad():
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            image_embeddings = image_features / image_features.norm(
                dim=-1, keepdim=True)
            text_embeddings = text_features / text_features.norm(dim=-1,
                                                                 keepdim=True)
        torch.cuda.empty_cache()

        if i == 0:
            acc_image_embeddings = image_embeddings.cpu().numpy()
            acc_text_embeddings = text_embeddings.cpu().numpy()
        else:
            tmp_image_embeddings = image_embeddings.cpu().numpy()
            tmp_text_embeddings = text_embeddings.cpu().numpy()
            acc_image_embeddings = np.concatenate(
                (acc_image_embeddings, tmp_image_embeddings), axis=0)
            acc_text_embeddings = np.concatenate(
                (acc_text_embeddings, tmp_text_embeddings), axis=0)

    acc_image_embeddings = torch.from_numpy(acc_image_embeddings).cuda()
    acc_text_embeddings = torch.from_numpy(acc_text_embeddings).cuda()
    acc_sim = acc_image_embeddings.mm(acc_text_embeddings.T)
    acc_sim = acc_sim.cpu().numpy()
    full_save_path = os.path.join(output_dir, meta_info['name'])
    np.save('{}'.format(full_save_path), acc_sim)
    return acc_sim
