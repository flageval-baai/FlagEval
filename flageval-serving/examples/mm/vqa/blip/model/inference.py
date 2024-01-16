import torch
import json
import os
import logging
from omegaconf import OmegaConf
from tqdm import tqdm
from lavis.models import load_preprocess
from lavis.common.registry import registry


def load_model_and_preprocess(name,
                              model_type,
                              config_path,
                              checkpoint_path,
                              is_eval=False,
                              device="cpu"):
    """
    Load model and its related preprocessors.

    List all available models and types in registry:
    >>> from lavis.models import model_zoo
    >>> print(model_zoo)

    Args:
        name (str): name of the model.
        model_type (str): type of the model.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu".

    Returns:
        model (torch.nn.Module): model.
        vis_processors (dict): preprocessors for visual inputs.
        txt_processors (dict): preprocessors for text inputs.
    """
    model_cls = registry.get_model_class(name)
    # load cfg
    cfg = OmegaConf.load(config_path)
    cfg.model.finetuned = checkpoint_path
    # load model
    model_cfg = cfg.model
    model = model_cls.from_config(model_cfg)

    if is_eval:
        model.eval()

    # load preprocess
    if cfg is not None:
        preprocess_cfg = cfg.preprocess

        vis_processors, txt_processors = load_preprocess(preprocess_cfg)
    else:
        vis_processors, txt_processors = None, None
        logging.info(f"""No default preprocess for model {name} ({model_type}).
                This can happen if the model is not finetuned on downstream datasets,
                or it is not intended for direct use without finetuning.
            """)

    if device == "cpu" or device == torch.device("cpu"):
        model = model.float()

    return model.to(device), vis_processors, txt_processors


def get_model(path, device, checkpoint_path):
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip_vqa",
        model_type="vqav2",
        config_path=path,
        is_eval=True,
        checkpoint_path=checkpoint_path,
        device=device)
    return model, vis_processors, txt_processors


def run(model, device, output_dir, dataloader, datasetname):
    result = []
    for images, questions, question_ids in tqdm(dataloader):
        images = images.to(device)
        answers = model.predict_answers(samples={
            "image": images,
            "text_input": questions
        },
                                        inference_method="generate")
        for answer, question_id in zip(answers, question_ids.tolist()):
            result.append({"question_id": question_id, "answer": answer})

    file_name = datasetname + ".json"
    save_path = os.path.join(output_dir, file_name)
    with open(save_path, "w") as json_file:
        json.dump(result, json_file)
