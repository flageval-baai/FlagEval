import argparse
import torch
import os
import json
from tqdm import tqdm
from retrying import retry
import requests

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math


def parse_args():
    parser = argparse.ArgumentParser(description='Model Adapter')
    parser.add_argument('--task', type=str, default='vqa')
    parser.add_argument('--server_ip', type=str, default="http://localhost")
    parser.add_argument('--server_port', type=int, default=5000)
    parser.add_argument('--timeout', type=int, default=1000)
    return parser.parse_args()


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def run(tokenizer, model, image_processor, url_template, meta_info, output_dir):
    temperature = 0
    top_p = None
    num_beams = 1
    conv_mode = "vicuna_v1"
    num_questions = meta_info['length']
    dataset_name = meta_info["name"]

    result = []
    for i in range(num_questions):
        data = requests.get(url_template.format(i)).json()
        idx = data["question_id"]
        image_file = data["img_path"]
        question_type = data['type']
        if question_type == 'multiple-choice':
            instruct = "\nAnswer with the option's letter from the given choices directly."
        elif question_type == 'multiple-response':
            instruct = "\nAnswer with the option's letters from the given choices directly. There might be multiple correct choices; indicate them by listing their letters together without spaces."
        elif question_type == 'fill-in-the-blank':
            instruct = "\nComplete each blank with a single word or phrase. If there is more than one blank, separate your answers with a semicolon (;)"
        else:
            instruct = "\nAnswer the question using a single word or phrase."
        qs = data["question"] + instruct
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[conv_mode].copy()
        # conv.system = "Answer the question using a single word or short phrase, give the answer directly, as short as possible"
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        print("Question:", qs)

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX,
            return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(image_file)
        image_tensor = image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids
                               != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids'
            )
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:],
                                         skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        print(outputs, "\n\n")
        result.append({"question_id": idx, "answer": outputs})
    with open(os.path.join(output_dir, dataset_name + ".json"), "w") as fout:
        json.dump(result, fout)


class ModelAdapter:

    def __init__(self, task, server_ip, server_port, timeout=1000):
        self.task = task
        self.server_ip = server_ip
        self.server_port = server_port
        self.timeout = timeout
        io_info = self.get_io_info()
        self.output_dir = io_info['output_dir']
        self.model_init(io_info['checkpoint_path'])
        self.meta_info = self.get_meta()

    def model_init(self, model_path: str):
        disable_torch_init()

        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, model_name)
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor

    @retry(stop_max_attempt_number=5, wait_fixed=500)
    def get_io_info(self):
        url = f"{self.server_ip}:{self.server_port}/io_info"
        io_info = requests.get(url, timeout=self.timeout).json()
        return io_info

    @retry(stop_max_attempt_number=5, wait_fixed=500)
    def get_meta(self):
        url = f"{self.server_ip}:{self.server_port}/meta_info"
        meta_info = requests.get(url, timeout=self.timeout).json()
        return meta_info

    def run(self):
        url_template = f"{self.server_ip}:{self.server_port}/get_data?index={{}}"
        run(self.tokenizer, self.model, self.image_processor, url_template,
            self.meta_info, self.output_dir)


if __name__ == "__main__":
    args = parse_args()
    model_adapter = ModelAdapter(task=args.task,
                                 server_ip=args.server_ip,
                                 server_port=args.server_port,
                                 timeout=args.timeout)
    model_adapter.run()
