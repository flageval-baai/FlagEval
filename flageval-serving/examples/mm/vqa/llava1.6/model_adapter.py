import argparse
import torch
from torch.utils.data import Dataset, DataLoader

import os
import json
from retrying import retry
import requests
import time
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token
from dataclasses import dataclass
import transformers
from PIL import Image
import re
from typing import List, Tuple
from accelerate import Accelerator


def remove_images_symbol(text):
    pattern = r'<image\s*\d+\>'
    result = re.sub(pattern, '', text)
    return result


class CustomDataset(Dataset):

    def __init__(
        self,
        tokenizer,
        image_processor,
        model_config,
        server_ip="http://localhost",
        server_port=5000,
        timeout=1000
    ):
        self.server_ip = server_ip
        self.server_port = server_port
        self.timeout = timeout
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        meta_info = self.get_meta()
        self.datasetname = meta_info['name']
        self.length = meta_info['length']

    @retry(stop_max_attempt_number=5, wait_fixed=500)
    def get_meta(self):
        url = f"{self.server_ip}:{self.server_port}/meta_info"
        meta_info = requests.get(url, timeout=self.timeout).json()
        return meta_info

    @retry(stop_max_attempt_number=5, wait_fixed=500)
    def get_data(self, index):
        url = f"{self.server_ip}:{self.server_port}/get_data?index={index}"
        response = requests.get(url)
        return response.json()

    def get_instruct(self, data):
        is_cn = any('\u4e00' <= char <= '\u9fff' for char in data["question"])
        question_type = data['type']
        if is_cn:
            if question_type == 'multiple-choice':
                instruct = "从给定的选项中直接选择答案的字母，不要多余的解释。"
            elif question_type == 'multiple-response':
                instruct = "请直接回答正确选项的字母，正确选项可能有多个"
            elif question_type == 'fill-in-the-blank':
                instruct = "在横线或者空白处直接填上答案，如果有多个空需要填，使用分号(;)分隔。直接给出答案，不需要多余的解释"
            elif question_type == 'yes-no':
                instruct = "直接回答'对'或'错'，不需要多余的解释"
            else:
                instruct = "用简短的句子或者词语回答问题，不需要多余的解释。"
        else:
            if question_type == 'multiple-choice':
                instruct = "Answer with the option's letter from the given choices directly."
            elif question_type == 'multiple-response':
                instruct = "Answer with the option's letters from the given choices directly. There might be multiple correct choices; indicate them by listing their letters together without spaces."
            elif question_type == 'fill-in-the-blank':
                instruct = "Complete each blank with a single word or phrase directly. If there is more than one blank, separate your answers with a semicolon (;)"
            elif question_type == 'yes-no':
                instruct = "Answer with 'yes' or 'no'."
            else:
                instruct = "Answer the question using a single word or phrase."
        return instruct

    def __getitem__(self, index):
        conv_mode = "vicuna_v1"
        data = self.get_data(index)
        question_id = data["question_id"]
        # only use the first image
        img_path = data["img_path"][0]

        instruct = self.get_instruct(data)
        qs = remove_images_symbol(data["question"]) + "\n" + instruct

        image = Image.open(img_path).convert('RGB')
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensor = self.image_processor.preprocess(
            image, return_tensors='pt'
        )['pixel_values'][0]

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        )
        return input_ids, image_tensor, question_id, qs

    def __len__(self):
        return self.length


@dataclass
class DataCollatorForVisualTextGeneration(object):
    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [
                torch.flip(_input_ids, [0]) for _input_ids in input_ids
            ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids, image_tensors, question_ids, questions = zip(*batch)
        input_ids = self.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        image_tensors = torch.stack(image_tensors, dim=0)
        return input_ids, image_tensors, question_ids, questions


# DataLoader
def create_data_loader(
    tokenizer, image_processor, model_config, batch_size=8, num_workers=8
):
    dataset = CustomDataset(tokenizer, image_processor, model_config)
    collator = DataCollatorForVisualTextGeneration(tokenizer=tokenizer)
    data_loader = DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    return data_loader


def parse_args():
    parser = argparse.ArgumentParser(description='Model Adapter')
    parser.add_argument('--task', type=str, default='vqa')
    parser.add_argument('--server_ip', type=str, default="http://localhost")
    parser.add_argument('--server_port', type=int, default=5000)
    parser.add_argument('--timeout', type=int, default=1000)
    return parser.parse_args()


class ModelAdapter:

    def __init__(self, task, server_ip, server_port, timeout=1000):
        self.task = task
        self.server_ip = server_ip
        self.server_port = server_port
        self.timeout = timeout
        io_info = self.get_io_info()
        self.output_dir = io_info['output_dir']
        self.accelerator = Accelerator()
        self.model_init(io_info['checkpoint_path'])
        self.meta_info = self.get_meta()

    def model_init(self, model_path: str):
        disable_torch_init()
        model_path = os.path.expanduser(model_path)
        model_name = "llava"
        with self.accelerator.main_process_first():
            tokenizer, model, image_processor, _ = load_pretrained_model(
                model_path,
                None,
                model_name,
                device_map={"": self.accelerator.process_index},
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
            )
            data_loader = create_data_loader(
                tokenizer, image_processor, model.config
            )
        model = model.to(torch.bfloat16)
        # set padding side to `left` for batch text generation
        model.config.tokenizer_padding_side = tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.unk_token
        data_loader = self.accelerator.prepare(data_loader)
        model = self.accelerator.prepare_model(model, evaluation_mode=True)
        self.tokenizer = tokenizer
        if hasattr(model, "module"):
            model = model.module
        self.model = model
        self.data_loader = data_loader

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
        temperature = 0
        result = []
        cnt = 0

        for input_ids, image_tensor, question_ids, questions in self.data_loader:
            if cnt == 1:
                start_time = time.perf_counter()
            cnt += 1
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.bfloat16),
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    max_new_tokens=1024,
                    attention_mask=attention_mask,
                    use_cache=True
                )

            outputs = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            for i, idx in enumerate(question_ids):
                self.accelerator.print(f"{questions[i]}\n{outputs[i]}\n\n")
                result.append(
                    {
                        "question_id": idx,
                        "answer": outputs[i].strip(),
                        "prompt": questions[i]
                    }
                )
        rank = self.accelerator.state.local_process_index
        with open(
            os.path.join(
                self.output_dir, f"{self.meta_info['name']}_rank{rank}.json"
            ), "w"
        ) as fout:
            json.dump(result, fout, indent=2)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            print(
                "self.accelerator.state.num_processes",
                self.accelerator.state.num_processes
            )
            results_collect = []
            id_set = set()
            for i in range(self.accelerator.state.num_processes):
                with open(
                    os.path.join(
                        self.output_dir,
                        f"{self.meta_info['name']}_rank{i}.json"
                    ), "r"
                ) as fin:
                    for ans in json.load(fin):
                        if ans['question_id'] not in id_set:
                            id_set.add(ans['question_id'])
                            results_collect.append(ans)
            total_time = time.perf_counter() - start_time
            print("Total time:", total_time)
            print("Average time:", total_time / cnt)
            print("results_collect number", len(results_collect))
            with open(
                os.path.join(
                    self.output_dir, self.meta_info["name"] + ".json"
                ), "w"
            ) as fout:
                json.dump(results_collect, fout, indent=2)
            print("finished")
        print("rank", rank, "finished")


if __name__ == "__main__":
    args = parse_args()
    model_adapter = ModelAdapter(
        task=args.task,
        server_ip=args.server_ip,
        server_port=args.server_port,
        timeout=args.timeout
    )
    model_adapter.run()
