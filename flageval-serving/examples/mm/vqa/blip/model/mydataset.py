import os
import json
from PIL import Image
from torch.utils.data import Dataset
from PIL import Image
import requests
from retrying import retry


class COCOVQAEvalDataset(Dataset):

    def __init__(self,
                 vis_processor,
                 text_processor,
                 server_ip="http://localhost",
                 server_port=5000,
                 timeout=1000):
        self.server_ip = server_ip
        self.server_port = server_port
        self.timeout = timeout
        self.text_processor = text_processor
        self.vis_processor = vis_processor
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

    def __getitem__(self, index):
        data = self.get_data(index)
        question = data["question"]
        question_id = data["question_id"]
        img_path = data["img_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.vis_processor["eval"](image)
        text_input = self.text_processor["eval"](question)

        return image, text_input, question_id

    def __len__(self):
        return self.length
