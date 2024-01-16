import os
import sys
import torch
import argparse
from retrying import retry
import requests

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from model.inference import get_model, run  # noqa E402
from model.mydataset import COCOVQAEvalDataset


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
        self.model_init(io_info['checkpoint_path'])

    def model_init(self, checkpoint_path: str):
        cfg_path = os.path.join(current_dir, "model", "blip_vqav2.yaml")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.vis_processors, self.txt_processors = get_model(
            cfg_path, self.device, checkpoint_path)
        self.test_dataset = COCOVQAEvalDataset(self.vis_processors,
                                               self.txt_processors,
                                               server_ip=self.server_ip,
                                               server_port=self.server_port,
                                               timeout=self.timeout)
        batch_size = 16
        self.dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=8,
                                                      pin_memory=True)

    @retry(stop_max_attempt_number=5, wait_fixed=500)
    def get_io_info(self):
        url = f"{self.server_ip}:{self.server_port}/io_info"
        io_info = requests.get(url, timeout=self.timeout).json()
        return io_info

    def run(self):
        run(self.model, self.device, self.output_dir, self.dataloader,
            self.test_dataset.datasetname)


if __name__ == "__main__":
    args = parse_args()
    model_adapter = ModelAdapter(task=args.task,
                                 server_ip=args.server_ip,
                                 server_port=args.server_port,
                                 timeout=args.timeout)
    model_adapter.run()
