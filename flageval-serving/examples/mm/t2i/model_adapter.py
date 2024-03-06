import argparse
from omegaconf import OmegaConf
import os
import os.path as osp
import sys
from retrying import retry
import requests

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from model.inference import load_model_from_config, run  # noqa E402


def parse_args():
    parser = argparse.ArgumentParser(description='Model Adapter')
    parser.add_argument('--task', type=str, default='v2i')
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
        self.meta_info = self.get_meta()

        self.config = OmegaConf.load(
            osp.join(current_dir, "model/v1-inference.yaml")
        )
        self.model_init(io_info['checkpoint_path'])

    def model_init(self, checkpoint_path):
        self.model = load_model_from_config(self.config, checkpoint_path)

    @retry(stop_max_attempt_number=5, wait_fixed=500)
    def get_meta(self):
        url = f"{self.server_ip}:{self.server_port}/meta_info"
        meta_info = requests.get(url, timeout=self.timeout).json()
        return meta_info

    @retry(stop_max_attempt_number=5, wait_fixed=500)
    def get_io_info(self):
        url = f"{self.server_ip}:{self.server_port}/io_info"
        io_info = requests.get(url, timeout=self.timeout).json()
        return io_info

    def run(self):
        url_template = f"{self.server_ip}:{self.server_port}/get_data?index={{}}"
        run(self.model, self.meta_info, url_template, self.output_dir)


if __name__ == '__main__':
    args = parse_args()
    model_adapter = ModelAdapter(
        task=args.task,
        server_ip=args.server_ip,
        server_port=args.server_port,
        timeout=args.timeout
    )
    model_adapter.run()
