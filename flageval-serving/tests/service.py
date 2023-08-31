import datetime
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from flageval.serving.service import NLPModelService, NLPEvalRequest, NLPEvalResponse, NLPCompletion


class ModelService:
    """
    Adjustable configuration values can be determined based on model requirements, such as seed, token length,
    temperature, number of GPUs, etc.
    """

    def __init__(self, model_path):
        self.model_path = model_path
        self.model_info = "model_name"
        self.seed = 1234
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.cuda()
        self.model.eval()

    def set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def request(self, request_info):
        inputs = self.tokenizer(request_info.get('prompt'), return_tensors="pt").to("cuda")
        self.seed = request_info.get('seed', 1234)
        self.set_seed()
        tokens = self.model.generate(
            **inputs,
            max_new_tokens=request_info.get('max_new_tokens', 100),
            temperature=request_info.get('temperature', 0.7),
            top_k=request_info.get('top_k_per_token'),
            top_p=request_info.get('top_p'),
            do_sample=True
        )
        completion_tokens = tokens[0][inputs['input_ids'].size(1):]
        completion = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)

        if "###" in completion:
            special_index = completion.index("###")
            completion = completion[: special_index]

        if "[UNK]" in completion:
            special_index = completion.index("[UNK]")
            completion = completion[:special_index]

        if "</s>" in completion:
            special_index = completion.index("</s>")
            completion = completion[: special_index]

        if len(completion) > 0 and completion[0] == " ":
            completion = completion[1:]

        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")

        answer = {"completions":
            [{
                "text": completion,
                "tokens": completion,
                "logprobs": [],
                "top_logprobs_dicts": []
            }],
            "model_info": str(self.model_info.split('/')[-1]).lower() + "-seed-" + str(self.seed),
            "input_length": len(inputs),
            "status": 200,
            "time": time
        }

        log = "[" + time + "] " + ' prompt: \n ' + \
              request_info.get('prompt') + '\n response: \n' + repr(completion)
        print(log)
        return answer


class Service(NLPModelService):
    def global_init(self, model_path: str):
        """
        init model
        """
        self.model_server = ModelService(model_path)

    def infer(self, req: NLPEvalRequest) -> NLPEvalResponse:
        raw_request = {
            "engine": req.engine,
            "prompt": req.prompt,
            "temperature": req.temperature,
            "num_return_sequences": req.max_return_sequences,
            "max_new_tokens": req.max_tokens,
            "top_p": req.top_p,
            "echo_prompt": req.echo_prompt,
            "top_k_per_token": req.top_k_per_token,
            "stop_sequences": req.stop_sequences,
        }

        temp = self.model_server.request(raw_request)

        return NLPEvalResponse(
            model_info=temp.get('model_info') or '',
            status=temp.get('status') or 200,
            input_length=temp.get('input_length') or 0,
            completions=[
                NLPCompletion(
                    text=item['text'],
                    tokens=item['tokens'],
                    logprobs=item['logprobs'],
                    top_logprobs_dicts=item['top_logprobs_dicts'],
                )
                for item in temp.get('completions') or []
            ]
        )
