import abc
import json

from dataclasses import dataclass, field
from typing import List, Dict, Any

from dataclasses_json import dataclass_json
from flask import request, Response, jsonify

from .base import ModelService


@dataclass
class NLPEvalRequest:
    engine: str = ""

    prompt: str = ""
    """What prompt do condition the language model on"""

    temperature: float = 1.0
    """Temperature parameter that governs diversity"""

    max_return_sequences: int = 1
    """Generate this many completions (by sampling from the model)"""

    max_tokens: int = 100
    """Maximum number of tokens to generate (per completion)"""

    top_p: float = 1
    """Same from tokens that occupy this probability mass (nucleus sampling)"""

    echo_prompt: bool = False
    """Should `prompt` be included as a prefix of each completion? (e.g., for
    evaluating perplexity of the prompt)"""

    top_k_per_token: int = 1
    """Take this many highest probability candidates per token in the completion"""

    stop_sequences: List[str] = field(default_factory=list)
    """Stop generating once we hit one of these strings."""

    extra: Dict[str, Any] = field(default_factory=dict)
    """Extra parameters required by model."""


@dataclass
class NLPCompletion:
    text: str
    tokens: str
    logprobs: List[str] = field(default_factory=list)
    top_logprobs_dicts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass_json
@dataclass
class NLPEvalResponse:
    completions: List[NLPCompletion]
    model_info: str = ''
    status: int = 200
    input_length: int = 0


class NLPModelService(ModelService):
    @abc.abstractmethod
    def infer(self, req: NLPEvalRequest) -> NLPEvalResponse:
        pass

    def parse_request(self) -> NLPEvalRequest:
        data = request.get_json()
        if isinstance(data, (str, bytes)):
            data = json.loads(data)

        return NLPEvalRequest(
            engine=data["engine"],
            prompt=data["prompt"],
            temperature=data["temperature"],
            max_return_sequences=data["num_return_sequences"],
            max_tokens=data["max_new_tokens"],
            top_p=data["top_p"],
            echo_prompt=data["echo_prompt"],
            top_k_per_token=data["top_k_per_token"],
            stop_sequences=data["stop_sequences"],
        )

    def write_response(self, response: NLPEvalResponse) -> Response:
        return jsonify(response.to_dict())
