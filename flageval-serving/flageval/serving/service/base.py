import abc
import logging

from typing import Any

from flask import Response, jsonify


logger = logging.getLogger(__name__)


class ModelService(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def global_init(self, model_path: str):
        """This will be invoked once during the boostrap of the process.
        """
        pass

    def do_infer(self) -> Response:
        request = self.parse_request()
        try:
            response = self.infer(request)
            return self.write_response(response)
        except Exception as e:
            logger.error("Uncaught error", exc_info=True)
            response = jsonify({"code": 500, "message": e.args[0]})
            response.status_code = 500
            return response

    @abc.abstractmethod
    def infer(self, req: Any) -> Any:
        pass

    @abc.abstractmethod
    def parse_request(self) -> Any:
        pass

    @abc.abstractmethod
    def write_response(self, response: Any) -> Response:
        pass

    @property
    def endpoint(self) -> str:
        return "/func"
