import abc
import importlib
import os
import sys

from typing import Any

from cached_property import cached_property

from .service.base import ModelService


ENV_FINDER_ATTR = "_FLAGEVALSERVING_FINDER_ATTR"


class Finder(metaclass=abc.ABCMeta):
    """Finder holds the global information for flageval serving, like;

    - Inference module
    - Model path
    """

    @abc.abstractmethod
    def set_model(self, p: str) -> None:
        pass

    @abc.abstractproperty
    def model(self) -> str:
        pass

    @abc.abstractmethod
    def set_service(self, p: str) -> None:
        pass

    @abc.abstractproperty
    def service(self) -> ModelService:
        pass

    @abc.abstractmethod
    def set_local_settings(self, m: str) -> None:
        pass

    @abc.abstractproperty
    def local_settings(self) -> Any:
        pass


class EnvironFinder(Finder):
    "A Finder implementation based on environment variables."
    ENV_SERVICE = "_FLAGEVALSERVING_SERVICE"
    ENV_MODEL = "_FLAGEVALSERVING_MODEL"
    ENV_LOCAL_SETTINGS = "_FLAGEVALSERVING_LOCAL_SETTINGS"

    def set_model(self, p: str) -> None:
        os.environ[self.ENV_MODEL] = p

    @cached_property
    def model(self) -> str:
        return os.environ[self.ENV_MODEL]

    def set_service(self, m: str) -> None:
        if m.endswith(".py"):
            dir_, f = os.path.split(m)
            m = f'{f[:-3]}:Service'
            sys.path.insert(0, dir_)
        os.environ[self.ENV_SERVICE] = m

    @cached_property
    def service(self) -> ModelService:
        service = os.environ[self.ENV_SERVICE]
        parts = service.split(":")
        if len(parts) == 2:
            module, cls_name = parts
        elif len(parts) == 1:
            module = parts[0]
            cls_name = "Service"
        else:
            raise ValueError(f"unrecognize service string {service}")


        cls = getattr(importlib.import_module(module), cls_name)
        if not issubclass(cls, ModelService):
            raise ValueError(f'{cls} is not a subclass of ModelService.')

        return cls()

    def set_local_settings(self, m: str):
        os.environ[self.ENV_LOCAL_SETTINGS] = m

    @cached_property
    def local_settings(self) -> Any:
        return importlib.import_module(os.environ[self.ENV_LOCAL_SETTINGS])


environ = EnvironFinder()


def set(ins_path: str):
    os.environ[ENV_FINDER_ATTR] = ins_path


def get() -> Finder:
    attr = os.environ[ENV_FINDER_ATTR]
    if ":" in attr:
        module, name = attr.split(":")
        m = importlib.import_module(module)
        return getattr(m, name)
    return importlib.import_module(attr)  # type: ignore
