import os

from flask import Flask

from . import signals
from .flagenv import FlagevalConfigLoader
from .flagenv import FlagevalExtInitializer


config_loader = FlagevalConfigLoader()
ext_initializer = FlagevalExtInitializer()


def create_app(name: str) -> Flask:
    """创建一个 Flask APP"""
    app = Flask(name)
    config_loader.init_app(app)
    return app


app: Flask = create_app(__name__)
"""flageval:class:`flask.Flask` 实例。部署时可以指定 ``flageval.serving.app:app`` 进行部署。
"""


@signals.ext_ready.connect
def _setup_apps_and_raise_ready(app):
    # No project module(blueprints) will be imported before this signal has been sent.
    signals.ready.send(app)


if "FLAG_NO_INIT" not in os.environ:  # pragma: no cover
    ext_initializer.init_app(app)
