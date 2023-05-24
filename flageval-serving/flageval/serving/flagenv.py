import inspect
import warnings

from typing import Optional

from flask import Blueprint
from flask import Flask

from . import signals
from . import util


class FlagevalConfigLoader(object):
    "Flageval 配置加载"

    def __init__(self, app: Optional[Flask] = None) -> None:
        if app is not None:  # pragma: no cover
            self.init_app(app)

    def init_app(self, app: Flask) -> None:
        settings = util.get_settings_module()
        for k, v in settings.__dict__.items():
            if k.startswith("_"):
                continue
            app.config[k] = v

        signals.config_ready.send(app)


class FlagevalExtInitializer(object):
    def __init__(self, app: Optional[Flask] = None) -> None:
        if app is not None:  # pragma: no cover
            self.init_app(app)

    def init_app(self, app: Flask) -> None:
        self._init_exts(app)
        signals.ext_ready.send(app)

    @staticmethod
    def _init_exts(app: Flask) -> None:
        from . import extensions

        for value in extensions.__dict__.values():
            if inspect.isclass(value):
                continue

            if not hasattr(value, "init_app"):
                continue

            if getattr(value, "__flag_ignore__", False):
                continue

            try:
                value.init_app(app)
            except (ValueError, KeyError) as e:  # pragma: no cover
                warnings.warn(
                    f"Initialize {value} failed with KeyError/ValueError: {e.args[0]}",
                    RuntimeWarning,
                )


@signals.ready.connect
def _load_apps(app):
    for pkg in app.config["INSTALLED_APPS"]:
        for m in util.SubmoduleLoader(pkg).iter_modules():
            for value in m.__dict__.values():
                if isinstance(value, Blueprint):
                    base_prefix = __make_url_prefix(app, value)
                    app.register_blueprint(value, url_prefix=base_prefix)

    signals.blueprint_loaded.send(app)


def __make_url_prefix(app, blueprint: Blueprint):
    base_prefix = app.config.get("BASE_PREFIX", "")
    blueprint_prefix = blueprint.url_prefix or ""
    return "{base_prefix}{blueprint_prefix}".format(
        base_prefix=base_prefix,
        blueprint_prefix=blueprint_prefix,
    )
