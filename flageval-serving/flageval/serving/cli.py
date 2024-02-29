import os
import sys

import click

from . import finder
from .service import settings


@click.group(
    name="flageval",
    context_settings=dict(
        auto_envvar_prefix="FLAGEVAL_SERVING",
    ),
)
@click.option(
    "--service",
    default="flageval.serving.service.demo_service:DemoService",
    help="Style of 'pkg.module:attr' to import a implementation of the flageval.sevice.base.ModelService",
    show_default=True,
)
@click.option(
    "--settings",
    default="local_settings",
    show_default=True,
    help="Style of 'pkg.module' to import a module to override default settings.",
)
@click.option(
    "--finder",
    "finder_attr",
    type=str,
    default="flageval.serving.finder:environ",
    show_default=True,
)
def cli(service, settings, finder_attr):
    "Flageval serving to bootstrap AI model for evaluating."

    finder.set(finder_attr)
    f = finder.get()

    # ready for merge local settings
    f.set_local_settings(settings)

    os.environ["FLAG_SETTINGS_MODULE"] = "flageval.serving.service.settings"
    f.set_service(service)

    # set current path
    sys.path.insert(0, ".")

    guess_project()


@cli.command()
@click.option("--host", default="127.0.0.1", help="Debug mode listen host.")
@click.option("--port", default=5000, type=int, help="Debug mode listen port.")
@click.argument("model", default=".")
def dev(host, port, model):
    "Serving model with debug mode."
    f = finder.get()
    f.set_model(model)

    f.service.global_init(f.model)
    stage = "development"
    _merge_local_settings(stage)

    os.environ["FLASK_ENV"] = "development"
    os.environ["STAGE"] = stage
    from .app import app

    click.echo("** Running flagevals-serving dev mode:", color=True)
    click.echo(f"**    Model path: {model}", color=True)
    click.echo(f"**    Listening on: {host}:{port}", color=True)
    app.run(host=host, port=port, debug=True, use_reloader=False)


@cli.command()
@click.option(
    "--stage",
    default="test",
    type=click.Choice(("test", "staging", "production")),
    show_choices=True,
)
@click.argument("model")
def run(stage, model):
    "Serving model with production model."
    settings.ENV = stage
    os.environ["FLASK_ENV"] = "production"
    os.environ["STAGE"] = stage

    finder.get().set_model(model)

    _merge_local_settings(stage)
    sys.argv = sys.argv[:1]
    sys.argv.append("--config=python:flageval.serving.service.guniconf")
    from gunicorn.app.wsgiapp import WSGIApplication

    WSGIApplication("%(prog)s OPTIONS").run()

class NoHelpOption(click.Option):
    def get_help_record(self, ctx):
        pass

@cli.command()
@click.option("--host", default="https://flageval.baai.ac.cn/", help="FlagEval host.")
@click.option("--token", required=True, help="Token to upload model.")
@click.option("--followlinks", default=False, help="Follow links.")
@click.argument("model_path", required=True)
def upload(host, token, followlinks, model_path):
    "Upload model for evaluating."
    from .flageval import FlagEvalUploader

    uploader = FlagEvalUploader(host, token, model_path, followlinks=followlinks)
    uploader.upload()


@cli.command()
@click.option("--host", default="https://flageval.baai.ac.cn/", help="FlagEval host.", cls=NoHelpOption)
@click.option("--token", required=True, default="", help="user's token.")
@click.argument("dst_path", default="")
def ls(host, token, dst_path):
    """list files in flageval.

\b
for example:
flageval-serving ls --token='your token' /remote_path
    """
    if token == "":
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        return

    from .flageval import FlagEvalManager
    manager = FlagEvalManager(host, token, "", dst_path)
    result, _ = manager.list_remote_files(dst_path)
    if len(result) == 0:
        print("error: " + dst_path + " not exists", file=sys.stderr)
    else:
        for item in result:
            print(item.path)

@cli.command()
@click.option("--host", default="https://flageval.baai.ac.cn/", help="FlagEval host.", cls=NoHelpOption)
@click.option("--token", required=True, default="", help="destination user's token.")
@click.option("--followlinks", default=False, help="Follow links.", cls=NoHelpOption)
@click.argument("src_path", default="")
@click.argument("dst_path", default="")
def cp(host, token, followlinks, src_path, dst_path):
    """copy file or directory to flageval.

\b
for example:
flageval-serving cp --token='your token' /path/fileordir /remote_path
    """
    if token == "":
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        return
    

    from .flageval import FlagEvalManager
    manager = FlagEvalManager(host, token, src_path, dst_path, followlinks = followlinks)
    manager.cp()


def _merge_local_settings(stage):
    from .service import settings

    f = finder.get()
    settings.ENV = stage
    click.echo(f"[LOCAL] Merge ENV = {getattr(settings, 'ENV')}")

    try:
        for key, val in f.local_settings.__dict__.items():
            if not key.startswith("_") and key.isupper():
                setattr(settings, key, val)
                click.echo(f"[LOCAL] Merge {key} = {getattr(settings, key)}")
    except ImportError as e:
        click.echo(f"Merge local settings failed due to {e.args[0]}")


def guess_project():
    if settings.DEFAULT_PROJECT_NAME != settings.PROJECT_NAME or True:
        return

    click.echo("NO PROJECT_NAME setted in settings, let me guess...")
    import os

    settings.PROJECT_NAME = os.path.split(os.getcwd())[-1]
    click.echo("Say.... %s!" % (settings.PROJECT_NAME))


def main():
    cli()
