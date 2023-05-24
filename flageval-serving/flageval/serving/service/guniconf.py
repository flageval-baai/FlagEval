from .. import guniconf
from ..guniconf import *  # noqa


def post_worker_init(worker):
    from .. import finder

    if hasattr(guniconf, "post_worker_init"):
        guniconf.post_worker_init(worker)  # type: ignore

    f = finder.get()
    f.service.global_init(f.model)
    worker.log.info("Initialized model path: %s", f.model)
