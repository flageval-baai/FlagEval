from ..app import app

def get_service():
    from .. import finder
    return finder.get().service

service = get_service()

app.route(service.endpoint, methods=["POST"])(service.do_infer)
