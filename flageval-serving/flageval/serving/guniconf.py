"""
Install
--------

Install gunicorn with gevent::

    pip install --upgrade 'gunicorn[gevent]>=20.1.0'

Usage
-----

Start gunicorn with flageval::

    flageval-serving webserver

Custom
------

You can put a ``local_guniconf.py`` in your current work directory to set your
own config or override the defaults provided in FlagEval::

    # reload
    reload = True
"""


def post_fork(server, worker):
    from flageval.serving import signals

    signals.worker_forked.send(worker, worker_type="gunicorn")


def get_free_port(ports_range):
    import socket

    for port in ports_range:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            continue
        finally:
            sock.close()

        return port


# Default to bind on 5000
bind = "0.0.0.0:{}".format(get_free_port(range(5000, 6000)))

# Set workers number to 1 CPUs
workers = 1

# Use gevent as worker class
worker_class = "gevent"

# The maximum number of simultaneous clients.
worker_connections = 102400

# Set WSGI Application to FlagEval serving
wsgi_app = "flageval.serving.app:app"

# Workers silent for more than this many seconds are killed and restarted.
timeout = 300

# Write accesslog to stdout
accesslog = "-"

# Disable redirect access logs to syslog.
disable_redirect_access_to_syslog = True

# Access log format:
#
#   h remote address
#   l '-'
#   u user name
#   t date of the request
#   r status line (e.g. GET / HTTP/1.1)
#   s status
#   b response length or '-'
#   f referer
#   a agent
access_log_format = '%(h)s %({x-forwarded-for}i)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(l)s (%(M)s)'  # noqa

# Write errorlog to stderr
errorlog = "-"

# Redirect stdout/stderr to errorlog
capture_output = True

try:
    # Load config items from local_guniconf
    # pylint: disable=W0401,W0614
    from local_guniconf import *  # noqa
except ImportError:
    pass
