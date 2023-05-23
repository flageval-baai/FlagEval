"""
信号
====

由 Flageval 定义的信号，使用方法参见 `官方文档 <http://flask.pocoo.org/docs/1.0/signals/>`_ 。
"""
from blinker import Namespace


flageval_signals = Namespace()


config_ready = flageval_signals.signal("flageval-config-ready")
"""
Flageval 已经加载完配置，:class:`flask.Flask` 实例作为发送者（sender）。
"""

ext_ready = flageval_signals.signal("flageval-ext-ready")
"""
Flageval 完成扩展初始化。

:class:`flask.Flask` 实例作为发送者（sender）。
"""


ready = flageval_signals.signal("flageval-ready")
"""
Flageval 准备完毕开始加载蓝图，:class:`flask.Flask` 实例作为发送者（sender）。
"""

blueprint_loaded = flageval_signals.signal("flageval-blueprint-loaded")
"""
Flageval 已经加载完 Blueprint。
"""


worker_forked = flageval_signals.signal("flageval-worker-forked")
"""Web worker or any else worker(Celery/faust) forked.
"""
