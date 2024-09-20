#!/usr/bin/env python
import re
import importlib
from . import bases
from . import plotting
import functools


__PREFIX_DOTS_PAT = re.compile(r'^\.+')


def prefix_dots_count(s):
    m = __PREFIX_DOTS_PAT.match(s)
    return len(m.group(0)) if m else 0


@functools.cache
def import_class(class_name, package=None):
    mod_cls = class_name.rsplit('.', maxsplit=1)
    if len(mod_cls) == 2:
        mod, cls = mod_cls
        mod = mod or '.'
    else:
        mod = '.'
        cls = mod_cls[0]

    mod = importlib.import_module(mod, package)  # 经测试, 当mod是绝对路径时, 不会使用 package 的值.
    return getattr(mod, cls)
