__author__ = "Yixiao Chen"

try:
    from ._version import version as __version__
except ImportError:
    __version__ = 'unkown'

__all__ = [
    "iterate",
    "model",
    "scf",
    "task",
    # "tools" # collection of command line scripts, should not be imported by user
]

def __getattr__(name):
    from importlib import import_module
    if name in __all__:
        return import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
