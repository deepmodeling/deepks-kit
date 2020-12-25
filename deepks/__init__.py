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