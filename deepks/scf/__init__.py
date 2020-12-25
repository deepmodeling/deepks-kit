__all__ = [
    "scf",
    "grad",
    "run",
    "stats",
    "fields",
    "penalty",
]

def __getattr__(name):
    from importlib import import_module
    if name in __all__:
        return import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
