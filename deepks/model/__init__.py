__all__ = [
    "model",
    "reader",
    "train",
    "test",
    "model_enn",
    "preprocess"
]

def __getattr__(name):
    from importlib import import_module
    if name == "CorrNet":
        from .model import CorrNet
        return CorrNet
    if name in __all__:
        return import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
