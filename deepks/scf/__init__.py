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


def DSCF(mol, model, xc="HF", **kwargs):
    """A wrap function to create NN SCF object (RDSCF or UDSCF)"""
    from .scf import RDSCF, UDSCF
    if mol.spin == 0:
        return RDSCF(mol, model, xc, **kwargs)
    else:
        return UDSCF(mol, model, xc, **kwargs)

DeepSCF = DSCF