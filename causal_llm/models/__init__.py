from .loader import load_model, get_available_models

__all__ = ["load_model", "get_available_models"]


def __getattr__(name):
    if name in ("load_model", "get_available_models"):
        from . import loader
        return getattr(loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
