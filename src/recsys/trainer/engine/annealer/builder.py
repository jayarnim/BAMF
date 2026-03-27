from .annealer.registry import ANNEALER_REGISTRY


def annealer_builder(cfg):
    cls = ANNEALER_REGISTRY[cfg.name]
    return cls(**cfg.params)