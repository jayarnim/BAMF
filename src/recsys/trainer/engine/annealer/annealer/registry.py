from .linear import LinearAnnealer


ANNEALER_REGISTRY = {
    "linear": LinearAnnealer,
}