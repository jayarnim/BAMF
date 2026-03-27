from ..config.model import (
    CombCfg,
    ContextCfg,
)


def auto(cfg):
    model = cfg["model"]["name"]
    if "comb" in model:
        return comb(cfg)
    elif "context" in model:
        return context(cfg)
    else:
        raise ValueError(f"invalid model name in .yaml config")


def comb(cfg):
    return CombCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        embedding_dim=cfg["model"]["embedding_dim"],
        dist=cfg["model"]["dist"],
        hyper_approx=cfg["model"]["hyper_approx"],
        hyper_prior=cfg["model"]["hyper_prior"],
        beta=cfg["model"]["beta"],
        comb=cfg["model"]["comb"],
    )


def context(cfg):
    return ContextCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        embedding_dim=cfg["model"]["embedding_dim"],
        dist=cfg["model"]["dist"],
        hyper_approx=cfg["model"]["hyper_approx"],
        hyper_prior=cfg["model"]["hyper_prior"],
        beta=cfg["model"]["beta"],
    )