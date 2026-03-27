from ..config.model import CombBimodalCfg, CombUserCfg, CombItemCfg, ContextBimodalCfg, ContextUserCfg, ContextItemCfg


def model(cfg):
    cls = cfg["model"]["name"]

    if cls=="comb_bimodal":
        return comb_bimodal(cfg)
    elif cls=="comb_user":
        return comb_user(cfg)
    elif cls=="comb_item":
        return comb_item(cfg)
    elif cls=="context_bimodal":
        return context_bimodal(cfg)
    elif cls=="context_user":
        return context_user(cfg)
    elif cls=="context_item":
        return context_item(cfg)
    else:
        raise ValueError("invalid model name in .yaml config")


def comb_bimodal(cfg):
    return CombBimodalCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"],
    )


def comb_user(cfg):
    return CombUserCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"],
    )


def comb_item(cfg):
    return CombItemCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"],
    )


def context_bimodal(cfg):
    return ContextBimodalCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"],
    )


def context_user(cfg):
    return ContextUserCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"],
    )


def context_item(cfg):
    return ContextItemCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"],
    )