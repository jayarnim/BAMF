from .schema import schema
from ..config.trainer import (
    LossCfg,
    AnnealerCfg,
    OptimizerCfg,
    EngineCfg,
    MonitorCfg,
    TrainerCfg,
)


def loss(cfg):
    return LossCfg(
        name=cfg["loss"]["name"],
        params=cfg["loss"].get("params") or dict(),
    )

def annealer(cfg):
    return AnnealerCfg(
        name=cfg["annealer"]["name"],
        params=cfg["annealer"].get("params") or dict(),
    )

def optimizer(cfg):
    return OptimizerCfg(
        name=cfg["optimizer"]["name"],
        params=cfg["optimizer"].get("params") or dict(),
    )

def engine(cfg):
    return EngineCfg(
        strategy=cfg["strategy"],
        loss=loss(cfg),
        annealer=annealer(cfg),
        optimizer=optimizer(cfg),
    )

def monitor(cfg):
    return MonitorCfg(
        schema=schema(cfg),
        metric=cfg["monitor"]["metric"],
        k=cfg["monitor"]["k"],
        delta=cfg["monitor"]["delta"],
        patience=cfg["monitor"]["patience"],
        warmup=cfg["monitor"]["warmup"],
    )

def trainer(cfg):
    return TrainerCfg(
        num_epochs=cfg["trainer"]["num_epochs"],
        engine=engine(cfg),
        monitor=monitor(cfg),
    )