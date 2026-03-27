from dataclasses import dataclass
from typing import Literal
from .schema import SchemaCfg


@dataclass
class LossCfg:
    name: Literal["bce", "bpr", "climf"]
    params: dict


@dataclass
class AnnealerCfg:
    name: Literal["linear"]
    params: dict


@dataclass
class OptimizerCfg:
    name: Literal["adagrad", "adam", "adamw"]
    params: dict


@dataclass
class EngineCfg:
    strategy: Literal["pointwise", "pairwise", "listwise"]
    loss: LossCfg
    annealer: AnnealerCfg
    optimizer: OptimizerCfg


@dataclass
class MonitorCfg:
    metric: Literal["hit_ratio", "precision", "recall", "map", "ndcg"]
    k: int
    delta: float
    patience: int
    warmup: int
    schema: SchemaCfg


@dataclass
class TrainerCfg:
    num_epochs: int
    engine: EngineCfg
    monitor: MonitorCfg