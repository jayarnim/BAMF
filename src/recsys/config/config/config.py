from dataclasses import dataclass
from typing import Literal, Union
from .pipeline import PipelineCfg
from .trainer import TrainerCfg
from .evaluator import EvaluatorCfg
from .schema import SchemaCfg
from .model import CombBimodalCfg, CombUserCfg, CombItemCfg, ContextBimodalCfg, ContextUserCfg, ContextItemCfg


@dataclass
class Config:
    model: Union[CombBimodalCfg, CombUserCfg, CombItemCfg, ContextBimodalCfg, ContextUserCfg, ContextItemCfg]
    schema: SchemaCfg
    pipeline: PipelineCfg
    trainer: TrainerCfg
    evaluator: EvaluatorCfg
    strategy: Literal["pointwise", "pairwise", "listwise"]
    model_cls: Literal["comb_bimodal", "comb_user", "comb_item", "context_bimodal", "context_user", "context_item"]
    comb: Literal["att", "cat", "mean", "prod", "sum"]
    dataset: str
    seed: int