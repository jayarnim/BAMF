from dataclasses import dataclass


@dataclass
class CombCfg:
    num_users: int
    num_items: int
    embedding_dim: int
    dist: str
    hyper_approx: float
    hyper_prior: float
    beta: float
    comb: float


@dataclass
class ContextCfg:
    num_users: int
    num_items: int
    embedding_dim: int
    dist: str
    hyper_approx: float
    hyper_prior: float
    beta: float