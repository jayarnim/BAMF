from dataclasses import dataclass


@dataclass
class CombBimodalCfg:
    num_users: int
    num_items: int
    params: dict


@dataclass
class CombUserCfg:
    num_users: int
    num_items: int
    params: dict


@dataclass
class CombItemCfg:
    num_users: int
    num_items: int
    params: dict


@dataclass
class ContextBimodalCfg:
    num_users: int
    num_items: int
    params: dict


@dataclass
class ContextUserCfg:
    num_users: int
    num_items: int
    params: dict


@dataclass
class ContextItemCfg:
    num_users: int
    num_items: int
    params: dict