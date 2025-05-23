from enum import Enum
from typing import TypedDict


class SchedulerTypes(str, Enum):
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    STEP = "step"
    COSINE_ANNEALING = "cosine_annealing"
    ONE_CYCLE = "one_cycle"
    EXPONENTIAL = "exponential"
    NONE = "none"


class ActivationTypes(str, Enum):
    GELU = "gelu"
    RELU = "relu"
    SILU = "silu"


class Hyperparameters(TypedDict):
    hidden_sizes: list[int]
    freq_hidden_sizes: list[int] | None
    other_hidden_sizes: list[int] | None
    dropout_rate: float
    learning_rate: float
    activation: ActivationTypes
    lr_scheduler_type: SchedulerTypes
    weight_decay: float
    epochs: int
    patience: int
    batch_size: int


class ModelDict(TypedDict):
    model_name: str
    label: str
    hparams: Hyperparameters
