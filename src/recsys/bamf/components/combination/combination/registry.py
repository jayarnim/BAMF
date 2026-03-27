from .prod import ElementwiseProduct
from .cat import Concatenation
from .sum import ElementwiseSum
from .mean import ElementwiseMean
from .att import Attention


COMBINATION_REGISTRY = {
    "prod": ElementwiseProduct,
    "cat": Concatenation,
    "sum": ElementwiseSum,
    "mean": ElementwiseMean,
    "att": Attention,
}