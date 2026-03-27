from .user import IDXEmbeddingWithHistory as user
from .item import IDXEmbeddingWithHistory as item
from .bimodal import IDXEmbeddingWithHistory as bimodal


EMBEDDING_REGISTRY = {
    "user": user,
    "item": item,
    "bimodal": bimodal,
}