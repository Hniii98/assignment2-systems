import importlib.metadata
from .model import BasicsTransformerLM
from .optimizer import AdamW
from .nn_utils import cross_entropy
try:
    __version__ = importlib.metadata.version("cs336-basics")
except importlib.metadata.PackageNotFoundError:
    pass
