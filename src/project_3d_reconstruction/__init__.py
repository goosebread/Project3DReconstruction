import importlib.metadata
from typing import Final

DISTRIBUTION_NAME: Final = "project-3d-reconstruction"

__version__ = importlib.metadata.version(DISTRIBUTION_NAME)
