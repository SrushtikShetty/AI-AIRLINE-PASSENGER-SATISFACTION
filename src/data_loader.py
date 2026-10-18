import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_PATH

logger = logging.getLogger(__name__)


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the supplied CSV and report a useful error when it is unavailable."""
    source = Path(path) if path is not None else DATA_PATH
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")
    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError(f"Dataset contains no rows: {source}")
    logger.info("Loaded %s rows and %s columns from %s", *frame.shape, source)
    return frame