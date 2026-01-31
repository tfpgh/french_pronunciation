from pathlib import Path

import pandas as pd
from loguru import logger

SPLITS = ["test", "train"]
ORIGINAL_DATASET_PATH = Path("/storage/tpenner/cv-corpus-24.0-2025-12-05/fr")
FORMATTED_DATASET_PATH = Path("/storage/tpenner/french_pronunciation_dataset/")

if not ORIGINAL_DATASET_PATH.exists():
    raise Exception("Original dataset path doesn't exist")

FORMATTED_DATASET_PATH.mkdir()
for split in SPLITS:
    logger.info(f"Processing {split} split")
    (FORMATTED_DATASET_PATH / split).mkdir()

    df = pd.read_table(ORIGINAL_DATASET_PATH / f"{split}.tsv")
    logger.info(df)
