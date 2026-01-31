from pathlib import Path
import pandas as pd

SPLITS = ["train", "test"]
ORIGINAL_DATASET_PATH = Path("/storage/tpenner/cv-corpus-24.0-2025-12-05/fr")
FORMATTED_DATASET_PATH = Path("/storage/tpenner/french_pronunciation_dataset/")

if not ORIGINAL_DATASET_PATH.exists():
    raise Exception("Original dataset path doesn't exist")

FORMATTED_DATASET_PATH.mkdir()
for split in SPLITS:
    (FORMATTED_DATASET_PATH / split).mkdir()
