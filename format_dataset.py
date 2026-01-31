from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

SPLITS = ["test", "train"]
ORIGINAL_DATASET_PATH = Path("/storage/tpenner/cv-corpus-24.0-2025-12-05/fr")
FORMATTED_DATASET_PATH = Path("/storage/tpenner/french_pronunciation_dataset/")

if not ORIGINAL_DATASET_PATH.exists():
    raise Exception("Original dataset path doesn't exist")

FORMATTED_DATASET_PATH.mkdir(exist_ok=True)
for split in SPLITS:
    logger.info(f"Processing {split} split")

    (FORMATTED_DATASET_PATH / split).mkdir(exist_ok=True)

    split_path = ORIGINAL_DATASET_PATH / f"{split}.tsv"
    df = pd.read_table(split_path)
    for segment in tqdm(df.itertuples(), total=len(df), unit="segments"):
        client_id = str(segment.client_id)  # pyright: ignore[reportAttributeAccessIssue]
        file_name = str(segment.path)  # pyright: ignore[reportAttributeAccessIssue]
        sentence_text = str(segment.sentence)  # pyright: ignore[reportAttributeAccessIssue]

        speaker_dir = FORMATTED_DATASET_PATH / split / client_id
        speaker_dir.mkdir(exist_ok=True)

        original_audio_path = ORIGINAL_DATASET_PATH / "clips" / file_name
        new_audio_path = speaker_dir / file_name.split("_")[-1]
        new_text_path = new_audio_path.with_suffix(".lab")

        if new_audio_path.exists() and new_text_path.exists():
            continue
        elif new_audio_path.exists() and not new_text_path.exists():
            logger.warning(f"Missing text file for {new_audio_path}")
        elif not new_audio_path.exists() and new_text_path.exists():
            logger.warning(f"Missing audio file for {new_text_path}")

        if not new_audio_path.exists():
            # Hard link from original audio to path in dataset
            new_audio_path.hardlink_to(original_audio_path)

        if not new_text_path.exists():
            with open(new_text_path, "w") as f:
                # Common voice recomends removing smart apostrophes
                f.write(sentence_text.replace("‘", "'").replace("’", "'").strip())
