import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import praatio.utilities.constants
import torch
import torch.multiprocessing as mp
from joblib import Parallel, delayed
from loguru import logger
from praatio import textgrid
from torch.utils.data import DataLoader, Dataset
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
)
from transformers.utils.logging import disable_progress_bar

SPLITS = ["test", "train"]

DATASET_PATH = Path("/storage/tpenner/french_pronunciation_dataset")
MFA_OUTPUT_PATH = Path("/storage/tpenner/french_pronunciation_mfa_output")
OUTPUT_PATH = Path("/storage/tpenner/french_pronunciation_embeddings")

# These are all determined by the model
MODEL_NAME = "facebook/w2v-bert-2.0"
SAMPLE_RATE = 16000
FRAME_RATE = 0.020
HIDDEN_DIM = 1024
LAYERS_TO_SAVE = [4, 6, 8, 12]

NUM_GPUS = 4
TEXTGRID_PROCESS_COUNT = 32

MMAP_FLUSH_FREQUENCY = 1000  # Flush mmaps every x segments per GPU process


@dataclass
class Phoneme:
    ipa_char: str
    start: float
    end: float


@dataclass
class Sample:
    sample_id: str
    speaker_id: str
    audio_path: Path
    tg_path: Path

    alignment: list[Phoneme] | None = None
    offset: int | None = None


@dataclass
class Metadata:
    global_idx: int
    sample_id: str
    speaker_id: str
    phoneme: str
    phoneme_idx: int
    num_phonemes: int
    start_sec: float
    end_sec: float


class AudioDataset(Dataset):
    def __init__(
        self, samples: list[Sample], model_name: str, sample_rate: int
    ) -> None:
        self.samples = samples
        self.sample_rate = sample_rate
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[dict, int]:
        item = self.samples[idx]

        decoder = AudioDecoder(
            item.audio_path,
            sample_rate=self.sample_rate,
            num_channels=1,
        )
        audio = decoder.get_all_samples().data.squeeze().numpy()

        inputs = self.feature_extractor(
            audio, sampling_rate=self.sample_rate, return_tensors="pt"
        )

        return {k: v.squeeze(0) for k, v in inputs.items()}, idx


def get_split_samples(split: str) -> list[Sample]:
    samples = []
    audio_dir = DATASET_PATH / split
    tg_dir = MFA_OUTPUT_PATH / split

    for speaker_dir in audio_dir.iterdir():
        if not speaker_dir.is_dir():
            continue

        speaker_id = speaker_dir.name
        tg_speaker_dir = tg_dir / speaker_id

        if not tg_speaker_dir.exists():
            continue

        for audio_path in speaker_dir.glob("*.mp3"):
            tg_path = tg_speaker_dir / f"{audio_path.stem}.TextGrid"
            if not tg_path.exists():
                continue

            samples.append(
                Sample(
                    sample_id=f"{speaker_id}/{audio_path.stem}",
                    speaker_id=speaker_id,
                    audio_path=audio_path,
                    tg_path=tg_path,
                )
            )

    return samples


def parse_textgrid(path: Path) -> list[Phoneme]:
    tg = textgrid.openTextgrid(str(path), includeEmptyIntervals=False)
    tier = tg.getTier("phones")

    phonemes = []

    for entry in tier.entries:
        assert isinstance(entry, praatio.utilities.constants.Interval)

        phonemes.append(
            Phoneme(
                ipa_char=entry.label,
                start=entry.start,
                end=entry.end,
            )
        )

    return phonemes


def load_audio(path: Path) -> np.ndarray:
    decoder = AudioDecoder(
        path,
        sample_rate=SAMPLE_RATE,
        num_channels=1,
    )

    return decoder.get_all_samples().data.squeeze().numpy()


def gpu_worker(
    gpu_id: int,
    all_shards: list[list[Sample]],
    mmap_paths: dict,
    total_phonemes: int,
) -> None:
    torch.set_num_threads(1)

    disable_progress_bar()

    work_items = all_shards[gpu_id]

    device = torch.device(f"cuda:{gpu_id}")

    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()  # pyright: ignore[reportArgumentType]

    mmaps = {
        layer: np.memmap(
            path, dtype=np.float16, mode="r+", shape=(total_phonemes, HIDDEN_DIM)
        )
        for layer, path in mmap_paths.items()
    }

    dataset = AudioDataset(work_items, MODEL_NAME, SAMPLE_RATE)
    loader = DataLoader(
        dataset,
        batch_size=1,  # Keep it 1 to match your original logic
        shuffle=False,  # Order matters for index alignment
        num_workers=6,  # 4 Loaders + 1 Main process = 5 cores used per GPU (20 total used across system)
        prefetch_factor=2,  # Buffer 2 batches per worker
        pin_memory=True,  # Faster transfer to GPU
    )

    for i, (inputs, idx_tensor) in enumerate(
        tqdm(loader, desc=f"GPU {gpu_id}", unit="samples")
    ):
        idx = idx_tensor.item()
        original_item = work_items[idx]

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        num_frames = hidden_states[0].shape[1]

        assert original_item.alignment is not None and original_item.offset is not None
        for phoneme_idx, phoneme in enumerate(original_item.alignment):
            start_frame = max(0, min(int(phoneme.start / FRAME_RATE), num_frames - 1))
            end_frame = max(
                start_frame + 1, min(int(phoneme.end / FRAME_RATE), num_frames)
            )

            global_idx = original_item.offset + phoneme_idx

            for layer in LAYERS_TO_SAVE:
                mean_emb = hidden_states[layer][0, start_frame:end_frame].mean(0)
                mmaps[layer][global_idx] = mean_emb.cpu().numpy().astype(np.float16)

        if i % MMAP_FLUSH_FREQUENCY == 0:
            for mmap in mmaps.values():
                mmap.flush()

    for mmap in mmaps.values():
        mmap.flush()


def process_split(split: str) -> None:
    logger.info(f"Processing split {split}")

    split_output_path = OUTPUT_PATH / split
    split_output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Finding samples and parsing alignments")
    samples = get_split_samples(split)

    alignments = Parallel(n_jobs=TEXTGRID_PROCESS_COUNT, prefer="processes")(
        delayed(parse_textgrid)(sample.tg_path) for sample in samples
    )

    for sample, alignment in zip(samples, alignments):
        sample.alignment = alignment

    # Offsets
    offset = 0
    for sample in samples:
        sample.offset = offset

        assert sample.alignment is not None
        offset += len(sample.alignment)
    total_phonemes = offset

    logger.info(f"Found {len(samples):,} samples with {total_phonemes:,} phonemes")

    logger.info("Building metadata")
    metadata: list[Metadata] = []
    sequences: dict[str, list[str]] = {}

    for sample in samples:
        assert sample.alignment is not None and sample.offset is not None
        seq = [p.ipa_char for p in sample.alignment]
        sequences[sample.sample_id] = seq

        for i, phoneme in enumerate(sample.alignment):
            metadata.append(
                Metadata(
                    global_idx=sample.offset + i,
                    sample_id=sample.sample_id,
                    speaker_id=sample.speaker_id,
                    phoneme=phoneme.ipa_char,
                    phoneme_idx=i,
                    num_phonemes=len(sample.alignment),
                    start_sec=phoneme.start,
                    end_sec=phoneme.end,
                )
            )

    pd.DataFrame(asdict(md) for md in metadata).to_parquet(
        split_output_path / "metadata.parquet"
    )
    with open(split_output_path / "phoneme_sequences.json", "w") as f:
        json.dump(sequences, f, default=asdict)

    logger.info("Creating memmaps")
    mmap_paths: dict[int, Path] = {}
    for layer in LAYERS_TO_SAVE:
        path = split_output_path / f"layer_{layer:02d}.npy"
        mmap_paths[layer] = path
        np.memmap(
            path, dtype=np.float16, mode="w+", shape=(total_phonemes, HIDDEN_DIM)
        ).flush()

    logger.info("Generating GPU shards")
    samples_per_gpu = len(samples) // NUM_GPUS
    shards: list[list[Sample]] = []
    for i in range(NUM_GPUS):
        start = i * samples_per_gpu
        end = start + samples_per_gpu if i < NUM_GPUS - 1 else len(samples)
        shards.append(samples[start:end])

    logger.info(f"Launching {NUM_GPUS} GPU workers")
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        gpu_worker,
        args=(shards, mmap_paths, total_phonemes),
        nprocs=NUM_GPUS,
        join=True,
    )

    logger.success(f"Done with {split} split!")


if __name__ == "__main__":
    OUTPUT_PATH.mkdir(exist_ok=True)
    for split in SPLITS:
        process_split(split)
