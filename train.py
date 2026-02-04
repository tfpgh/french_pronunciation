import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

AUDIO_EMBEDDINGS_PATH = Path("/storage/tpenner/french_pronunciation_embeddings")
CHECKPOINT_PATH = Path("/storage/tpenner/french_pronunciation_checkpoints")

AUDIO_EMBEDDING_LAYER = 6  # Internal layer to use from the audio embedding model
AUDIO_EMBEDDING_DIM = 1024
SHARED_EMBEDDING_DIM = 256

TEXT_ENCODER_DIM = 512
TEXT_ENCODER_LAYERS = 4
TEXT_ENCODER_HEADS = 8
TEXT_ENCODER_DROPOUT = 0.1

BATCH_SIZE = 2048
POOL_FACTOR = 50
LEARNING_RATE = 3e-4
NUM_EPOCHS = 50
INIT_TEMPERATURE = 0.07


class PhonemeDataset(Dataset):
    def __init__(self, split: str, phoneme_to_idx: dict[str, int]) -> None:
        split_path = AUDIO_EMBEDDINGS_PATH / split

        logger.info(f"Loading metadata and sequences for {split} split")
        metadata_df = pd.read_parquet(split_path / "metadata.parquet")
        with open(split_path / "phoneme_sequences.json") as f:
            raw_sequences = json.load(f)

        self.lengths = (
            metadata_df["sample_id"].apply(lambda x: len(raw_sequences[x])).values
        )

        # Get rid of iloc lookups
        logger.info(f"Convert pandas metadata to numpy for {split} split")
        self.sample_ids = metadata_df["sample_id"].values
        self.positions = metadata_df["phoneme_idx"].values
        self.global_idxs = metadata_df["global_idx"].values

        # Preprocess tokens to integers
        logger.info(f"Preprocess tokens to integers for {split} split")
        self.preprocessed_sequences = {
            k: [phoneme_to_idx[p] for p in seq] for k, seq in raw_sequences.items()
        }

        logger.info(
            f"Loading layer {AUDIO_EMBEDDING_LAYER} embeddings for {split} split"
        )
        num_phonemes = len(metadata_df)
        self.audio_embeddings = np.memmap(
            split_path / f"layer_{AUDIO_EMBEDDING_LAYER:02d}.npy",
            dtype=np.float16,
            mode="r",
            shape=(num_phonemes, AUDIO_EMBEDDING_DIM),
        )

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        sample_id = self.sample_ids[idx]
        sequence_ids = self.preprocessed_sequences[sample_id]

        return {
            "sequence_ids": torch.tensor(sequence_ids, dtype=torch.long),
            "position": self.positions[idx],
            "audio_emb": torch.from_numpy(
                self.audio_embeddings[self.global_idxs[idx]].copy()
            ),
        }


class DistributedSampler:
    def __init__(
        self,
        dataset: PhonemeDataset,  # Pyright doesn't like typing as Dataset
    ):
        self.dataset = dataset
        self.epoch = 0
        self.rank = dist.get_rank()
        self.num_replicas = dist.get_world_size()

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        total_size = len(self.dataset) - (len(self.dataset) % self.num_replicas)
        indices = indices[:total_size]

        indices = indices[self.rank :: self.num_replicas]

        pool_size = BATCH_SIZE * POOL_FACTOR
        lengths = self.dataset.lengths

        for i in range(0, len(indices), pool_size):
            pool = indices[i : i + pool_size]
            pool.sort(key=lambda idx: lengths[idx], reverse=True)

            for j in range(0, len(pool) - BATCH_SIZE + 1, BATCH_SIZE):
                yield pool[j : j + BATCH_SIZE]

    def __len__(self):
        return (len(self.dataset) // self.num_replicas) // BATCH_SIZE

    def set_epoch(self, epoch: int):
        self.epoch = epoch


def collate_fn(batch):
    sequences = [item["sequence_ids"] for item in batch]
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0)

    mask = padded_seqs != 0

    positions = torch.tensor([item["position"] for item in batch], dtype=torch.long)
    audio_embs = torch.stack([item["audio_emb"] for item in batch])

    return {
        "sequence_ids": padded_seqs,
        "mask": mask,
        "positions": positions,
        "audio_emb": audio_embs,
    }


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, TEXT_ENCODER_DIM, padding_idx=0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TEXT_ENCODER_DIM,
            nhead=TEXT_ENCODER_HEADS,
            dim_feedforward=TEXT_ENCODER_DIM * 4,
            dropout=TEXT_ENCODER_DROPOUT,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=TEXT_ENCODER_LAYERS
        )
        self.proj = nn.Linear(TEXT_ENCODER_DIM, SHARED_EMBEDDING_DIM)

    def forward(
        self, sequence_ids: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        x = self.embedding(sequence_ids)
        x = self.transformer(x, src_key_padding_mask=~mask)

        batch_idx = torch.arange(x.size(0), device=x.device)
        x = x[batch_idx, positions]

        return self.proj(x)


class AudioProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(AUDIO_EMBEDDING_DIM, AUDIO_EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(AUDIO_EMBEDDING_DIM, SHARED_EMBEDDING_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PronunciationModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size)
        self.audio_proj = AudioProjection()

        self.log_temp = nn.Parameter(torch.tensor([np.log(INIT_TEMPERATURE)]))

    def forward(
        self,
        sequence_ids: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
        audio_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_emb = self.text_encoder(sequence_ids, mask, positions)
        audio_emb = self.audio_proj(audio_emb)
        temp = self.log_temp.exp().clamp(min=0.01, max=1.0)
        return text_emb, audio_emb, temp


def contrastive_loss(
    text_emb: torch.Tensor, audio_emb: torch.Tensor, temp: torch.Tensor
) -> torch.Tensor:
    # Normalize local
    text_emb = F.normalize(text_emb, dim=-1)
    audio_emb = F.normalize(audio_emb, dim=-1)

    # Gather from all GPUs
    all_text = torch.cat(dist_nn.all_gather(text_emb), dim=0)
    all_audio = torch.cat(dist_nn.all_gather(audio_emb), dim=0)

    logits_t2a = text_emb @ all_audio.T / temp
    logits_a2t = audio_emb @ all_text.T / temp

    batch_size = text_emb.size(0)
    rank = dist.get_rank()
    start_idx = rank * batch_size
    labels = torch.arange(start_idx, start_idx + batch_size, device=text_emb.device)

    loss_t2a = F.cross_entropy(logits_t2a, labels)
    loss_a2t = F.cross_entropy(logits_a2t, labels)

    return (loss_t2a + loss_a2t) / 2


def build_vocab() -> dict[str, int]:
    with open(AUDIO_EMBEDDINGS_PATH / "train" / "phoneme_sequences.json") as f:
        sequences = json.load(f)

    phonemes = set()
    for seq in sequences.values():
        phonemes.update(seq)

    # 0 = pad, then phonemes
    vocab = {"<pad>": 0}
    for i, p in enumerate(sorted(phonemes), start=1):
        vocab[p] = i

    return vocab


def average_metrics(val, world_size):
    if isinstance(val, float):
        val = torch.tensor(val, device=torch.cuda.current_device())
    dist.all_reduce(val, op=dist.ReduceOp.SUM)
    return val.item() / world_size


def train() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])

    logger.info(f"Starting process with rank {local_rank}")

    logger.remove()
    if local_rank == 0:
        logger.add(sys.stderr, level="INFO")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)

    CHECKPOINT_PATH.mkdir(exist_ok=True)

    logger.info("Building vocabulary")
    phoneme_to_idx = build_vocab()
    logger.info(f"Vocab size: {len(phoneme_to_idx)}")

    with open(CHECKPOINT_PATH / "phoneme_vocab.json", "w") as f:
        json.dump(phoneme_to_idx, f)

    logger.info("Loading datasets")
    train_dataset = PhonemeDataset("train", phoneme_to_idx)
    test_dataset = PhonemeDataset("test", phoneme_to_idx)

    dist.barrier()

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=3,
        persistent_workers=True,
        pin_memory=True,
    )

    test_sampler = DistributedSampler(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=3,
        persistent_workers=True,
        pin_memory=True,
    )

    logger.info(f"Train: {len(train_dataset):,} phonemes")
    logger.info(f"Test: {len(test_dataset):,} phonemes")

    model = PronunciationModel(len(phoneme_to_idx)).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameter count: {num_params:,}")

    # For AMP
    scaler = GradScaler()

    best_test_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0

        temp = None

        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch+1}", disable=local_rank != 0
        ):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad()

            with autocast(device_type="cuda", dtype=torch.float16):
                text_emb, audio_emb, temp = model(**batch)
                loss = contrastive_loss(text_emb, audio_emb, temp)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        world_size = dist.get_world_size()
        avg_train_loss = average_metrics(train_loss, world_size)

        scheduler.step()

        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(
                test_loader, desc=f"Epoch {epoch+1} [test]", disable=local_rank != 0
            ):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with autocast(device_type="cuda", dtype=torch.float16):
                    text_emb, audio_emb, temp = model(**batch)
                    loss = contrastive_loss(text_emb, audio_emb, temp)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        avg_test_loss = average_metrics(test_loss, world_size)

        if temp is None:
            temp_str = "None"
        else:
            temp_str = f"{temp.item():.4f}"

        logger.info(
            f"Epoch {epoch+1}: train={avg_train_loss:.4f} test={avg_test_loss:.4f} temp={temp_str}"
        )

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            if local_rank == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "train_loss": avg_train_loss,
                        "test_loss": avg_test_loss,
                        "vocab": phoneme_to_idx,
                    },
                    CHECKPOINT_PATH / "best.pt",
                )
                logger.success(f"Saved best model (test_loss={test_loss:.4f})")

        dist.barrier()  # Don't start until save is done


if __name__ == "__main__":
    train()
