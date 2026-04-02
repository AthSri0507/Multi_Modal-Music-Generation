"""Milestone 4 trainer for emotion-conditioned music GAN."""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset

from src.config.config import (
    M4_GAN_AUX_CE_WEIGHT,
    M4_GAN_BATCH_SIZE,
    M4_GAN_BETA1,
    M4_GAN_BETA2,
    M4_GAN_D_LR,
    M4_GAN_DISCRIMINATOR_HIDDEN_DIMS,
    M4_GAN_EMOTION_EMBED_DIM_D,
    M4_GAN_EMOTION_EMBED_DIM_G,
    M4_GAN_G_LR,
    M4_GAN_GENERATOR_HIDDEN_DIMS,
    M4_GAN_NOISE_DIM,
    M4_GAN_NUM_EMOTIONS,
    M4_GAN_R1_WEIGHT,
    M4_GAN_USE_SPECTRAL_NORM,
)
from src.music_generation.dataset import EmotionConditionedMusicDataset, MusicLabelEncoder
from src.music_generation.losses import (
    discriminator_hinge_loss,
    generator_hinge_loss,
    r1_regularization,
)
from src.music_generation.models import MusicDiscriminator, MusicGenerator
from src.music_generation.models import build_generator

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class MusicGanTrainerConfig:
    train_npz: str
    val_npz: str | None = None
    output_dir: str = "artifacts/m4_gan"
    device: str = "auto"
    seed: int = 42

    epochs: int = 100
    batch_size: int = M4_GAN_BATCH_SIZE
    val_ratio: float = 0.15

    num_emotions: int = M4_GAN_NUM_EMOTIONS
    noise_dim: int = M4_GAN_NOISE_DIM
    g_emotion_embed_dim: int = M4_GAN_EMOTION_EMBED_DIM_G
    d_emotion_embed_dim: int = M4_GAN_EMOTION_EMBED_DIM_D
    g_hidden_dims: Tuple[int, int, int] = M4_GAN_GENERATOR_HIDDEN_DIMS
    d_hidden_dims: Tuple[int, int, int] = M4_GAN_DISCRIMINATOR_HIDDEN_DIMS
    use_spectral_norm: bool = M4_GAN_USE_SPECTRAL_NORM
    generator_variant: str = "mlp"  # mlp | residual | attention
    seq_len: int = 120
    feature_dim: int = 4

    g_lr: float = M4_GAN_G_LR
    d_lr: float = M4_GAN_D_LR
    beta1: float = M4_GAN_BETA1
    beta2: float = M4_GAN_BETA2

    n_critic: int = 2
    r1_weight: float = M4_GAN_R1_WEIGHT
    r1_interval: int = 16
    aux_ce_weight: float = M4_GAN_AUX_CE_WEIGHT
    grad_clip_norm: float = 5.0

    sample_every_epochs: int = 10
    sample_per_emotion: int = 16
    log_every_steps: int = 50

    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 1e-3


class _ArrayMusicDataset(Dataset):
    """In-memory dataset used for train/val split without separate files."""

    def __init__(self, vectors: np.ndarray, labels: np.ndarray, encoder: MusicLabelEncoder) -> None:
        self.vectors = vectors.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.label_encoder = encoder

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    def __getitem__(self, idx: int):
        x = torch.tensor(self.vectors[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_datasets(cfg: MusicGanTrainerConfig) -> tuple[Dataset, Dataset, MusicLabelEncoder]:
    train_ds_full = EmotionConditionedMusicDataset(cfg.train_npz)
    encoder = train_ds_full.label_encoder

    if cfg.val_npz:
        val_ds = EmotionConditionedMusicDataset(cfg.val_npz, label_encoder=encoder)
        return train_ds_full, val_ds, encoder

    labels = train_ds_full.labels
    idx = np.arange(len(train_ds_full))

    if len(np.unique(labels)) < 2:
        raise ValueError("Need at least 2 emotion classes for GAN training split")

    train_idx, val_idx = train_test_split(
        idx,
        test_size=cfg.val_ratio,
        random_state=cfg.seed,
        stratify=labels,
    )

    train_ds = Subset(train_ds_full, train_idx.tolist())
    val_ds = Subset(train_ds_full, val_idx.tolist())
    return train_ds, val_ds, encoder


def _build_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _emotion_consistency_and_diversity(
    generator: MusicGenerator,
    discriminator: MusicDiscriminator,
    device: torch.device,
    num_emotions: int,
    noise_dim: int,
    samples_per_emotion: int,
) -> Dict[str, float]:
    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        labels = []
        vectors = []
        for emotion_id in range(num_emotions):
            y = torch.full((samples_per_emotion,), emotion_id, device=device, dtype=torch.long)
            z = torch.randn(samples_per_emotion, noise_dim, device=device)
            fake = generator(z, y)
            vectors.append(fake)
            labels.append(y)

        fake_all = torch.cat(vectors, dim=0)
        y_all = torch.cat(labels, dim=0)

        out = discriminator(fake_all, y_all)
        pred = torch.argmax(out.emotion_logits, dim=1)
        consistency = (pred == y_all).float().mean().item()

        # Diversity proxy: mean std across vector dimensions.
        diversity_std = fake_all.std(dim=0).mean().item()

        # Pairwise cosine distance proxy for mode collapse (lower means collapse).
        normalized = F.normalize(fake_all, p=2, dim=1)
        sim = normalized @ normalized.t()
        off_diag = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)]
        mean_pairwise_cosine_distance = (1.0 - off_diag).mean().item()

    return {
        "emotion_consistency": float(consistency),
        "diversity_std": float(diversity_std),
        "mean_pairwise_cosine_distance": float(mean_pairwise_cosine_distance),
    }


def _save_generated_samples(
    generator: MusicGenerator,
    device: torch.device,
    num_emotions: int,
    noise_dim: int,
    samples_per_emotion: int,
    out_path: Path,
) -> None:
    generator.eval()
    all_vectors: List[np.ndarray] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for emotion_id in range(num_emotions):
            y = torch.full((samples_per_emotion,), emotion_id, device=device, dtype=torch.long)
            z = torch.randn(samples_per_emotion, noise_dim, device=device)
            fake = generator(z, y)
            all_vectors.append(fake.cpu().numpy().astype(np.float32))
            all_labels.extend([emotion_id] * samples_per_emotion)

    vectors = np.concatenate(all_vectors, axis=0)
    labels = np.asarray(all_labels, dtype=np.int64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, vectors=vectors, labels=labels)


def run_music_gan_training(cfg: MusicGanTrainerConfig) -> Dict[str, object]:
    """Run full GAN training loop for Milestone 4."""
    _seed_everything(cfg.seed)
    device = _pick_device(cfg.device)

    out_dir = Path(cfg.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    samples_dir = out_dir / "samples"
    logs_dir = out_dir / "logs"
    for p in (out_dir, ckpt_dir, samples_dir, logs_dir):
        p.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, label_encoder = _build_datasets(cfg)
    train_loader = _build_loader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = _build_loader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # Infer vector dim from a sample batch.
    x0, _ = next(iter(train_loader))
    vector_dim = int(x0.shape[1])

    num_emotions_from_data = len(label_encoder.label_to_id)
    if cfg.num_emotions != num_emotions_from_data:
        logger.warning(
            "num_emotions from config (%d) differs from dataset (%d). Using dataset value.",
            cfg.num_emotions,
            num_emotions_from_data,
        )
    num_emotions = num_emotions_from_data

    G = build_generator(
        variant=cfg.generator_variant,
        noise_dim=cfg.noise_dim,
        num_emotions=num_emotions,
        emotion_embedding_dim=cfg.g_emotion_embed_dim,
        output_dim=vector_dim,
        hidden_dims=cfg.g_hidden_dims,
        seq_len=cfg.seq_len,
        feature_dim=cfg.feature_dim,
    ).to(device)

    D = MusicDiscriminator(
        input_dim=vector_dim,
        num_emotions=num_emotions,
        emotion_embedding_dim=cfg.d_emotion_embed_dim,
        hidden_dims=cfg.d_hidden_dims,
        spectral_norm=cfg.use_spectral_norm,
    ).to(device)

    g_opt = Adam(G.parameters(), lr=cfg.g_lr, betas=(cfg.beta1, cfg.beta2))
    d_opt = Adam(D.parameters(), lr=cfg.d_lr, betas=(cfg.beta1, cfg.beta2))

    history: List[Dict[str, float | int]] = []
    best_score = -1e9
    best_epoch = -1
    patience_counter = 0

    logger.info(
        "Starting M4 GAN training | train_rows=%d val_rows=%d vector_dim=%d classes=%d device=%s",
        len(train_ds),
        len(val_ds),
        vector_dim,
        num_emotions,
        device,
    )

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        G.train()
        D.train()

        d_adv_sum = 0.0
        d_aux_sum = 0.0
        d_r1_sum = 0.0
        g_adv_sum = 0.0
        g_aux_sum = 0.0
        d_updates = 0
        g_updates = 0

        for step, (real_x, real_y) in enumerate(train_loader, start=1):
            global_step += 1
            real_x = real_x.to(device)
            real_y = real_y.to(device)
            bsz = real_x.shape[0]

            # Discriminator update
            y_fake = torch.randint(low=0, high=num_emotions, size=(bsz,), device=device)
            z = torch.randn(bsz, cfg.noise_dim, device=device)
            fake_x = G(z, y_fake)

            real_out = D(real_x, real_y)
            fake_out = D(fake_x.detach(), y_fake)

            d_adv = discriminator_hinge_loss(real_out.adv_logits, fake_out.adv_logits)
            d_aux = F.cross_entropy(real_out.emotion_logits, real_y)
            d_loss = d_adv + cfg.aux_ce_weight * d_aux

            d_r1 = torch.tensor(0.0, device=device)
            if cfg.r1_weight > 0 and (global_step % max(1, cfg.r1_interval) == 0):
                real_x_gp = real_x.detach().clone().requires_grad_(True)
                real_out_gp = D(real_x_gp, real_y)
                d_r1 = r1_regularization(real_x_gp, real_out_gp.adv_logits)
                d_loss = d_loss + cfg.r1_weight * d_r1

            d_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(D.parameters(), cfg.grad_clip_norm)
            d_opt.step()

            d_adv_sum += float(d_adv.detach().item())
            d_aux_sum += float(d_aux.detach().item())
            d_r1_sum += float(d_r1.detach().item())
            d_updates += 1

            # Generator update every n_critic steps.
            if global_step % max(1, cfg.n_critic) == 0:
                y_gen = torch.randint(low=0, high=num_emotions, size=(bsz,), device=device)
                z = torch.randn(bsz, cfg.noise_dim, device=device)
                gen_x = G(z, y_gen)
                gen_out = D(gen_x, y_gen)

                g_adv = generator_hinge_loss(gen_out.adv_logits)
                g_aux = F.cross_entropy(gen_out.emotion_logits, y_gen)
                g_loss = g_adv + cfg.aux_ce_weight * g_aux

                g_opt.zero_grad(set_to_none=True)
                g_loss.backward()
                if cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(G.parameters(), cfg.grad_clip_norm)
                g_opt.step()

                g_adv_sum += float(g_adv.detach().item())
                g_aux_sum += float(g_aux.detach().item())
                g_updates += 1

            if cfg.log_every_steps > 0 and (step % cfg.log_every_steps == 0 or step == len(train_loader)):
                logger.info(
                    "Epoch %d step %d/%d | d_adv=%.4f d_aux=%.4f g_adv=%.4f g_aux=%.4f",
                    epoch,
                    step,
                    len(train_loader),
                    d_adv_sum / max(1, d_updates),
                    d_aux_sum / max(1, d_updates),
                    g_adv_sum / max(1, g_updates),
                    g_aux_sum / max(1, g_updates),
                )

        # Validation proxies.
        val_consistency = _emotion_consistency_and_diversity(
            generator=G,
            discriminator=D,
            device=device,
            num_emotions=num_emotions,
            noise_dim=cfg.noise_dim,
            samples_per_emotion=max(4, cfg.sample_per_emotion // 2),
        )

        # Lightweight real-label validation through discriminator auxiliary head.
        D.eval()
        with torch.no_grad():
            aux_correct = 0
            aux_total = 0
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                out = D(x_val, y_val)
                pred = torch.argmax(out.emotion_logits, dim=1)
                aux_correct += int((pred == y_val).sum().item())
                aux_total += int(y_val.numel())
            val_aux_acc = aux_correct / max(1, aux_total)

        # Composite score to select best checkpoint.
        val_score = (
            1.0 * val_consistency["emotion_consistency"]
            + 0.25 * val_consistency["mean_pairwise_cosine_distance"]
            + 0.25 * val_aux_acc
        )

        row: Dict[str, float | int] = {
            "epoch": epoch,
            "d_adv": d_adv_sum / max(1, d_updates),
            "d_aux": d_aux_sum / max(1, d_updates),
            "d_r1": d_r1_sum / max(1, d_updates),
            "g_adv": g_adv_sum / max(1, g_updates),
            "g_aux": g_aux_sum / max(1, g_updates),
            "val_emotion_consistency": val_consistency["emotion_consistency"],
            "val_diversity_std": val_consistency["diversity_std"],
            "val_pairwise_cosine_distance": val_consistency["mean_pairwise_cosine_distance"],
            "val_aux_accuracy": val_aux_acc,
            "val_score": val_score,
            "epoch_seconds": time.time() - t0,
        }
        history.append(row)

        logger.info(
            "Epoch %d summary | val_score=%.4f consistency=%.4f diversity=%.4f aux_acc=%.4f",
            epoch,
            val_score,
            row["val_emotion_consistency"],
            row["val_pairwise_cosine_distance"],
            row["val_aux_accuracy"],
        )

        # Save samples periodically.
        if cfg.sample_every_epochs > 0 and (epoch % cfg.sample_every_epochs == 0 or epoch == 1):
            sample_path = samples_dir / f"epoch_{epoch:03d}_samples.npz"
            _save_generated_samples(
                generator=G,
                device=device,
                num_emotions=num_emotions,
                noise_dim=cfg.noise_dim,
                samples_per_emotion=cfg.sample_per_emotion,
                out_path=sample_path,
            )

        # Always save last checkpoint.
        last_ckpt = ckpt_dir / "last_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "generator_state_dict": G.state_dict(),
                "discriminator_state_dict": D.state_dict(),
                "g_optimizer_state_dict": g_opt.state_dict(),
                "d_optimizer_state_dict": d_opt.state_dict(),
                "label_to_id": label_encoder.label_to_id,
                "vector_dim": vector_dim,
                "config": asdict(cfg),
                "history": history,
            },
            last_ckpt,
        )

        improved = val_score > (best_score + cfg.early_stopping_min_delta)
        if improved:
            best_score = val_score
            best_epoch = epoch
            patience_counter = 0
            best_ckpt = ckpt_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "generator_state_dict": G.state_dict(),
                    "discriminator_state_dict": D.state_dict(),
                    "g_optimizer_state_dict": g_opt.state_dict(),
                    "d_optimizer_state_dict": d_opt.state_dict(),
                    "label_to_id": label_encoder.label_to_id,
                    "vector_dim": vector_dim,
                    "best_val_score": best_score,
                    "config": asdict(cfg),
                    "history": history,
                },
                best_ckpt,
            )
        else:
            patience_counter += 1

        if patience_counter >= cfg.early_stopping_patience:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    summary = {
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "final_metrics": history[-1] if history else {},
        "history": history,
        "label_to_id": label_encoder.label_to_id,
        "artifacts": {
            "output_dir": str(out_dir),
            "best_checkpoint": str((ckpt_dir / "best_model.pt").as_posix()),
            "last_checkpoint": str((ckpt_dir / "last_model.pt").as_posix()),
            "samples_dir": str(samples_dir.as_posix()),
        },
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Training complete. Best epoch=%d best_val_score=%.4f", best_epoch, best_score)
    return summary
