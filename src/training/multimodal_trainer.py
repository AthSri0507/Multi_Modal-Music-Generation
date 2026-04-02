"""Phase-1 trainer for multimodal emotion classification."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.data.multimodal_dataloader import (
    HDF5MultimodalDataset,
    build_multimodal_dataloader,
)
from src.models.multimodal_classifier import MultimodalEmotionClassifier
from src.training.losses import build_loss
from src.training.metrics import compute_classification_metrics
from src.training.metrics import compute_confusion_matrix_payload
from src.config.config import PHASE_1_CANONICAL_EMOTIONS, PHASE_1_4CLASS_EMOTIONS

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


@dataclass
class TrainerConfig:
    text_h5: str
    audio_h5: str
    output_dir: str = "artifacts/m3_phase1"
    batch_size: int = 64
    epochs: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    patience: int = 4
    min_delta: float = 1e-4
    loss_name: str = "cross_entropy"
    use_weighted_sampler: bool = False
    use_class_weights: bool = True
    focal_gamma: float = 2.0
    seed: int = 42
    device: str = "auto"
    use_tensorboard: bool = True
    log_every_steps: int = 50
    class_weight_clip_min: float = 0.5
    class_weight_clip_max: float = 3.0
    num_classes: int | None = None  # If None, use PHASE_1_CANONICAL_EMOTIONS


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _pick_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        remain = max(1, total_steps - warmup_steps)
        progress = float(step - warmup_steps) / float(remain)
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


def _class_weights_from_dataset(
    dataset: HDF5MultimodalDataset,
    num_classes: int,
    device: torch.device,
    clip_min: float,
    clip_max: float,
) -> torch.Tensor:
    counts = dataset.get_class_counts() or {}
    arr = np.ones(num_classes, dtype=np.float32)
    for cls_id, count in counts.items():
        arr[int(cls_id)] = 1.0 / max(1, int(count))
    arr = arr / arr.sum() * num_classes
    arr = np.clip(arr, clip_min, clip_max)
    arr = arr / arr.mean()
    return torch.tensor(arr, dtype=torch.float32, device=device)


def _epoch_pass(
    model: MultimodalEmotionClassifier,
    loader: DataLoader,
    device: torch.device,
    criterion,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    log_every_steps: int = 50,
    epoch_num: int = 0,
    split_name: str = "train",
) -> Dict[str, float | list]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []
    y_score: List[List[float]] = []
    t0 = time.time()
    num_batches = max(1, len(loader))

    for step, (text_x, audio_x, y) in enumerate(loader, start=1):
        text_x = text_x.to(device)
        audio_x = audio_x.to(device)
        y = y.to(device)

        logits = model(text_x, audio_x)
        loss = criterion(logits, y)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += float(loss.item()) * y.size(0)

        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        y_score.extend(probs.detach().cpu().tolist())

        if log_every_steps > 0 and (step % log_every_steps == 0 or step == num_batches):
            elapsed = time.time() - t0
            per_step = elapsed / max(step, 1)
            eta_s = per_step * max(0, num_batches - step)
            logger.info(
                "Epoch %d [%s] step %d/%d (%.1f%%) loss=%.4f eta=%dm%02ds",
                epoch_num,
                split_name,
                step,
                num_batches,
                100.0 * step / num_batches,
                float(loss.item()),
                int(eta_s // 60),
                int(eta_s % 60),
            )

    avg_loss = total_loss / max(1, len(loader.dataset))
    return {
        "loss": float(avg_loss),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
    }


def run_phase1_training(cfg: TrainerConfig) -> Dict[str, object]:
    """Run full Phase-1 training, validation, and test evaluation."""
    _seed_everything(cfg.seed)
    device = _pick_device(cfg.device)

    out_dir = Path(cfg.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    for p in (out_dir, ckpt_dir, log_dir):
        p.mkdir(parents=True, exist_ok=True)

    # Use the canonical Phase-1 set based on num_classes parameter
    if cfg.num_classes == 4:
        canonical_labels = set(PHASE_1_4CLASS_EMOTIONS)
    else:
        # Default to 6-class
        canonical_labels = set(PHASE_1_CANONICAL_EMOTIONS)

    train_ds = HDF5MultimodalDataset(
        cfg.text_h5,
        cfg.audio_h5,
        split="train",
        allowed_labels=canonical_labels if canonical_labels else None,
    )
    label_encoder = train_ds.label_encoder
    val_ds = HDF5MultimodalDataset(
        cfg.text_h5,
        cfg.audio_h5,
        split="val",
        label_encoder=label_encoder,
    )
    test_ds = HDF5MultimodalDataset(
        cfg.text_h5,
        cfg.audio_h5,
        split="test",
        label_encoder=label_encoder,
    )

    train_loader = build_multimodal_dataloader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=not cfg.use_weighted_sampler,
        weighted_sampling=cfg.use_weighted_sampler,
    )
    val_loader = build_multimodal_dataloader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = build_multimodal_dataloader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    num_classes = len(label_encoder.label_to_id)
    model = MultimodalEmotionClassifier(num_classes=num_classes).to(device)

    class_weights = None
    if cfg.use_class_weights and not cfg.use_weighted_sampler:
        class_weights = _class_weights_from_dataset(
            train_ds,
            num_classes,
            device,
            cfg.class_weight_clip_min,
            cfg.class_weight_clip_max,
        )

    logger.info(
        "Train rows after label cleanup: %d | classes=%d | weighted_sampler=%s | class_weights=%s | lr=%.6f",
        len(train_ds),
        num_classes,
        cfg.use_weighted_sampler,
        bool(class_weights is not None),
        cfg.learning_rate,
    )
    criterion = build_loss(
        name=cfg.loss_name,
        class_weights=class_weights,
        focal_gamma=cfg.focal_gamma,
    )

    optimizer = Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = _build_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    writer = None
    if cfg.use_tensorboard and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))

    best_val_f1 = -1.0
    best_epoch = -1
    patience_count = 0
    history: List[Dict[str, object]] = []

    id_to_label = label_encoder.id_to_label

    for epoch in range(1, cfg.epochs + 1):
        logger.info("Starting epoch %d/%d", epoch, cfg.epochs)
        train_pass = _epoch_pass(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            scheduler,
            log_every_steps=cfg.log_every_steps,
            epoch_num=epoch,
            split_name="train",
        )
        val_pass = _epoch_pass(
            model,
            val_loader,
            device,
            criterion,
            optimizer=None,
            scheduler=None,
            log_every_steps=max(1, math.ceil(len(val_loader) / 4)),
            epoch_num=epoch,
            split_name="val",
        )

        train_m = compute_classification_metrics(
            train_pass["y_true"], train_pass["y_pred"], id_to_label, y_score=train_pass["y_score"]
        )
        val_m = compute_classification_metrics(
            val_pass["y_true"], val_pass["y_pred"], id_to_label, y_score=val_pass["y_score"]
        )

        row = {
            "epoch": epoch,
            "train_loss": train_pass["loss"],
            "val_loss": val_pass["loss"],
            "train_accuracy": train_m["accuracy"],
            "val_accuracy": val_m["accuracy"],
            "train_f1_macro": train_m["f1_macro"],
            "val_f1_macro": val_m["f1_macro"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        logger.info(
            "Epoch %d summary: train_loss=%.4f val_loss=%.4f train_acc=%.4f val_acc=%.4f train_f1=%.4f val_f1=%.4f",
            epoch,
            row["train_loss"],
            row["val_loss"],
            row["train_accuracy"],
            row["val_accuracy"],
            row["train_f1_macro"],
            row["val_f1_macro"],
        )

        if writer is not None:
            writer.add_scalar("loss/train", row["train_loss"], epoch)
            writer.add_scalar("loss/val", row["val_loss"], epoch)
            writer.add_scalar("acc/train", row["train_accuracy"], epoch)
            writer.add_scalar("acc/val", row["val_accuracy"], epoch)
            writer.add_scalar("f1/train_macro", row["train_f1_macro"], epoch)
            writer.add_scalar("f1/val_macro", row["val_f1_macro"], epoch)
            writer.add_scalar("lr", row["lr"], epoch)

        improved = row["val_f1_macro"] > (best_val_f1 + cfg.min_delta)
        if improved:
            best_val_f1 = float(row["val_f1_macro"])
            best_epoch = epoch
            patience_count = 0
            ckpt_path = ckpt_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "label_to_id": label_encoder.label_to_id,
                    "val_f1_macro": best_val_f1,
                    "config": asdict(cfg),
                },
                ckpt_path,
            )
        else:
            patience_count += 1

        if patience_count >= cfg.patience:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    if writer is not None:
        writer.flush()
        writer.close()

    # Evaluate best checkpoint on test split.
    best_ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_pass = _epoch_pass(
        model,
        test_loader,
        device,
        criterion,
        optimizer=None,
        scheduler=None,
        log_every_steps=max(1, math.ceil(len(test_loader) / 4)),
        epoch_num=best_epoch if best_epoch > 0 else 0,
        split_name="test",
    )
    test_m = compute_classification_metrics(
        test_pass["y_true"], test_pass["y_pred"], id_to_label, y_score=test_pass["y_score"]
    )

    summary = {
        "config": asdict(cfg),
        "device": str(device),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_f1_macro": best_val_f1,
        "best_val_accuracy": max((h["val_accuracy"] for h in history), default=0.0),
        "history": history,
        "test_metrics": test_m,
        "test_confusion": compute_confusion_matrix_payload(
            test_pass["y_true"],
            test_pass["y_pred"],
            id_to_label,
        ),
        "artifacts": {
            "best_checkpoint": str((ckpt_dir / "best_model.pt").as_posix()),
            "history_json": str((log_dir / "history.json").as_posix()),
            "summary_json": str((out_dir / "summary.json").as_posix()),
            "test_confusion_json": str((out_dir / "test_confusion.json").as_posix()),
        },
    }

    with open(log_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "test_confusion.json", "w", encoding="utf-8") as f:
        json.dump(summary["test_confusion"], f, indent=2)

    return summary
