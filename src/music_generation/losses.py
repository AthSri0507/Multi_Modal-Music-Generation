"""GAN losses and regularizers for music generation training."""

from __future__ import annotations

import torch


def discriminator_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discriminator: E[max(0, 1 - D(x))] + E[max(0, 1 + D(G(z)))]."""
    loss_real = torch.relu(1.0 - real_logits).mean()
    loss_fake = torch.relu(1.0 + fake_logits).mean()
    return loss_real + loss_fake


def generator_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Hinge loss for generator: -E[D(G(z))]."""
    return -fake_logits.mean()


def r1_regularization(real_inputs: torch.Tensor, real_logits: torch.Tensor) -> torch.Tensor:
    """R1 gradient penalty on real inputs.

    Returns mean(||grad D(x)||^2) over batch.
    """
    grad_outputs = torch.ones_like(real_logits)
    grads = torch.autograd.grad(
        outputs=real_logits,
        inputs=real_inputs,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.reshape(grads.shape[0], -1)
    return (grads.pow(2).sum(dim=1)).mean()
