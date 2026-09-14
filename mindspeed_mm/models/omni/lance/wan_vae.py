"""Frozen Wan2.2 VAE encoder used by native Lance data preparation."""

from pathlib import Path
from typing import Iterable, List, Union

import torch
from einops import rearrange

from .wan_vae2_2 import Wan2_2_VAE


class LanceWanVAE:
    """Small MindSpeed-MM wrapper around the released 48-channel Wan2.2 VAE."""

    latent_channels = 48
    spatial_downsample = 16
    temporal_downsample = 4

    def __init__(
        self,
        checkpoint: Union[str, Path],
        *,
        device: Union[str, torch.device] = "npu",
        dtype: torch.dtype = torch.bfloat16,
        sample_posterior: bool = True,
        encoder_only: bool = True,
    ) -> None:
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("Wan2.2 VAE checkpoint not found: {}".format(checkpoint))
        self.device = torch.device(device)
        self.dtype = dtype
        self.sample_posterior = bool(sample_posterior)
        self.vae = Wan2_2_VAE(
            vae_pth=str(checkpoint),
            device=self.device,
            dtype=dtype,
        )
        if encoder_only and hasattr(self.vae.model, "decoder"):
            del self.vae.model.decoder

    @torch.no_grad()
    def encode(self, samples: Iterable[torch.Tensor]) -> List[torch.Tensor]:
        """Encode CTHW tensors and return THWC normalized latent tensors."""

        values = []
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            for sample in samples:
                mean, log_variance = self.vae.encode(
                    sample.to(device=self.device, dtype=self.dtype).unsqueeze(0)
                )
                if self.sample_posterior:
                    standard_deviation = torch.exp(0.5 * log_variance)
                    latent = mean + standard_deviation * torch.randn_like(standard_deviation)
                else:
                    latent = mean
                values.append(
                    rearrange(latent, "b c t h w -> b t h w c")[0].to(
                        device="cpu", dtype=torch.bfloat16
                    )
                )
        return values

    @torch.no_grad()
    def encode_distribution(
        self, samples: Iterable[torch.Tensor]
    ) -> List[tuple[torch.Tensor, torch.Tensor]]:
        """Return normalized THWC posterior mean and log variance."""

        values = []
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            for sample in samples:
                mean, log_variance = self.vae.encode(
                    sample.to(device=self.device, dtype=self.dtype).unsqueeze(0)
                )
                values.append((
                    rearrange(mean, "b c t h w -> b t h w c")[0].to(
                        device="cpu", dtype=torch.bfloat16
                    ),
                    rearrange(log_variance, "b c t h w -> b t h w c")[0].to(
                        device="cpu", dtype=torch.bfloat16
                    ),
                ))
        return values

    def close(self) -> None:
        del self.vae
        if self.device.type == "npu" and hasattr(torch, "npu"):
            torch.npu.empty_cache()
