"""Stateful, bridge-free datasets for native Lance training.

The pre-encoded format is intended for high-throughput training: frozen ViT and
Wan-VAE work is performed once by preprocessing, rather than on every epoch and
every data-parallel rank.  Each file contains one validated
``LanceTrainingBatch`` or a mapping accepted by that dataclass.
"""

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from mindspeed_mm.fsdp.utils.register import data_register
from mindspeed_mm.models.omni.lance.data import LancePreparedSample, pack_preencoded_samples
from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig
from mindspeed_mm.models.omni.lance.sequence import LanceSegment
from mindspeed_mm.models.omni.lance.training_lance import LanceTrainingBatch


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    return value


def _tiny_config(raw: Mapping[str, Any]) -> LanceNativeConfig:
    values = dict(raw)
    for key in ("mrope_section", "latent_patch_size", "vit_fullatt_block_indexes"):
        if key in values:
            values[key] = tuple(values[key])
    return LanceNativeConfig(**values)


class LancePreencodedDataset(Dataset):
    """Map-style dataset over trusted, pre-encoded ``.pt`` packed batches."""

    def __init__(
        self,
        files: Sequence[str],
        root: str | None = None,
        *,
        resample_timesteps: bool = True,
        timestep_sampling: str = "sigmoid_normal",
        timestep_uniform_probability: float = 0.0,
        fixed_noise_seed: int | None = None,
        disable_posterior_sampling: bool = False,
    ) -> None:
        root_path = Path(root).expanduser().resolve() if root else None
        resolved = []
        for value in files:
            path = Path(value).expanduser()
            if not path.is_absolute() and root_path is not None:
                path = root_path / path
            path = path.resolve()
            if path.is_dir():
                resolved.extend(sorted(path.glob("*.pt")))
            else:
                resolved.append(path)
        missing = [str(path) for path in resolved if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing native Lance batches: {}".format(", ".join(missing)))
        if not resolved:
            raise ValueError("native Lance pre-encoded dataset is empty")
        self.files = tuple(resolved)
        self.resample_timesteps = bool(resample_timesteps)
        self.timestep_sampling = str(timestep_sampling)
        self.timestep_uniform_probability = float(timestep_uniform_probability)
        if self.timestep_sampling not in ("sigmoid_normal", "uniform", "mixture"):
            raise ValueError(
                "timestep_sampling must be 'sigmoid_normal', 'uniform', or 'mixture'"
            )
        if not 0.0 <= self.timestep_uniform_probability <= 1.0:
            raise ValueError("timestep_uniform_probability must be in [0, 1]")
        if (
            self.timestep_sampling != "mixture"
            and self.timestep_uniform_probability != 0.0
        ):
            raise ValueError(
                "timestep_uniform_probability is only used by mixture sampling"
            )
        self.fixed_noise_seed = (
            None if fixed_noise_seed is None else int(fixed_noise_seed)
        )
        self.disable_posterior_sampling = bool(disable_posterior_sampling)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        # Training manifests and batch files are operator-controlled artifacts.
        value = torch.load(self.files[index], map_location="cpu", weights_only=False)
        if isinstance(value, LanceTrainingBatch):
            batch = value
        elif isinstance(value, Mapping):
            batch = LanceTrainingBatch(**dict(value))
        else:
            raise TypeError("{} does not contain a LanceTrainingBatch".format(self.files[index]))
        batch.resample_timesteps = self.resample_timesteps
        batch.timestep_sampling = self.timestep_sampling
        batch.timestep_uniform_probability = self.timestep_uniform_probability
        if self.disable_posterior_sampling:
            batch.latent_log_variance = None
        if self.fixed_noise_seed is not None and batch.clean_latents is not None:
            # Generate in FP32 on CPU for deterministic behavior across hosts,
            # then cast to the stored latent dtype.  The index offset gives
            # every packed batch a stable but distinct noise tensor.
            generator = torch.Generator(device="cpu").manual_seed(
                self.fixed_noise_seed + int(index)
            )
            batch.noise = torch.randn(
                batch.clean_latents.shape,
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).to(dtype=batch.clean_latents.dtype)
        return {"lance_batch": batch, "batch_path": str(self.files[index])}


class LanceSyntheticDataset(Dataset):
    """Deterministic joint CE/MSE data used only for runtime smoke tests."""

    def __init__(self, config: LanceNativeConfig, length: int = 32, seed: int = 2025) -> None:
        self.config = config
        self.length = int(length)
        self.seed = int(seed)
        if self.length <= 0:
            raise ValueError("synthetic dataset length must be positive")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        generator = torch.Generator().manual_seed(self.seed + int(index))
        sample = LancePreparedSample(
            sample_id="synthetic-{}".format(index),
            segments=(
                LanceSegment(4, "causal", "text", "understanding"),
                LanceSegment(4, "noise", "vae", "generation"),
            ),
            token_ids=torch.tensor([1, 2, 3, 4, 0, 0, 0, 0], dtype=torch.long),
            text_indexes=torch.tensor([0, 1, 2, 3], dtype=torch.long),
            position_ids=torch.arange(8, dtype=torch.long).repeat(3, 1),
            vae_indexes=torch.tensor([4, 5, 6, 7], dtype=torch.long),
            clean_latents=torch.randn(4, self.config.patch_latent_dim, generator=generator),
            latent_position_ids=torch.tensor([0, 1, 2, 3], dtype=torch.long),
            timesteps=torch.full((4,), 0.5),
            noise=torch.randn(4, self.config.patch_latent_dim, generator=generator),
            ce_indexes=torch.tensor([0, 1, 2], dtype=torch.long),
            ce_labels=torch.tensor([2, 3, 4], dtype=torch.long),
            ce_weights=torch.ones(3),
            mse_indexes=torch.tensor([4, 5, 6, 7], dtype=torch.long),
        )
        use_ascend = hasattr(torch, "npu") and torch.npu.is_available()
        packed = pack_preencoded_samples(
            (sample,),
            self.config,
            max_tokens=8,
            attention_backend="ascend" if use_ascend else "reference",
        )
        return {"lance_batch": packed.batch, "batch_path": sample.sample_id}


def build_lance_dataset(basic_param, preprocess_param, dataset_param=None):
    basic = _plain(basic_param)
    preprocess = _plain(preprocess_param)
    dataset_param = _plain(dataset_param or {})
    data_format = str(dataset_param.get("format", basic.get("format", "preencoded")))
    if data_format == "synthetic":
        raw_config = _plain(dataset_param.get("native_config", preprocess.get("native_config", {})))
        config = _tiny_config(raw_config) if raw_config else LanceNativeConfig.for_variant(
            str(dataset_param.get("variant", preprocess.get("variant", "video")))
        )
        return LanceSyntheticDataset(
            config,
            length=int(dataset_param.get("length", basic.get("length", 32))),
            seed=int(dataset_param.get("seed", basic.get("seed", 2025))),
        )
    if data_format != "preencoded":
        raise ValueError("native_lance format must be 'preencoded' or 'synthetic'")
    files = dataset_param.get("files") or basic.get("files") or basic.get("dataset")
    if isinstance(files, str):
        files = [files]
    if not files:
        raise ValueError("native_lance preencoded format requires basic_parameters.files")
    return LancePreencodedDataset(
        files,
        root=dataset_param.get("dataset_dir", basic.get("dataset_dir")),
        resample_timesteps=bool(dataset_param.get("resample_timesteps", True)),
        timestep_sampling=str(
            dataset_param.get("timestep_sampling", "sigmoid_normal")
        ),
        timestep_uniform_probability=float(
            dataset_param.get("timestep_uniform_probability", 0.0)
        ),
        fixed_noise_seed=dataset_param.get("fixed_noise_seed"),
        disable_posterior_sampling=bool(
            dataset_param.get("disable_posterior_sampling", False)
        ),
    )


data_register.register("native_lance")(build_lance_dataset)
