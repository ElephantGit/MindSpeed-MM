"""Entry point for fully native Lance pretraining on MindSpeed-MM FSDP2."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torchdata.stateful_dataloader import StatefulDataLoader

from mindspeed_mm.fsdp.data import build_mm_dataset
from mindspeed_mm.fsdp.data.dataloader.sampler import BaseRandomBatchSampler
from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.params.argument import Arguments, parse_args
from mindspeed_mm.fsdp.tasks.lance.train_engine import LanceTrainEngine
from mindspeed_mm.fsdp.train.trainer import Trainer
from mindspeed_mm.fsdp.utils.device import get_device_type


def _single_packed_batch(values):
    if len(values) != 1:
        raise ValueError("native Lance expects one pre-packed sequence per micro batch")
    return values[0]


class LanceTrainer(Trainer):
    def get_dataloader(self):
        if self.args.training.micro_batch_size != 1:
            raise ValueError("native Lance packed training requires micro_batch_size=1")
        dataset = build_mm_dataset(self.args.data.dataset_param)
        config = self.args.data.dataloader_param.to_dict()
        workers = int(config.get("num_workers", 0))
        group = get_parallel_state().get_dp_group()
        sampler = BaseRandomBatchSampler(
            dataset,
            batch_size=1,
            num_replicas=group.size(),
            rank=group.rank(),
            shuffle=bool(config.get("shuffle", True)),
            seed=self.args.training.seed,
            drop_last=bool(config.get("drop_last", True)),
            data_sharding=bool(config.get("data_sharding", False)),
        )
        return StatefulDataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=_single_packed_batch,
            num_workers=workers,
            pin_memory=bool(config.get("pin_memory", True)),
            pin_memory_device=get_device_type(),
            prefetch_factor=config.get("prefetch_factor", 2) if workers else None,
            persistent_workers=bool(config.get("persistent_workers", workers > 0)) if workers else False,
        )

    def build_train_engine(self, *args, **kwargs):
        return LanceTrainEngine(*args, **kwargs)


if __name__ == "__main__":
    trainer = LanceTrainer(parse_args(Arguments))
    trainer.train()
