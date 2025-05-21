from src import Trainer, TrainerConfig, NeRFDataset, NeRFDatasetConfig
import dataclasses
import tyro


@dataclasses.dataclass
class AppConfig:
    train: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)
    dataset: NeRFDatasetConfig = dataclasses.field(default_factory=NeRFDatasetConfig)


if __name__ == '__main__':
    cfg = tyro.cli(AppConfig)
    trainer = Trainer(config=cfg.train)

    if cfg.train.mode == 'train':
        trainer.train(
            train_dataset=NeRFDataset(config=cfg.dataset, dataset_type="train"),
            valid_dataset=NeRFDataset(config=cfg.dataset, dataset_type="val"),
            max_epochs=20,
        )
    elif cfg.train.mode == 'test':
        trainer.test(
            test_dataset=NeRFDataset(config=cfg.dataset, dataset_type="test"),
        )
    else:
        raise ValueError(f"Unknown mode {cfg.train.mode}")
