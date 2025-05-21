from src import *
import dataclasses
import tyro


@dataclasses.dataclass
class AppConfig:
    train: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)
    dataset: NeRFDatasetConfig = dataclasses.field(default_factory=NeRFDatasetConfig)


if __name__ == '__main__':
    cfg = tyro.cli(AppConfig)
    trainer = Trainer(config=cfg.train)
    # trainer.train(
    #     train_dataset=NeRFDataset(config=cfg.dataset, type="train"),
    #     valid_dataset=NeRFDataset(config=cfg.dataset, type="val"),
    #     max_epochs=20,
    # )
    trainer.test(
        test_dataset=NeRFDataset(config=cfg.dataset, type="test"),
    )
