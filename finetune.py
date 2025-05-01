import argparse
import pytorch_lightning as pl
import torch

from wrappers import StyleTTS2TrainingWrapper
from meldataset import build_dataloader
from config import TrainConfig
from utils import get_data_path_list

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--num_workers', type=int, required=False, help='Number of workers for dataloader', default=2)
    parser.add_argument('--gpus', type=int, help='Amount of gpus', default=0)

    args = parser.parse_args()

    config = TrainConfig.from_json(args.config)

    model = StyleTTS2TrainingWrapper(config_path=args.config)

    train_list, val_list = get_data_path_list(
        train_path=config.data_params.train_data,
        val_path=config.data_params.val_data
    )

    train_loader = build_dataloader(
        train_list,
        config.data_params.root_path,
        config.model_params.vocab,
        batch_size=config.batch_size,
        num_workers=args.num_workers
    )

    val_loader = build_dataloader(
        val_list,
        config.data_params.root_path,
        config.model_params.vocab,
        validation=True,
        batch_size=config.batch_size,
        num_workers=args.num_workers
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="styletts2-{epoch:02d}",
        save_top_k=3,
        monitor="train/total_loss",
        mode="min",
        save_last=True,
        every_n_epochs=config.save_freq,
    )

    logger = TensorBoardLogger("logs", name="styletts2")

    steps_per_epoch = len(train_loader)
    model.set_steps_per_epoch(steps_per_epoch)

    if args.gpus != 0:
        trainer = pl.Trainer(
            devices=args.gpus,
            accelerator="gpu",
            max_epochs=config.epochs,
            log_every_n_steps=config.log_interval,
            logger=logger,
            callbacks=[checkpoint_callback],
        )
    else:
        trainer = pl.Trainer(
            accelerator="cpu",
            max_epochs=config.epochs,
            log_every_n_steps=config.log_interval,
            logger=logger,
            callbacks=[checkpoint_callback],
        )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
