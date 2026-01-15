import os, wandb
import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
import torchmetrics

from torchmetrics.classification import accuracy

from lightning.pytorch.loggers import WandbLogger

import hydra
from hydra.utils import instantiate
from hydra.utils import get_original_cwd, to_absolute_path

from omegaconf import DictConfig, OmegaConf






@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):

    data_module = instantiate(cfg.data)
    data_module.prepare_data()
    data_module.setup()

    os.environ.setdefault("WANDB_API_KEY", "05a459841786c9f37d82bc178a0d72c923c25156")
    os.environ["WANDB_START_METHOD"] = "thread"
    os.environ.setdefault("WANDB_INIT_TIMEOUT", "60")
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)

    logger = WandbLogger(
        project="Klasyfikator_Chmur",
        name="run-multimodal-v1",   
        log_model="all"
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )

    MODEL_CKPT_PATH = 'model/'
    MODEL_CKPT = 'model-{epoch:02d}-{val_loss:.2f}'

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=MODEL_CKPT_PATH,
        filename=MODEL_CKPT,
        save_top_k=3,
        mode='min')

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    api = wandb.Api()
    artifact_path = 'rafi_mys-politechnika-warszawska/wandb-multimodal/model-xtm5fags:v3'
    artifact = api.artifact(artifact_path, type='model')
    artifact_dir = artifact.download()

    #print(f"Model pobrany do: {artifact_dir}")

    # 3. Znalezienie pliku .ckpt
    checkpoint_file = [f for f in os.listdir(artifact_dir) if f.endswith('.ckpt')][0]
    checkpoint_path = os.path.join(artifact_dir, checkpoint_file)

    model = instantiate(cfg.network)

    #classifier = TransferResNet152LitModel.load_from_checkpoint(checkpoint_path)

    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="gpu",
        devices=1,
        logger=logger,
        log_every_n_steps = 10,
        callbacks=[early_stopping, checkpoint_callback, lr_monitor],
        fast_dev_run=False,
        enable_progress_bar=cfg.trainer.enable_progress_bar
    )

    trainer.fit(model=model, datamodule=data_module)

if __name__ == "__main__":
    main()
