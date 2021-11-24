import os, sys
import warnings

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

# os.chdir("../../..")

import copy
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters


## Load data
data = pd.read_csv('output/formatted_omi_vol.csv', index_col=0)

# Create dataset and dataloaders
max_prediction_length = 5
max_encoder_length = 252

# Split train and valid
columns = [
    'Symbol', 'date', 'log_vol', 'open_to_close', 'days_from_start', 'day_of_week', 'day_of_month', 
    'week_of_year', 'month', 'Region'
    ]
training_cutoff = 2016

# Dev fixes
data = data.reset_index(drop=True)
data[['day_of_week', 'day_of_month', 'week_of_year', 'month']] = data[['day_of_week', 'day_of_month', 'week_of_year', 'month']].astype(str)

training = TimeSeriesDataSet(
    data[lambda x: x.year < training_cutoff],
    time_idx="days_from_start",        # Check here original was data, I'm gonna use days_from_start
    target="log_vol",
    group_ids=['Symbol'],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=['Region'],
    # static_reals=['days_from_start'],   # observed?
    time_varying_known_categoricals=['day_of_week', 'day_of_month', 'week_of_year', 'month'],   # Str?
    # variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
    time_varying_known_reals=['days_from_start'],  # known input?
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "log_vol",
        "open_to_close",
    ],
    target_normalizer=GroupNormalizer(
        groups=['Symbol'], transformation="softplus"
    ),  # use softplus and normalize by group
    # add_relative_time_idx=True,
    # add_target_scales=True,
    add_encoder_length=False,
    allow_missing_timesteps=True, # we need to delete this    
)

# Validation
validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

# create dataloaders for model
batch_size = 64  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)


## Train model
pl.seed_everything(42)

# configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=50,
    gpus=1,
    weights_summary="top",
    gradient_clip_val=0.01,
    limit_train_batches=30,  # coment in for training, running valiation every 30 batches
    # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    callbacks=[lr_logger],
    logger=logger,
)


tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.01,
    hidden_size=160,
    attention_head_size=1,
    dropout=0.3,
    hidden_continuous_size=160,
    output_size=3,  # 7 quantiles by default
    loss=QuantileLoss(quantiles=(0.1, 0.5, 0.9)),
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    reduce_on_plateau_patience=4,
    embedding_sizes={'day_of_week': (7, 160), 'day_of_month': (31, 160), 'week_of_year': (53, 160), 'month': (12, 160), 'Region': (4, 160)}
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# fit network
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)


## Eval

# load the best model according to the validation loss
# (given that we use early stopping, this is not necessarily the last epoch)
best_model_path = trainer.checkpoint_callback.best_model_path   # ''
best_model_path = 'lightning_logs/default/version_1/checkpoints/epoch=4-step=149.ckpt'
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# raw predictions are a dictionary from which all kind of information including quantiles can be extracted
raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)

for idx in range(10):  # plot 10 examples
    best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)

    plt.show()

print('de')