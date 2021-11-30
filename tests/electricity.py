import os, sys
import warnings

from pytorch_forecasting.data.encoders import EncoderNormalizer

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

# os.chdir("../../..")

import copy
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, TorchNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

import matplotlib.pyplot as plt


## Load data

from pytorch_forecasting.data.examples import get_stallion_data

# data = get_stallion_data()
data = pd.read_csv('output/hourly_electricity.csv', index_col=0)

# cols = ['time_idx', 'power_usage', 'categorical_id']
cols = ['time_idx', 'power_usage', 'hour', 'day_of_week', 'hours_from_start', 'categorical_id']
data['time_idx'] = (data['hours_from_start'] - data['hours_from_start'].min() + 1).astype(int)
# data['day_of_week'] = data['day_of_week'].astype(str).astype("category")
data = data[cols]

max_prediction_length = 24
max_encoder_length = 168
training_cutoff = data["time_idx"].max() - max_prediction_length
# training_cutoff = 32279

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="power_usage",
    group_ids=["categorical_id"],
    min_encoder_length=max_encoder_length,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["categorical_id"],
    # static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
    # time_varying_known_categoricals=["day_of_week"],
    # variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
    time_varying_known_reals=['hour', 'day_of_week', "hours_from_start"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "power_usage",
        # "log_volume",
        # "industry_volume",
    ],
    target_normalizer=GroupNormalizer(
        groups=["categorical_id"], transformation="softplus"
    ),  # use softplus and normalize by group
    # target_normalizer=EncoderNormalizer(),
    # add_relative_time_idx=True,
    # add_target_scales=True,
    add_encoder_length=False,
)

## Dev

# # convert the dataset to a dataloader
# dataloader = training.to_dataloader(batch_size=4)

# # and load the first batch
# x, y = next(iter(dataloader))
# print("x =", x)
# print("\ny =", y)
# print("\nsizes of x =")
# for key, value in x.items():
#     print(f"\t{key} = {value.size()}")

# print('dfe')

# sys.exit()


# create validation set (predict=True) which means to predict the last max_prediction_length points in time
# for each series
validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

# create dataloaders for model
batch_size = 64  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)


## Train the Temporal Fusion Transformer
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
    learning_rate=0.03,
    hidden_size=160,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=160,
    output_size=3,  # 7 quantiles by default
    loss=QuantileLoss(quantiles=(0.1, 0.5, 0.9)),
    # log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    reduce_on_plateau_patience=4,
    optimizer='adam',
    embedding_sizes={'categorical_id': (369, 160)},
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
best_model_path = 'lightning_logs/default/version_4/checkpoints/epoch=21-step=659.ckpt'
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# calcualte mean absolute error on validation set
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)
(actuals - predictions).abs().mean()

# raw predictions are a dictionary from which all kind of information including quantiles can be extracted
raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)

for idx in range(10):  # plot 10 examples
    best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)

    plt.show()

    print('de')