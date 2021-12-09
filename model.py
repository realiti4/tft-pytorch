import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

from temporal import TemporalFusionTransformer
from utils.tf_wrapper import tf_wrapper
from data_formatters.volatility import VolatilityFormatter
from data_formatters.electricity import ElectricityFormatter
from data_formatters.traffic import TrafficFormatter
from data_formatters.favorita import FavoritaFormatter

from pytorch_forecasting.metrics import QuantileLoss



class tft:
    def __init__(self, wrapper) -> None:
        self.device = 'cpu'
        self.fp16 = False
        self.wrapper = wrapper
        self.fixed_params = self.wrapper.fixed_params

        # Params
        self.lr = 0.01
        self.batch_size = self.wrapper.batch_size
        self.quantiless = [0.1, 0.5, 0.9]

        # Network and Function
        self.net = TemporalFusionTransformer(
            batch_size=self.batch_size,
            wrapper=self.wrapper,
            device=self.device,

            # Dev - The below values are not functional
            learning_rate=0.01,
            hidden_size=160,
            attention_head_size=1,
            dropout=0.3,
            hidden_continuous_size=160,
            output_size=3,  # 7 quantiles by default
            loss=QuantileLoss(quantiles=(0.1, 0.5, 0.9)),
            log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
            reduce_on_plateau_patience=4,
        ).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)
        self.loss_func = QuantileLoss(quantiles=self.quantiless)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # Load something
        self.load = False
        if self.load:
            self._load()

    def fit(
        self,
        epochs,
        train_dataloader,
        val_dataloader,
        limit_batch=None    # How many batces per train
    ):
        
        # Eval first
        self.evaluate(0, val_dataloader)
        
        for e in range(epochs):
            self.train(e, train_dataloader, limit_batch=limit_batch)

            val_loss = self.evaluate(e, val_dataloader)
            # val_loss = 0

            # Checkpoint
            self._save(e, val_loss)
    
    def train(self, epoch, train_dataloader, limit_batch=None):
        """
            1 Epoch training loop
        """
        #reset iterator
        dataiter = iter(train_dataloader)

        losses= 0

        # Dev
        total_size = len(train_dataloader) * self.batch_size if not limit_batch else limit_batch * self.batch_size

        with tqdm(
            total=total_size, desc=f'Training Epoch {epoch}',
            # bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
            # ascii=' ='
            ) as pbar:

            for i, batch in enumerate(dataiter):
                x, y = batch
                        
                #reset gradients
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=False):
                    out = self.net(x.to(self.device))     # [0] > tuple to dict

                    loss = self.loss_func(out, y.to(self.device).squeeze(2))

                #backpropagation
                # loss.backward()
                self.scaler.scale(loss).backward()

                # Gradient Clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.01)
                
                #update the parameters
                # self.optimizer.step()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Metrics
                losses += loss.item()

                pbar.update(self.batch_size)
                
                if i % 10 == 0:
                    pbar.set_postfix(Loss=(losses / 10), Val_Loss=2)
                    losses = 0

                if limit_batch:
                    if limit_batch - 1 == i:
                        break

    def evaluate(self, e, val_dataloader):
        """
            1 Epoch evaluating loop
        """
        print('Evaluating')
        loss = 0
        dataiter = iter(val_dataloader)
        self.net.eval()
        plotted = False
        
        for i, batch in enumerate(dataiter):
            x, y = batch
            if x.size(1) == self.fixed_params['total_time_steps']:
                with torch.no_grad():
                    out = self.net(x.to(self.device))
                loss += self.loss_func(out, y.to(self.device).squeeze(2)).item()

                # Dev
                if not plotted:
                    self.plot_func(x, out)
                    plotted = True
            else:
                print('lan!')

        val_loss = loss / (i + 1)
        print('Val_loss:')
        print(val_loss)
        self.net.train()
        return val_loss

    def plot_func(self, x, out):
        """
            Plotter
        """
        
        num_encoder = self.fixed_params['num_encoder_steps']
        total_time = self.fixed_params['total_time_steps']

        select = [63]
        if self.load:
            select = range(100)

        # If shuffled show n different results for fun
        for i in select:

            x_test = x[i, :, 0].cpu().numpy()
                        
            test = out[i].cpu().numpy()

            plt.plot(np.arange(num_encoder), x_test[:num_encoder])
            plt.plot(np.arange(num_encoder, total_time), x_test[num_encoder:])
            plt.plot(np.arange(num_encoder, total_time), test)
            plt.show()

    def _save(self, epochs, val_loss=0):
        path = 'output/test.pt'
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': val_loss,
            }, path)

    def _load(self):
        path = 'output/electricity.pth'
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Model is loaded from: {path}')



if __name__ == '__main__':
    wrapper = tf_wrapper(
        'output/hourly_electricity.csv',
        'output/electricity',
        ElectricityFormatter(),
        batch_size=256,
        test=False,
    )
    # wrapper = tf_wrapper(
    #     'output/formatted_omi_vol.csv',
    #     'output/volatility/',
    #     VolatilityFormatter(),
    #     batch_size=256,
    # )
    # wrapper = tf_wrapper(
    #     'output/traffic.csv',
    #     'output/traffic/',
    #     TrafficFormatter(),
    #     batch_size=64,
    #     test=False,
    # )
    # wrapper = tf_wrapper(
    #     'temporal3.parquet',
    #     'output/favorita/',
    #     FavoritaFormatter(),
    #     batch_size=128,
    #     test=False,
    # )
    train_dataloader, val_dataloader = wrapper.make_dataset()

    model = tft(wrapper)
    model.fit(
        epochs=100,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        limit_batch=1200,
    )
