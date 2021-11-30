import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

from temporal import TemporalFusionTransformer
from dataset import make_dataset

from pytorch_forecasting.metrics import QuantileLoss



class tft:
    def __init__(self) -> None:
        self.device = 'cpu'
        self.fp16 = False

        # Params
        self.lr = 0.01
        self.quantiless = [0.1, 0.5, 0.9]

        # Network and Function
        self.net = TemporalFusionTransformer(
            None,
            device=self.device,

            learning_rate=0.01,
            hidden_size=160,
            attention_head_size=1,
            dropout=0.3,
            hidden_continuous_size=160,
            output_size=3,  # 7 quantiles by default
            loss=QuantileLoss(quantiles=(0.1, 0.5, 0.9)),
            log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
            reduce_on_plateau_patience=4,
        )

        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)
        self.loss_func = QuantileLoss(quantiles=self.quantiless)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        self._create_datasets()

    def fit(self, epochs):
        
        for e in range(epochs):
            self.train(e)
    
    def train(self, epoch):
        """
            1 Epoch training loop
        """
        #reset iterator
        dataiter = iter(self.train_dataloader)

        losses= 0

        # Dev
        batch_size = 64

        with tqdm(
            total=len(self.train_dataloader) * batch_size, desc=f'Training Epoch {epoch}',
            # bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
            ) as pbar:

            for i in range(len(self.train_dataloader)):
                time.sleep(0.05)

                pbar.update(batch_size)
                pbar.set_postfix(Loss=1, Val_Loss=2)
                # print('de')
        
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

            if i % 100 == 0:
                print(losses / 100)
                losses = 0

    def evaluate(self):
        """
            1 Epoch evaluating loop
        """
        print('Evaluating')
        loss = 0
        dataiter = iter(self.val_dataloader)
        self.net.eval()
        plotted = False
        
        for i, batch in enumerate(dataiter):
            x, y = batch
            if x.size(1) == 257:
                with torch.no_grad():
                    out = self.net(x.to(self.device))
                loss += self.loss_func(out, y.to(self.device).squeeze(2)).item()

                # Dev
                if not plotted:
                    x_test = x[63, :, 0].cpu().numpy()
                    
                    test = out[63].cpu().numpy()

                    plt.plot(np.arange(252), x_test[:252])
                    plt.plot(np.arange(252, 257), x_test[252:])
                    plt.plot(np.arange(252, 257), test)
                    plt.show()
                    plotted = True
            else:
                print('lan!')

        print('Val_loss:')
        print(loss / (i + 1))
        self.net.train()

    def _create_datasets(self):
        # Create Dataset
        data, labels, valid_data, valid_labels, params, fixed_params = make_dataset()

        # Train
        dataset = TensorDataset(torch.Tensor(data), torch.Tensor(labels))
        self.train_dataloader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        # Valid
        val_dataset = TensorDataset(torch.Tensor(valid_data), torch.Tensor(valid_labels))
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=64,
            pin_memory=True,
            drop_last=True,
        )

if __name__ == '__main__':
    model = tft()
    model.fit(
        epochs=100,
    )        