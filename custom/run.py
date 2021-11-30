import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from temporal import TemporalFusionTransformer
from dataset import make_dataset
from utils.tf_wrapper import tf_wrapper
from data.volatility import VolatilityFormatter
from data_formatters.electricity import ElectricityFormatter

from pytorch_forecasting.metrics import QuantileLoss

# # Create Dataset
# data, labels, valid_data, valid_labels, params, fixed_params = make_dataset()

# # Train
# dataset = TensorDataset(torch.Tensor(data), torch.Tensor(labels))
# train_dataloader = DataLoader(
#     dataset,
#     batch_size=64,
#     shuffle=True,
#     pin_memory=True,
#     drop_last=True,
# )
# # Valid
# val_dataset = TensorDataset(torch.Tensor(valid_data), torch.Tensor(valid_labels))
# val_dataloader = DataLoader(
#     val_dataset,
#     batch_size=64,
#     pin_memory=True,
#     drop_last=True,
# )

# Initiliaze Model
device = 'cuda'
# wrapper = tf_wrapper(
#     'output/formatted_omi_vol.csv',
#     'output/volatility/',
#     VolatilityFormatter(),
# )
wrapper = tf_wrapper(
    'output/hourly_electricity.csv',
    'output/electricity',
    ElectricityFormatter(),
)
train_dataloader, val_dataloader = wrapper.make_dataset()


model = TemporalFusionTransformer(
    # params,
    device=device,
    wrapper=wrapper,

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

def eval(model, dataloader):
    print('Evaluating')
    loss = 0
    dataiter = iter(dataloader)
    model.eval()
    plotted = False
    
    for i, batch in enumerate(dataiter):
        x, y = batch
        if x.size(1) == 257:
            with torch.no_grad():
                out = model(x.to(device))
            loss += loss_func(out, y.to(device).squeeze(2)).item()

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
    model.train()

def train(model, dataloader):
    #reset iterator
    dataiter = iter(train_dataloader)

    losses= 0
    
    for i, batch in enumerate(dataiter):
        x, y = batch
                
        #reset gradients
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=False):
            out = model(x.to(device))     # [0] > tuple to dict

            loss = loss_func(out, y.to(device).squeeze(2))

        #backpropagation
        # loss.backward()
        scaler.scale(loss).backward()

        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        
        #update the parameters
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        losses += loss.item()

        if i % 100 == 0:
            print(losses / 100)
            losses = 0


# Training
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_func = QuantileLoss(quantiles=(0.1, 0.5, 0.9))
scaler = torch.cuda.amp.GradScaler(enabled=False)

model.to(device)
# eval(model, val_dataloader)

for epoch in range(50):
    # Train
    train(model, train_dataloader)

    # Eval
    # eval(model, val_dataloader)

    print(f'Epoch {epoch} has ended!')
    # break

print('de')