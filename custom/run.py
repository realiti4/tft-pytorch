import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from temporal import TemporalFusionTransformer
from dataset import make_dataset

# Create Dataset
data, labels, params, fixed_params = make_dataset()

dataset = TensorDataset(torch.Tensor(data), torch.Tensor(labels))
train_dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    pin_memory=True,
)

# Initiliaze Model
model = TemporalFusionTransformer()


# Training
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    #reset iterator
    dataiter = iter(train_dataloader)
    
    for i, batch in enumerate(dataiter):
                
        #reset gradients
        optimizer.zero_grad()

        loss, out = model.training_step(batch, i)     # [0] > tuple to dict

        #backpropagation
        loss.backward()
        
        #update the parameters
        optimizer.step()

        print(loss.item())

print('de')