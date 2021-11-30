# tft-pytorch
Pytorch Temporal Fusion Transformers

WIP: 

- check `model.py` for now. It is working but messy.
- I haven't included download scripts yet, but it is the same as the original one. You can use that for now. The model needs `hourly_electricity.csv`, `formatted_omi_vol.csv` etc.

The main model is taken from pytorch-forecasting. The problem with libraries like that it can be really hard to debug what is going on if you want to understand. I also couldn't replicate the original results. This repo is great if you want more control over your data processing and understand the implementation.
