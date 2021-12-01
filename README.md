# tft-pytorch
Pytorch Temporal Fusion Transformers

WIP: 

- Check `model.py` for now. It is working but messy.
- Check `data_formatters/script_download_data.py` to download datasets. See the last 2 lines and change the name to download which dataset you want to download, then run it.


The main model is taken from pytorch-forecasting. The problem with libraries like that it can be really hard to debug what is going on if you want to understand. I also couldn't replicate the original results. This repo is great if you want more control over your data processing and understand the implementation.
