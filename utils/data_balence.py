import pandas as pd
import numpy as np


def try_balance_data(df, valence=6, arousal=3, stress=3):
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df)
    buff1 = df[df.arousal <= arousal]
    buff2 = df[df.stress <= stress]
    buff3 = df[(df.valence >= valence)]

    balance = pd.concat((df, buff3, buff1, buff2, buff3), axis=0).reset_index(drop=True)
    # balance = balance[balance.stress <= 5.5].reset_index(drop=True)
    return balance


def draw(df, in_dims):
    df.iloc[:, -in_dims:].hist(figsize=(8, 8), bins=50, xlabelsize=8, ylabelsize=8)


def denorm(tensor):
    mean = (0.485, 0.456, 0.406),
    std = (0.229, 0.224, 0.225),
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


# def compute_std():
#     mean = 0.
#     std = 0.
#     for images, _ in loader:
#         batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
#         images = images.view(batch_samples, images.size(1), -1)
#         mean += images.mean(2).sum(0)
#         std += images.std(2).sum(0)
#
#     mean /= len(loader.dataset)
#     std /= len(loader.dataset)