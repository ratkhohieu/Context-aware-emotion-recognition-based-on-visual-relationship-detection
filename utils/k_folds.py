import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold

from .metrics import test_scikit_ap


def split_data(df_data1, df_data2, number_fold):
    if df_data2 is not None:
        df_data1 = pd.read_csv(df_data1, index_col=False)
        df_data2 = pd.read_csv(df_data2, index_col=False)
        data = pd.concat((df_data1, df_data2), axis=0)
    else:
        data = pd.read_csv(df_data1, index_col=False)

    kf = GroupKFold(n_splits=number_fold)
    df_train = {}
    df_split = {}
    for fold, (train_index, test_index) in enumerate(kf.split(data, data, data.iloc[:, 0])):
        df_train[fold] = data.iloc[train_index].reset_index(drop=True)
        df_split[fold] = data.iloc[test_index].reset_index(drop=True)

    return df_train, df_split


def unfreeze(model, percent=0.25):
    l = int(np.ceil(len(model._modules.keys()) * percent))
    l = list(model._modules.keys())[-l:]
    print(f"unfreezing these layer {l}", )
    for name in l:
        for params in model._modules[name].parameters():
            params.requires_grad_(True)


def check_freeze(model):
    for name, layer in model._modules.items():
        s = []
        for l in layer.parameters():
            s.append(l.requires_grad)
        print(name, all(s))


def k_fold_data(train_data, number_fold=5):
    kf = KFold(n_splits=number_fold, shuffle=True)
    index_train = {}
    index_valid = {}
    for fold, (train_index, valid_index) in enumerate(kf.split(train_data)):
        print("TRAIN:", train_index, "VALID:", valid_index)
        index_train[fold] = train_index
        index_valid[fold] = valid_index
    return index_train, index_valid


def ensemble_results(test_cat, ind2cat, path='./weight/results/?.npy'):
    list_npy = glob.glob(path)
    result = []
    for i in list_npy:
        i = np.load(i)
        result.append(np.expand_dims(i, axis=0))
    result = np.concatenate(result, axis=0).mean(axis=0)
    test_scikit_ap(result, test_cat.transpose(), ind2cat)
