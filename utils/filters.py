import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.cluster import KMeans


def filter_one(result, n_cluster):
    y_pred = KMeans(n_clusters=n_cluster, random_state=1).fit_predict(result)
    unique, counts = np.unique(y_pred, return_counts=True)
    d = dict(zip(unique, counts))

    # max_value = max(d.values())
    # key = [k for k, v in d.items() if v == max_value]
    #
    # if len(key) == 1:
    #     return result[(y_pred == key)].mean(axis=0)
    # else:
    #     final_result = []
    #     for k in key:
    #         final_result.append(result[(y_pred == k)])
    #     final_result = np.concatenate(final_result, axis=0)
    #     return final_result.mean(axis=0)
    # import pdb; pdb.set_trace()
    sort_orders = sorted(d.items(), key=lambda x: x[1], reverse=True)
    final_result = []
    for k, _ in sort_orders[:]:
        final_result.append(result[(y_pred == k)])
    final_result = np.concatenate(final_result, axis=0)
    return final_result.mean(axis=0)


def filters_all(df_result_face, n_cluster):
    df = pd.read_csv(df_result_face)
    video_name = df['video_name'].unique()

    out_result = []
    for video in video_name:
        df_frame = df[df['video_name'] == video].iloc[:, -4:-1]
        result_frame = np.array(df_frame)
        result_video = filter_one(result_frame, n_cluster)
        out_result.append(result_video)
    out_result = np.asarray(out_result)
    # import pdb; pdb.set_trace()
    df_out = pd.DataFrame(columns=['Valence', 'Arousal', 'Stress'], data=out_result)
    return df_out


def fillter_peak(pred, emotion):
    peak = []
    weak = []
    for ind, i in enumerate(pred):
        if emotion[ind]:
            weak.append(i)
        else:
            peak.append(i)
    #     import pdb; pdb.set_trace()
    if len(peak) != 0 and len(weak) != 0 and len(peak) / len(pred) > 0.2:
        pred_out = np.mean(peak, axis=0)
    else:
        pred_out = np.mean(pred, axis=0)

    return pred_out


def cluster_peak_emotion(df):
    df['emotion'] = df['emotion'].apply(lambda x: True if x == 4 else False)
    video_name = df['video_name'].unique()

    out_combine = []
    for idx, video in enumerate(video_name):
        sd = df[df['video_name'] == video]
        pred_all = np.array([sd['pred_va'], sd['pred_ar'], sd['pred_st']])
        pred_all = pred_all.T
        emotion = np.array(sd['emotion'])
        out_combine.append(fillter_peak(pred_all, emotion))

    return np.array(out_combine)


def filter_before_train(df_out_emo, df_train):
    """

    :param df_out_emo:'test.csv'
    :param df_train:'./dataset/train_faces.csv'
    :return:
    """
    df_out_emo = pd.read_csv(df_out_emo)
    df_train = pd.read_csv(df_train)
    video_name = df_out_emo.video_name.unique()
    df_buff = pd.DataFrame()
    for video in video_name:
        df_frame = df_out_emo[df_out_emo['video_name'] == video]
        if len(df_frame[df_frame['emotion'] == 4]) / len(df_frame) < 1:
            df_frame = df_frame.drop(df_frame[df_frame['emotion'] == 4].index)

        df_buff = pd.concat((df_buff, df_frame), axis=0)
    new_df_train = df_train.iloc[df_buff.index]
    return new_df_train.reset_index(drop=True)
