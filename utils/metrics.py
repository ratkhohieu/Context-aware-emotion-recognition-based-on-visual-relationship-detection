import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def test_scikit_ap(cat_preds, cat_labels, ind2cat):
    ''' Calculate average precision per emotion category using sklearn library.
    :param cat_preds: Categorical emotion predictions.
    :param cat_labels: Categorical emotion labels.
    :param ind2cat: Dictionary converting integer index to categorical emotion.
    :return: Numpy array containing average precision per emotion category.
    '''
    ap = np.zeros(26, dtype=np.float32)
    for i in range(26):
        ap[i] = average_precision_score(cat_labels[i, :], cat_preds[i, :])
        print('Category %16s %.5f' % (ind2cat[i], ap[i]))
    print('Mean AP %.5f' % (ap.mean()))
    return ap


def test_vad(cont_preds, cont_labels, ind2vad):
    ''' Calcaulate VAD (valence, arousal, dominance) errors.
    :param cont_preds: Continuous emotion predictions.
    :param cont_labels: Continuous emotion labels.
    :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance).
    :return: Numpy array containing mean absolute error per continuous emotion dimension.
    '''
    vad = np.zeros(3, dtype=np.float32)
    for i in range(3):
        vad[i] = np.mean(np.abs(cont_preds[i, :] - cont_labels[i, :]))
        print('Continuous %10s %.5f' % (ind2vad[i], vad[i]))
    print('Mean VAD Error %.5f' % (vad.mean()))
    return vad


def vad_mean_square_error(cont_preds, cont_labels, ind2vad):
    ''' Calcaulate VAD (valence, arousal, dominance) errors.
    :param cont_preds: Continuous emotion predictions.
    :param cont_labels: Continuous emotion labels.
    :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance).
    :return: Numpy array containing mean absolute error per continuous emotion dimension.
    '''
    vad = np.zeros(3, dtype=np.float32)
    for i in range(3):
        vad[i] = np.mean(np.square(cont_preds[i, :] - cont_labels[i, :]))
        print('Continuous %10s %.5f' % (ind2vad[i], vad[i]))
    print('Mean VAD Error %.5f' % (vad.mean()))
    return vad

def get_thresholds(cat_preds, cat_labels):
    ''' Calculate thresholds where precision is equal to recall. These thresholds are then later for inference.
    :param cat_preds: Categorical emotion predictions.
    :param cat_labels: Categorical emotion labels.
    :return: Numpy array containing thresholds per emotion category where precision is equal to recall.
    '''
    thresholds = np.zeros(26, dtype=np.float32)
    for i in range(26):
        p, r, t = precision_recall_curve(cat_labels[i, :], cat_preds[i, :])
        for k in range(len(p)):
            if p[k] == r[k]:
                thresholds[i] = t[k]
                break
    return thresholds


def pre_ind2cat():
    cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
           'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
           'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy',
           'Yearning']
    cat2ind = {}
    ind2cat = {}
    for idx, emotion in enumerate(cat):
        cat2ind[emotion] = idx
        ind2cat[idx] = emotion

    vad = ['Valence', 'Arousal', 'Dominance']
    ind2vad = {}
    for idx, continuous in enumerate(vad):
        ind2vad[idx] = continuous
    return cat2ind, ind2cat, ind2vad
