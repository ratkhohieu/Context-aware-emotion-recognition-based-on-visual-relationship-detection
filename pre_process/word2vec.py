import json

import numpy as np
import torchtext


def create_cat_labels(path_annotations):
    cat_2017 = path_annotations
    with open(cat_2017, 'r') as COCO:
        js = json.loads(COCO.read())
        dist_categories = js['categories']

    list_ct = []
    list_id = []
    for i in dist_categories:
        for k, v in i.items():
            if k == 'name':
                list_ct.append(v)
            if k == 'id':
                list_id.append(v)
    return dict(zip(list_id, list_ct))


def word_embedding_categories(dict_categories):
    glove = torchtext.vocab.GloVe(name="6B",  # trained on Wikipedia 2014 corpus
                                  dim=300)  # embedding size = 100
    di = dict_categories
    for k, v in di.items():
        emmbedding_label = []
        for word in v.split(' '):
            emmbedding_label.append(np.array(glove[word]))

        di[k] = np.mean(emmbedding_label, axis=0)
    return di
