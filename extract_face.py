import ast
import os

import cv2
import numpy as np
import pandas as pd
from facenet_pytorch import MTCNN
from tqdm import tqdm

from pre_process.dataloader import new_coordinates_after_resize_img


def frame_to_face(frame, box_a):
    im = cv2.imread(frame)[..., ::-1]
    # print(im.shape)
    # import pdb; pdb.set_trace()
    im = im[box_a[1]:box_a[3], box_a[0]:box_a[2]]
    mtcnn = MTCNN(margin=10, keep_all=True, post_process=False, device='cuda:0')
    try:
        boxes, probs = mtcnn.detect(im, landmarks=False)
        h, w, _ = im.shape
    except:
        im = cv2.imread(frame)[..., ::-1]
        boxes, probs = mtcnn.detect(im, landmarks=False)
        h, w, _ = im.shape
    if boxes is not None:
        main_bbox = boxes[np.argmax(probs, axis=0)].astype(int)
        # mask = im.copy()
        im = im[main_bbox[1]:main_bbox[3], main_bbox[0]:main_bbox[2]]

        x1, y1, x2, y2 = main_bbox
        x1_new, y1_new = new_coordinates_after_resize_img((w, h), (320, 320), (x1, y1))
        x2_new, y2_new = new_coordinates_after_resize_img((w, h), (320, 320), (x2, y2))

        # import pdb;
        # pdb.set_trace()
        return im, str([x1_new, y1_new, x2_new, y2_new])
    else:
        return None, str([0, 0, 0, 0])


if __name__ == '__main__':

    root = './emotic/'
    for fold in ['val']:
        df = pd.read_csv(f'./emotic/emotic_pre/{fold}.csv', index_col=0)

        df['face_file'] = root + df['Folder'] + '/' + df['Filename']
        df['face_numpy'] = None
        df['bbox_face_320'] = None
        for i, im_path in tqdm(enumerate(df['face_file']), total=len(df['face_file'])):
            row = df.iloc[i]
            box_a = ast.literal_eval(row['BBox'])
            face, mask = frame_to_face(im_path, box_a)
            df['bbox_face_320'][i] = mask

            path_npy = os.path.join(root, 'faces', fold, f'{i}.npy')
            df['face_numpy'][i] = path_npy
            if face is not None:
                np.save(path_npy, face)
            else:
                np.save(path_npy, [])
        df.to_csv(f'./emotic/emotic_pre/{fold}_face_mask.csv')
