import glob
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from facenet_pytorch import MTCNN
from torchvision import transforms
from tqdm import tqdm


def get_emotion(path):
    return path.split("/")[-2]


def get_video_name(path):
    return path.split("/")[-1]


def get_dataset(path):
    return path.split("/")[-3]


def create_df_early(part):
    videos = glob.glob(f'/media/sven/HUNG/CAER/CAER-S/{part}/*/*.png')

    df = pd.DataFrame()
    df['file'] = videos
    df['emotion'] = df['file'].apply(lambda x: get_emotion(x))
    df['dataset'] = df['file'].apply(lambda x: get_dataset(x))
    df['image'] = df['file'].apply(lambda x: get_video_name(x))
    return df


def frame_to_face(frame, mtcnn):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face, prob = mtcnn(image, return_prob=True)
    return face, prob


def main():
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=10,
                                                                 box_score_thresh=0.4, box_nms_thresh=0.4,
                                                                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5)
    model.to(device)
    model.eval()
    mtcnn = MTCNN(margin=10, keep_all=True, post_process=False, device='cuda:0')
    test_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(size=(320, 320)),
                                         transforms.ToTensor()])

    for part in ['train', 'test']:
        df = create_df_early(part)
        list_face = []
        boxes = []

        for i, df_row in tqdm(enumerate(df.iloc), total=len(df)):
            image = cv2.imread(df_row['file'])
            emotion = df_row['emotion']
            #############################
            try:
                face, prob = frame_to_face(image, mtcnn)
                face = face[np.argmax(prob, axis=0)].permute(1, 2, 0)
                path = '/media/sven/HUNG/CAER/extracted_feature/face'
                name = df_row['image']
                name_face = f'face_{name}_{emotion}'
                cv2.imwrite(os.path.join(path, name_face), np.array(face).astype('uint8')[..., ::-1])
                list_face.append(os.path.join(path, name_face))
            except:
                path = '/media/sven/HUNG/CAER/extracted_feature/face'
                name = df_row['image']
                name_face = f'face_{name}'
                cv2.imwrite(os.path.join(path, name_face), np.zeros((160, 160, 3)))
                list_face.append(os.path.join(path, name_face))
            ############################
            try:
                out = model(test_transform(image).unsqueeze(0).to(device))
                label = out[0]['labels']
                box = out[0]['boxes'][label == 1][0].detach().cpu().numpy()
                # import pdb; pdb.set_trace()
                boxes.append(str([box[0], box[1], box[2], box[3]]))
            except:
                boxes.append(str([0, 0, 0, 0]))

        df['face'] = list_face
        df['box_body'] = boxes

        df.to_csv(f'/media/sven/HUNG/CAER/extracted_feature/{part}.csv')


if __name__ == '__main__':
    main()
