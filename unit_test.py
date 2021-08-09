# import ast
# import os
#
# import cv2
# import numpy as np
# import pandas as pd
# from facenet_pytorch import MTCNN
# from torch.utils.data import Dataset
# from pose.body import *
# import torch
# DIR_INPUT = './emotic/emotic_pre'
# train_df = pd.read_csv(f'{DIR_INPUT}/test.csv', index_col=0)
#
# from tqdm import tqdm
#
#
# class Emotic_CSVDataset(Dataset):
#
#     def __init__(self, data_df, data_src, transform):
#         super().__init__()
#         self.data_df = data_df
#         self.data_src = data_src
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.data_df)
#
#     def __getitem__(self, index):
#         row = self.data_df.iloc[index]
#         image_context = cv2.imread(os.path.join(self.data_src, row['Folder'], row['Filename']))
#         bbox = ast.literal_eval(row['BBox'])
#         image_body = image_context[bbox[1]:bbox[3], bbox[0]:bbox[2]]
#         # image_context = cv2.resize(image_context,(128, 128))
#         # image_body = cv2.resize(image_body,(128, 128))
#         image_context[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0
#         cat_labels = ast.literal_eval(row['Categorical_Labels'])
#         cont_labels = ast.literal_eval(row['Continuous_Labels'])
#
#         return image_body, image_context, cat_labels, cont_labels
#
#
# # os.makedirs('./emotic/faces', exist_ok=True)
# # train = Emotic_CSVDataset(train_df, './emotic', None)
# # mtcnn = MTCNN(margin=10, keep_all=True, post_process=False, device='cuda:0')
# # train_df['Path'] = './emotic' + '/' + 'faces' + '/' + 'face_' + train_df['Filename']
# # list_faces = []
#
#
# # for ind, (image_body, image_context, cat_labels, cont_labels) in tqdm(enumerate(train), total=len(train)):
# #     image = cv2.cvtColor(image_body, cv2.COLOR_BGR2RGB)
# #     # import pdb; pdb.set_trace()
# #
# #     try:
# #         face = mtcnn(image)
# #     except:
# #         image = cv2.resize(image, (224, 224))
# #         face = mtcnn(image)
# #     boxes, probs = mtcnn.detect(image, landmarks=False)
# #
# #     if face is None:
# #         list_faces.append(None)
# #     elif face.size(0) == 1:
# #         cv2.imwrite(train_df.Path[ind], np.array(face[0].permute(1, 2, 0))[..., ::-1])
# #     elif face.size(0) > 1:
# #         cv2.imwrite(train_df.Path[ind], np.array(face[np.argmax(probs)].permute(1, 2, 0))[..., ::-1])
#
#
# # os.makedirs('./emotic/keys_pose', exist_ok=True)
# # train = Emotic_CSVDataset(train_df, './emotic', None)
# # list_key = []
# #
# # for ind, (image_body, _, _, _) in tqdm(enumerate(train), total=len(train)):
# #
# #     estimator = BodyPoseEstimator(pretrained=True)
# #     keypoints = estimator(image_body)
# #     # import pdb; pdb.set_trace()
# #     if keypoints is None:
# #         list_key.append(np.zeros(18, 3))
# #     elif keypoints.shape[0] == 1:
# #         list_key.append(keypoints[0])
# #     elif keypoints.shape[0] > 1:
# #         list_key.append(keypoints[0])
# #
# # np.save('./emotic/keys_pose/pose_key.npy', list_key)
#
#
# os.makedirs('./emotic/mask', exist_ok=True)
# train = Emotic_CSVDataset(train_df, './emotic', None)
#
# train_df['Path'] = './emotic' + '/' + 'mask' + '/' + 'mask_' + train_df['Filename']
#
#
# for ind, (image_body, image_context, cat_labels, cont_labels) in tqdm(enumerate(train), total=len(train)):
#     cv2.imwrite(train_df.Path[ind], np.array(image_context))

# class SKConv(nn.Module):
#     def __init__(self, features, M=2, G=2, r=16, stride=1, L=32):
#         """ Constructor
#         Args:
#             features: input channel dimensionality.
#             M: the number of branchs.
#             G: num of convolution groups.
#             r: the ratio for compute d, the length of z.
#             stride: stride, default 1.
#             L: the minimum dim of the vector z in paper, default 32.
#         """
#         super(SKConv, self).__init__()
#         d = max(int(features / r), L)
#         self.M = M
#         self.features = features
#         self.convs = nn.ModuleList([])
#         for i in range(M):
#             self.convs.append(nn.Sequential(
#                 nn.Conv1d(features, features, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, groups=G,
#                           bias=False),
#                 nn.BatchNorm1d(features),
#                 nn.ReLU(inplace=False)
#             ))
#         self.gap = nn.AdaptiveAvgPool1d((1))
#         self.fc = nn.Sequential(nn.Conv1d(features, d, kernel_size=1, stride=1, bias=False),
#                                 nn.BatchNorm1d(d),
#                                 nn.ReLU(inplace=False))
#         self.fcs = nn.ModuleList([])
#         for i in range(M):
#             self.fcs.append(
#                 nn.Conv1d(d, features, kernel_size=1, stride=1)
#             )
#         self.softmax = nn.Softmax(dim=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.shortcut = nn.Sequential()
#
#     def forward(self, x):
#
#         residual = x
#         batch_size = x.shape[0]
#
#         feats = [conv(x) for conv in self.convs]
#         feats = torch.cat(feats, dim=1)
#         feats = feats.view(batch_size, self.M, self.features, feats.shape[2])
#
#         feats_U = torch.sum(feats, dim=1)
#
#         feats_S = self.gap(feats_U)
#         feats_Z = self.fc(feats_S)
#
#         attention_vectors = [fc(feats_Z) for fc in self.fcs]
#         attention_vectors = torch.cat(attention_vectors, dim=1)
#         attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1)
#         attention_vectors = self.softmax(attention_vectors)
#
#         feats_V = torch.sum(feats * attention_vectors, dim=1)
#
#         return self.relu(feats_V + self.shortcut(residual))
#
#
# model = SKConv(features=10)
# x = torch.randn(4, 10, 50)
#
# print(model(x).shape)


# import time
#
# import torchvision
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.utils.data import DataLoader
# from tqdm import tqdm, trange
#
# from models import *
# from pre_process import *
# from utils import *
#
# warnings.filterwarnings("ignore")
#
# parser = argparse.ArgumentParser(description='MuSE Training')
# parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
# parser.add_argument('--batch_size', default=12, type=int, help='batch size')
# parser.add_argument('--num_epochs', default=20, type=int, help='number epochs')
# parser.add_argument('--weight_decay', type=float, default=5e-4)
#
# args = parser.parse_args()
#
# device = "cuda:0" if torch.cuda.is_available() else 'cpu'
# scaler = torch.cuda.amp.GradScaler()
#
#
# def batch_filled(sel, num_boxes):
#     index = 0
#     a = []
#     for i in num_boxes:
#         buffed = torch.zeros(5 - i, 1024).to(device)
#         buffed = torch.cat((sel[index:i + index, :], buffed), dim=0)
#         index = index + i
#         a.append(buffed.unsqueeze(dim=0))
#
#     result = torch.cat(a)
#     return result
#
#
# def main():
#     seed_everything()
#     train_context, train_body, train_cat, train_cont = load_data_npy(mode='train')
#     # val_context, val_body, val_cat, val_cont = load_data_npy(mode='valid')
#     test_context, test_body, test_cat, test_cont = load_data_npy(mode='test')
#
#     cat2ind, ind2cat, ind2vad = pre_ind2cat()
#     root = './emotic'
#     train_dataset = Emotic_CSVDataset(cat2ind=cat2ind, transform=train_transform, root=root,
#                                       data_src='emotic_pre/train.csv')
#     train_loader = DataLoader(dataset=train_dataset,
#                               batch_size=args.batch_size,
#                               num_workers=4,
#                               shuffle=True,
#                               drop_last=True)
#
#
#     test_dataset = Emotic_CSVDataset(cat2ind=cat2ind, transform=test_transform, root=root,
#                                      data_src='emotic_pre/test.csv', mode_cat=None)
#
#     test_loader = DataLoader(dataset=test_dataset,
#                              batch_size=args.batch_size,
#                              num_workers=4,
#                              shuffle=False)
#
#     # model_hook = Sentiment_MLP(256, 4, 50, N=10, num_sentiment=4096)
#     model = Emotest(1024 * 5)
#
#     model_faster = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=5,
#                                                                         box_score_thresh=0.3, box_nms_thresh=0.4,
#                                                                         box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5, )
#
#     if torch.cuda.device_count() > 1:
#         print("Let's use", torch.cuda.device_count(), "GPUs!")
#         model = nn.DataParallel(model)
#
#     model_faster.to(device)
#
#     # model_hook.to(device)
#     model.to(device)
#
#     criterion1 = DiscreteLoss(weight_type='dynamic', device=device)
#     # criterion1 = nn.MultiLabelMarginLoss()
#     criterion2 = ContinuousLoss_SL1()
#     # criterion1 = nn.BCEWithLogitsLoss()
#     # criterion1 = FocalLoss()
#
#     optimizer = torch.optim.Adam(list(model.parameters()),
#                                  lr=args.lr, weight_decay=args.weight_decay)
#
#     start_time = time.time()
#     scheduler_steplr = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=0, last_epoch=-1)
#
#     optimizer.zero_grad()
#     optimizer.step()
#     best_scores = 10
#
#     with trange(args.num_epochs, total=args.num_epochs, desc='Epoch') as t:
#         for epoch in t:
#             if optimizer.param_groups[0]['lr'] < 5e-4:
#                 unfreeze(model_context, percent=0.1)
#                 unfreeze(model_body, percent=0.1)
#
#             t.set_description('Epoch %i' % epoch)
#             model.train()
#             model_faster.eval()
#             # print(epoch, optimizer.param_groups[0]['lr'], '\n')
#             cost_list = 0
#             scheduler_steplr.step(epoch)
#
#             for batch_idx, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
#                 images_context = sample['image_context'].to(device)
#                 labels_cat = sample['cat_label'].to(device).long()
#                 outputs = []
#                 hook = model_faster.backbone.register_forward_hook(
#                     lambda self, input, output: outputs.append(output))
#                 res = model_faster(images_context)
#
#                 hook.remove()
#                 selected_rois = model_faster.roi_heads.box_roi_pool(
#                     outputs[0], [r['boxes'] for r in res], [i.shape[-2:] for i in images_context])
#
#                 sel = model_faster.roi_heads.box_head(selected_rois)
#
#                 num_boxes = []
#                 for i in res:
#                     num_boxes.append(len(i['boxes']))
#                 features_for_fc = batch_filled(sel, num_boxes)
#
#                 optimizer.zero_grad()
#
#                 pred_cat = model(features_for_fc.view(features_for_fc.size(0), -1))
#                 pred_cat = pred_cat.sigmoid()
#
#                 with torch.cuda.amp.autocast():
#                     cat_loss_batch = criterion1(pred_cat, labels_cat)
#                     loss = cat_loss_batch
#
#                 cost_list += loss.item()
#
#                 scaler.scale(loss).backward()
#                 # UPDATE MODEL PARAMETERS
#                 scaler.step(optimizer)
#
#                 # Updates the scale for next iteration
#                 scaler.update()
#
#                 t.set_postfix(Loss=f'{cost_list / (batch_idx + 1):04f}',
#                               Batch=f'{batch_idx + 1:03d}/{len(train_loader):03d}',
#                               Lr=optimizer.param_groups[0]['lr'])
#
#             model.eval()
#             model_faster.eval()
#             with torch.no_grad():
#
#                 cat_preds = []
#                 cat_labels = []
#
#                 for batch_idx, sample in tqdm(enumerate(test_loader), total=len(test_loader), desc='Test_mode'):
#                     images_context = sample['image_context'].to(device)
#                     labels_cat = sample['cat_label'].to(device).long()
#                     outputs = []
#                     hook = model_faster.backbone.register_forward_hook(
#                         lambda self, input, output: outputs.append(output))
#                     res = model_faster(images_context)
#
#                     hook.remove()
#                     selected_rois = model_faster.roi_heads.box_roi_pool(
#                         outputs[0], [r['boxes'] for r in res], [i.shape[-2:] for i in images_context])
#
#                     sel = model_faster.roi_heads.box_head(selected_rois)
#
#                     num_boxes = []
#                     for i in res:
#                         num_boxes.append(len(i['boxes']))
#                     features_for_fc = batch_filled(sel, num_boxes)
#
#                     optimizer.zero_grad()
#
#                     pred_cat = model(features_for_fc.view(features_for_fc.size(0), -1))
#                     pred_cat = pred_cat.sigmoid()
#
#                     cat_preds.append(pred_cat.detach().cpu())
#                     cat_labels.append(labels_cat.detach().cpu())
#
#                 cat_preds = np.concatenate(cat_preds, axis=0).transpose()
#                 cat_labels = np.concatenate(cat_labels, axis=0).transpose()
#
#                 tqdm.write('\n')
#                 test_scikit_ap(cat_preds, cat_labels, ind2cat)
#
#                 # print('Thresholds= ', thresholds)
#
#         elapsed = (time.time() - start_time) / 60
#         print(f'Total Training Time: {elapsed:.2f} min')
#
#     del model
#     torch.cuda.empty_cache()
#
#
# if __name__ == '__main__':
#     # path = './emotic/annotations_coco/instances_train2017.json'
#     # print(word_embedding_categories(create_cat_labels(path_annotations=path)))
#     main()


# import time
#
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.utils.data import DataLoader
# from tqdm import tqdm, trange
#
# from models import *
# from pre_process import *
# from utils import *
#
# warnings.filterwarnings("ignore")
#
# parser = argparse.ArgumentParser(description='MuSE Training')
# parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
# parser.add_argument('--batch_size', default=26, type=int, help='batch size')
# parser.add_argument('--num_epochs', default=20, type=int, help='number epochs')
# parser.add_argument('--weight_decay', type=float, default=5e-4)
#
# args = parser.parse_args()
#
# device = "cuda:0" if torch.cuda.is_available() else 'cpu'
# scaler = torch.cuda.amp.GradScaler()
#
#
# def main():
#     seed_everything()
#     train_context, train_body, train_cat, train_cont = load_data_npy(mode='train')
#     # val_context, val_body, val_cat, val_cont = load_data_npy(mode='valid')
#     # train_context = np.concatenate((train_context, val_context), axis=0)
#     # train_body = np.concatenate((train_body, val_body), axis=0)
#     # train_cat = np.concatenate((train_cat, val_cat), axis=0)
#     # import pdb; pdb.set_trace()
#     test_context, test_body, test_cat, test_cont = load_data_npy(mode='test')
#
#     cat2ind, ind2cat, ind2vad = pre_ind2cat()
#     root = './emotic'
#     train_dataset = Emotic_Dataset_addhook(train_context, train_body, train_cat, train_cont, train_transform,
#                                            mode_unnorm=False,
#                                            hook_faster=load_data_hook_faster('train'),
#                                            hook_sentiment=load_data_sentiment('train'),
#                                            )
#
#     train_loader = DataLoader(dataset=train_dataset,
#                               batch_size=args.batch_size,
#                               num_workers=4,
#                               shuffle=True,
#                               drop_last=False)
#
#     # valid_dataset = Emotic_Dataset_addhook(val_context, val_body, val_cat, val_cont, test_transform,
#     #                                        mode_unnorm=False,
#     #                                        hook_faster=load_data_hook_faster('val'),
#     #                                        hook_sentiment=load_data_sentiment('val'))
#     # valid_loader = DataLoader(dataset=valid_dataset,
#     #                           batch_size=args.batch_size,
#     #                           num_workers=4,
#     #                           shuffle=False)
#
#     test_dataset = Emotic_Dataset_addhook(test_context, test_body, test_cat, test_cont, test_transform,
#                                           mode_unnorm=False,
#                                           hook_faster=load_data_hook_faster('test'),
#                                           hook_sentiment=load_data_sentiment('test'),
#                                           )
#     test_loader = DataLoader(dataset=test_dataset,
#                              batch_size=args.batch_size,
#                              num_workers=4,
#                              shuffle=False)
#
#     model_context, model_body = prep_models(model_dir='proj/debug_exp/models', context_model='resnet50')
#
#     model_hook = Sentiment_MLP(256, 4, 50, N=10, num_sentiment=4096)
#     model = Emotest(4096)
#     model_context = nn.Sequential(*(list(model_context.children())[:-1]))
#     model_body = nn.Sequential(*(list(model_body.children())[:-1]))
#
#     # if torch.cuda.device_count() > 1:
#     #     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     #     model = nn.DataParallel(model)
#
#     model_body.to(device)
#     model_context.to(device)
#     model_hook.to(device)
#     model.to(device)
#
#     criterion1 = DiscreteLoss(weight_type='dynamic', device=device)
#     # criterion1 = nn.MultiLabelMarginLoss()
#     criterion2 = ContinuousLoss_SL1()
#     # criterion1 = nn.BCEWithLogitsLoss()
#     # criterion1 = FocalLoss()
#
#     optimizer = torch.optim.AdamW(
#         (list(model.parameters()) + list(model_context.parameters()) + list(model_body.parameters()) + list(
#             model_hook.parameters())),
#         lr=args.lr, weight_decay=args.weight_decay)
#
#     start_time = time.time()
#     scheduler_steplr = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=0, last_epoch=-1)
#
#     # lmbda = lambda epoch: 0.65 ** epoch
#     # # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
#     # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#
#     optimizer.zero_grad()
#     optimizer.step()
#     best_scores = 10
#
#     for name, p in model_body.named_parameters():
#         p.requires_grad = False
#
#     for name, p in model_context.named_parameters():
#         p.requires_grad = False
#
#     with trange(args.num_epochs, total=args.num_epochs, desc='Epoch') as t:
#         for epoch in t:
#             # if optimizer.param_groups[0]['lr'] < 5e-4:
#             #     unfreeze(model_context, percent=0.2)
#             #     unfreeze(model_body, percent=0.2)
#
#             t.set_description('Epoch %i' % epoch)
#             model.train()
#             model_context.train()
#             model_body.train()
#             model_hook.train()
#             # print(epoch, optimizer.param_groups[0]['lr'], '\n')
#             cost_list = 0
#             scheduler_steplr.step(epoch)
#
#             for batch_idx, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
#                 labels_cat = sample['cat_label'].to(device).long()
#
#                 features_object = sample['features_object'].to(device).float()
#                 bbox = sample['bbox'].to(device).float()
#                 word_embedding = sample['labels'].to(device).float()
#
#                 sentiment_object = sample['sentiment_objects'].to(device).float()
#                 # import pdb; pdb.set_trace()
#                 optimizer.zero_grad()
#
#                 pred_hook = model_hook(features_object, bbox, word_embedding, sentiment_object)
#                 pred_cat = model(pred_hook)
#
#                 with torch.cuda.amp.autocast():
#                     cat_loss_batch = criterion1(pred_cat, labels_cat)
#                     # cont_loss_batch = criterion2(pred_cont * 10, labels_cont * 10)
#                     loss = cat_loss_batch
#
#                 cost_list += loss.item()
#
#                 scaler.scale(loss).backward()
#                 # UPDATE MODEL PARAMETERS
#                 scaler.step(optimizer)
#
#                 # Updates the scale for next iteration
#                 scaler.update()
#
#                 t.set_postfix(Loss=f'{cost_list / (batch_idx + 1):04f}',
#                               Batch=f'{batch_idx + 1:03d}/{len(train_loader):03d}',
#                               Lr=optimizer.param_groups[0]['lr'])
#
#             model.eval()
#             model_context.eval()
#             model_body.eval()
#             model_hook.eval()
#
#             with torch.no_grad():
#                 # cat_preds = []
#                 # cat_labels = []
#                 # cont_preds = []
#                 # cont_labels = []
#                 # for batch_idx, sample in tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid_mode'):
#                 #     images_context = sample['image_context'].to(device)
#                 #     images_body = sample['image_body'].to(device)
#                 #
#                 #     labels_cat = sample['cat_label'].to(device).long()
#                 #     labels_cont = sample['cont_label'].to(device)
#                 #
#                 #     features_object = sample['features_object'].to(device).float()
#                 #     bbox = sample['bbox'].to(device).float()
#                 #     word_embedding = sample['labels'].to(device).float()
#                 #
#                 #     sentiment_object = sample['sentiment_objects'].to(device).float()
#                 #
#                 #     pred_context = model_context(images_context)
#                 #     pred_body = model_body(images_body)
#                 #     pred_hook = model_hook(features_object, bbox, word_embedding, sentiment_object)
#                 #     pred_cat, pred_cont = model(pred_context, pred_body, pred_hook)
#                 #     pred_cat = pred_cat.sigmoid()
#                 #
#                 #     cat_preds.append(pred_cat.detach().cpu())
#                 #     cat_labels.append(labels_cat.detach().cpu())
#                 #     cont_preds.append(pred_cont.detach().cpu())
#                 #     cont_labels.append(labels_cont.detach().cpu())
#                 #
#                 # # import pdb; pdb.set_trace();
#                 #
#                 # cat_preds = np.concatenate(cat_preds, axis=0).transpose()
#                 # cat_labels = np.concatenate(cat_labels, axis=0).transpose()
#                 # cont_preds = (np.concatenate(cont_preds, axis=0) * 10).transpose()
#                 # cont_labels = (np.concatenate(cont_labels, axis=0) * 10).transpose()
#                 #
#                 # tqdm.write('\n')
#                 # test_scikit_ap(cat_preds, cat_labels, ind2cat)
#                 # test_vad(cont_preds, cont_labels, ind2vad)
#                 # thresholds = get_thresholds(cat_preds, cat_labels)
#
#                 cat_preds = []
#                 cat_labels = []
#                 cont_preds = []
#                 cont_labels = []
#
#                 for batch_idx, sample in tqdm(enumerate(test_loader), total=len(test_loader), desc='Test_mode'):
#                     labels_cat = sample['cat_label'].to(device).long()
#
#                     features_object = sample['features_object'].to(device).float()
#                     bbox = sample['bbox'].to(device).float()
#                     word_embedding = sample['labels'].to(device).float()
#
#                     sentiment_object = sample['sentiment_objects'].to(device).float()
#
#                     pred_hook = model_hook(features_object, bbox, word_embedding, sentiment_object)
#
#                     pred_cat = model(pred_hook)
#                     pred_cat = pred_cat.sigmoid()
#
#                     cat_preds.append(pred_cat.detach().cpu())
#                     cat_labels.append(labels_cat.detach().cpu())
#
#                 cat_preds = np.concatenate(cat_preds, axis=0).transpose()
#                 cat_labels = np.concatenate(cat_labels, axis=0).transpose()
#
#                 tqdm.write('\n')
#                 test_scikit_ap(cat_preds, cat_labels, ind2cat)
#
#                 # print('Thresholds= ', thresholds)
#
#         elapsed = (time.time() - start_time) / 60
#         print(f'Total Training Time: {elapsed:.2f} min')
#
#     del model
#     torch.cuda.empty_cache()
#
#
# if __name__ == '__main__':
#     # path = './emotic/annotations_coco/instances_train2017.json'
#     # print(word_embedding_categories(create_cat_labels(path_annotations=path)))
#     main()


# class EmbraceNet(nn.Module):
#
#     def __init__(self, device, input_size_list, embracement_size=256, bypass_docking=False):
#         """
#         Initialize an EmbraceNet module.
#         Args:
#           device: A "torch.device()" object to allocate internal parameters of the EmbraceNet module.
#           input_size_list: A list of input sizes.
#           embracement_size: The length of the output of the embracement layer ("c" in the paper).
#           bypass_docking: Bypass docking step, i.e., connect the input data directly to the embracement layer. If True, input_data must have a shape of [batch_size, embracement_size].
#         """
#         super(EmbraceNet, self).__init__()
#
#         self.device = device
#         self.input_size_list = input_size_list
#         self.embracement_size = embracement_size
#         self.bypass_docking = bypass_docking
#
#         if (not bypass_docking):
#             for i, input_size in enumerate(input_size_list):
#                 setattr(self, 'docking_%d' % (i), nn.Linear(input_size, embracement_size))
#
#     def forward(self, input_list, availabilities=None, selection_probabilities=None):
#         """
#         Forward input data to the EmbraceNet module.
#         Args:
#           input_list: A list of input data. Each input data should have a size as in input_size_list.
#           availabilities: A 2-D tensor of shape [batch_size, num_modalities], which represents the availability of data for each modality. If None, it assumes that data of all modalities are available.
#           selection_probabilities: A 2-D tensor of shape [batch_size, num_modalities], which represents probabilities that output of each docking layer will be selected ("p" in the paper). If None, the same probability of being selected will be used for each docking layer.
#         Returns:
#           A 2-D tensor of shape [batch_size, embracement_size] that is the embraced output.
#         """
#         import pdb; pdb.set_trace()
#         # check input data
#         assert len(input_list) == len(self.input_size_list)
#         num_modalities = len(input_list)
#         batch_size = input_list[0].shape[0]
#
#         # docking layer
#         docking_output_list = []
#         if (self.bypass_docking):
#             docking_output_list = input_list
#         else:
#             for i, input_data in enumerate(input_list):
#                 x = getattr(self, 'docking_%d' % (i))(input_data)
#                 x = nn.functional.relu(x)
#                 docking_output_list.append(x)
#
#         # check availabilities
#         if (availabilities is None):
#             availabilities = torch.ones(batch_size, len(input_list), dtype=torch.float, device=self.device)
#         else:
#             availabilities = availabilities.float()
#
#         # adjust selection probabilities
#         if (selection_probabilities is None):
#             selection_probabilities = torch.ones(batch_size, len(input_list), dtype=torch.float, device=self.device)
#         selection_probabilities = torch.mul(selection_probabilities, availabilities)
#
#         probability_sum = torch.sum(selection_probabilities, dim=-1, keepdim=True)
#         selection_probabilities = torch.div(selection_probabilities, probability_sum)
#
#         # stack docking outputs
#         docking_output_stack = torch.stack(docking_output_list,
#                                            dim=-1)  # [batch_size, embracement_size, num_modalities]
#
#         # embrace
#         modality_indices = torch.multinomial(selection_probabilities, num_samples=self.embracement_size,
#                                              replacement=True)  # [batch_size, embracement_size]
#         modality_toggles = nn.functional.one_hot(modality_indices,
#                                                  num_classes=num_modalities).float()  # [batch_size, embracement_size, num_modalities]
#
#         embracement_output_stack = torch.mul(docking_output_stack, modality_toggles)
#         embracement_output = torch.sum(embracement_output_stack, dim=-1)  # [batch_size, embracement_size]
#
#         return embracement_output
#
#
# model = EmbraceNet(device='cuda:0', input_size_list=[50, 32])
# model.to('cuda:0')
#
# a = torch.randn((4, 10, 50)).to('cuda:0')
# b = torch.randn((4, 10, 32)).to('cuda:0')
#
# print(model(input_list=[a, b]).shape)
# from pre_process import prepare_models
#
# model = prepare_models.prepare_model_relation()

import math

# from models.transformer import TransformerEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F


#
# model = TransformerEncoder(embed_dim=30,
#                            num_heads=5,
#                            layers=max(5, 3),
#                            attn_dropout=0.1,
#                            relu_dropout=0.1,
#                            res_dropout=0.1,
#                            embed_dropout=0.1,
#                            attn_mask=True)
#
# a = torch.randn(2, 4, 30)
# b = torch.randn(2, 4, 30)
# c = torch.randn(2, 4, 30)
# print(model(a,b,c).shape)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class SPModule(torch.nn.Module):
    """
    Module for spatial branch computation.
    """

    def __init__(self):
        super(SPModule, self).__init__()
        self.sp_head_channels = 7756
        sp_head_channels = 7756
        self.ssar_key_net = nn.Linear(7756, 1024, bias=False)  # 7756, 1024 (7756 = 5408 + cfg.DATASETS.WORD_DIM + 2048)
        self.ssar_value_net = nn.Linear(sp_head_channels, sp_head_channels, bias=False)  # 7756, 7756
        self.ssar_attn_dropout = nn.Dropout(p=0.7, inplace=True)
        self.ssar_layernorm = nn.LayerNorm(sp_head_channels)  # 7756





    def refinement(self, net_human_embed):
        '''
        refinement module
        :param net_human_embed: spatial branch embedding, size [N, 7756], N is batch size
        :return: weights: size [N, 7756]
        '''

        key_embed = self.ssar_key_net(net_human_embed)  # key and query share weights, n*1024
        query_embed = self.ssar_key_net(net_human_embed)  # n*1024
        value_embed = self.ssar_value_net(net_human_embed)  # n*7756

        weights = torch.matmul(query_embed, torch.t(key_embed))  # key_embed.permute(1, 0), n*n
        weights = weights / math.sqrt(key_embed.size(-1))  # math.sqrt(key_embed.size(-1)) = sqrt(1024) = 32
        softmax_attention = nn.Softmax(dim=1)(weights)

        weights = torch.matmul(softmax_attention, value_embed)  # n*7756
        weights = F.relu(weights)
        weights = self.ssar_attn_dropout(weights)
        weights = self.ssar_layernorm(net_human_embed + weights)

        return weights

    def forward(self, net_word_sp):

        X_1 = self.refinement(net_word_sp)
        X_2 = self.refinement(X_1)

model = SPModule()
a = model(torch.tensor(2,7756))
print(a.shape)

