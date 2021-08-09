import pandas as pd
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from pre_process import *
from utils import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='MuSE Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')

args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else 'cpu'
scaler = torch.cuda.amp.GradScaler()


def hook_faster_rcnn(model, image_input):
    outputs = []
    hook = model.backbone.register_forward_hook(
        lambda self, input, output: outputs.append(output))
    res = model(image_input)
    hook.remove()

    selected_rois = model.roi_heads.box_roi_pool(
        outputs[0], [r['boxes'] for r in res], [i.shape[-2:] for i in image_input])

    out_pool = nn.AdaptiveAvgPool2d((1, 1))

    result = {
        'features_object': np.array(out_pool(selected_rois).view(selected_rois.size(0), -1).detach().cpu()),
        'bounding_box': np.array(res[0]['boxes'].detach().cpu()),
        'labes': np.array(res[0]['labels'].detach().cpu())
    }
    return result


def main():
    seed_everything()
    # train_context, train_body, train_cat, train_cont = load_data_npy(mode='train')
    # val_context, val_body, val_cat, val_cont = load_data_npy(mode='valid')
    # test_context, test_body, test_cat, test_cont = load_data_npy(mode='test')

    # cat2ind, ind2cat, ind2vad = pre_ind2cat()

    root = '/media/sven/HUNG/CAER/extracted_feature/'
    # train_transform = transforms.Compose([transforms.ToPILImage(),
    #                                       transforms.Resize(size=(320, 320)),
    #                                       # transforms.RandomHorizontalFlip(),
    #                                       # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    #                                       transforms.ToTensor()])
    #
    # test_transform = transforms.Compose([transforms.ToPILImage(),
    #                                      transforms.Resize(size=(320, 320)),
    #                                      transforms.ToTensor()])
    #
    # train_dataset = Emotic_PreDataset(train_context, train_body, train_cat, train_cont, train_transform,
    #                                   root=root,
    #                                   data_src='emotic_pre/train.csv', mode_unnorm=True)
    # train_loader = DataLoader(dataset=train_dataset,
    #                           batch_size=args.batch_size,
    #                           num_workers=4,
    #                           shuffle=False,
    #                           drop_last=False)
    #
    # valid_dataset = Emotic_PreDataset(val_context, val_body, val_cat, val_cont, test_transform,
    #                                   root=root,
    #                                   data_src='emotic_pre/val.csv', mode_unnorm=True)
    # valid_loader = DataLoader(dataset=valid_dataset,
    #                           batch_size=args.batch_size,
    #                           num_workers=4,
    #                           shuffle=False)
    #
    # test_dataset = Emotic_PreDataset(test_context, test_body, test_cat, test_cont, test_transform,
    #                                  root=root,
    #                                  data_src='emotic_pre/test.csv', mode_unnorm=True)
    # test_loader = DataLoader(dataset=test_dataset,
    #                          batch_size=args.batch_size,
    #                          num_workers=4,
    #                          shuffle=False)

    # load a model pre-trained pre-trained on COCO

    train_dataset = Caer_Dataset(path_df='/media/sven/HUNG/CAER/extracted_feature/train.csv', transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=False,
                              drop_last=False)

    test_dataset = Caer_Dataset(path_df='/media/sven/HUNG/CAER/extracted_feature/test.csv', transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             num_workers=4,
                             shuffle=False)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=10,
                                                                 box_score_thresh=0.4, box_nms_thresh=0.4,
                                                                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5, )

    model.to(device)
    model.eval()

    # for k, v in {'train': train_loader, 'test': test_loader}.items():
    #     results = []
    #     for batch_idx, sample in tqdm(enumerate(v), total=len(v)):
    #         images_context = sample['image_context'].to(device)
    #         outputs = []
    #         hook = model.backbone.register_forward_hook(
    #             lambda self, input, output: outputs.append(output))
    #         res = model(images_context)
    #         # import pdb;
    #         # pdb.set_trace()
    #         try:
    #             assert len(res[0]['labels'] != 0)
    #         except:
    #             # TODO: need to process when can't detect anything, box, label, features are zeros
    #             print('Zeros_box')
    #             result = {
    #                 'features_object': np.zeros((1, 1024)),
    #                 'bounding_box': np.zeros((1, 4)),
    #                 'labes': np.zeros(1)
    #             }
    #             results.append(result)
    #             continue
    #
    #         hook.remove()
    #         selected_rois = model.roi_heads.box_roi_pool(
    #             outputs[0], [r['boxes'] for r in res], [i.shape[-2:] for i in images_context])
    #
    #         # out_pool = nn.AdaptiveAvgPool2d((1, 1))
    #         #
    #         # result = {
    #         #     'features_object': np.array(out_pool(selected_rois).view(selected_rois.size(0), -1).detach().cpu()),
    #         #     'bounding_box': np.array(res[0]['boxes'].detach().cpu()),
    #         #     'labes': np.array(res[0]['labels'].detach().cpu())
    #         # }
    #
    #         result = {
    #             'features_object': np.array(model.roi_heads.box_head(selected_rois).detach().cpu()),
    #             'bounding_box': np.array(res[0]['boxes'].detach().cpu()),
    #             'labes': np.array(res[0]['labels'].detach().cpu())
    #         }
    #
    #         results.append(result)
    #     path_npy = os.path.join(root, 'hook_faster')
    #     os.makedirs(path_npy, exist_ok=True)
    #     np.save(os.path.join(path_npy, f'{k}_faster_320_1024.npy'), results)
    #     print(len(results))

    for k, v in {'train': train_loader, 'test': test_loader}.items():
        results = []
        df = pd.read_csv(f'/media/sven/HUNG/CAER/extracted_feature/{k}.csv')
        for batch_idx, sample in tqdm(enumerate(v), total=len(v)):
            images_context = sample['image_context'].to(device)
            outputs = []
            hook = model.backbone.register_forward_hook(
                lambda self, input, output: outputs.append(output))
            res = model(images_context)
            # import pdb;
            # pdb.set_trace()
            try:
                assert len(res[0]['labels'] != 0)
            except:
                # TODO: need to process when can't detect anything, box, label, features are zeros
                print('Zeros_box')
                result = {
                    'features_object': np.zeros((1, 1024)),
                    'bounding_box': np.zeros((1, 4)),
                    'labes': np.zeros(1)
                }
                results.append(result)
                continue

            hook.remove()
            selected_rois = model.roi_heads.box_roi_pool(
                outputs[0], [r['boxes'] for r in res], [i.shape[-2:] for i in images_context])

            # out_pool = nn.AdaptiveAvgPool2d((1, 1))
            #
            # result = {
            #     'features_object': np.array(out_pool(selected_rois).view(selected_rois.size(0), -1).detach().cpu()),
            #     'bounding_box': np.array(res[0]['boxes'].detach().cpu()),
            #     'labes': np.array(res[0]['labels'].detach().cpu())
            # }

            result = {
                'features_object': np.array(model.roi_heads.box_head(selected_rois).detach().cpu()),
                'bounding_box': np.array(res[0]['boxes'].detach().cpu()),
                'labes': np.array(res[0]['labels'].detach().cpu())
            }

            name_npy_file = df.iloc[batch_idx]['image']
            emotion = df.iloc[batch_idx]['emotion']
            path_npy = os.path.join(root, 'hook_faster')
            os.makedirs(path_npy, exist_ok=True)
            np.save(os.path.join(path_npy, f'{emotion}_{name_npy_file}.npy'), result)
            results.append(os.path.join(path_npy, f'{emotion}_{name_npy_file}.npy'))
        df['hook_faster'] = results
        df.to_csv(f'{k}_hook.csv')


if __name__ == '__main__':
    main()
