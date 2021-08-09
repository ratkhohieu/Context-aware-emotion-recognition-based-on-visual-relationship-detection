import time

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from models import *
from pre_process import *
from utils import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='MuSE Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=26, type=int, help='batch size')
parser.add_argument('--num_epochs', default=20, type=int, help='number epochs')
parser.add_argument('--weight_decay', type=float, default=5e-4)

args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else 'cpu'


# scaler = torch.cuda.amp.GradScaler()


def main():
    # seed_everything()
    # import pdb; pdb.set_trace()
    # model = load_face_model()
    train_context_all, train_body_all, train_cat_all, train_cont_all = load_data_npy(mode='train')
    val_context, val_body, val_cat, val_cont = load_data_npy(mode='valid')
    test_context, test_body, test_cat, test_cont = load_data_npy(mode='test')

    index_filled = np.load('index_filled_train.npy')
    index_filled_val = np.load('index_filled_val.npy')
    train_context_all, train_body_all, train_cat_all, train_cont_all = \
        train_context_all[index_filled], train_body_all[index_filled], \
        train_cat_all[index_filled], train_cont_all[index_filled]

    val_context, val_body, val_cat, val_cont = val_context[index_filled_val], val_body[index_filled_val], \
                                               val_cat[index_filled_val], val_cont[index_filled_val]

    index_train, index_valid = k_fold_data(train_context_all, number_fold=5)
    cat2ind, ind2cat, ind2vad = pre_ind2cat()
    root = './emotic'
    # import pdb; pdb.set_trace()
    for fold in range(5):
        # fold = 4

        print(f'============++++++ FOLD {fold} ++++++======================')

        train_dataset = Emotic_Dataset_addhookv3(train_context_all[index_train[fold]],
                                                 train_body_all[index_train[fold]],
                                                 train_cat_all[index_train[fold]],
                                                 train_cont_all[index_train[fold]], train_transform,
                                                 mode_unnorm=False,
                                                 hook_faster=load_data_hook_faster('train')[index_train[fold]],
                                                 hook_sentiment=load_data_sentiment('train')[index_train[fold]],
                                                 index_train=index_train[fold],
                                                 mask_context=False,
                                                 data_src='./emotic/emotic_pre/train_face_mask.csv'
                                                 )
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=4,
                                  shuffle=True,
                                  drop_last=True)

        valid_dataset = Emotic_Dataset_addhookv3(val_context, val_body, val_cat, val_cont, train_transform,
                                                 mode_unnorm=False,
                                                 hook_faster=load_data_hook_faster('val'),
                                                 hook_sentiment=load_data_sentiment('val'),
                                                 mask_context=False,
                                                 data_src='./emotic/emotic_pre/val_face_mask.csv'
                                                 )
        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=4,
                                  shuffle=True,
                                  drop_last=True)

        test_dataset = Emotic_Dataset_addhookv3(test_context, test_body, test_cat, test_cont, test_transform,
                                                mode_unnorm=False,
                                                hook_faster=load_data_hook_faster('test'),
                                                hook_sentiment=load_data_sentiment('test'),
                                                mask_context=False,
                                                data_src='./emotic/emotic_pre/test_face_mask.csv'
                                                )
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=4,
                                 shuffle=False)

        # model = Emotic_v16(num_context_features=2048, num_body_features=2048, num_hook=256,
        #                    num_box=4096, fold=fold, weight=True)

        model = torch.load(f'./weight/checkpoints/{fold}_model_f32.pth')
        # import pdb; pdb.set_trace()

        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     model = nn.DataParallel(model)

        model.to(device)

        # df = pd.DataFrame(test_cat)
        # corr = np.array(df.corr())
        criterion1 = DiscreteLoss(weight_type='dynamic', device=device)
        # criterion2 = DependentLoss(alpha=corr)

        # criterion2 = nn.MultiLabelSoftMarginLoss(weight=class_weights)
        # criterion2 = nn.SmoothL1Loss()
        # criterion1 = nn.BCEWithLogitsLoss()
        # criterion3 = FocalLoss(logits=True)

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        start_time = time.time()
        scheduler_steplr = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=1e-4, last_epoch=-1)

        optimizer.zero_grad()
        optimizer.step()
        best_scores = 0
        # import pdb; pdb.set_trace()
        for name, p in model.module.model_body.named_parameters():
            p.requires_grad = False

        for name, p in model.module.model_context.named_parameters():
            p.requires_grad = False

        for name, p in model.module.model_face.named_parameters():
            p.requires_grad = False

        with trange(args.num_epochs, total=args.num_epochs, desc='Epoch') as t:
            for epoch in t:
                if optimizer.param_groups[0]['lr'] < 5e-4:
                    unfreeze(model.module.model_context, percent=0.1)
                    unfreeze(model.module.model_body, percent=0.1)
                    # unfreeze(model.module.model_face, percent=0.2)

                t.set_description('Epoch %i' % epoch)

                cost_list = 0
                scheduler_steplr.step(epoch)

                model.train()
                for batch_idx, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
                    images_context = sample['image_context'].to(device).float()
                    images_body = sample['image_body'].to(device).float()
                    image_face = sample['image_face'].to(device).float()

                    labels_cat = sample['cat_label'].to(device).float()

                    features_object = sample['features_object'].to(device).float()
                    # bbox = sample['bbox'].to(device).float()
                    word_embedding = sample['labels'].to(device).float()

                    sentiment_object = sample['sentiment_objects'].to(device).float()

                    sp_box = sample['pair_box'].to(device).float()

                    bbox = sample['bbox'].to(device).float()
                    weight_area = get_area_feature(bbox, 320, 320)

                    optimizer.zero_grad()

                    pred_cat, _, _ = model(images_context, images_body, features_object, sp_box,
                                           word_embedding, sentiment_object, image_face, weight_area)

                    pred_cat = pred_cat.sigmoid()
                    cat_loss_batch = criterion1(pred_cat, labels_cat)
                    loss = cat_loss_batch
                    cost_list += loss.item()

                    loss.backward()
                    # UPDATE MODEL PARAMETERS
                    optimizer.step()

                    t.set_postfix(Loss=f'{cost_list / (batch_idx + 1):04f}',
                                  Batch=f'{batch_idx + 1:03d}/{len(train_loader):03d}',
                                  Lr=optimizer.param_groups[0]['lr'])

                model.eval()
                with torch.no_grad():

                    cat_preds = []
                    cat_labels = []
                    out_weights = []
                    out_att_mask = []

                    for batch_idx, sample in tqdm(enumerate(test_loader), total=len(test_loader), desc='Test_mode'):
                        images_context = sample['image_context'].to(device).float()
                        images_body = sample['image_body'].to(device).float()
                        image_face = sample['image_face'].to(device).float()

                        labels_cat = sample['cat_label'].to(device).float()

                        features_object = sample['features_object'].to(device).float()
                        # bbox = sample['bbox'].to(device).float()
                        word_embedding = sample['labels'].to(device).float()

                        sentiment_object = sample['sentiment_objects'].to(device).float()

                        sp_box = sample['pair_box'].to(device).float()

                        bbox = sample['bbox'].to(device).float()
                        weight_area = get_area_feature(bbox, 320, 320)

                        pred_cat, impact_weight, att_mask = model(images_context, images_body, features_object, sp_box,
                                                                  word_embedding,
                                                                  sentiment_object, image_face, weight_area)

                        pred_cat = pred_cat.sigmoid()

                        cat_preds.append(pred_cat.detach().cpu())
                        cat_labels.append(labels_cat.detach().cpu())
                        out_weights.append(impact_weight.detach().cpu())
                        out_att_mask.append(att_mask.detach().cpu())

                    cat_preds = np.concatenate(cat_preds, axis=0).transpose()
                    cat_labels = np.concatenate(cat_labels, axis=0).transpose()
                    out_weights = np.concatenate(out_weights, axis=0)
                    out_att_mask = np.concatenate(out_att_mask, axis=0)
                    tqdm.write('\n')
                    ap = test_scikit_ap(cat_preds, cat_labels, ind2cat)

                    if ap.mean() > best_scores:
                        best_scores = ap.mean()
                        os.makedirs('./weight/results', exist_ok=True)
                        np.save(f'./weight/results/{fold}.npy', cat_preds)
                        os.makedirs('./weight/checkpoints', exist_ok=True)
                        # np.save(f'./weight/results/impact_weight_{fold}.npy', out_weights)
                        # np.save(f'./weight/results/out_att_mask_{fold}.npy', out_att_mask)

                        # torch.save(model.module.model_context, f'./weight/checkpoints/{fold}_context.pth')
                        # torch.save(model.module.model_body, f'./weight/checkpoints/{fold}_body.pth')
                        # torch.save(model, f'./weight/checkpoints/{fold}_model_f32_v2.pth')
                    # print('Thresholds= ', thresholds)

                model.train()
                for batch_idx, sample in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                    images_context = sample['image_context'].to(device).float()
                    images_body = sample['image_body'].to(device).float()
                    image_face = sample['image_face'].to(device).float()

                    labels_cat = sample['cat_label'].to(device).float()

                    features_object = sample['features_object'].to(device).float()
                    # bbox = sample['bbox'].to(device).float()
                    word_embedding = sample['labels'].to(device).float()

                    sentiment_object = sample['sentiment_objects'].to(device).float()

                    sp_box = sample['pair_box'].to(device).float()

                    bbox = sample['bbox'].to(device).float()
                    weight_area = get_area_feature(bbox, 320, 320)

                    optimizer.zero_grad()

                    pred_cat, _, _ = model(images_context, images_body, features_object, sp_box,
                                           word_embedding, sentiment_object, image_face, weight_area)

                    pred_cat = pred_cat.sigmoid()
                    cat_loss_batch = criterion1(pred_cat, labels_cat)
                    loss = cat_loss_batch

                    cost_list += loss.item()

                    loss.backward()
                    # UPDATE MODEL PARAMETERS
                    optimizer.step()

                    t.set_postfix(Loss=f'{cost_list / (batch_idx + 1):04f}',
                                  Batch=f'{batch_idx + 1:03d}/{len(valid_loader):03d}',
                                  Lr=optimizer.param_groups[0]['lr'])

                model.eval()
                with torch.no_grad():

                    cat_preds = []
                    cat_labels = []
                    out_weights = []
                    out_att_mask = []

                    for batch_idx, sample in tqdm(enumerate(test_loader), total=len(test_loader), desc='Test_mode'):
                        images_context = sample['image_context'].to(device).float()
                        images_body = sample['image_body'].to(device).float()
                        image_face = sample['image_face'].to(device).float()

                        labels_cat = sample['cat_label'].to(device).float()

                        features_object = sample['features_object'].to(device).float()
                        # bbox = sample['bbox'].to(device).float()
                        word_embedding = sample['labels'].to(device).float()

                        sentiment_object = sample['sentiment_objects'].to(device).float()

                        sp_box = sample['pair_box'].to(device).float()

                        bbox = sample['bbox'].to(device).float()
                        weight_area = get_area_feature(bbox, 320, 320)

                        pred_cat, impact_weight, att_mask = model(images_context, images_body, features_object, sp_box,
                                                                  word_embedding,
                                                                  sentiment_object, image_face, weight_area)

                        pred_cat = pred_cat.sigmoid()

                        cat_preds.append(pred_cat.detach().cpu())
                        cat_labels.append(labels_cat.detach().cpu())
                        out_weights.append(impact_weight.detach().cpu())
                        out_att_mask.append(att_mask.detach().cpu())

                    cat_preds = np.concatenate(cat_preds, axis=0).transpose()
                    cat_labels = np.concatenate(cat_labels, axis=0).transpose()
                    out_weights = np.concatenate(out_weights, axis=0)
                    out_att_mask = np.concatenate(out_att_mask, axis=0)
                    tqdm.write('\n')
                    ap = test_scikit_ap(cat_preds, cat_labels, ind2cat)

                    if ap.mean() > best_scores:
                        best_scores = ap.mean()
                        os.makedirs('./weight/results', exist_ok=True)
                        np.save(f'./weight/results/{fold}.npy', cat_preds)
                        os.makedirs('./weight/checkpoints', exist_ok=True)
                        # np.save(f'./weight/results/impact_weight_{fold}_{mode_list}.npy', out_weights)
                        # np.save(f'./weight/results/out_att_mask_{fold}_{mode_list}.npy', out_att_mask)

                        # torch.save(model.module.model_context, f'./weight/checkpoints/{fold}_context.pth')
                        # torch.save(model.module.model_body, f'./weight/checkpoints/{fold}_body.pth')
                        # torch.save(model, f'./weight/checkpoints/{fold}_model_f32_v2.pth')
                    # print('Thresholds= ', thresholds)

            elapsed = (time.time() - start_time) / 60
            print(f'Total Training Time: {elapsed:.2f} min')

        del model
        torch.cuda.empty_cache()
    ensemble_results(test_cat, ind2cat=ind2cat)
    print('***************************##########################*********************************')
    print('***************************##########################*********************************')
    print('***************************##########################*********************************')
    print('***************************##########################*********************************')
    print('***************************##########################*********************************')


if __name__ == '__main__':
    main()
