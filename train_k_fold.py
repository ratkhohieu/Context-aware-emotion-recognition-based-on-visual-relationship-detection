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
scaler = torch.cuda.amp.GradScaler()


def main():
    seed_everything()
    train_context_all, train_body_all, train_cat_all, train_cont_all = load_data_npy(mode='train')
    # val_context, val_body, val_cat, val_cont = load_data_npy(mode='valid')
    test_context, test_body, test_cat, test_cont = load_data_npy(mode='test')


    # train_context_all = np.concatenate((train_context_all, val_context), axis=0)
    # train_body_all = np.concatenate((train_body_all, val_body), axis=0)
    # train_cat_all = np.concatenate((train_cat_all, val_cat), axis=0)
    # train_cont_all = np.concatenate((train_cont_all, val_cont), axis=0)


    index_train, index_valid = k_fold_data(train_context_all)
    cat2ind, ind2cat, ind2vad = pre_ind2cat()
    root = './emotic'
    # import pdb; pdb.set_trace()
    for fold in range(5):
        print(f'============++++++ FOLD {fold} ++++++======================')
        # train_dataset = Emotic_Dataset_transform_od(train_context_all[index_train[fold]],
        #                                             train_body_all[index_train[fold]],
        #                                             train_cat_all[index_train[fold]],
        #                                             train_cont_all[index_train[fold]],
        #                                             train_transform,
        #                                             mode_unnorm=False,
        #                                             hook_faster=load_data_hook_faster('train'),
        #                                             hook_sentiment=load_data_sentiment('train'),
        #                                             mode='TRAIN')
        #
        # train_loader = DataLoader(dataset=train_dataset,
        #                           batch_size=args.batch_size,
        #                           num_workers=4,
        #                           shuffle=True,
        #                           drop_last=False)
        #
        # # valid_dataset = Emotic_Dataset_addhook(train_context_all[index_valid[fold]],
        # #                                        train_body_all[index_valid[fold]],
        # #                                        train_cat_all[index_valid[fold]],
        # #                                        train_cont_all[index_valid[fold]],
        # #                                        test_transform,
        # #                                        mode_unnorm=False,
        # #                                        hook_faster=load_data_hook_faster('val'),
        # #                                        hook_sentiment=load_data_sentiment('val'))
        # # valid_loader = DataLoader(dataset=valid_dataset,
        # #                           batch_size=args.batch_size,
        # #                           num_workers=4,
        # #                           shuffle=False)
        #
        # test_dataset = Emotic_Dataset_transform_od(test_context, test_body, test_cat, test_cont, test_transform,
        #                                            mode_unnorm=False,
        #                                            hook_faster=load_data_hook_faster('test'),
        #                                            hook_sentiment=load_data_sentiment('test'),
        #                                            mode='TEST')
        # test_loader = DataLoader(dataset=test_dataset,
        #                          batch_size=args.batch_size,
        #                          num_workers=4,
        #                          shuffle=False)
        train_dataset = Emotic_Dataset_addhook(train_context_all[index_train[fold]], train_body_all[index_train[fold]],
                                               train_cat_all[index_train[fold]],
                                               train_cont_all[index_train[fold]], train_transform,
                                               mode_unnorm=False,
                                               hook_faster=load_data_hook_faster('train'),
                                               hook_sentiment=load_data_sentiment('train'))

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=4,
                                  shuffle=True,
                                  drop_last=True)

        test_dataset = Emotic_Dataset_addhook(test_context, test_body, test_cat, test_cont, test_transform,
                                              mode_unnorm=False,
                                              hook_faster=load_data_hook_faster('test'),
                                              hook_sentiment=load_data_sentiment('test'), )
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=4,
                                 shuffle=False)

        model_context, model_body = prep_models(model_dir='proj/debug_exp/models', context_model='resnet50')

        model_hook = Sentiment_MLP(1024, 4, 50, N=10, num_sentiment=4096, embracement_size=128)
        model = Emotic_v4(list(model_context.children())[-1].in_features,
                          list(model_body.children())[-1].in_features,
                          num_hook=128, cont_vad=False)

        print(model_hook)
        print(model)

        model_context = nn.Sequential(*(list(model_context.children())[:-1]))
        model_body = nn.Sequential(*(list(model_body.children())[:-1]))

        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     model = nn.DataParallel(model)

        model_body.to(device)
        model_context.to(device)
        model_hook.to(device)
        model.to(device)

        criterion1 = DiscreteLoss(weight_type='dynamic', device=device)
        # criterion1 = nn.MultiLabelMarginLoss()
        criterion2 = nn.SmoothL1Loss()
        # criterion1 = nn.BCEWithLogitsLoss()
        criterion3 = FocalLoss(logits=True)

        optimizer = torch.optim.SGD(
            (list(model.parameters()) + list(model_context.parameters()) + list(model_body.parameters()) + list(
                model_hook.parameters())),
            lr=args.lr, weight_decay=args.weight_decay)

        start_time = time.time()
        scheduler_steplr = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=0, last_epoch=-1)

        # lmbda = lambda epoch: 0.65 ** epoch
        # # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        optimizer.zero_grad()
        optimizer.step()
        best_scores = 0

        for name, p in model_body.named_parameters():
            p.requires_grad = False

        for name, p in model_context.named_parameters():
            p.requires_grad = False

        with trange(args.num_epochs, total=args.num_epochs, desc='Epoch') as t:
            for epoch in t:
                if optimizer.param_groups[0]['lr'] < 5e-4:
                    unfreeze(model_context, percent=0.3)
                    unfreeze(model_body, percent=0.3)

                t.set_description('Epoch %i' % epoch)
                model.train()
                model_context.train()
                model_body.train()
                model_hook.train()
                # print(epoch, optimizer.param_groups[0]['lr'], '\n')
                cost_list = 0
                scheduler_steplr.step(epoch)

                for batch_idx, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
                    images_context = sample['image_context'].to(device)
                    images_body = sample['image_body'].to(device)

                    labels_cat = sample['cat_label'].to(device).float()
                    labels_cont = sample['cont_label'].to(device)

                    features_object = sample['features_object'].to(device).float()
                    bbox = sample['bbox'].to(device).float()
                    word_embedding = sample['labels'].to(device).float()

                    sentiment_object = sample['sentiment_objects'].to(device).float()
                    # import pdb; pdb.set_trace()
                    optimizer.zero_grad()

                    pred_context = model_context(images_context)
                    pred_body = model_body(images_body)
                    pred_hook = model_hook(features_object, bbox, word_embedding, sentiment_object)
                    pred_cat, context, body, hook = model(pred_context, pred_body, pred_hook)
                    # pred_cat = model(pred_context, pred_body, pred_hook)
                    pred_cat = pred_cat.sigmoid()
                    context = context.sigmoid()
                    body = body.sigmoid()
                    hook = hook.sigmoid()

                    with torch.cuda.amp.autocast():
                        cat_loss_batch = criterion1(pred_cat, labels_cat)
                        cat_loss_context = multitask_selection_loss(context, labels_cat)
                        cat_loss_body = multitask_selection_loss(body, labels_cat)
                        cat_loss_hook = multitask_selection_loss(hook, labels_cat)
                        if math.isnan(cat_loss_hook):
                            import pdb;
                            pdb.set_trace()
                        # import pdb; pdb.set_trace()
                        # cont_loss_batch = criterion2(pred_cont * 10, labels_cont * 10)
                        # if optimizer.param_groups[0]['lr'] < 4e-4:
                        if epoch > 1:
                            loss = cat_loss_batch - (cat_loss_context + cat_loss_body + cat_loss_hook)
                        else:
                            loss = cat_loss_batch - (cat_loss_context + cat_loss_body)

                    cost_list += loss.item()

                    scaler.scale(loss).backward()
                    # UPDATE MODEL PARAMETERS
                    scaler.step(optimizer)

                    # Updates the scale for next iteration
                    scaler.update()

                    t.set_postfix(Loss=f'{cost_list / (batch_idx + 1):04f}',
                                  Batch=f'{batch_idx + 1:03d}/{len(train_loader):03d}',
                                  Lr=optimizer.param_groups[0]['lr'])

                model.eval()
                model_context.eval()
                model_body.eval()
                model_hook.eval()

                with torch.no_grad():
                    # cat_preds = []
                    # cat_labels = []
                    #
                    # for batch_idx, sample in tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid_mode'):
                    #     images_context = sample['image_context'].to(device)
                    #     images_body = sample['image_body'].to(device)
                    #
                    #     labels_cat = sample['cat_label'].to(device).long()
                    #
                    #     features_object = sample['features_object'].to(device).float()
                    #     bbox = sample['bbox'].to(device).float()
                    #     word_embedding = sample['labels'].to(device).float()
                    #
                    #     sentiment_object = sample['sentiment_objects'].to(device).float()
                    #
                    #     pred_context = model_context(images_context)
                    #     pred_body = model_body(images_body)
                    #     pred_hook = model_hook(features_object, bbox, word_embedding, sentiment_object)
                    #     pred_cat = model(pred_context, pred_body, pred_hook)
                    #     pred_cat = pred_cat.sigmoid()
                    #
                    #     cat_preds.append(pred_cat.detach().cpu())
                    #     cat_labels.append(labels_cat.detach().cpu())
                    #
                    # # import pdb; pdb.set_trace();
                    #
                    # cat_preds = np.concatenate(cat_preds, axis=0).transpose()
                    # cat_labels = np.concatenate(cat_labels, axis=0).transpose()
                    #
                    # tqdm.write('\n')
                    # test_scikit_ap(cat_preds, cat_labels, ind2cat)

                    cat_preds = []
                    cat_labels = []
                    cont_preds = []
                    cont_labels = []

                    for batch_idx, sample in tqdm(enumerate(test_loader), total=len(test_loader), desc='Test_mode'):
                        images_context = sample['image_context'].to(device)
                        images_body = sample['image_body'].to(device)

                        labels_cat = sample['cat_label'].to(device).float()
                        labels_cont = sample['cont_label'].to(device)

                        features_object = sample['features_object'].to(device).float()
                        bbox = sample['bbox'].to(device).float()
                        word_embedding = sample['labels'].to(device).float()

                        sentiment_object = sample['sentiment_objects'].to(device).float()

                        pred_context = model_context(images_context)
                        pred_body = model_body(images_body)
                        pred_hook = model_hook(features_object, bbox, word_embedding, sentiment_object)

                        pred_cat, _, _, _ = model(pred_context, pred_body, pred_hook)
                        # pred_cat = model(pred_context, pred_body, pred_hook)
                        pred_cat = pred_cat.sigmoid()

                        cat_preds.append(pred_cat.detach().cpu())
                        cat_labels.append(labels_cat.detach().cpu())
                        # cont_preds.append(pred_cont.detach().cpu())
                        # cont_labels.append(labels_cont.detach().cpu())

                    cat_preds = np.concatenate(cat_preds, axis=0).transpose()
                    cat_labels = np.concatenate(cat_labels, axis=0).transpose()
                    # cont_preds = np.concatenate(cont_preds, axis=0).transpose()
                    # cont_labels = np.concatenate(cont_labels, axis=0).transpose()

                    # import pdb; pdb.set_trace()
                    tqdm.write('\n')
                    ap = test_scikit_ap(cat_preds, cat_labels, ind2cat)
                    # test_vad(cont_preds, cont_labels, ind2vad)
                    if ap.mean() > best_scores:
                        best_scores = ap.mean()
                        os.makedirs('./weight/results', exist_ok=True)
                        np.save(f'./weight/results/{fold}.npy', cat_preds)
                    # print('Thresholds= ', thresholds)

            elapsed = (time.time() - start_time) / 60
            print(f'Total Training Time: {elapsed:.2f} min')

        del model
        torch.cuda.empty_cache()
    ensemble_results(test_cat, ind2cat=ind2cat)


if __name__ == '__main__':
    # path = './emotic/annotations_coco/instances_train2017.json'
    # print(word_embedding_categories(create_cat_labels(path_annotations=path)))
    main()
