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

        train_dataset = Emotic_Dataset_addhook(train_context_all[index_train[fold]], train_body_all[index_train[fold]],
                                               train_cat_all[index_train[fold]],
                                               train_cont_all[index_train[fold]], train_transform,
                                               mode_unnorm=False,
                                               hook_faster=load_data_hook_faster('train'),
                                               hook_sentiment=load_data_sentiment('train'),
                                               hook_sentiment_output=load_data_output_sentiment('train'))
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=4,
                                  shuffle=True,
                                  drop_last=True)

        test_dataset = Emotic_Dataset_addhook(test_context, test_body, test_cat, test_cont, test_transform,
                                              mode_unnorm=False,
                                              hook_faster=load_data_hook_faster('test'),
                                              hook_sentiment=load_data_sentiment('test'),
                                              hook_sentiment_output=load_data_output_sentiment('test'))
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=4,
                                 shuffle=False)

        model = Emotic_v8(num_context_features=2048, num_body_features=2048, num_hook=128, cont_vad=False,
                          embracement_size=128, num_box=4)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        model.to(device)

        criterion1 = DiscreteLoss(weight_type='dynamic', device=device)
        # criterion1 = nn.MultiLabelSoftMarginLoss()
        criterion2 = nn.SmoothL1Loss()
        # criterion1 = nn.BCEWithLogitsLoss()
        # criterion3 = FocalLoss(logits=True)

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        start_time = time.time()
        scheduler_steplr = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=0, last_epoch=-1)

        optimizer.zero_grad()
        optimizer.step()
        best_scores = 0
        # import pdb; pdb.set_trace()
        for name, p in model.module.model_body.named_parameters():
            p.requires_grad = False

        for name, p in model.module.model_context.named_parameters():
            p.requires_grad = False

        with trange(args.num_epochs, total=args.num_epochs, desc='Epoch') as t:
            for epoch in t:
                if optimizer.param_groups[0]['lr'] < 5e-4:
                    unfreeze(model.module.model_context, percent=1)
                    unfreeze(model.module.model_body, percent=1)

                t.set_description('Epoch %i' % epoch)
                model.train()

                cost_list = 0
                scheduler_steplr.step(epoch)

                for batch_idx, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
                    images_context = sample['image_context'].to(device)
                    images_body = sample['image_body'].to(device)

                    labels_cat = sample['cat_label'].to(device).float()

                    features_object = sample['features_object'].to(device).float()
                    bbox = sample['bbox'].to(device).float()
                    word_embedding = sample['labels'].to(device).float()

                    sentiment_object = sample['sentiment_objects'].to(device).float()
                    sentiment_output = sample['sentiment_objects_output'].to(device).float()
                    optimizer.zero_grad()

                    pred_cat, context, body, hook, _, _ = model(images_context, images_body,
                                                                features_object, bbox,
                                                                word_embedding, sentiment_object, sentiment_output)

                    pred_cat = pred_cat.sigmoid()
                    context = context.sigmoid()
                    body = body.sigmoid()
                    hook = hook.sigmoid()

                    with torch.cuda.amp.autocast():
                        cat_loss_batch = criterion1(pred_cat, labels_cat)
                        cat_loss_context = multitask_selection_loss(context, labels_cat)
                        cat_loss_body = multitask_selection_loss(body, labels_cat)
                        cat_loss_hook = multitask_selection_loss(hook, labels_cat)

                        if epoch > 3:
                            if math.isnan(cat_loss_hook):
                                import pdb;
                                pdb.set_trace()

                            loss = cat_loss_batch - (cat_loss_context + cat_loss_body + cat_loss_hook)
                        else:
                            loss = cat_loss_batch

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

                with torch.no_grad():

                    cat_preds = []
                    cat_labels = []
                    out_weights = []
                    out_sentiments = []
                    for batch_idx, sample in tqdm(enumerate(test_loader), total=len(test_loader), desc='Test_mode'):
                        images_context = sample['image_context'].to(device)
                        images_body = sample['image_body'].to(device)

                        labels_cat = sample['cat_label'].to(device).float()

                        features_object = sample['features_object'].to(device).float()
                        bbox = sample['bbox'].to(device).float()
                        word_embedding = sample['labels'].to(device).float()

                        sentiment_object = sample['sentiment_objects'].to(device).float()
                        sentiment_output = sample['sentiment_objects_output'].to(device).float()
                        pred_cat, _, _, _, out_weight, out_sentiment = model(images_context, images_body,
                                                                             features_object, bbox, word_embedding,
                                                                             sentiment_object, sentiment_output)

                        pred_cat = pred_cat.sigmoid()

                        cat_preds.append(pred_cat.detach().cpu())
                        cat_labels.append(labels_cat.detach().cpu())
                        out_weights.append(out_weight.detach().cpu())
                        out_sentiments.append(out_sentiment.detach().cpu())
                    # import pdb; pdb.set_trace()
                    cat_preds = np.concatenate(cat_preds, axis=0).transpose()
                    cat_labels = np.concatenate(cat_labels, axis=0).transpose()
                    out_weights = np.concatenate(out_weights, axis=0)
                    out_sentiments = np.concatenate(out_sentiments, axis=0)

                    tqdm.write('\n')
                    ap = test_scikit_ap(cat_preds, cat_labels, ind2cat)

                    if ap.mean() > best_scores:
                        best_scores = ap.mean()
                        # os.makedirs('./weight/results', exist_ok=True)
                        # np.save(f'./weight/results/{fold}.npy', cat_preds)
                        os.makedirs('./weight/checkpoints', exist_ok=True)
                        torch.save(model.state_dict(), f'./weight/checkpoints/{fold}_v8_sentiment.pth')
                        np.save('impact_weight.npy', out_weights)
                        np.save('sentiment_out.npy', out_sentiments)
                    # print('Thresholds= ', thresholds)

            elapsed = (time.time() - start_time) / 60
            print(f'Total Training Time: {elapsed:.2f} min')

        del model
        torch.cuda.empty_cache()
    ensemble_results(test_cat, ind2cat=ind2cat)


if __name__ == '__main__':
    main()
