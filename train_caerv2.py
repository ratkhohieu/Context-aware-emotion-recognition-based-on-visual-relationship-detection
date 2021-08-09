import time

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from pre_process import *
from utils import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='MuSE Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=12, type=int, help='batch size')
parser.add_argument('--num_epochs', default=20, type=int, help='number epochs')
parser.add_argument('--weight_decay', type=float, default=5e-4)

args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else 'cpu'
scaler = torch.cuda.amp.GradScaler()


def main():
    # seed_everything()
    df = pd.read_csv('/media/sven/HUNG/CAER/extracted_feature/train.csv')
    index_train, index_valid = k_fold_data(df, number_fold=5)
    for fold in range(5):
        train_dataset = Caer_Dataset_addhook(path_df='/media/sven/HUNG/CAER/extracted_feature/train.csv',
                                             transform=train_transform,
                                             hook_faster='./train_hook.csv',
                                             hook_sentiment='./train_hook_sentiment.csv',
                                             index_train=index_train[fold])
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=4,
                                  shuffle=True,
                                  drop_last=True)

        test_dataset = Caer_Dataset_addhook(path_df='/media/sven/HUNG/CAER/extracted_feature/test.csv',
                                            transform=test_transform,
                                            hook_faster='./test_hook.csv',
                                            hook_sentiment='./test_hook_sentiment.csv'
                                            )
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=4,
                                 shuffle=False)

        # model = CaerNet_v2(num_context_features=2048, num_body_features=2048, num_hook=256,
        #                    num_box=4096, fold=4, weight=True)
        model = torch.load(f'./weight/checkpoints/{fold}_caer_model.pth')
        if torch.cuda.device_count() > 1:
            # print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        model.to(device)

        # criterion1 = DiscreteLoss(weight_type='dynamic', device=device)
        criterion1 = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        start_time = time.time()
        scheduler_steplr = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=0, last_epoch=-1)

        optimizer.zero_grad()
        optimizer.step()
        best_scores = 0
        # import pdb; pdb.set_trace()
        # for name, p in model.model_body.named_parameters():
        #     p.requires_grad = False
        #
        # for name, p in model.model_context.named_parameters():
        #     p.requires_grad = False

        with trange(args.num_epochs, total=args.num_epochs, desc='Epoch') as t:
            for epoch in t:

                t.set_description('Epoch %i' % epoch)
                model.train()

                cost_list = 0
                scheduler_steplr.step(epoch)

                for batch_idx, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
                    images_context = sample['image_context'].to(device).float()
                    images_body = sample['image_body'].to(device).float()
                    image_face = sample['image_face'].to(device).float()

                    labels_cat = sample['cat_label'].to(device).long()
                    features_object = sample['features_object'].to(device).float()
                    # bbox = sample['bbox'].to(device).float()
                    word_embedding = sample['labels'].to(device).float()

                    sentiment_object = sample['sentiment_objects'].to(device).float()

                    sp_box = sample['pair_box'].to(device).float()

                    bbox = sample['bbox'].to(device).float()
                    weight_area = get_area_feature(bbox, 320, 320).to(device).float()

                    optimizer.zero_grad()

                    pred_cat, _, _ = model(images_context, images_body, features_object, sp_box,
                                           word_embedding, sentiment_object, image_face, weight_area)

                    with torch.cuda.amp.autocast():
                        cat_loss_batch = criterion1(pred_cat, labels_cat)

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
                    correct = 0
                    total = 0
                    for batch_idx, sample in tqdm(enumerate(test_loader), total=len(test_loader), desc='Test_mode'):
                        images_context = sample['image_context'].to(device).float()
                        images_body = sample['image_body'].to(device).float()
                        image_face = sample['image_face'].to(device).float()

                        labels_cat = sample['cat_label'].to(device).long()

                        features_object = sample['features_object'].to(device).float()
                        # bbox = sample['bbox'].to(device).float()
                        word_embedding = sample['labels'].to(device).float()

                        sentiment_object = sample['sentiment_objects'].to(device).float()

                        sp_box = sample['pair_box'].to(device).float()

                        bbox = sample['bbox'].to(device).float()
                        weight_area = get_area_feature(bbox, 320, 320).to(device).float()

                        pred_cat, _, _ = model(images_context, images_body, features_object, sp_box,
                                               word_embedding,
                                               sentiment_object, image_face, weight_area)

                        pred_cat = F.softmax(pred_cat)
                        _, predicted = pred_cat.max(1)

                        correct += predicted.eq(labels_cat).sum().item()
                        total += labels_cat.shape[0]

                        out = pred_cat.detach().cpu().numpy()
                        cat_preds.append(out)
                    print(f'Acc : {100. * correct / total}')

                    if (correct / total) > best_scores:
                        best_scores = correct / total
                        os.makedirs('./weight/results_caer', exist_ok=True)
                        np.save(f'./weight/results_caer/{fold}_caer.npy', cat_preds)
                        # os.makedirs('./weight/checkpoints', exist_ok=True)
                        # np.save('impact_weight.npy', out_weights)
                        #
                        # torch.save(model.model_body, f'./weight/checkpoints/caer_model_body.pth')
                        # torch.save(model.model_context, f'./weight/checkpoints/caer_model_context.pth')
                        torch.save(model, f'./weight/checkpoints/{fold}_caer_model.pth')
                    # print('Thresholds= ', thresholds)

            elapsed = (time.time() - start_time) / 60
            print(f'Total Training Time: {elapsed:.2f} min')

        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
