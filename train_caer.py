import time

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from models import *
from pre_process import *
from utils import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='MuSE Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--batch_size', default=12, type=int, help='batch size')
parser.add_argument('--num_epochs', default=20, type=int, help='number epochs')
parser.add_argument('--weight_decay', type=float, default=5e-4)

args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else 'cpu'
scaler = torch.cuda.amp.GradScaler()


def main():
    # seed_everything()

    train_dataset = Caer_Dataset(path_df='/media/sven/HUNG/CAER/extracted_feature/train.csv', transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=True,
                              drop_last=True)

    test_dataset = Caer_Dataset(path_df='/media/sven/HUNG/CAER/extracted_feature/test.csv', transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             num_workers=4,
                             shuffle=False)

    # model = CaerNet()
    model = torch.load(f'./weight/checkpoints/caer_model.pth')

    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    # criterion1 = DiscreteLoss(weight_type='dynamic', device=device)
    criterion1 = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_time = time.time()
    scheduler_steplr = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=1e-4, last_epoch=-1)

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
            # if optimizer.param_groups[0]['lr'] < 5e-4:
            #     unfreeze(model.model_context, percent=1)
            #     unfreeze(model.model_body, percent=1)

            t.set_description('Epoch %i' % epoch)
            model.train()

            cost_list = 0
            scheduler_steplr.step(epoch)

            for batch_idx, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
                images_context = sample['image_context'].to(device).float()
                images_body = sample['image_body'].to(device).float()
                image_face = sample['image_face'].to(device).float()

                labels_cat = sample['label'].to(device).long()

                optimizer.zero_grad()

                pred_cat = model(images_context, images_body)

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

                    labels_cat = sample['label'].to(device).float()

                    pred_cat = model(images_context, images_body)

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
                    np.save(f'./weight/results_caer/caer.npy', cat_preds)
                    # os.makedirs('./weight/checkpoints', exist_ok=True)
                    # np.save('impact_weight.npy', out_weights)
                    #
                    # torch.save(model.model_body, f'./weight/checkpoints/caer_model_body.pth')
                    # torch.save(model.model_context, f'./weight/checkpoints/caer_model_context.pth')
                    torch.save(model, f'./weight/checkpoints/caer_model.pth')

                # print('Thresholds= ', thresholds)

        elapsed = (time.time() - start_time) / 60
        print(f'Total Training Time: {elapsed:.2f} min')

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
