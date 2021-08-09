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
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--num_epochs', default=20, type=int, help='number epochs')
parser.add_argument('--weight_decay', type=float, default=5e-4)

args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else 'cpu'
scaler = torch.cuda.amp.GradScaler()


def main():
    seed_everything()
    data_src = "./emotic/emotic_pre"
    train_cat = np.load(os.path.join(data_src, 'train_cat_256.npy'))
    test_cat = np.load(os.path.join(data_src, 'test_cat_256.npy'))

    cat2ind, ind2cat, ind2vad = pre_ind2cat()
    # root = '/home/sven/Documents/Emotic/emotic'
    train_dataset = Emotic_Face_Dataset(train_cat, train_transform, data_src='./emotic/emotic_pre/train_face.csv')

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=True,
                              drop_last=True)

    test_dataset = Emotic_Face_Dataset(test_cat, test_transform, data_src='./emotic/emotic_pre/test_face.csv')
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             num_workers=4,
                             shuffle=False)

    model = Emotic_face()

    model.to(device)

    criterion1 = DiscreteLoss(weight_type='dynamic', device=device)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr, weight_decay=args.weight_decay)

    start_time = time.time()

    scheduler_steplr = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=0, last_epoch=-1)

    optimizer.zero_grad()
    optimizer.step()
    best_scores = 10

    # for name, p in model.model_face.named_parameters():
    #     p.requires_grad = False

    with trange(args.num_epochs, total=args.num_epochs, desc='Epoch') as t:
        for epoch in t:
            # if optimizer.param_groups[0]['lr'] < 5e-4:
            #     unfreeze(model.model_face, percent=1)

            t.set_description('Epoch %i' % epoch)
            model.train()

            # print(epoch, optimizer.param_groups[0]['lr'], '\n')
            cost_list = 0

            scheduler_steplr.step(epoch)

            for batch_idx, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
                images_face = sample['image_face'].to(device)

                labels_cat = sample['cat_label'].to(device)

                # import pdb; pdb.set_trace()
                optimizer.zero_grad()
                pred_cat = model(images_face)

                pred_cat = pred_cat.sigmoid()

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
                cat_labels = []

                for batch_idx, sample in tqdm(enumerate(test_loader),
                                              total=len(test_loader), desc='Test_mode'):
                    images_face = sample['image_face'].to(device)
                    labels_cat = sample['cat_label'].to(device)

                    pred_cat = model(images_face)

                    pred_cat = pred_cat.sigmoid()

                    cat_preds.append(pred_cat.detach().cpu())
                    cat_labels.append(labels_cat.detach().cpu())

                cat_preds = np.concatenate(cat_preds, axis=0).transpose()
                cat_labels = np.concatenate(cat_labels, axis=0).transpose()

                tqdm.write('\n')
                test_scikit_ap(cat_preds, cat_labels, ind2cat)

                # print('Thresholds= ', thresholds)

        elapsed = (time.time() - start_time) / 60
        print(f'Total Training Time: {elapsed:.2f} min')

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
