import time
import warnings

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from warmup_scheduler import GradualWarmupScheduler

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
    train_context, train_body, train_cat, train_cont = load_data_npy(mode='train')
    val_context, val_body, val_cat, val_cont = load_data_npy(mode='valid')
    test_context, test_body, test_cat, test_cont = load_data_npy(mode='test')

    cat2ind, ind2cat, ind2vad = pre_ind2cat()
    # root = '/home/sven/Documents/Emotic/emotic'
    root = './emotic'
    train_dataset = Emotic_PreDataset(train_context, train_body, train_cat, train_cont, train_transform,
                                      root=root,
                                      data_src='emotic_pre/train.csv')

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=True,
                              drop_last=False)

    valid_dataset = Emotic_PreDataset(val_context, val_body, val_cat, val_cont, test_transform,
                                      root=root,
                                      data_src='emotic_pre/val.csv')
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=False)

    test_dataset = Emotic_PreDataset(test_context, test_body, test_cat, test_cont, test_transform,
                                     root=root,
                                     data_src='emotic_pre/test.csv')
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             num_workers=4,
                             shuffle=False)

    model_context, model_body = prep_models(model_dir='proj/debug_exp/models', context_model='resnet50')
    model_depth = model_body
    # model_face = load_face_model()
    model = Emotic_depth(
        list(model_context.children())[-1].in_features,
        list(model_body.children())[-1].in_features,
        list(model_depth.children())[-1].in_features,
    )
    model_context = nn.Sequential(*(list(model_context.children())[:-1]))
    model_body = nn.Sequential(*(list(model_body.children())[:-1]))
    model_depth = model_body

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    model_body.to(device)
    model_context.to(device)
    model.to(device)
    model_depth.to(device)
    criterion1 = DiscreteLoss(weight_type='dynamic', device=device)
    criterion2 = ContinuousLoss_SL1()
    # criterion1 = nn.BCEWithLogitsLoss()
    # criterion1 = FocalLoss()

    optimizer = torch.optim.Adam(
        (list(model.parameters()) + list(model_context.parameters())
         + list(model_body.parameters())) + list(model_depth.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)

    start_time = time.time()

    scheduler_steplr = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=0, last_epoch=-1)


    # lmbda = lambda epoch: 0.65 ** epoch
    # # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    optimizer.zero_grad()
    optimizer.step()
    best_scores = 10

    for name, p in model_body.named_parameters():
        p.requires_grad = False

    for name, p in model_context.named_parameters():
        p.requires_grad = False

    for name, p in model_depth.named_parameters():
        p.requires_grad = False

    with trange(args.num_epochs, total=args.num_epochs, desc='Epoch') as t:
        for epoch in t:
            if optimizer.param_groups[0]['lr'] < 5e-4:
                unfreeze(model_context, percent=0.1)
                unfreeze(model_body, percent=0.1)
                unfreeze(model_depth, percent=0.1)

            t.set_description('Epoch %i' % epoch)
            model.train()
            model_context.train()
            model_body.train()
            model_depth.train()
            # print(epoch, optimizer.param_groups[0]['lr'], '\n')
            cost_list = 0

            scheduler_steplr.step(epoch)

            for batch_idx, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
                images_context = sample['image_context'].to(device)
                images_body = sample['image_body'].to(device)
                images_depth = sample['image_depth'].to(device)

                labels_cat = sample['cat_label'].to(device)
                labels_cont = sample['cont_label'].to(device)
                # import pdb; pdb.set_trace()
                optimizer.zero_grad()

                pred_context = model_context(images_context)
                pred_body = model_body(images_body)
                pred_depth = model_depth(images_depth)
                pred_cat, pred_cont = model(pred_context, pred_body, pred_depth)
                # pred_cat = pred_cat.sigmoid()

                with torch.cuda.amp.autocast():
                    cat_loss_batch = criterion1(pred_cat, labels_cat)
                    # cont_loss_batch = criterion2(pred_cont * 10, labels_cont * 10)
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
            model_context.eval()
            model_body.eval()
            model_depth.eval()
            with torch.no_grad():
                cat_preds = []
                cat_labels = []
                cont_preds = []
                cont_labels = []
                for batch_idx, sample in tqdm(enumerate(valid_loader),
                                              total=len(valid_loader), desc='Valid_mode'):
                    images_context = sample['image_context'].to(device)
                    images_body = sample['image_body'].to(device)
                    images_depth = sample['image_depth'].to(device)
                    labels_cat = sample['cat_label'].to(device)
                    labels_cont = sample['cont_label'].to(device)

                    pred_context = model_context(images_context)
                    pred_body = model_body(images_body)
                    pred_depth = model_depth(images_depth)
                    pred_cat, pred_cont = model(pred_context, pred_body, pred_depth)
                    # pred_cat = pred_cat.sigmoid()

                    cat_preds.append(pred_cat.detach().cpu())
                    cat_labels.append(labels_cat.detach().cpu())
                    cont_preds.append(pred_cont.detach().cpu())
                    cont_labels.append(labels_cont.detach().cpu())
                # import pdb; pdb.set_trace();
                cat_preds = np.concatenate(cat_preds, axis=0).transpose()
                cat_labels = np.concatenate(cat_labels, axis=0).transpose()
                cont_preds = (np.concatenate(cont_preds, axis=0) * 10).transpose()
                cont_labels = (np.concatenate(cont_labels, axis=0) * 10).transpose()

                tqdm.write('\n')
                test_scikit_ap(cat_preds, cat_labels, ind2cat)
                test_vad(cont_preds, cont_labels, ind2vad)
                # thresholds = get_thresholds(cat_preds, cat_labels)

                cat_preds = []
                cat_labels = []
                cont_preds = []
                cont_labels = []

                for batch_idx, sample in tqdm(enumerate(test_loader),
                                              total=len(test_loader), desc='Test_mode'):
                    images_context = sample['image_context'].to(device)
                    images_body = sample['image_body'].to(device)
                    images_depth = sample['image_depth'].to(device)
                    labels_cat = sample['cat_label'].to(device)
                    labels_cont = sample['cont_label'].to(device)

                    pred_context = model_context(images_context)
                    pred_body = model_body(images_body)
                    pred_depth = model_depth(images_depth)
                    pred_cat, pred_cont = model(pred_context, pred_body, pred_depth)
                    # pred_cat = pred_cat.sigmoid()

                    cat_preds.append(pred_cat.detach().cpu())
                    cat_labels.append(labels_cat.detach().cpu())
                    cont_preds.append(pred_cont.detach().cpu())
                    cont_labels.append(labels_cont.detach().cpu())

                cat_preds = np.concatenate(cat_preds, axis=0).transpose()
                cat_labels = np.concatenate(cat_labels, axis=0).transpose()
                cont_preds = np.concatenate(cont_preds, axis=0).transpose()
                cont_labels = np.concatenate(cont_labels, axis=0).transpose()

                tqdm.write('\n')
                test_scikit_ap(cat_preds, cat_labels, ind2cat)
                test_vad(cont_preds, cont_labels, ind2vad)

                # print('Thresholds= ', thresholds)

        elapsed = (time.time() - start_time) / 60
        print(f'Total Training Time: {elapsed:.2f} min')

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
