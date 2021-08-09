from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from pre_process import *
from utils import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='MuSE Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=26, type=int, help='batch size')
parser.add_argument('--num_epochs', default=10, type=int, help='number epochs')
parser.add_argument('--weight_decay', type=float, default=5e-4)

args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else 'cpu'


# scaler = torch.cuda.amp.GradScaler()


def main():
    # seed_everything()
    train_context_all, train_body_all, train_cat_all, train_cont_all = load_data_npy(mode='train')
    # val_context, val_body, val_cat, val_cont = load_data_npy(mode='valid')
    # test_context, test_body, test_cat, test_cont = load_data_npy(mode='test')

    # train_context_all = np.concatenate((train_context_all, val_context), axis=0)
    # train_body_all = np.concatenate((train_body_all, val_body), axis=0)
    # train_cat_all = np.concatenate((train_cat_all, val_cat), axis=0)
    # train_cont_all = np.concatenate((train_cont_all, val_cont), axis=0)

    index_train = np.arange(len(train_context_all))
    cat2ind, ind2cat, ind2vad = pre_ind2cat()
    root = './emotic'
    # import pdb; pdb.set_trace()

    for fold in range(5):
        # fold = 4

        print(f'============++++++ FOLD {fold} ++++++======================')

        train_dataset = Emotic_Dataset_addhookv3(train_context_all, train_body_all, train_cat_all, train_cont_all,
                                                 train_transform,
                                                 mode_unnorm=False,
                                                 hook_faster=load_data_hook_faster('train'),
                                                 hook_sentiment=load_data_sentiment('train'),
                                                 mask_context=False,
                                                 index_train=index_train,
                                                 data_src='./emotic/emotic_pre/train_face_mask.csv'
                                                 )
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=4,
                                  shuffle=True,
                                  drop_last=True)

        # valid_dataset = Emotic_Dataset_addhookv3(val_context, val_body, val_cat, val_cont, train_transform,
        #                                          mode_unnorm=False,
        #                                          hook_faster=load_data_hook_faster('val'),
        #                                          hook_sentiment=load_data_sentiment('val'),
        #                                          mask_context=False,
        #                                          data_src='./emotic/emotic_pre/val_face_mask.csv'
        #                                          )
        # valid_loader = DataLoader(dataset=valid_dataset,
        #                           batch_size=args.batch_size,
        #                           num_workers=4,
        #                           shuffle=True,
        #                           drop_last=True)
        #
        # test_dataset = Emotic_Dataset_addhookv3(test_context, test_body, test_cat, test_cont, test_transform,
        #                                         mode_unnorm=False,
        #                                         hook_faster=load_data_hook_faster('test'),
        #                                         hook_sentiment=load_data_sentiment('test'),
        #                                         mask_context=False,
        #                                         data_src='./emotic/emotic_pre/test_face_mask.csv'
        #                                         )
        # test_loader = DataLoader(dataset=test_dataset,
        #                          batch_size=args.batch_size,
        #                          num_workers=4,
        #                          shuffle=False)

        # model = Emotic_v16(num_context_features=2048, num_body_features=2048, num_hook=256,
        #                    num_box=4096, fold=fold, weight=True)

        model = torch.load(f'./weight/checkpoints/{fold}_model_f32.pth')
        # import pdb; pdb.set_trace()

        # if torch.cuda.device_count() > 1:
        #     # print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     model = nn.DataParallel(model)

        model.to(device)

        best_scores = 0

        # model.eval()
        # with torch.no_grad():
        #
        #     cat_preds = []
        #     cat_labels = []
        #
        #     for batch_idx, sample in tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Test_mode'):
        #         images_context = sample['image_context'].to(device).float()
        #         images_body = sample['image_body'].to(device).float()
        #         image_face = sample['image_face'].to(device).float()
        #
        #         labels_cat = sample['cat_label'].to(device).float()
        #
        #         features_object = sample['features_object'].to(device).float()
        #
        #         word_embedding = sample['labels'].to(device).float()
        #
        #         sentiment_object = sample['sentiment_objects'].to(device).float()
        #
        #         sp_box = sample['pair_box'].to(device).float()
        #
        #         bbox = sample['bbox'].to(device).float()
        #         weight_area = get_area_feature(bbox, 320, 320)
        #
        #         pred_cat, impact_weight, att_mask = model(images_context, images_body, features_object, sp_box,
        #                                                   word_embedding,
        #                                                   sentiment_object, image_face, weight_area)
        #
        #         pred_cat = pred_cat.sigmoid()
        #
        #         cat_preds.append(pred_cat.detach().cpu())
        #         cat_labels.append(labels_cat.detach().cpu())
        #
        #     cat_preds = np.concatenate(cat_preds, axis=0).transpose()
        #     cat_labels = np.concatenate(cat_labels, axis=0).transpose()
        #
        #     tqdm.write('\n')
        #     ap = test_scikit_ap(cat_preds, cat_labels, ind2cat)
        #
        #     if ap.mean() > best_scores:
        #         best_scores = ap.mean()
        #         os.makedirs('./weight/stacks', exist_ok=True)
        #         np.save(f'./weight/stacks/{fold}_stack_val.npy', cat_preds)

        model.eval()
        with torch.no_grad():
            best_scores = 0
            cat_preds = []
            cat_labels = []

            for batch_idx, sample in tqdm(enumerate(train_loader), total=len(train_loader), desc='Test_mode'):
                images_context = sample['image_context'].to(device).float()
                images_body = sample['image_body'].to(device).float()
                image_face = sample['image_face'].to(device).float()

                labels_cat = sample['cat_label'].to(device).float()

                features_object = sample['features_object'].to(device).float()

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

            cat_preds = np.concatenate(cat_preds, axis=0).transpose()
            cat_labels = np.concatenate(cat_labels, axis=0).transpose()

            tqdm.write('\n')
            ap = test_scikit_ap(cat_preds, cat_labels, ind2cat)

            if ap.mean() > best_scores:
                best_scores = ap.mean()
                os.makedirs('./weight/stacks', exist_ok=True)
                np.save(f'./weight/stacks/{fold}_stack_train.npy', cat_preds)


class StackNet(nn.Module):
    def __init__(self, ):
        super(StackNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(2, 1))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2, 1))

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 1))

        # self.conv4 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 1))

        #
        # self.bn1 = nn.BatchNorm2d(8)
        # self.act1 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(in_channels=8, out_channels=26, kernel_size=(3, 1))
        #
        # self.bn2 = nn.BatchNorm2d(26)
        # self.act2 = nn.ReLU(inplace=True)
        # #         self.conv3 = nn.Conv2d(in_channels=1 , out_channels=8, kernel_size=(3,1))
        self.fc1 = nn.Linear(32 * 26*2, 128)
        self.fc2 = nn.Linear(128, 26)
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.act1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.bn2(x)
        # x = self.act2(x)
        # import pdb; pdb.set_trace()
        x = self.fc1(x.view(x.size(0), -1))
        x = self.fc2(x)
        x = self.act3(x)
        return x


class Stack_Val_Emo(Dataset):
    def __init__(self, data, val_label):
        super(Stack_Val_Emo, self).__init__()
        self.data = torch.tensor(data).unsqueeze(dim=1)
        self.val_label = torch.tensor(val_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.val_label[index]


def train():
    train_list = glob.glob('./weight/stacks/*train.npy')
    data_val = []
    for i in train_list:
        data_val.append(np.load(i))

    test_list = glob.glob('./weight/stacks/*test.npy')
    data_test = []
    for i in test_list:
        data_test.append(np.load(i))

    _, _, train_cat_all, _ = load_data_npy(mode='train')
    _, _, val_cat, _ = load_data_npy(mode='valid')
    _, _, test_cat, _ = load_data_npy(mode='test')
    cat2ind, ind2cat, ind2vad = pre_ind2cat()

    train_dataset = Stack_Val_Emo(np.array(data_val).transpose(2, 0, 1), train_cat_all)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=26,
                              num_workers=4,
                              shuffle=True)

    test_dataset = Stack_Val_Emo(np.array(data_test).transpose(2, 0, 1), test_cat)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=26,
                             num_workers=4,
                             shuffle=False)

    model = StackNet()
    model.to(device)
    criterion1 = DiscreteLoss(weight_type='dynamic', device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler_steplr = CosineAnnealingLR(optimizer, 1000, eta_min=1e-4, last_epoch=-1)
    optimizer.zero_grad()
    optimizer.step()
    with trange(1000, total=1000, desc='Epoch') as t:
        for epoch in range(1000):
            out_put = []
            cost_list = 0
            model.train()
            scheduler_steplr.step(epoch)
            for batch_data, batch_label in train_loader:
                optimizer.zero_grad()
                batch_data = batch_data.to(device)
                batch_label = batch_label.to(device)
                pred_cat = model(batch_data)
                loss = criterion1(pred_cat, batch_label)
                loss.backward()
                # UPDATE MODEL PARAMETERS
                optimizer.step()
                cost_list += loss.item()
            t.set_postfix(Loss=f'{cost_list / (len(train_loader) + 1):04f}')

            model.eval()
            for batch_data, batch_label in test_loader:
                batch_data = batch_data.to(device)
                # batch_label = batch_label.to(device)
                pred_cat = model(batch_data)
                out_put.append(pred_cat.detach().cpu())
            out_put = np.concatenate(out_put, axis=0).transpose()
            # import pdb; pdb.set_trace()
            tqdm.write('\n')
            ap = test_scikit_ap(out_put, test_cat.transpose(), ind2cat)


if __name__ == '__main__':
    # main()
    train()
