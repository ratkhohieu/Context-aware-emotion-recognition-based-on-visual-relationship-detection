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


def mainv1():
    seed_everything()
    train_context, train_body, train_cat, train_cont = load_data_npy(mode='train')
    val_context, val_body, val_cat, val_cont = load_data_npy(mode='valid')
    test_context, test_body, test_cat, test_cont = load_data_npy(mode='test')

    root = './emotic'

    train_transform = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(size=(224, 224)),
                                          # transforms.RandomHorizontalFlip(),
                                          # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                          transforms.ToTensor()])
    train_dataset = Emotic_PreDataset(train_context, train_body, train_cat, train_cont, train_transform,
                                      root=root,
                                      data_src='emotic_pre/train.csv', mode_unnorm=True)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=False,
                              drop_last=False)

    valid_dataset = Emotic_PreDataset(val_context, val_body, val_cat, val_cont, train_transform,
                                      root=root,
                                      data_src='emotic_pre/val.csv', mode_unnorm=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=False)

    test_dataset = Emotic_PreDataset(test_context, test_body, test_cat, test_cont, train_transform,
                                     root=root,
                                     data_src='emotic_pre/test.csv', mode_unnorm=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             num_workers=4,
                             shuffle=False)

    # load a model pre-trained pre-trained on COCO
    model = load_sentiment_model()
    # model = nn.Sequential(*list(model.children())[:-1])
    # import pdb; pdb.set_trace()
    model.to(device)
    model.eval()

    results = []
    for k, v in {'train': train_loader, 'test': test_loader, 'val': valid_loader}.items():
        for batch_idx, sample in tqdm(enumerate(v), total=len(v)):
            sentiment_object = sample['image_body'].to(device)
            hook = model.fc7_1.register_forward_hook(
                lambda self, input, output: results.append(output.detach().cpu().numpy()))
            res = model(sentiment_object)
            hook.remove()

        path_npy = os.path.join(root, 'visual_sentiment')
        os.makedirs(path_npy, exist_ok=True)
        np.save(os.path.join(path_npy, f'{k}_sentiment_objects.npy'), results)


def mainv2():
    seed_everything()
    train_context, train_body, train_cat, train_cont = load_data_npy(mode='train')
    val_context, val_body, val_cat, val_cont = load_data_npy(mode='valid')
    test_context, test_body, test_cat, test_cont = load_data_npy(mode='test')

    root = './emotic'

    train_transform = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(size=(224, 224)),
                                          transforms.ToTensor()])

    train_dataset = Emotic_Sentiment(train_context, train_body, train_transform,
                                     mode_unnorm=False,
                                     hook_faster=load_data_hook_faster('train'))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=False,
                              drop_last=False)

    valid_dataset = Emotic_Sentiment(val_context, val_body, train_transform,
                                     mode_unnorm=False,
                                     hook_faster=load_data_hook_faster('val'))
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=False)

    test_dataset = Emotic_Sentiment(test_context, test_body, train_transform,
                                    mode_unnorm=False,
                                    hook_faster=load_data_hook_faster('test'))
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             num_workers=4,
                             shuffle=False)

    # load a model pre-trained pre-trained on COCO
    model = load_sentiment_model()

    model.to(device)
    model.eval()

    for k, v in {'train': train_loader, 'test': test_loader, 'val': valid_loader}.items():
        results = []
        sentiment_elements = []
        for batch_idx, sample in tqdm(enumerate(v), total=len(v)):
            sentiment_objects = sample['objects']
            for element in sentiment_objects:
                # import pdb; pdb.set_trace()
                element = element.to(device)
                # hook = model.fc7_1.register_forward_hook(
                #     lambda self, input, output: results.append(output.detach().cpu().numpy()))
                res = model(element)
                results.append(res.detach().cpu().numpy())
                # hook.remove()
            sentiment_elements.append({'objects': np.array(results)})
            results = []
        path_npy = os.path.join(root, 'visual_sentiment')
        os.makedirs(path_npy, exist_ok=True)
        np.save(os.path.join(path_npy, f'{k}_sentiment_final_out.npy'), sentiment_elements)
        print(len(sentiment_elements))

def mainv3():
    seed_everything()
    train_transform = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(size=(224, 224)),
                                          transforms.ToTensor()])

    root = '/media/sven/HUNG/CAER/extracted_feature/'
    train_dataset = Emotic_Sentimentv2(path_df='./train_hook.csv', transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=False,
                              drop_last=False)

    test_dataset = Emotic_Sentimentv2(path_df='./test_hook.csv', transform=train_transform)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             num_workers=4,
                             shuffle=False)

    model = load_sentiment_model()
    model.to(device)
    model.eval()

    for k, v in {'train': train_loader, 'test': test_loader}.items():

        # results = []
        df = pd.read_csv(f'/media/sven/HUNG/CAER/extracted_feature/{k}.csv')
        sentiment_elements = []

        for batch_idx, sample in tqdm(enumerate(v), total=len(v)):
            sentiment_objects = sample['objects']
            results = []
            for element in sentiment_objects:
                element = element.to(device)
                hook = model.fc7_1.register_forward_hook(
                    lambda self, input, output: results.append(output.detach().cpu().numpy()))
                res = model(element)
                hook.remove()

            name_npy_file = df.iloc[batch_idx]['image']
            emotion = df.iloc[batch_idx]['emotion']
            path_npy = os.path.join(root, 'hook_sentiment')
            os.makedirs(path_npy, exist_ok=True)
            np.save(os.path.join(path_npy, f'{emotion}_{name_npy_file}.npy'), results)
            sentiment_elements.append(os.path.join(path_npy, f'{emotion}_{name_npy_file}.npy'))

        df['hook_sentiment'] = sentiment_elements
        df.to_csv(f'{k}_hook_sentiment.csv')


if __name__ == '__main__':
    mainv3()
