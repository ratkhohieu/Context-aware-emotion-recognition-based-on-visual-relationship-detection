# PlacesCNN to predict the scene category, attribute, and class activation map in a single pass
# by Bolei Zhou, sep 2, 2017
# updated, making it compatible to pytorch 1.x in a hacky way


from torch.utils.data import DataLoader
from torchvision import transforms as trn
from tqdm import tqdm

from pre_process.dataloader import *
from torch.nn import functional as F


# hacky way to deal with the Pytorch 1.0 update
def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module


def load_labels():
    # prepare all the labels
    # scene category relevant
    file_name_category = 'places365/categories_places365.txt'
    if not os.access(file_name_category, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = 'places365/IO_places365.txt'
    if not os.access(file_name_IO, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
        os.system('wget ' + synset_url)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) - 1)  # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = 'places365/labels_sunattribute.txt'
    if not os.access(file_name_attribute, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
        os.system('wget ' + synset_url)
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = 'places365/W_sceneattribute_wideresnet18.npy'
    if not os.access(file_name_W, os.W_OK):
        synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
        os.system('wget ' + synset_url)
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute


def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def returnTF():
    # load the image transformer
    tf = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf


def load_model():
    # this model has a last conv feature map as 14x14

    model_file = 'places365/wideresnet18_places365.pth.tar'
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
        os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')

    from places365 import wideresnet
    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)

    model.eval()

    # the following is deprecated, everything is migrated to python36

    ## if you encounter the UnicodeDecodeError when use python3 to load the model, add the following line will fix it. Thanks to @soravux
    # from functools import partial
    # import pickle
    # pickle.load = partial(pickle.load, encoding="latin1")
    # pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    # model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)

    model.eval()
    # hook the feature extractor
    features_names = ['layer4', 'avgpool']  # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    return model


# load the labels
classes, labels_IO, labels_attribute, W_attribute = load_labels()
classes = [x.replace('/outdoor', '').replace('/indoor', '').replace('/', ' ').replace('_', ' ').replace('-', ' ') for x
           in classes]
# load the model

model = load_model()
device = "cuda:0" if torch.cuda.is_available() else 'cpu'
model.to(device)
# load the transformer
tf = returnTF()  # image transformer

# get the softmax weight
params = list(model.parameters())
weight_softmax = params[-2].data
weight_softmax[weight_softmax < 0] = 0

# load the test image

# img = Image.open('test.jpg')
# input_img = V(tf(img).unsqueeze(0))

seed_everything()
train_context, train_body, train_cat, train_cont = load_data_npy(mode='train')
val_context, val_body, val_cat, val_cont = load_data_npy(mode='valid')
test_context, test_body, test_cat, test_cont = load_data_npy(mode='test')

cat2ind, ind2cat, ind2vad = pre_ind2cat()
# root = '/home/sven/Documents/Emotic/emotic'
root = './emotic'

train_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize(size=(224, 224)),
                                      # transforms.RandomHorizontalFlip(),
                                      # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                      transforms.ToTensor()])

test_transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor()])

train_dataset = Emotic_PreDataset(train_context, train_body, train_cat, train_cont, train_transform,
                                  root=root,
                                  data_src='emotic_pre/train.csv')

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=1,
                          num_workers=4,
                          shuffle=True,
                          drop_last=False)

valid_dataset = Emotic_PreDataset(val_context, val_body, val_cat, val_cont, test_transform,
                                  root=root,
                                  data_src='emotic_pre/val.csv')
valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=1,
                          num_workers=4,
                          shuffle=False)

test_dataset = Emotic_PreDataset(test_context, test_body, test_cat, test_cont, test_transform,
                                 root=root,
                                 data_src='emotic_pre/test.csv')
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=1,
                         num_workers=4,
                         shuffle=False)

for k, v in {'train': train_loader, 'test': test_loader, 'val': valid_loader}.items():
    data_results = []
    
    for batch_idx, sample in tqdm(enumerate(v), total=len(v)):
        # import pdb; pdb.set_trace()
        features_blobs = []
        images_context = sample['image_context'].to(device)
        # forward pass
        logit = model.forward(images_context)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        # probs = probs.detach().cpu().numpy()
        idx = idx.detach().cpu().numpy()
        # output the scene attributes
        responses_attribute = W_attribute.dot(features_blobs[1])
        idx_a = np.argsort(responses_attribute)

        list_pred = [labels_attribute[idx_a[i]] for i in range(-1, -5, -1)]
        list_pred.append(classes[idx[0]])
        data_results.append(list_pred)

    path_npy = os.path.join(root, 'places_attributes')
    os.makedirs(path_npy, exist_ok=True)
    np.save(os.path.join(path_npy, f'{k}_places_attributes.npy'), data_results)
    print(len(data_results))
    data_results = []
