import ast

from albumentations import *
from torch.utils.data import Dataset
from torchvision import transforms

from utils import *
from .word2vec import *

train_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize(size=(320, 320)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                      transforms.ToTensor(),
                                      transforms.RandomErasing(),
                                      ])

test_transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize(size=(320, 320)),
                                     transforms.ToTensor()])

face_transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize(size=(48, 48)),
                                     transforms.ToTensor()])


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def upsample_data(train_context_all, train_body_all, train_cat_all, train_cont_all):
    list_ind = []
    for ind, i in enumerate(train_cat_all):
        if i[4] == 1 or i[15] == 1 or i[10] == 1 or i[25] == 1 or i[21] == 1:
            list_ind.append(ind)
    return train_context_all[list_ind, :], train_body_all[list_ind, :], train_cat_all[list_ind, :], train_cont_all[
                                                                                                    list_ind, :]


def load_data_npy(mode):
    data_src = "./emotic/emotic_pre"
    if mode == 'train':
        # Load training preprocessed data
        train_context = np.load(os.path.join(data_src, 'train_context_256.npy'))
        train_body = np.load(os.path.join(data_src, 'train_body_256.npy'))
        train_cat = np.load(os.path.join(data_src, 'train_cat_256.npy'))
        train_cont = np.load(os.path.join(data_src, 'train_cont_256.npy'))
        return train_context, train_body, train_cat, train_cont

    elif mode == 'valid':
        # Load validation preprocessed data
        val_context = np.load(os.path.join(data_src, 'val_context_256.npy'))
        val_body = np.load(os.path.join(data_src, 'val_body_arr.npy'))
        val_cat = np.load(os.path.join(data_src, 'val_cat_arr.npy'))
        val_cont = np.load(os.path.join(data_src, 'val_cont_arr.npy'))
        return val_context, val_body, val_cat, val_cont

    elif mode == 'test':
        # Load testing preprocessed data
        test_context = np.load(os.path.join(data_src, 'test_context_256.npy'))
        test_body = np.load(os.path.join(data_src, 'test_body_256.npy'))
        test_cat = np.load(os.path.join(data_src, 'test_cat_256.npy'))
        test_cont = np.load(os.path.join(data_src, 'test_cont_256.npy'))
        return test_context, test_body, test_cat, test_cont


def load_data_hook_faster(mode):
    data_src = "./emotic/hook_faster"

    return np.load(os.path.join(data_src, f'{mode}_faster_320_1024.npy'), allow_pickle=True)


def load_data_sentiment(mode):
    data_src = "./emotic/visual_sentiment"

    return np.load(os.path.join(data_src, f'{mode}_sentiment_all_objects.npy'), allow_pickle=True)


def load_data_output_sentiment(mode):
    data_src = "./emotic/visual_sentiment"

    return np.load(os.path.join(data_src, f'{mode}_sentiment_final_out.npy'), allow_pickle=True)


def load_data_places(mode):
    data_src = "./emotic/places_attributes"

    return np.load(os.path.join(data_src, f'{mode}_places_attributes.npy'), allow_pickle=True)


def fill_if_need(sample, sentiment=False, places=True):
    # TODO maybe add places attribute for dataset( currently num_places = 5)
    # if places:
    #     if sample['places_attribute'].shape[0] != 10:
    #         buffed = np.zeros((10 - sample['places_attribute'].shape[0], 50))
    #         sample['places_attribute'] = np.concatenate((sample['places_attribute'], buffed), axis=0)
    if sentiment:
        if sample['sentiment_objects'].shape[0] != 10:
            buffed = np.zeros((10 - sample['sentiment_objects'].shape[0], 4096))
            sample['sentiment_objects'] = np.concatenate((sample['sentiment_objects'], buffed), axis=0)
    else:
        if sample['features_object'].shape[0] != 10:
            buffed = np.zeros((10 - sample['features_object'].shape[0], 1024))
            sample['features_object'] = np.concatenate((sample['features_object'], buffed), axis=0)

            buffed = np.zeros((10 - sample['bbox'].shape[0], 4))
            sample['bbox'] = np.concatenate((sample['bbox'], buffed), axis=0)

            buffed = np.zeros((10 - sample['labels'].shape[0], 50))
            sample['labels'] = np.concatenate((sample['labels'], buffed), axis=0)

    return sample


def new_coordinates_after_resize_img(original_size, new_size, original_coordinate):
    original_size = np.array(original_size)
    new_size = np.array(new_size)
    original_coordinate = np.array(original_coordinate)
    xy = original_coordinate / (original_size / new_size)
    x, y = int(xy[0]), int(xy[1])
    return x, y


class Emotic_Dataset(Dataset):
    def __init__(self, x_context, x_body, y_cat, y_cont, transform):
        super(Emotic_Dataset, self).__init__()
        self.x_context = x_context
        self.x_body = x_body
        self.y_cat = y_cat
        self.y_cont = y_cont
        self.transform = transform

    def __len__(self):
        return len(self.y_cat)

    def __getitem__(self, index):
        image_context = self.x_context[index]
        image_body = self.x_body[index]
        cat_label = self.y_cat[index]
        cont_label = self.y_cont[index] / 10

        if self.transform:
            argumented = self.transform(image=image_context, image_body=image_body)
            image_context = argumented['image']
            image_body = argumented['image_body']

        sample = {'image_context': image_context, 'image_body': image_body, 'cat_label': cat_label,
                  'cont_label': cont_label}
        return sample


class Emotic_PreDataset(Dataset):
    ''' Custom Emotic dataset class. Use preprocessed data stored in npy files. '''

    def __init__(self, x_context, x_body, y_cat, y_cont, transform, mode_unnorm=False):
        super(Emotic_PreDataset, self).__init__()

        context_mean = [0.4690646, 0.4407227, 0.40508908]
        context_std = [0.2514227, 0.24312855, 0.24266963]
        body_mean = [0.43832874, 0.3964344, 0.3706214]
        body_std = [0.24784276, 0.23621225, 0.2323653]
        # depth_mean = [0.5502, 0.2946, 0.3966]
        # depth_std = [0.2929, 0.2387, 0.1359]
        context_norm = [context_mean, context_std]
        body_norm = [body_mean, body_std]
        # depth_norm = [depth_mean, depth_std]
        # self.df = pd.read_csv(os.path.join(root, data_src), index_col=0)

        # self.df['Path'] = root + '/' + 'depth' + '/' + 'depth_' + self.df['Filename']
        # self.df['Path'] = self.df['Path'].apply(lambda x: x.replace('jpg', 'png'))
        #
        # self.df['Faces'] = root + '/' + 'faces' + '/' + 'face_' + self.df['Filename']

        self.x_context = x_context
        self.x_body = x_body
        self.y_cat = y_cat
        self.y_cont = y_cont
        self.transform = transform
        self.context_norm = transforms.Normalize(context_norm[0], context_norm[1])
        self.body_norm = transforms.Normalize(body_norm[0], body_norm[1])
        # self.depth_norm = transforms.Normalize(depth_norm[0], depth_norm[1])
        # self.list_depth = self.df.Path.values
        # self.list_faces = self.df.Faces.values

        self.mode_unnorm = mode_unnorm

    def __len__(self):
        return len(self.y_cat)

    def __getitem__(self, index):
        image_context = self.x_context[index]
        image_body = self.x_body[index]
        cat_label = self.y_cat[index]
        cont_label = self.y_cont[index]

        # image_depth = cv2.imread(self.list_depth[index])[..., ::-1]
        #
        # try:
        #     image_faces = cv2.imread(self.list_faces[index])[..., ::-1]
        # except:
        #     image_faces = torch.zeros(image_depth.shape)

        sample = {'image_context': self.context_norm(self.transform(image_context)),
                  'image_body': self.body_norm(self.transform(image_body)),
                  # 'image_depth': self.depth_norm(self.transform(image_depth)),
                  # 'image_face': self.context_norm(self.transform(image_faces)),
                  'cat_label': torch.tensor(cat_label, dtype=torch.float32),
                  'cont_label': torch.tensor(cont_label, dtype=torch.float32) / 10.0}

        if self.mode_unnorm:
            return {'image_context': self.transform(image_context),
                    'image_body': self.transform(image_body)
                    }

        return sample


class Emotic_Mask_Dataset(Dataset):
    ''' Custom Emotic dataset class. Use preprocessed data stored in npy files. '''

    def __init__(self, x_context, x_body, y_cat, y_cont, transform, root, data_src):
        super(Emotic_Mask_Dataset, self).__init__()

        context_mean = [0.4690646, 0.4407227, 0.40508908]
        context_std = [0.2514227, 0.24312855, 0.24266963]
        body_mean = [0.43832874, 0.3964344, 0.3706214]
        body_std = [0.24784276, 0.23621225, 0.2323653]
        depth_mean = [0.5502, 0.2946, 0.3966]
        depth_std = [0.2929, 0.2387, 0.1359]
        context_norm = [context_mean, context_std]
        body_norm = [body_mean, body_std]
        depth_norm = [depth_mean, depth_std]

        self.df = pd.read_csv(os.path.join(root, data_src), index_col=0)

        # self.df['Path'] = root + '/' + 'depth' + '/' + 'depth_' + self.df['Filename']
        # self.df['Path'] = self.df['Path'].apply(lambda x: x.replace('jpg', 'png'))
        #
        # self.df['Faces'] = root + '/' + 'faces' + '/' + 'face_' + self.df['Filename']
        self.df['Mask'] = root + '/' + 'mask' + '/' + 'mask_' + self.df['Filename']

        self.x_context = x_context
        self.x_body = x_body
        self.y_cat = y_cat
        self.y_cont = y_cont
        self.transform = transform
        self.context_norm = transforms.Normalize(context_norm[0], context_norm[1])
        self.body_norm = transforms.Normalize(body_norm[0], body_norm[1])
        self.depth_norm = transforms.Normalize(depth_norm[0], depth_norm[1])

        # self.list_depth = self.df.Path.values
        # self.list_faces = self.df.Faces.values
        self.list_mask = self.df.Mask.values

    def __len__(self):
        return len(self.y_cat)

    def __getitem__(self, index):
        image_context = self.x_context[index]
        image_body = self.x_body[index]
        cat_label = self.y_cat[index]
        cont_label = self.y_cont[index]

        # image_depth = cv2.imread(self.list_depth[index])[..., ::-1]
        #
        # try:
        #     image_faces = cv2.imread(self.list_faces[index])[..., ::-1]
        # except:
        #     image_faces = torch.zeros(image_depth.shape)

        image_mask = cv2.imread(self.list_mask[index])[..., ::-1]

        sample = {'image_context': self.context_norm(self.transform(image_context)),
                  'image_body': self.body_norm(self.transform(image_body)),
                  # 'image_depth': self.depth_norm(self.transform(image_depth)),
                  # 'image_face': self.context_norm(self.transform(image_faces)),
                  'image_mask': self.context_norm(self.transform(image_mask)),
                  'cat_label': torch.tensor(cat_label, dtype=torch.float32),
                  'cont_label': torch.tensor(cont_label, dtype=torch.float32) / 10.0}

        return sample


class Emotic_CSVDataset(Dataset):
    ''' Custom Emotic dataset class. Use csv files and generated data at runtime. '''

    def __init__(self, cat2ind, transform, data_src, data_face, root, mode_cat='emotic_pre/val.csv',
                 hook_faster=None, hook_sentiment=None, transform2=None):

        super(Emotic_CSVDataset, self).__init__()

        self.data_src = data_src
        self.transform = transform
        self.cat2ind = cat2ind
        self.transform2 = transform2

        context_mean = [0.4690646, 0.4407227, 0.40508908]
        context_std = [0.2514227, 0.24312855, 0.24266963]
        body_mean = [0.43832874, 0.3964344, 0.3706214]
        body_std = [0.24784276, 0.23621225, 0.2323653]

        context_norm = [context_mean, context_std]
        body_norm = [body_mean, body_std]

        self.context_norm = transforms.Normalize(context_norm[0], context_norm[1])
        self.body_norm = transforms.Normalize(body_norm[0], body_norm[1])
        if mode_cat:
            self.add_df = pd.read_csv(os.path.join(root, mode_cat), index_col=0)
            self.df = pd.read_csv(os.path.join(root, data_src), index_col=0)
            self.df = pd.concat((self.df, self.add_df), axis=0).reset_index()
        else:
            self.df = pd.read_csv(os.path.join(root, data_src), index_col=0)
        self.df['Path_image'] = root + '/' + self.df['Folder'] + '/' + self.df['Filename']

        self.hook_faster_sample = hook_faster
        self.dict_word = word_embedding_categories(
            create_cat_labels('./emotic/annotations_coco/instances_val2017.json'))

        self.hook_sentiment = hook_sentiment

        # self.df_face = pd.read_csv(os.path.join(data_face), index_col=0)
        # self.face_norm = transforms.Normalize(body_norm[0], body_norm[1])
        # self.list_face = self.df_face.face_numpy.values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]

        image_context = cv2.imread(row['Path_image'])[..., ::-1]
        main_bbox = ast.literal_eval(row['BBox'])
        image_body = image_context[main_bbox[1]:main_bbox[3], main_bbox[0]:main_bbox[2]]
        image_context = image_context / 255.0
        mask = np.ones(image_context.shape, dtype='float')
        cv2.rectangle(mask, (main_bbox[0], main_bbox[1]), (main_bbox[2], main_bbox[3]), (0, 0, 0), -1)
        image_context = (mask * image_context).astype('float')
        image_context = (image_context * 255).astype('uint8')

        cat_labels = ast.literal_eval(row['Categorical_Labels'])
        cont_labels = ast.literal_eval(row['Continuous_Labels'])

        one_hot_cat_labels = self.cat_to_one_hot(cat_labels)

        sample = {'image_context': self.context_norm(self.transform(image_context)),
                  'image_body': self.body_norm(self.transform(image_body)),

                  'cat_label': torch.tensor(one_hot_cat_labels, dtype=torch.float32),
                  'cont_label': torch.tensor(cont_labels, dtype=torch.float32) / 10.0}

        if self.hook_faster_sample is not None:
            relation_boxes = []
            features_object = self.hook_faster_sample[index]['features_object']
            bbox = self.hook_faster_sample[index]['bounding_box']
            labels = self.hook_faster_sample[index]['labes']

            # print(main_bbox)
            for box in np.array(bbox):
                relation_boxes.append(get_spt_features(np.expand_dims(np.array(main_bbox), axis=0).astype(int),
                                                       np.expand_dims(np.array(box), axis=0).astype(int), 256, 256))
            # print(np.array(relation_boxes).shape)
            word_embedding = []
            for i in labels:
                if i == 0:
                    word_embedding.append(np.zeros(self.dict_word[1].shape))
                else:
                    word_embedding.append(self.dict_word[i])
            sample_hook = {'features_object': features_object,
                           'bbox': np.array(relation_boxes).squeeze(axis=1),
                           'labels': np.array(word_embedding)
                           }

            sample_hook = random_fill(sample_hook)
            sample.update(sample_hook)
        if self.hook_sentiment is not None:
            sentiment_objects = self.hook_sentiment[index]['objects']
            sentiment_objects = np.squeeze(sentiment_objects, axis=1)
            sample_sentiment = {
                'sentiment_objects': sentiment_objects
            }
            sample_sentiment = random_fill(sample_sentiment, sentiment=True)
            sample.update(sample_sentiment)

        # if self.df_face is not None:
        #     row = self.df_face.loc[index]
        #
        #     image_face = np.load(row['face_numpy'], allow_pickle=True)
        #     if len(image_face.shape) == 0:
        #         image_face = np.zeros((48, 48, 3), dtype='uint8')
        #
        #     try:
        #         sample_face = {'image_face': self.face_norm(self.transform2(image_face))}
        #     except:
        #         image_face = np.zeros((48, 48, 3), dtype='uint8')
        #         sample_face = {'image_face': self.face_norm(self.transform2(image_face))}
        #     sample.update(sample_face)
        # return sample

    def cat_to_one_hot(self, cat):
        one_hot_cat = np.zeros(26)
        for em in cat:
            one_hot_cat[self.cat2ind[em]] = 1
        return one_hot_cat


class Emotic_Dataset_addhook(Dataset):
    ''' Custom Emotic dataset class. Use preprocessed data stored in npy files. '''

    def __init__(self, x_context, x_body, y_cat, y_cont, transform, mode_unnorm=False,
                 hook_faster=None, hook_sentiment=None, hook_sentiment_output=None):
        super(Emotic_Dataset_addhook, self).__init__()

        context_mean = [0.4690646, 0.4407227, 0.40508908]
        context_std = [0.2514227, 0.24312855, 0.24266963]
        body_mean = [0.43832874, 0.3964344, 0.3706214]
        body_std = [0.24784276, 0.23621225, 0.2323653]

        context_norm = [context_mean, context_std]
        body_norm = [body_mean, body_std]

        self.x_context = x_context
        self.x_body = x_body
        self.y_cat = y_cat
        self.y_cont = y_cont
        self.transform = transform
        self.context_norm = transforms.Normalize(context_norm[0], context_norm[1])
        self.body_norm = transforms.Normalize(body_norm[0], body_norm[1])

        self.mode_unnorm = mode_unnorm

        self.hook_faster_sample = hook_faster
        self.dict_word = word_embedding_categories(
            create_cat_labels('./emotic/annotations_coco/instances_val2017.json'))

        self.hook_sentiment = hook_sentiment
        self.hook_sentiment_output = hook_sentiment_output

    def __len__(self):
        return len(self.y_cat)

    def __getitem__(self, index):
        image_context = self.x_context[index]
        image_body = self.x_body[index]
        cat_label = self.y_cat[index]
        cont_label = self.y_cont[index]

        sample = {'image_context': self.context_norm(self.transform(image_context)),
                  'image_body': self.body_norm(self.transform(image_body)),

                  'cat_label': torch.tensor(cat_label, dtype=torch.float32),
                  'cont_label': torch.tensor(cont_label, dtype=torch.float32) / 10.0}

        if self.mode_unnorm:
            return {'image_context': self.transform(image_context)}

        if self.hook_faster_sample is not None:

            features_object = self.hook_faster_sample[index]['features_object']
            bbox = self.hook_faster_sample[index]['bounding_box']
            labels = self.hook_faster_sample[index]['labes']

            word_embedding = []
            for i in labels:
                if i == 0:
                    word_embedding.append(np.zeros((self.dict_word[1].shape)))
                else:
                    word_embedding.append(self.dict_word[i])
            sample_hook = {'features_object': features_object,
                           'bbox': bbox,
                           'labels': np.array(word_embedding)
                           }

            sample_hook = random_fill(sample_hook)
            sample.update(sample_hook)
        if self.hook_sentiment is not None:
            sentiment_objects = self.hook_sentiment[index]['objects']
            sentiment_objects = np.squeeze(sentiment_objects, axis=1)
            sample_sentiment = {
                'sentiment_objects': sentiment_objects
            }
            sample_sentiment = random_fill(sample_sentiment, sentiment=True)
            sample.update(sample_sentiment)

        if self.hook_sentiment_output is not None:
            sentiment_objects_output = self.hook_sentiment_output[index]['objects']
            sentiment_objects_output = np.squeeze(sentiment_objects_output, axis=1)
            sample_sentiment = {
                'sentiment_objects_output': sentiment_objects_output
            }
            sample_sentiment = random_fill(sample_sentiment, sentiment_output=True)
            sample.update(sample_sentiment)

        return sample


class Emotic_Sentiment(Dataset):
    ''' Custom Emotic dataset class. Use preprocessed data stored in npy files. '''

    def __init__(self, x_context, x_body, transform, mode_unnorm=False,
                 hook_faster=None, df_caer=None):
        super(Emotic_Sentiment, self).__init__()

        context_mean = [0.4690646, 0.4407227, 0.40508908]
        context_std = [0.2514227, 0.24312855, 0.24266963]
        body_mean = [0.43832874, 0.3964344, 0.3706214]
        body_std = [0.24784276, 0.23621225, 0.2323653]

        context_norm = [context_mean, context_std]
        body_norm = [body_mean, body_std]

        self.x_context = x_context
        self.x_body = x_body
        self.transform = transform
        self.context_norm = transforms.Normalize(context_norm[0], context_norm[1])
        self.body_norm = transforms.Normalize(body_norm[0], body_norm[1])

        self.mode_unnorm = mode_unnorm

        self.hook_faster_sample = hook_faster
        self.df_caer = pd.read_csv(df_caer)['hook_faster']

    def __len__(self):
        if self.x_context is not None:
            return len(self.x_context)
        else:
            return len(self.df_caer)

    def __getitem__(self, index):
        image_context = self.x_context[index]
        image_body = self.x_body[index]
        # image_context = cv2.resize(image_context, (224, 224))
        if self.hook_faster_sample is not None:
            bbox = self.hook_faster_sample[index]['bounding_box']
            objects = []

            for box in bbox:
                box = box.astype(int)
                try:

                    object_element = image_context[box[1]:box[3], box[0]:box[2]]
                    object_element = self.transform(object_element)
                    objects.append(object_element)
                except:
                    objects.append(self.transform(torch.zeros((224, 224, 3))))
            sample = {'objects': objects}
            return sample

        if self.mode_unnorm:
            return {'image_context': self.transform(image_context),
                    'image_body': self.transform(image_body)
                    }

        if self.df_caer is not None:
            bbox = np.load(self.df_caer[index], allow_pickle=True).item()['features_object']
            objects = []

            for box in bbox:
                box = box.astype(int)
                try:
                    object_element = image_context[box[1]:box[3], box[0]:box[2]]
                    object_element = self.transform(object_element)
                    objects.append(object_element)
                except:
                    objects.append(self.transform(torch.zeros((224, 224, 3))))
            sample = {'objects': objects}
            return sample


class Emotic_Sentimentv2(Dataset):
    ''' Custom Emotic dataset class. Use preprocessed data stored in npy files. '''

    def __init__(self, path_df, transform):
        super(Emotic_Sentimentv2, self).__init__()

        context_mean = [0.4690646, 0.4407227, 0.40508908]
        context_std = [0.2514227, 0.24312855, 0.24266963]
        body_mean = [0.43832874, 0.3964344, 0.3706214]
        body_std = [0.24784276, 0.23621225, 0.2323653]
        face_mean = [0.485, 0.456, 0.406]
        face_std = [0.229, 0.224, 0.225]
        context_norm = [context_mean, context_std]
        body_norm = [body_mean, body_std]
        face_norm = [face_mean, face_std]

        self.transform = transform
        self.context_norm = transforms.Normalize(context_norm[0], context_norm[1])
        self.body_norm = transforms.Normalize(body_norm[0], body_norm[1])
        self.face_norm = transforms.Normalize(face_norm[0], face_norm[1])

        self.df = pd.read_csv(path_df, index_col=0)
        self.image = self.df['file'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        image = self.image[index]

        image_context = cv2.imread(image)[..., ::-1]
        image_context = cv2.resize(image_context, (320, 320))
        try:
            bbox = (np.load(self.df['hook_faster'][index], allow_pickle=True)).item()['bounding_box']
        except:
            bbox = np.array([0, 0, 0, 0])
        objects = []

        for box in bbox:
            box = box.astype(int)
            try:
                object_element = image_context[box[1]:box[3], box[0]:box[2]]
                object_element = self.transform(object_element)
                objects.append(object_element)
            except:
                objects.append(self.transform(torch.zeros((224, 224, 3))))
        sample = {'objects': objects}
        return sample


class Emotic_Dataset_transform_od(Dataset):
    ''' Custom Emotic dataset class. Use preprocessed data stored in npy files. '''

    def __init__(self, x_context, x_body, y_cat, y_cont, transform, mode_unnorm=False,
                 hook_faster=None, hook_sentiment=None, mode='TEST'):
        super(Emotic_Dataset_transform_od, self).__init__()

        context_mean = [0.4690646, 0.4407227, 0.40508908]
        context_std = [0.2514227, 0.24312855, 0.24266963]
        body_mean = [0.43832874, 0.3964344, 0.3706214]
        body_std = [0.24784276, 0.23621225, 0.2323653]

        context_norm = [context_mean, context_std]
        body_norm = [body_mean, body_std]

        self.x_context = x_context
        self.x_body = x_body
        self.y_cat = y_cat
        self.y_cont = y_cont
        self.transform = transform
        self.context_norm = transforms.Normalize(context_norm[0], context_norm[1])
        self.body_norm = transforms.Normalize(body_norm[0], body_norm[1])

        self.mode_unnorm = mode_unnorm

        self.hook_faster_sample = hook_faster
        self.dict_word = word_embedding_categories(
            create_cat_labels('./emotic/annotations_coco/instances_val2017.json'))

        self.hook_sentiment = hook_sentiment
        self.mode = mode

    def __len__(self):
        return len(self.y_cat)

    def __getitem__(self, index):
        image_context = self.x_context[index]
        image_body = self.x_body[index]
        cat_label = self.y_cat[index]
        cont_label = self.y_cont[index]

        sample = {'image_context': image_context,
                  'image_body': self.body_norm(self.transform(image_body)),

                  'cat_label': torch.tensor(cat_label, dtype=torch.float32),
                  'cont_label': torch.tensor(cont_label, dtype=torch.float32) / 10.0}

        if self.mode_unnorm:
            return {'image_context': self.transform(image_context)}

        if self.hook_faster_sample is not None:

            features_object = self.hook_faster_sample[index]['features_object']
            bbox = self.hook_faster_sample[index]['bounding_box']
            labels = self.hook_faster_sample[index]['labes']

            word_embedding = []
            for i in labels:
                if i == 0:
                    word_embedding.append(np.zeros((self.dict_word[1].shape)))
                else:
                    word_embedding.append(self.dict_word[i])
            sample_hook = {'features_object': features_object,
                           'bbox': bbox,
                           'labels': np.array(word_embedding)
                           }

            sample_hook = fill_if_need(sample_hook)
            sample.update(sample_hook)
        if self.hook_sentiment is not None:
            sentiment_objects = self.hook_sentiment[index]['objects']
            sentiment_objects = np.squeeze(sentiment_objects, axis=1)
            sample_sentiment = {
                'sentiment_objects': sentiment_objects
            }
            sample_sentiment = fill_if_need(sample_sentiment, sentiment=True)
            sample.update(sample_sentiment)

        sample['image_context'], sample['bbox'], sample['labels'], _ = transform_od(sample['image_context'],
                                                                                    sample['bbox'], sample['labels'],
                                                                                    split=self.mode)

        return sample


def places2vec(list_pred, glove):
    # emmbedding_label = []

    di = []
    for ind, i in enumerate(list_pred):
        emmbedding_label = []
        for word in i.split(' '):
            emmbedding_label.append(np.array(glove[word]))

        di.append(np.mean(emmbedding_label, axis=0))
    return np.array(di)


class Emotic_Dataset_places(Dataset):
    ''' Custom Emotic dataset class. Use preprocessed data stored in npy files. '''

    def __init__(self, x_context, x_body, y_cat, y_cont, transform, mode_unnorm=False,
                 hook_faster=None, hook_sentiment=None, hook_places=None):
        super(Emotic_Dataset_places, self).__init__()

        context_mean = [0.4690646, 0.4407227, 0.40508908]
        context_std = [0.2514227, 0.24312855, 0.24266963]
        body_mean = [0.43832874, 0.3964344, 0.3706214]
        body_std = [0.24784276, 0.23621225, 0.2323653]

        context_norm = [context_mean, context_std]
        body_norm = [body_mean, body_std]

        self.x_context = x_context
        self.x_body = x_body
        self.y_cat = y_cat
        self.y_cont = y_cont
        self.transform = transform
        self.context_norm = transforms.Normalize(context_norm[0], context_norm[1])
        self.body_norm = transforms.Normalize(body_norm[0], body_norm[1])

        self.mode_unnorm = mode_unnorm

        self.hook_faster_sample = hook_faster
        self.dict_word = word_embedding_categories(
            create_cat_labels('./emotic/annotations_coco/instances_val2017.json'))

        self.hook_sentiment = hook_sentiment
        self.hook_places = hook_places
        self.glove = torchtext.vocab.GloVe(name="6B",  # trained on Wikipedia 2014 corpus
                                           dim=50)  # embedding size = 100

    def __len__(self):
        return len(self.y_cat)

    def __getitem__(self, index):
        image_context = self.x_context[index]
        image_body = self.x_body[index]
        cat_label = self.y_cat[index]
        cont_label = self.y_cont[index]

        sample = {'image_context': self.context_norm(self.transform(image_context)),
                  'image_body': self.body_norm(self.transform(image_body)),
                  'cat_label': torch.tensor(cat_label, dtype=torch.float32),
                  'cont_label': torch.tensor(cont_label, dtype=torch.float32) / 10.0}

        if self.mode_unnorm:
            return {'image_context': self.transform(image_context)}

        if self.hook_faster_sample is not None:

            features_object = self.hook_faster_sample[index]['features_object']
            bbox = self.hook_faster_sample[index]['bounding_box']
            labels = self.hook_faster_sample[index]['labes']

            word_embedding = []
            for i in labels:
                if i == 0:
                    word_embedding.append(np.zeros((self.dict_word[1].shape)))
                else:
                    word_embedding.append(self.dict_word[i])
            sample_hook = {'features_object': features_object,
                           'bbox': bbox,
                           'labels': np.array(word_embedding)
                           }

            sample_hook = random_fill(sample_hook)
            sample.update(sample_hook)
        if self.hook_sentiment is not None:
            sentiment_objects = self.hook_sentiment[index]['objects']
            sentiment_objects = np.squeeze(sentiment_objects, axis=1)
            sample_sentiment = {
                'sentiment_objects': sentiment_objects
            }
            sample_sentiment = random_fill(sample_sentiment, sentiment=True)
            sample.update(sample_sentiment)

        if self.hook_places is not None:
            places_word = self.hook_places[index]
            sample_places = {
                'places_attribute': places2vec(places_word, self.glove)
            }
            # sample_places = fill_if_need(sample_places, )
            sample.update(sample_places)
        return sample


class Emotic_CSVDatasetv2(Dataset):
    ''' Custom Emotic dataset class. Use csv files and generated data at runtime. '''

    def __init__(self, cat2ind, transform, data_src, root, mode_cat='emotic_pre/val.csv',
                 hook_faster=None, hook_sentiment=None):

        super(Emotic_CSVDatasetv2, self).__init__()

        self.data_src = data_src
        self.transform = transform
        self.cat2ind = cat2ind

        context_mean = [0.4690646, 0.4407227, 0.40508908]
        context_std = [0.2514227, 0.24312855, 0.24266963]
        body_mean = [0.43832874, 0.3964344, 0.3706214]
        body_std = [0.24784276, 0.23621225, 0.2323653]

        context_norm = [context_mean, context_std]
        body_norm = [body_mean, body_std]

        self.context_norm = transforms.Normalize(context_norm[0], context_norm[1])
        self.body_norm = transforms.Normalize(body_norm[0], body_norm[1])
        if mode_cat:
            self.add_df = pd.read_csv(os.path.join(root, mode_cat), index_col=0)
            self.df = pd.read_csv(os.path.join(root, data_src), index_col=0)
            self.df = pd.concat((self.df, self.add_df), axis=0).reset_index()
        else:
            self.df = pd.read_csv(os.path.join(root, data_src), index_col=0)
        self.df['Path_image'] = root + '/' + self.df['Folder'] + '/' + self.df['Filename']

        self.hook_faster_sample = hook_faster
        self.dict_word = word_embedding_categories(
            create_cat_labels('./emotic/annotations_coco/instances_val2017.json'))

        self.hook_sentiment = hook_sentiment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]

        image_context = cv2.imread(row['Path_image'])[..., ::-1]
        main_bbox = ast.literal_eval(row['BBox'])
        image_body = image_context[main_bbox[1]:main_bbox[3], main_bbox[0]:main_bbox[2]]
        cat_labels = ast.literal_eval(row['Categorical_Labels'])
        cont_labels = ast.literal_eval(row['Continuous_Labels'])

        one_hot_cat_labels = self.cat_to_one_hot(cat_labels)

        x1, y1, x2, y2 = main_bbox
        h, w, _ = image_context.shape
        x1_new, y1_new = new_coordinates_after_resize_img((w, h), (320, 320), (x1, y1))
        x2_new, y2_new = new_coordinates_after_resize_img((w, h), (320, 320), (x2, y2))

        sample = {'image_context': self.context_norm(self.transform(image_context)),
                  'image_body': self.body_norm(self.transform(image_body)),
                  'main_box': torch.tensor([x1_new, y1_new, x2_new, y2_new]),
                  'cat_label': torch.tensor(one_hot_cat_labels, dtype=torch.float32)}

        if self.hook_faster_sample is not None:
            relation_boxes = []
            features_object = self.hook_faster_sample[index]['features_object']
            bbox = self.hook_faster_sample[index]['bounding_box']
            labels = self.hook_faster_sample[index]['labes']

            # print(main_bbox)
            for box in np.array(bbox):
                relation_boxes.append(get_spt_features(np.expand_dims(np.array(main_bbox), axis=0).astype(int),
                                                       np.expand_dims(np.array(box), axis=0).astype(int), 320, 320))
            # print(np.array(relation_boxes).shape)
            word_embedding = []
            for i in labels:
                if i == 0:
                    word_embedding.append(np.zeros(self.dict_word[1].shape))
                else:
                    word_embedding.append(self.dict_word[i])

            list_pair_object_main = []
            for c in np.array(bbox):
                im_copy = np.array(sample['image_context']).transpose(1, 2, 0).copy()
                mask = np.zeros((320, 320, 3))
                xo1, yo1, xo2, yo2 = c.astype(int)

                cv2.rectangle(mask, (xo1, yo1), (xo2, yo2), (1, 1, 1), -1)
                cv2.rectangle(mask, (x1_new, y1_new), (x2_new, y2_new), (1, 1, 1), -1)

                im_copy = im_copy * mask
                list_pair_object_main.append(im_copy)

            sample_hook = {
                'features_object': features_object,
                'bbox': np.array(bbox),
                'labels': np.array(word_embedding),
                'pair_o_m': np.array(list_pair_object_main),
            }

            sample_hook = random_fill(sample_hook)
            sample.update(sample_hook)
        if self.hook_sentiment is not None:
            sentiment_objects = self.hook_sentiment[index]['objects']
            sentiment_objects = np.squeeze(sentiment_objects, axis=1)
            sample_sentiment = {
                'sentiment_objects': sentiment_objects
            }
            sample_sentiment = random_fill(sample_sentiment, sentiment=True)
            sample.update(sample_sentiment)
        return sample

    def cat_to_one_hot(self, cat):
        one_hot_cat = np.zeros(26)
        for em in cat:
            one_hot_cat[self.cat2ind[em]] = 1
        return one_hot_cat


def random_fill(sample, features, mode_zero=False, num_sample=10):
    # TODO maybe add places attribute for dataset( currently num_places = 5)

    if mode_zero:
        for i in features:
            # import pdb; pdb.set_trace()
            buff = np.zeros(sample[i].shape)
            while sample[i].shape[0] < num_sample:
                sample[i] = np.concatenate((sample[i], buff), axis=0)

            sample[i] = sample[i][:num_sample, :]
    else:
        for i in features:
            while sample[i].shape[0] < num_sample:
                sample[i] = np.tile(sample[i], (2, 1))
            sample[i] = sample[i][:num_sample, :]

    return sample


class Emotic_Face_Dataset(Dataset):
    ''' Custom Emotic dataset class. Use preprocessed data stored in npy files. '''

    def __init__(self, y_cat, transform, data_src):
        super(Emotic_Face_Dataset, self).__init__()

        context_mean = [0.4690646, 0.4407227, 0.40508908]
        context_std = [0.2514227, 0.24312855, 0.24266963]
        body_mean = [0.43832874, 0.3964344, 0.3706214]
        body_std = [0.24784276, 0.23621225, 0.2323653]
        context_norm = [context_mean, context_std]
        body_norm = [body_mean, body_std]

        self.df = pd.read_csv(os.path.join(data_src), index_col=0)
        self.face_norm = transforms.Normalize(body_norm[0], body_norm[1])
        self.list_face = self.df.face_numpy.values
        self.y_cat = y_cat

        self.transform = transform

    def __len__(self):
        return len(self.y_cat)

    def __getitem__(self, index):
        row = self.df.loc[index]

        image_face = np.load(row['face_numpy'], allow_pickle=True)
        if len(image_face.shape) == 0:
            image_face = np.zeros((48, 48, 3), dtype='uint8')

        cat_label = self.y_cat[index]
        try:
            sample = {'image_face': self.face_norm(self.transform(image_face)),
                      'cat_label': torch.tensor(cat_label, dtype=torch.float32)}
        except:
            image_face = np.zeros((48, 48, 3), dtype='uint8')
            sample = {'image_face': self.face_norm(self.transform(image_face)),
                      'cat_label': torch.tensor(cat_label, dtype=torch.float32)}

        return sample


class Emotic_Dataset_addhookv2(Dataset):
    ''' Custom Emotic dataset class. Use preprocessed data stored in npy files. '''

    def __init__(self, x_context, x_body, y_cat, y_cont, transform, data_face, index_train, mode_unnorm=False,
                 hook_faster=None, hook_sentiment=None, hook_sentiment_output=None,
                 transform2=None):
        super(Emotic_Dataset_addhookv2, self).__init__()

        context_mean = [0.4690646, 0.4407227, 0.40508908]
        context_std = [0.2514227, 0.24312855, 0.24266963]
        body_mean = [0.43832874, 0.3964344, 0.3706214]
        body_std = [0.24784276, 0.23621225, 0.2323653]

        context_norm = [context_mean, context_std]
        body_norm = [body_mean, body_std]

        self.x_context = x_context
        self.x_body = x_body
        self.y_cat = y_cat
        self.y_cont = y_cont
        self.transform = transform
        self.transform2 = transform2

        self.context_norm = transforms.Normalize(context_norm[0], context_norm[1])
        self.body_norm = transforms.Normalize(body_norm[0], body_norm[1])

        self.mode_unnorm = mode_unnorm

        self.hook_faster_sample = hook_faster
        self.dict_word = word_embedding_categories(
            create_cat_labels('./emotic/annotations_coco/instances_val2017.json'))

        self.hook_sentiment = hook_sentiment
        self.hook_sentiment_output = hook_sentiment_output

        self.df_face = pd.read_csv(os.path.join(data_face), index_col=0)
        # import pdb; pdb.set_trace()
        if index_train is not None:
            self.df_face = self.df_face.loc[index_train].reset_index()
        self.face_norm = transforms.Normalize(body_norm[0], body_norm[1])
        self.list_face = self.df_face.face_numpy.values

    def __len__(self):
        return len(self.y_cat)

    def __getitem__(self, index):
        image_context = self.x_context[index]
        image_body = self.x_body[index]
        cat_label = self.y_cat[index]
        cont_label = self.y_cont[index]

        sample = {'image_context': self.context_norm(self.transform(image_context)),
                  'image_body': self.body_norm(self.transform(image_body)),

                  'cat_label': torch.tensor(cat_label, dtype=torch.float32),
                  'cont_label': torch.tensor(cont_label, dtype=torch.float32) / 10.0}

        if self.mode_unnorm:
            return {'image_context': self.transform(image_context)}

        if self.hook_faster_sample is not None:

            features_object = self.hook_faster_sample[index]['features_object']
            bbox = self.hook_faster_sample[index]['bounding_box']
            labels = self.hook_faster_sample[index]['labes']

            word_embedding = []
            for i in labels:
                if i == 0:
                    word_embedding.append(np.zeros((self.dict_word[1].shape)))
                else:
                    word_embedding.append(self.dict_word[i])
            sample_hook = {'features_object': features_object,
                           'bbox': bbox,
                           'labels': np.array(word_embedding)
                           }

            sample_hook = random_fill(sample_hook)
            sample.update(sample_hook)
        if self.hook_sentiment is not None:
            sentiment_objects = self.hook_sentiment[index]['objects']
            sentiment_objects = np.squeeze(sentiment_objects, axis=1)
            sample_sentiment = {
                'sentiment_objects': sentiment_objects
            }
            sample_sentiment = random_fill(sample_sentiment, sentiment=True)
            sample.update(sample_sentiment)

        if self.hook_sentiment_output is not None:
            sentiment_objects_output = self.hook_sentiment_output[index]['objects']
            sentiment_objects_output = np.squeeze(sentiment_objects_output, axis=1)
            sample_sentiment = {
                'sentiment_objects_output': sentiment_objects_output
            }
            sample_sentiment = random_fill(sample_sentiment, sentiment_output=True)
            sample.update(sample_sentiment)

        if self.df_face is not None:
            row = self.df_face.iloc[index]

            image_face = np.load(row['face_numpy'], allow_pickle=True)
            if len(image_face.shape) == 0:
                image_face = np.zeros((48, 48, 3), dtype='uint8')

            try:
                sample_face = {'image_face': self.face_norm(self.transform2(image_face))}
            except:
                image_face = np.zeros((48, 48, 3), dtype='uint8')
                sample_face = {'image_face': self.face_norm(self.transform2(image_face))}
            sample.update(sample_face)

        return sample


class Emotic_Dataset_addhookv3(Dataset):
    ''' Custom Emotic dataset class. Use preprocessed data stored in npy files. '''

    def __init__(self, x_context, x_body, y_cat, y_cont, transform, mode_unnorm=False,
                 hook_faster=None, hook_sentiment=None, hook_sentiment_output=None, index_train=None,
                 three_streams=False, mask_context=False, data_src=None, num_sample=10):
        super(Emotic_Dataset_addhookv3, self).__init__()

        context_mean = [0.4690646, 0.4407227, 0.40508908]
        context_std = [0.2514227, 0.24312855, 0.24266963]
        body_mean = [0.43832874, 0.3964344, 0.3706214]
        body_std = [0.24784276, 0.23621225, 0.2323653]

        context_norm = [context_mean, context_std]
        body_norm = [body_mean, body_std]

        self.x_context = x_context
        self.x_body = x_body
        self.y_cat = y_cat
        self.y_cont = y_cont
        self.transform = transform
        self.context_norm = transforms.Normalize(context_norm[0], context_norm[1])
        self.body_norm = transforms.Normalize(body_norm[0], body_norm[1])

        self.mode_unnorm = mode_unnorm

        self.hook_faster_sample = hook_faster
        self.dict_word = word_embedding_categories(
            create_cat_labels('./emotic/annotations_coco/instances_val2017.json'))

        if index_train is not None:
            self.df = pd.read_csv('./emotic/emotic_pre/train.csv')
            self.df = self.df.loc[index_train].reset_index()

            if data_src:
                self.df_face = pd.read_csv(os.path.join(data_src), index_col=0)
                self.df_face = self.df_face.loc[index_train].reset_index()
                self.face_norm = transforms.Normalize(body_norm[0], body_norm[1])
                self.list_face = self.df_face.face_numpy.values
                self.face_transform = face_transform
            else:
                self.df_face = None

        else:
            self.df = pd.read_csv('./emotic/emotic_pre/test.csv')
            if data_src:
                self.df_face = pd.read_csv(os.path.join(data_src), index_col=0)
                self.face_norm = transforms.Normalize(body_norm[0], body_norm[1])
                self.list_face = self.df_face.face_numpy.values
                self.face_transform = face_transform
            else:
                self.df_face = None

        self.hook_sentiment = hook_sentiment
        self.hook_sentiment_output = hook_sentiment_output
        self.three_streams = three_streams

        self.mask_context = mask_context
        self.num_sample = num_sample

    def __len__(self):
        return len(self.y_cat)

    def __getitem__(self, index):

        image_context = self.x_context[index]
        image_body = self.x_body[index]
        cat_label = self.y_cat[index]

        sample = {'image_context': self.context_norm(self.transform(image_context)),
                  'image_body': self.body_norm(self.transform(image_body)),
                  'cat_label': torch.tensor(cat_label, dtype=torch.float32)}

        if self.mode_unnorm:
            sample['image_context'] = self.transform(image_context)
            sample['image_body'] = self.transform(image_body)

        if self.hook_faster_sample is not None:

            features_object = self.hook_faster_sample[index]['features_object']
            bbox = self.hook_faster_sample[index]['bounding_box']
            labels = self.hook_faster_sample[index]['labes']

            word_embedding = []
            for i in labels:
                if i == 0:
                    word_embedding.append(np.zeros((self.dict_word[1].shape)))
                else:
                    word_embedding.append(self.dict_word[i])

            sample_hook = {'features_object': features_object,
                           'bbox': bbox,
                           'labels': np.array(word_embedding)
                           }

            sample_hook = random_fill(sample_hook, features=['features_object', 'bbox', 'labels'],
                                      num_sample=self.num_sample)
            sample.update(sample_hook)

        if self.hook_sentiment is not None:
            sentiment_objects = self.hook_sentiment[index]['objects']
            sentiment_objects = np.squeeze(sentiment_objects, axis=1)
            sample_sentiment = {
                'sentiment_objects': sentiment_objects
            }
            sample_sentiment = random_fill(sample_sentiment, features=['sentiment_objects'], num_sample=self.num_sample)
            sample.update(sample_sentiment)

        if self.hook_sentiment_output is not None:
            sentiment_objects_output = self.hook_sentiment_output[index]['objects']
            sentiment_objects_output = np.squeeze(sentiment_objects_output, axis=1)
            sample_sentiment = {
                'sentiment_objects_output': sentiment_objects_output
            }
            sample_sentiment = random_fill(sample_sentiment, features=['sentiment_objects_output'],
                                           num_sample=self.num_sample)
            sample.update(sample_sentiment)

        if self.df is not None:
            row = self.df.iloc[index]
            main_bbox = ast.literal_eval(row['BBox'])
            w, h = ast.literal_eval(row['Image Size'])
            x1, y1, x2, y2 = main_bbox
            # h, w, _ = image_context.shape
            x1_new, y1_new = new_coordinates_after_resize_img((w, h), (320, 320), (x1, y1))
            x2_new, y2_new = new_coordinates_after_resize_img((w, h), (320, 320), (x2, y2))

            sample_box = {'main_box': np.array([x1_new, y1_new, x2_new, y2_new], dtype=float)}
            sample.update(sample_box)
            sample_box = random_fill(sample_box, features=['main_box'], num_sample=self.num_sample)

            spatial_features = []
            for human, object_ in zip(sample_box['main_box'], sample['bbox']):
                spatial = generate_spatial(human, object_)
                spatial_features.append(spatial)
            sample_pair_box = {'pair_box': np.array(spatial_features)}
            sample.update(sample_pair_box)

        if self.three_streams:
            mask = np.zeros((320, 320, 3))
            im_copy = np.array(sample['image_context']).transpose(1, 2, 0).copy()
            for c in np.array(sample['bbox']):
                xo1, yo1, xo2, yo2 = c.astype(int)

                cv2.rectangle(mask, (xo1, yo1), (xo2, yo2), (1, 1, 1), -1)
            x1_new, y1_new, x2_new, y2_new = sample['main_box'].astype(int)
            cv2.rectangle(mask, (x1_new, y1_new), (x2_new, y2_new), (1, 1, 1), -1)
            im_copy = im_copy * mask
            sample_image_object = {'image_object': torch.tensor(im_copy).permute(2, 0, 1)}
            sample.update(sample_image_object)

            mask = np.zeros((320, 320, 3))
            im_copy = np.array(sample['image_context']).transpose(1, 2, 0).copy()
            for c in np.array(sample['bbox']):
                xo1, yo1, xo2, yo2 = c.astype(int)

                cv2.rectangle(mask, (xo1, yo1), (xo2, yo2), (1, 1, 1), -1)
            x1_new, y1_new, x2_new, y2_new = sample['main_box'].astype(int)
            cv2.rectangle(mask, (x1_new, y1_new), (x2_new, y2_new), (1, 1, 1), -1)
            im_copy = im_copy * mask
            sample_image_object = {'image_object': torch.tensor(im_copy).permute(2, 0, 1)}
            sample.update(sample_image_object)

            sample_image_scene = {'image_scene': sample['image_context'] - torch.tensor(im_copy).permute(2, 0, 1)}
            sample.update(sample_image_scene)

        if self.mask_context:
            mask = np.zeros((320, 320, 3))
            im_copy = np.array(sample['image_context']).transpose(1, 2, 0).copy()
            x1_new, y1_new, x2_new, y2_new = sample['main_box'].astype(int)
            cv2.rectangle(mask, (x1_new, y1_new), (x2_new, y2_new), (1, 1, 1), -1)
            im_copy = im_copy * mask
            sample['image_context'] = sample['image_context'] - torch.tensor(im_copy).permute(2, 0, 1)

        if self.df_face is not None:
            row = self.df_face.loc[index]

            image_face = np.load(row['face_numpy'], allow_pickle=True)

            box_face = ast.literal_eval(row['bbox_face_320'])

            if len(image_face.shape) == 0:
                image_face = np.zeros((48, 48, 3), dtype='uint8')
            try:
                sample['image_face'] = self.face_norm(self.face_transform(image_face))
                if self.mask_context:
                    sample['image_body'][:, box_face[1]:box_face[3], box_face[0]:box_face[2]] = 0
            except:
                image_face = np.zeros((48, 48, 3), dtype='uint8')
                sample['image_face'] = self.face_norm(self.face_transform(image_face))

        return sample


class Caer_Dataset(Dataset):
    def __init__(self, path_df, transform, mode_unnorm=False):
        super(Caer_Dataset, self).__init__()

        context_mean = [0.4690646, 0.4407227, 0.40508908]
        context_std = [0.2514227, 0.24312855, 0.24266963]
        body_mean = [0.43832874, 0.3964344, 0.3706214]
        body_std = [0.24784276, 0.23621225, 0.2323653]
        face_mean = [0.485, 0.456, 0.406]
        face_std = [0.229, 0.224, 0.225]
        context_norm = [context_mean, context_std]
        body_norm = [body_mean, body_std]
        face_norm = [face_mean, face_std]

        self.categories = {'Anger': 0, 'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5,
                           'Surprise': 6}
        self.transform = transform
        self.context_norm = transforms.Normalize(context_norm[0], context_norm[1])
        self.body_norm = transforms.Normalize(body_norm[0], body_norm[1])
        self.face_norm = transforms.Normalize(face_norm[0], face_norm[1])

        self.df = pd.read_csv(path_df, index_col=0)
        self.faces = self.df['face'].values
        self.body = self.df['box_body'].values
        self.image = self.df['file'].values
        self.cat_label = self.df['emotion'].values

        self.mode_unnorm = mode_unnorm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        face = self.faces[index]
        body = ast.literal_eval(self.body[index])
        image = self.image[index]
        label = self.categories[self.cat_label[index]]
        image_context = cv2.imread(image)[..., ::-1]
        image_face = cv2.imread(face)[..., ::-1]

        image_body = cv2.resize(image_context, (320, 320))[int(body[1]):int(body[3]), int(body[0]):int(body[2])]

        if self.mode_unnorm:
            try:
                sample = {'image_context': self.transform(image_context),
                          'image_body': self.transform(image_body),
                          'image_face': self.transform(image_face),
                          'label': label}

            except:
                sample = {'image_context': self.context_norm(self.transform(image_context)),
                          'image_body': torch.zeros((3, 320, 320)),
                          'image_face': self.face_norm(self.transform(image_face)),
                          'label': label}
            return sample

        else:
            try:
                sample = {'image_context': self.context_norm(self.transform(image_context)),
                          'image_body': self.body_norm(self.transform(image_body)),
                          'image_face': self.face_norm(self.transform(image_face)),
                          'label': label}
            except:
                sample = {'image_context': self.context_norm(self.transform(image_context)),
                          'image_body': torch.zeros((3, 320, 320)),
                          'image_face': self.face_norm(self.transform(image_face)),
                          'label': label}

        return sample


class Caer_Dataset_addhook(Dataset):
    def __init__(self, path_df, transform, hook_faster, hook_sentiment, mode_unnorm=False, index_train=None):
        super(Caer_Dataset_addhook, self).__init__()

        context_mean = [0.4690646, 0.4407227, 0.40508908]
        context_std = [0.2514227, 0.24312855, 0.24266963]
        body_mean = [0.43832874, 0.3964344, 0.3706214]
        body_std = [0.24784276, 0.23621225, 0.2323653]
        face_mean = [0.485, 0.456, 0.406]
        face_std = [0.229, 0.224, 0.225]
        context_norm = [context_mean, context_std]
        body_norm = [body_mean, body_std]
        face_norm = [face_mean, face_std]

        self.categories = {'Anger': 0, 'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5,
                           'Surprise': 6}
        self.transform = transform
        self.context_norm = transforms.Normalize(context_norm[0], context_norm[1])
        self.body_norm = transforms.Normalize(body_norm[0], body_norm[1])
        self.face_norm = transforms.Normalize(face_norm[0], face_norm[1])
        self.face_transform = face_transform

        self.df = pd.read_csv(path_df, index_col=0)
        if index_train is not None:
            self.df = self.df.loc[index_train].reset_index()
            self.hook_faster = pd.read_csv(hook_faster).loc[index_train].reset_index()['hook_faster']
            self.hook_sentiment = pd.read_csv(hook_sentiment).loc[index_train].reset_index()['hook_sentiment']
        else:
            self.hook_faster = pd.read_csv(hook_faster)['hook_faster']
            self.hook_sentiment = pd.read_csv(hook_sentiment)['hook_sentiment']
        self.faces = self.df['face'].values
        self.body = self.df['box_body'].values
        self.image = self.df['file'].values
        self.cat_label = self.df['emotion'].values

        self.mode_unnorm = mode_unnorm

        self.dict_word = word_embedding_categories(
            create_cat_labels('./emotic/annotations_coco/instances_val2017.json'))
        self.num_sample = 10

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        face = self.faces[index]
        body = ast.literal_eval(self.body[index])
        image = self.image[index]
        label = self.categories[self.cat_label[index]]
        image_context = cv2.imread(image)[..., ::-1]
        image_face = cv2.imread(face)[..., ::-1]

        image_body = cv2.resize(image_context, (320, 320))[int(body[1]):int(body[3]), int(body[0]):int(body[2])]

        if self.mode_unnorm:
            try:
                sample = {'image_context': self.transform(image_context),
                          'image_body': self.transform(image_body),
                          'image_face': self.face_transform(image_face),
                          'cat_label': label}

            except:
                sample = {'image_context': self.context_norm(self.transform(image_context)),
                          'image_body': torch.zeros((3, 320, 320)),
                          'image_face': self.face_norm(self.face_transform(image_face)),
                          'cat_label': label}
        else:
            try:
                sample = {'image_context': self.context_norm(self.transform(image_context)),
                          'image_body': self.body_norm(self.transform(image_body)),
                          'image_face': self.face_norm(self.face_transform(image_face)),
                          'cat_label': label}
            except:
                sample = {'image_context': self.context_norm(self.transform(image_context)),
                          'image_body': torch.zeros((3, 320, 320)),
                          'image_face': self.face_norm(self.face_transform(image_face)),
                          'cat_label': label}

        if self.hook_faster is not None:
            try:
                self.hook_faster_sample = np.load(self.hook_faster[index], allow_pickle=True).item()
            except:
                self.hook_faster_sample = {
                    'features_object': np.zeros((1, 1024)),
                    'bounding_box': np.zeros((1, 4)),
                    'labes': np.zeros(1)
                }
            features_object = self.hook_faster_sample['features_object']
            bbox = self.hook_faster_sample['bounding_box']
            labels = self.hook_faster_sample['labes']

            word_embedding = []
            for i in labels:
                if i == 0:
                    word_embedding.append(np.zeros((self.dict_word[1].shape)))
                else:
                    word_embedding.append(self.dict_word[i])

            sample_hook = {'features_object': features_object,
                           'bbox': bbox,
                           'labels': np.array(word_embedding)
                           }

            sample_hook = random_fill(sample_hook, features=['features_object', 'bbox', 'labels'],
                                      num_sample=self.num_sample)
            sample.update(sample_hook)

        if self.hook_sentiment is not None:
            # import pdb; pdb.set_trace()

            sentiment_objects = np.load(self.hook_sentiment[index], allow_pickle=True)
            sentiment_objects = np.squeeze(sentiment_objects, axis=1)
            sample_sentiment = {
                'sentiment_objects': sentiment_objects
            }
            sample_sentiment = random_fill(sample_sentiment, features=['sentiment_objects'], num_sample=self.num_sample)
            sample.update(sample_sentiment)

        if self.df is not None:
            sample_box = {'main_box': np.array(ast.literal_eval(self.body[index])).astype(float)}
            sample.update(sample_box)
            sample_box = random_fill(sample_box, features=['main_box'], num_sample=self.num_sample)

            spatial_features = []
            for human, object_ in zip(sample_box['main_box'], sample['bbox']):
                spatial = generate_spatial(human, object_)
                spatial_features.append(spatial)
            sample_pair_box = {'pair_box': np.array(spatial_features)}
            sample.update(sample_pair_box)
        return sample
