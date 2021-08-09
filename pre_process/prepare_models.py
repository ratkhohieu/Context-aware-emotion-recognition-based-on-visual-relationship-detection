import os

import torch
import torch.nn as nn
import torchvision.models as models

from models.vgg import VGG
from models.vgg19 import KitModel


def prep_models(context_model='resnet50', body_model='resnet50', model_dir='./', file_name='UnbiasedEmo_best'):
    ''' Download imagenet pretrained models for context_model and body_model.
    :param context_model: Model to use for conetxt features.
    :param body_model: Model to use for body features.
    :param model_dir: Directory path where to store pretrained models.
    :return: Yolo model after loading model weights
    '''
    if context_model == 'resnet50':
        print('Model EmotionNet Loaded')
        save_file = f'./weight/models/{file_name}.pth.tar'
        checkpoint = torch.load(save_file)

        model_body = models.resnet50(pretrained=True)
        model_context = models.resnet50(pretrained=True)
        # import pdb; pdb.set_trace()
        new_state_dict = {}
        for key, values in checkpoint['state_dict'].items():
            new_state_dict[key.replace('module.', '')] = values

        del new_state_dict['fc.weight']
        del new_state_dict['fc.bias']

        model_body.load_state_dict(new_state_dict, strict=False)
        model_context.load_state_dict(new_state_dict, strict=False)

        print('completed preparing body model')
        return model_context, model_body
    else:
        model_name = '%s_places365.pth.tar' % context_model
        model_file = os.path.join(model_dir, model_name)
        if not os.path.exists(model_file):
            download_command = 'wget ' + 'http://places2.csail.mit.edu/models_places365/' + model_name + ' -O ' + model_file
            os.system(download_command)

        save_file = os.path.join(model_dir, '%s_places365_py36.pth.tar' % context_model)
        from functools import partial
        import pickle
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)
        torch.save(model, save_file)

        # create the network architecture
        model_context = models.__dict__[context_model](num_classes=365)
        checkpoint = torch.load(save_file, map_location=lambda storage,
                                                               loc: storage)  # model trained in GPU could be deployed in CPU machine like this!
        if context_model == 'densenet161':
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            state_dict = {str.replace(k, 'norm.', 'norm'): v for k, v in state_dict.items()}
            state_dict = {str.replace(k, 'conv.', 'conv'): v for k, v in state_dict.items()}
            state_dict = {str.replace(k, 'normweight', 'norm.weight'): v for k, v in state_dict.items()}
            state_dict = {str.replace(k, 'normrunning', 'norm.running'): v for k, v in state_dict.items()}
            state_dict = {str.replace(k, 'normbias', 'norm.bias'): v for k, v in state_dict.items()}
            state_dict = {str.replace(k, 'convweight', 'conv.weight'): v for k, v in state_dict.items()}
        else:
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name
        model_context.load_state_dict(state_dict)
        model_context.eval()
        model_context.cpu()
        torch.save(model_context, os.path.join(model_dir, 'context_model' + '.pth'))
        model_body = models.__dict__[body_model](pretrained=True)
        model_body.cpu()
        torch.save(model_body, os.path.join(model_dir, 'body_model' + '.pth'))

        print('completed preparing body model')
        return model_context, model_body


def load_model_pair(fold=0, data='emotic'):
    if data == 'emotic':
        model_body = torch.load(f'./weight/checkpoints/{fold}_body.pth')
        model_context = torch.load(f'./weight/checkpoints/{fold}_context.pth')
    else:
        model_body = torch.load(f'./weight/checkpoints/caer_model_body.pth')
        model_context = torch.load(f'./weight/checkpoints/caer_model_context.pth')
    # import pdb; pdb.set_trace()
    print('completed preparing body model')
    return model_context, model_body


def load_face_model():
    # save_file = os.path.join('./weight/models/', 'multitask_fold_all.pth')
    # checkpoint = torch.load(save_file)
    #
    # model = densenet169(pretrained=False)
    # new_state_dict = {}
    # for key, values in checkpoint.items():
    #     new_state_dict[key.replace('module.backbone.', '')] = values
    #
    # model.load_state_dict(new_state_dict, strict=False)
    # model = nn.Sequential(*list(model.children())[:-1])
    new_state_dict = torch.load('./weight/models/PrivateTest_model.t7')

    model = VGG('VGG19')

    model.load_state_dict(new_state_dict['net'], strict=True)

    return model


def load_sentiment_model():
    save_file = os.path.join('./weight/models/', 'vgg19_finetuned_all.pth')
    model = KitModel(save_file)
    return model


def prepare_model_relation(model_name='resnet50', model_dir='./', file_name='UnbiasedEmo_best'):
    ''' Download imagenet pretrained models for context_model and body_model.
    :param context_model: Model to use for conetxt features.
    :param body_model: Model to use for body features.
    :param model_dir: Directory path where to store pretrained models.
    :return: Yolo model after loading model weights
    '''
    if model_name == 'resnet50':
        print('Model EmotionNet Loaded')
        save_file = f'./weight/models/{file_name}.pth.tar'
        checkpoint = torch.load(save_file)

        model_object = models.resnet50(pretrained=True)

        new_state_dict = {}
        for key, values in checkpoint['state_dict'].items():
            new_state_dict[key.replace('module.', '')] = values

        del new_state_dict['fc.weight']
        del new_state_dict['fc.bias']

        model_object.load_state_dict(new_state_dict, strict=False)

        weight = model_object.conv1.weight.clone()
        model_object.conv1 = nn.Conv2d(30, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        with torch.no_grad():
            for i in range(10):
                model_object.conv1.weight[:, (i * 3):(i * 3 + 3)] = weight

        model_object = nn.Sequential(*(list(model_object.children())[:-1]))
        print('completed preparing body model')
        return model_object

    else:
        model_base = 'resnet18'
        model_name = '%s_places365.pth.tar' % model_base
        model_file = os.path.join(model_dir, model_name)
        if not os.path.exists(model_file):
            download_command = 'wget ' + 'http://places2.csail.mit.edu/models_places365/' + model_name + ' -O ' + model_file
            os.system(download_command)

        save_file = os.path.join(model_dir, '%s_places365_py36.pth.tar' % model_base)
        from functools import partial
        import pickle
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)
        torch.save(model, save_file)

        # create the network architecture
        model = models.__dict__[model_base](num_classes=365)
        checkpoint = torch.load(save_file, map_location=lambda storage,
                                                               loc: storage)  # model trained in GPU could be deployed in CPU machine like this!
        if model == 'densenet161':
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            state_dict = {str.replace(k, 'norm.', 'norm'): v for k, v in state_dict.items()}
            state_dict = {str.replace(k, 'conv.', 'conv'): v for k, v in state_dict.items()}
            state_dict = {str.replace(k, 'normweight', 'norm.weight'): v for k, v in state_dict.items()}
            state_dict = {str.replace(k, 'normrunning', 'norm.running'): v for k, v in state_dict.items()}
            state_dict = {str.replace(k, 'normbias', 'norm.bias'): v for k, v in state_dict.items()}
            state_dict = {str.replace(k, 'convweight', 'conv.weight'): v for k, v in state_dict.items()}
        else:
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name
        model.load_state_dict(state_dict)
        model.eval()
        model.cpu()
        # torch.save(model, os.path.join(model_dir, 'context_model' + '.pth'))

        weight = model.conv1.weight.clone()
        model.conv1 = nn.Conv2d(30, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        with torch.no_grad():
            for i in range(10):
                model.conv1.weight[:, (i * 3):(i * 3 + 3)] = weight

        model = nn.Sequential(*(list(model.children())[:-1]))
        print('completed preparing body model')
        return model


if __name__ == '__main__':
    model = prepare_model_relation()
