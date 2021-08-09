import torch.nn.functional as F

from pre_process.prepare_models import *
from .cbam1d import ChannelAttention, SpatialAttention, SKConv, ChannelAttentionv2


class Emotic(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features=2048, num_body_features=2048, cont_vad=False):
        super(Emotic, self).__init__()
        self.model_context, self.model_body = prep_models(model_dir='proj/debug_exp/models',
                                                          context_model='resnet50',
                                                          file_name='model_best')
        self.model_context = nn.Sequential(*(list(self.model_context.children())[:-1]))
        self.model_body = nn.Sequential(*(list(self.model_body.children())[:-1]))

        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((self.num_context_features + self.num_body_features), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.6)

        self.fc_cat = nn.Linear(256, 26)
        self.fc_cont = nn.Linear(256, 3)
        self.relu = nn.ReLU()
        self.cont_vad = cont_vad

    def forward(self, x_context, x_body):
        # import pdb; pdb.set_trace()
        x_context = self.model_context(x_context)
        x_body = self.model_body(x_body)
        context_features = x_context.view(x_context.size(0), -1)
        body_features = x_body.view(x_body.size(0), -1)
        fuse_features = torch.cat((context_features, body_features), 1)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)
        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        cat_out = self.fc_cat(fuse_out)
        if self.cont_vad:
            cont_out = self.fc_cont(fuse_out)
            return cat_out, cont_out
        else:
            return cat_out


class Emotest(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features):
        super(Emotest, self).__init__()
        self.num_context_features = num_context_features

        self.fc1 = nn.Linear(self.num_context_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.d1 = nn.Dropout(p=0.4)
        self.fc_cat = nn.Linear(512, 26)

        self.relu = nn.ReLU()

    def forward(self, x_context):
        # import pdb; pdb.set_trace()
        context_features = x_context
        fuse_features = context_features.view(context_features.size(0), -1)
        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)

        cat_out = self.fc_cat(fuse_out)

        return cat_out


class Emotic_depth(nn.Module):

    def __init__(self, num_context_features, num_body_features, num_depth_features):
        super(Emotic_depth, self).__init__()
        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.num_depth_features = num_depth_features
        self.fc1 = nn.Linear((self.num_context_features + num_body_features + num_depth_features), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.6)
        self.fc_cat = nn.Linear(256, 26)
        self.fc_cont = nn.Linear(256, 3)
        self.relu = nn.ReLU()

    def forward(self, x_context, x_body, x_depth):
        context_features = x_context
        body_features = x_body
        depth_features = x_depth
        fuse_features = torch.cat((context_features, body_features, depth_features), 1)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)
        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)
        cat_out = self.fc_cat(fuse_out)
        cont_out = self.fc_cont(fuse_out)
        return cat_out, cont_out


class Emotic_face_body(nn.Module):

    def __init__(self, num_context_features, num_face_features, num_body_features):
        super(Emotic_face_body, self).__init__()
        self.num_context_features = num_context_features
        self.num_face_features = num_face_features
        self.num_body_features = num_body_features

        self.fc1 = nn.Linear((self.num_context_features + num_face_features + num_body_features),
                             (self.num_context_features + num_face_features + num_body_features))
        self.bn1 = nn.BatchNorm1d((self.num_context_features + num_face_features + num_body_features))
        self.d1 = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear((self.num_context_features + num_face_features + num_body_features), 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.d2 = nn.Dropout(p=0.6)
        self.fc_cat = nn.Linear(256, 26)
        self.fc_cont = nn.Linear(256, 3)
        self.relu = nn.ReLU()

    def forward(self, x_context, x_face, x_body):
        context_features = x_context
        face_features = x_face
        body_features = x_body

        fuse_features = torch.cat((context_features, face_features, body_features), 1)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)
        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        fuse_out = self.fc2(fuse_out)
        fuse_out = self.bn2(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d2(fuse_out)

        cat_out = self.fc_cat(fuse_out)
        cont_out = self.fc_cont(fuse_out)
        return cat_out, cont_out


class Emotic_mask(nn.Module):

    def __init__(self, num_mask_features, num_face_features, num_body_features):
        super(Emotic_mask, self).__init__()
        self.num_mask_features = num_mask_features
        self.num_face_features = num_face_features
        self.num_body_features = num_body_features

        self.fc1 = nn.Linear((self.num_mask_features + num_face_features + num_body_features),
                             (self.num_mask_features + num_face_features + num_body_features))
        self.bn1 = nn.BatchNorm1d((self.num_mask_features + num_face_features + num_body_features))
        self.d1 = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear((self.num_mask_features + num_face_features + num_body_features), 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.d2 = nn.Dropout(p=0.6)
        self.fc_cat = nn.Linear(256, 26)
        self.fc_cont = nn.Linear(256, 3)
        self.relu = nn.ReLU()

    def forward(self, x_context, x_face, x_body):
        context_features = x_context
        face_features = x_face
        body_features = x_body

        fuse_features = torch.cat((context_features, face_features, body_features), 1)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)
        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        fuse_out = self.fc2(fuse_out)
        fuse_out = self.bn2(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d2(fuse_out)

        cat_out = self.fc_cat(fuse_out)
        cont_out = self.fc_cont(fuse_out)
        return cat_out, cont_out


class Emotic_v2(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_hook):
        super(Emotic_v2, self).__init__()

        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((num_context_features + num_body_features + num_hook),
                             (num_context_features + num_body_features + num_hook))
        self.bn1 = nn.BatchNorm1d((num_context_features + num_body_features + num_hook))
        self.d1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear((num_context_features + num_body_features + num_hook), 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.d2 = nn.Dropout(p=0.5)
        self.fc_cat = nn.Linear(256, 26)
        # self.fc_cont = nn.Linear(256, 3)
        self.relu = nn.ReLU()

    def forward(self, x_context, x_body, x_hook):
        context_features = x_context.view(x_context.size(0), -1)
        body_features = x_body.view(x_body.size(0), -1)
        hook_features = x_hook

        fuse_features = torch.cat((context_features, body_features, hook_features), 1)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)
        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)
        fuse_out = self.fc2(fuse_out)
        fuse_out = self.bn2(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d2(fuse_out)
        cat_out = self.fc_cat(fuse_out)
        # cont_out = self.fc_cont(fuse_out)
        return cat_out


class Simple_MLP(nn.Module):
    def __init__(self, num_features_object, num_bbox, num_word_embedding, N):
        super(Simple_MLP, self).__init__()
        self.num_features_object = num_features_object
        self.num_bbox = num_bbox
        self.num_word_embedding = num_word_embedding

        self.fc1 = nn.Linear(N * (self.num_features_object + self.num_bbox + self.num_word_embedding), 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.d1 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, features_object, bbox, word_embedding):
        all_in_features = torch.cat((features_object, bbox, word_embedding), dim=2)
        all_in_features = all_in_features.view(all_in_features.shape[0], -1)
        out = self.fc1(all_in_features)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.d1(out)
        return out


class Sentiment_MLP(nn.Module):
    def __init__(self, num_features_object, num_bbox, num_word_embedding, N, num_sentiment, embracement_size):
        super(Sentiment_MLP, self).__init__()
        self.num_features_object = num_features_object
        self.num_bbox = num_bbox
        self.num_word_embedding = num_word_embedding
        self.num_sentiment = num_sentiment

        self.fc1 = nn.Linear(
            (self.num_features_object + self.num_bbox + self.num_word_embedding + 256),
            embracement_size)
        self.bn1 = nn.BatchNorm1d(embracement_size)
        self.d1 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(self.num_sentiment, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.d2 = nn.Dropout(p=0.5)

        self.ca = ChannelAttention(10)
        self.sa = SpatialAttention()

        self.Embrace = EmbraceNet(
            input_size_list=[num_features_object * N, num_bbox * N, num_word_embedding * N, 256 * N],
            embracement_size=embracement_size)
        self.sk = SKConv(features=10)

    def forward(self, features_object, bbox, word_embedding, sentiment):
        list_tensor = torch.Tensor().cuda()

        for i in range(sentiment.size(1)):
            out_s = sentiment[:, i, :].view(sentiment.size(0), -1)
            out_s = self.fc2(out_s)
            out_s = self.bn2(out_s)
            out_s = self.relu(out_s)
            out_s = self.d2(out_s)
            out_s = out_s.unsqueeze(dim=1)
            list_tensor = torch.cat((list_tensor, out_s), dim=1)

        # embrace_features = self.Embrace(input_list=[features_object.view(features_object.size(0), -1),
        #                                             bbox.view(bbox.size(0), -1),
        #                                             word_embedding.view(word_embedding.size(0), -1),
        #                                             list_tensor.view(list_tensor.size(0), -1)])

        all_in_features = torch.cat((features_object, bbox, word_embedding, list_tensor), dim=2)
        all_in_features = self.ca(all_in_features) * all_in_features
        all_in_features = self.sa(all_in_features) * all_in_features
        # all_in_features = self.sk(all_in_features)
        all_in_features = all_in_features.sum(dim=1)
        all_in_features = all_in_features.view(all_in_features.shape[0], -1)

        out = self.fc1(all_in_features)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.d1(out)
        out = out
        return out


class FusionNetwork(nn.Module):
    def __init__(self, use_face=True, use_context=True, num_class=7, use_attention=True):
        super().__init__()

        # when attention is used, must provide both face and context
        if use_attention:
            assert use_face and use_context

        self.use_face = use_face
        self.use_context = use_context
        self.use_attention = use_attention

        if use_attention:
            # only compute weight if fusion attention is used
            self.face_stream_conv = CNN2DBlock(conv_num=2, cnn_num_kernels=[2048, 1, 1], cnn_kernel_size=1, bn=False,
                                               relu=False, maxpool=False)
            self.context_stream_conv = CNN2DBlock(conv_num=2, cnn_num_kernels=[2048, 1, 1], cnn_kernel_size=1,
                                                  bn=False, relu=False, maxpool=False)

        # when using both face and context, input to conv layer has shape 1x1x(256*2), otherwise, the shape is 1x1x256
        self.conv1 = nn.Conv2d(2048 * 2, 256, kernel_size=1) if (use_face and use_context) else nn.Conv2d(2048, 128,
                                                                                                          kernel_size=1)
        self.conv2 = nn.Conv2d(256, num_class, kernel_size=1)
        self.dropout = nn.Dropout2d()

    def forward(self, face=None, context=None):
        assert face is not None or context is not None
        # when attention is used, must provide both face and context
        if self.use_attention:
            assert face is not None and context is not None

            # compute weights
            face_weights = self.face_stream_conv(face)
            context_weights = self.context_stream_conv(context)

            weights = torch.cat([face_weights, context_weights], dim=1)
            weights = F.softmax(weights, dim=1)

            face = face * weights[:, 0, :].unsqueeze(dim=1)
            context = context * weights[:, 1, :].unsqueeze(dim=1)

        if self.use_context and self.use_face:
            features = torch.cat([face, context], dim=1)
        elif self.use_face:
            features = face
        else:
            features = context

        features = F.relu(self.conv1(features))
        features = self.dropout(features)
        features = self.conv2(features)

        return features


class CNN2DBlock(nn.Module):
    def __init__(self, conv_num, cnn_num_kernels, cnn_kernel_size=3, bn=True, relu=True, maxpool=True,
                 maxpool_kernel_size=2):
        """
        A CNN2DBlock with architecture (CNN + BN + ReLU + max-pool) x N
        Inputs:
        - conv_num (int): number of convolution layers
        - cnn_num_kernels (list of int with size conv_num + 1): the first elements is the number of channel of the inputs, the rest are theh number of kernels in each convolution layer
        - cnn_kernel_size (int): size of kernels
        - bn (boolean): use batch normalization or not
        - relu (boolean): use relu or not
        - maxpool (boolean): use maxpool layers or not (number of maxpool layers = conv_num - 1)
        - maxpool_kernel_size (int): size of maxpool kernel
        """
        super().__init__()
        padding = int((cnn_kernel_size - 1) / 2)
        self.convs = nn.ModuleList(
            [nn.Conv2d(cnn_num_kernels[i], cnn_num_kernels[i + 1], kernel_size=cnn_kernel_size, padding=padding) for i
             in range(conv_num)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(cnn_num_kernels[i + 1]) for i in range(conv_num)]) if bn else None
        self.relu = nn.ReLU() if relu else None
        self.maxpool = nn.MaxPool2d(maxpool_kernel_size) if maxpool else None

    def forward(self, x):
        n = len(self.convs)
        for i in range(n):
            x = self.convs[i](x)
            if self.bn is not None:
                x = self.bn[i](x)
            if self.relu is not None:
                x = self.relu(x)
            if self.maxpool is not None:
                x = self.maxpool(x)

        return x


class Emotic_v3(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_hook, cont_vad=False):
        super(Emotic_v3, self).__init__()

        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((num_context_features + num_body_features),
                             256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.5)
        self.fc_cat = nn.Linear(256 + num_hook, 26)

        self.relu = nn.ReLU()
        self.cont_vad = cont_vad
        if cont_vad:
            self.fc_cont = nn.Linear(256, 3)

    def forward(self, x_context, x_body, x_hook):
        context_features = x_context.view(x_context.size(0), -1)
        body_features = x_body.view(x_body.size(0), -1)
        hook_features = x_hook

        fuse_features = torch.cat((context_features, body_features), 1)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)
        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        cat_out = self.fc_cat(torch.cat((fuse_out, hook_features), 1))
        if self.cont_vad:
            cont_out = self.fc_cont(fuse_out)
            return cat_out, cont_out
        else:
            return cat_out


class EmbraceNet(nn.Module):

    def __init__(self, input_size_list, embracement_size=256, bypass_docking=False):
        """
        Initialize an EmbraceNet module.
        Args:
          device: A "torch.device()" object to allocate internal parameters of the EmbraceNet module.
          input_size_list: A list of input sizes.
          embracement_size: The length of the output of the embracement layer ("c" in the paper).
          bypass_docking: Bypass docking step, i.e., connect the input data directly to the embracement layer. If True, input_data must have a shape of [batch_size, embracement_size].
        """
        super(EmbraceNet, self).__init__()

        self.input_size_list = input_size_list
        self.embracement_size = embracement_size
        self.bypass_docking = bypass_docking

        if (not bypass_docking):
            for i, input_size in enumerate(input_size_list):
                setattr(self, 'docking_%d' % (i), nn.Linear(input_size, embracement_size))

    def forward(self, input_list, availabilities=None, selection_probabilities=None):
        """
        Forward input data to the EmbraceNet module.
        Args:
          input_list: A list of input data. Each input data should have a size as in input_size_list.
          availabilities: A 2-D tensor of shape [batch_size, num_modalities], which represents the availability of data for each modality. If None, it assumes that data of all modalities are available.
          selection_probabilities: A 2-D tensor of shape [batch_size, num_modalities], which represents probabilities that output of each docking layer will be selected ("p" in the paper). If None, the same probability of being selected will be used for each docking layer.
        Returns:
          A 2-D tensor of shape [batch_size, embracement_size] that is the embraced output.
        """

        # check input data
        assert len(input_list) == len(self.input_size_list)
        num_modalities = len(input_list)
        batch_size = input_list[0].shape[0]

        # docking layer
        docking_output_list = []
        if (self.bypass_docking):
            docking_output_list = input_list
        else:
            for i, input_data in enumerate(input_list):
                x = getattr(self, 'docking_%d' % (i))(input_data)
                x = nn.functional.relu(x)
                docking_output_list.append(x)

        # check availabilities
        if (availabilities is None):
            availabilities = torch.ones(batch_size, len(input_list), dtype=torch.float).cuda()
        else:
            availabilities = availabilities.float().cuda()

        # adjust selection probabilities
        if (selection_probabilities is None):
            selection_probabilities = torch.ones(batch_size, len(input_list), dtype=torch.float).cuda()
        selection_probabilities = torch.mul(selection_probabilities, availabilities)

        probability_sum = torch.sum(selection_probabilities, dim=-1, keepdim=True)
        selection_probabilities = torch.div(selection_probabilities, probability_sum)

        # stack docking outputs
        docking_output_stack = torch.stack(docking_output_list,
                                           dim=-1)  # [batch_size, embracement_size, num_modalities]

        # embrace
        modality_indices = torch.multinomial(selection_probabilities, num_samples=self.embracement_size,
                                             replacement=True)  # [batch_size, embracement_size]
        modality_toggles = nn.functional.one_hot(modality_indices,
                                                 num_classes=num_modalities).float()  # [batch_size, embracement_size, num_modalities]

        embracement_output_stack = torch.mul(docking_output_stack, modality_toggles)
        embracement_output = torch.sum(embracement_output_stack, dim=-1)  # [batch_size, embracement_size]

        return embracement_output


class Emotic_v4(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_hook, cont_vad=False):
        super(Emotic_v4, self).__init__()

        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((num_context_features + num_body_features), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.5)
        self.fc_cat = nn.Linear((256 + num_hook), 26)

        self.relu = nn.ReLU()
        self.cont_vad = cont_vad
        if cont_vad:
            self.fc_cont = nn.Linear((256 + num_hook), 3)

        self.fc3 = nn.Linear(self.num_context_features, 26)
        self.fc4 = nn.Linear(self.num_body_features, 26)
        self.fc5 = nn.Linear(num_hook, 26)

    def forward(self, x_context, x_body, x_hook):
        context_features = x_context.view(x_context.size(0), -1)
        body_features = x_body.view(x_body.size(0), -1)
        hook_features = x_hook

        fuse_features = torch.cat((context_features, body_features), 1)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)
        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        # fuse_out = self.fc2(torch.cat((fuse_out, hook_features), 1))
        # fuse_out = self.bn2(fuse_out)
        # fuse_out = self.relu(fuse_out)
        # fuse_out = self.d2(fuse_out)

        out_context = self.fc3(context_features)
        out_body = self.fc4(body_features)
        out_hook = self.fc5(hook_features)
        cat_out = self.fc_cat(torch.cat((fuse_out, hook_features), 1))
        if self.cont_vad:
            cont_out = self.fc_cont(fuse_out)
            return cat_out, cont_out, out_context, out_body, out_hook
        else:
            return cat_out, out_context, out_body, out_hook


class Emotic_v5(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_hook):
        super(Emotic_v5, self).__init__()

        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((num_context_features + num_body_features),
                             256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.6)
        self.relu = nn.ReLU()
        self.fc_cont = nn.Linear((256 + num_hook), 3)

        self.fc3 = nn.Linear(self.num_context_features, 3)
        self.fc4 = nn.Linear(self.num_body_features, 3)
        self.fc5 = nn.Linear(num_hook, 3)

    def forward(self, x_context, x_body, x_hook):
        context_features = x_context.view(x_context.size(0), -1)
        body_features = x_body.view(x_body.size(0), -1)
        hook_features = x_hook

        fuse_features = torch.cat((context_features, body_features), 1)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)
        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        out_context = self.fc3(context_features)
        out_body = self.fc4(body_features)
        out_hook = self.fc5(hook_features)
        cont_out = self.fc_cont(torch.cat((fuse_out, hook_features), 1))

        return cont_out, out_context, out_body, out_hook


class MLP_2layer(nn.Module):

    def __init__(self, in_features, out_features, p_drop=0.6):
        super(MLP_2layer, self).__init__()

        self.fc1 = nn.Linear(in_features, in_features // 8, bias=False)  # 6*6 from image dimension
        self.bn1 = nn.BatchNorm1d(in_features // 8)
        self.fc2 = nn.Linear(in_features // 8, out_features, bias=False)

        self.d1 = nn.Dropout(p=p_drop)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.d1(x)

        x = self.fc2(x)
        # x = self.bn1(x)
        # x = torch.sigmoid(x)

        return x


class MLP_1layer(nn.Module):

    def __init__(self, in_features, out_features, p_drop=0.6):
        super(MLP_1layer, self).__init__()

        self.fc1 = nn.Linear(in_features, out_features)  # 6*6 from image dimension
        self.bn1 = nn.BatchNorm1d(out_features)
        self.d1 = nn.Dropout(p=p_drop)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.d1(x)
        return x


class Objects_net(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_features_object, num_bbox, num_word_embedding, N, num_sentiment, num_body, embracement_size):
        super(Objects_net, self).__init__()
        self.num_features_object = num_features_object
        self.num_bbox = num_bbox
        self.num_word_embedding = num_word_embedding
        self.num_sentiment = num_sentiment
        self.num_body = num_body
        self.N = N
        self.fc1 = nn.Linear(
            (self.num_features_object + self.num_bbox + self.num_word_embedding + self.num_sentiment // 16 + num_body),
            embracement_size)
        self.bn1 = nn.BatchNorm1d(embracement_size)
        self.d1 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(self.num_sentiment, self.num_sentiment // 16)
        self.bn2 = nn.BatchNorm1d(self.num_sentiment // 16)
        self.d2 = nn.Dropout(p=0.5)

        self.ca = ChannelAttention(10)
        self.sa = SpatialAttention()

    def forward(self, features_object, bbox, word_embedding, sentiment, body):
        list_tensor = torch.Tensor().cuda()

        for i in range(sentiment.size(1)):
            out_s = sentiment[:, i, :].view(sentiment.size(0), -1)
            out_s = self.fc2(out_s)
            out_s = self.bn2(out_s)
            out_s = self.relu(out_s)
            out_s = self.d2(out_s)
            out_s = out_s.unsqueeze(dim=1)
            list_tensor = torch.cat((list_tensor, out_s), dim=1)
        # import pdb; pdb.set_trace()
        body = body.view(body.size(0), -1)
        body = body.unsqueeze(dim=1).repeat(1, self.N, 1)
        all_in_features = torch.cat((features_object, bbox, word_embedding, list_tensor, body), dim=2)
        all_in_features = self.ca(all_in_features) * all_in_features
        all_in_features = self.sa(all_in_features) * all_in_features

        all_in_features = all_in_features.sum(dim=1)
        all_in_features = all_in_features.view(all_in_features.shape[0], -1)

        out = self.fc1(all_in_features)
        out = self.bn1(out)
        out = self.relu(out)

        return out


class Objects_netv2(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_features_object, num_bbox, num_word_embedding, N, num_sentiment, num_body, embracement_size):
        super(Objects_netv2, self).__init__()
        self.num_features_object = num_features_object
        self.num_bbox = num_bbox
        self.num_word_embedding = num_word_embedding
        self.num_sentiment = num_sentiment
        self.num_body = num_body
        self.N = N

        self.mlp_object = MLP_2layer(1024, 128)
        self.mlp_sentiment = MLP_2layer(4096, 128)
        self.mlp_body = MLP_2layer(2048, 128)

        self.fc1 = nn.Linear(
            (128 + self.num_bbox + self.num_word_embedding + 128 + 128),
            embracement_size)
        self.bn1 = nn.BatchNorm1d(embracement_size)
        self.d1 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.ca = ChannelAttentionv2(10)
        self.sa = SpatialAttention()

    def forward(self, features_object, bbox, word_embedding, sentiment, body):
        list_sentiment = torch.Tensor().cuda()
        list_object = torch.Tensor().cuda()
        list_body = torch.Tensor().cuda()
        body = body.view(body.size(0), -1)
        body = body.unsqueeze(dim=1).repeat(1, self.N, 1)

        for i in range(sentiment.size(1)):
            out_s = sentiment[:, i, :].view(sentiment.size(0), -1)
            out_s = self.mlp_sentiment(out_s)
            out_s = out_s.unsqueeze(dim=1)
            list_sentiment = torch.cat((list_sentiment, out_s), dim=1)

        for i in range(features_object.size(1)):
            out_o = features_object[:, i, :].view(features_object.size(0), -1)
            out_o = self.mlp_object(out_o)
            out_o = out_o.unsqueeze(dim=1)
            list_object = torch.cat((list_object, out_o), dim=1)

        for i in range(body.size(1)):
            out_b = body[:, i, :].view(body.size(0), -1)
            out_b = self.mlp_body(out_b)
            out_b = out_b.unsqueeze(dim=1)
            list_body = torch.cat((list_body, out_b), dim=1)

        # import pdb; pdb.set_trace()

        all_in_features = torch.cat((list_object, bbox, word_embedding, list_sentiment, list_body), dim=2)
        all_in_features = self.ca(all_in_features) * all_in_features
        all_in_features = self.sa(all_in_features) * all_in_features

        all_in_features = all_in_features.sum(dim=1)
        all_in_features = all_in_features.view(all_in_features.shape[0], -1)

        out = self.fc1(all_in_features)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.d1(out)

        return out


class Emotic_v6(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_hook, num_places, cont_vad=False):
        super(Emotic_v6, self).__init__()

        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((num_context_features + num_body_features), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.6)
        self.fc_cat = nn.Linear((256 + num_hook + num_places), 26)

        self.relu = nn.ReLU()
        self.cont_vad = cont_vad
        if cont_vad:
            self.fc_cont = nn.Linear((256 + num_hook), 3)

        self.fc3 = nn.Linear(self.num_context_features, 26)
        self.fc4 = nn.Linear(self.num_body_features, 26)
        self.fc5 = nn.Linear(num_hook, 26)

    def forward(self, x_context, x_body, x_hook, x_places):
        context_features = x_context.view(x_context.size(0), -1)
        body_features = x_body.view(x_body.size(0), -1)
        hook_features = x_hook

        fuse_features = torch.cat((context_features, body_features), 1)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)
        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        # fuse_out = self.fc2(torch.cat((fuse_out, hook_features), 1))
        # fuse_out = self.bn2(fuse_out)
        # fuse_out = self.relu(fuse_out)
        # fuse_out = self.d2(fuse_out)

        out_context = self.fc3(context_features)
        out_body = self.fc4(body_features)
        out_hook = self.fc5(hook_features)
        cat_out = self.fc_cat(torch.cat((fuse_out, hook_features, x_places), 1))
        if self.cont_vad:
            cont_out = self.fc_cont(fuse_out)
            return cat_out, cont_out, out_context, out_body, out_hook
        else:
            return cat_out, out_context, out_body, out_hook


class Emotic_v7(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_hook, cont_vad=False, embracement_size=128,
                 num_box=4):
        super(Emotic_v7, self).__init__()

        self.model_context, self.model_body = prep_models(model_dir='proj/debug_exp/models',
                                                          context_model='resnet50')
        self.model_context = nn.Sequential(*(list(self.model_context.children())[:-1]))
        self.model_body = nn.Sequential(*(list(self.model_body.children())[:-1]))

        self.model_hook = Objects_netv2(1024, num_box, 300, N=10, num_sentiment=4096, num_body=2048,
                                        embracement_size=embracement_size)

        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((num_context_features + num_body_features), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.6)
        self.fc_cat = nn.Linear((256 + num_hook), 26)

        self.relu = nn.ReLU()
        self.cont_vad = cont_vad
        if cont_vad:
            self.fc_cont = nn.Linear((256 + num_hook), 3)

        self.fc3 = nn.Linear(self.num_context_features, 26)
        self.fc4 = nn.Linear(self.num_body_features, 26)
        self.fc5 = nn.Linear(num_hook, 26)

    def forward(self, images_context, images_body, features_object, bbox, word_embedding, sentiment_object):

        x_context = self.model_context(images_context)
        x_body = self.model_body(images_body)
        hook_features = self.model_hook(features_object, bbox, word_embedding, sentiment_object, x_body)

        context_features = x_context.view(x_context.size(0), -1)
        body_features = x_body.view(x_body.size(0), -1)

        fuse_features = torch.cat((context_features, body_features), 1)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)

        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        out_context = self.fc3(context_features)
        out_body = self.fc4(body_features)
        out_hook = self.fc5(hook_features)

        cat_out = self.fc_cat(torch.cat((fuse_out, hook_features), 1))

        if self.cont_vad:
            cont_out = self.fc_cont(fuse_out)
            return cat_out, cont_out, out_context, out_body, out_hook
        else:
            return cat_out, out_context, out_body, out_hook


class Sentiment_MLPv2(nn.Module):
    def __init__(self, num_features_object, num_bbox, num_word_embedding, N, num_sentiment, embracement_size):
        super(Sentiment_MLPv2, self).__init__()
        self.num_features_object = num_features_object
        self.num_bbox = num_bbox
        self.num_word_embedding = num_word_embedding
        self.num_sentiment = num_sentiment

        self.fc1 = nn.Linear(
            (self.num_features_object + self.num_bbox + self.num_word_embedding + 30),
            embracement_size)
        self.bn1 = nn.BatchNorm1d(embracement_size)
        self.d1 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.ca = ChannelAttention(10)
        self.sa = SpatialAttention()

        self.Embrace = EmbraceNet(
            input_size_list=[num_features_object * N, num_bbox * N, num_word_embedding * N, 300],
            embracement_size=embracement_size)

    def forward(self, features_object, bbox, word_embedding, sentiment):
        sentiment = sentiment.repeat(1, 1, 10)
        # import pdb; pdb.set_trace()
        embrace_features = self.Embrace(input_list=[features_object.view(features_object.size(0), -1),
                                                    bbox.view(bbox.size(0), -1),
                                                    word_embedding.view(word_embedding.size(0), -1),
                                                    sentiment.view(sentiment.size(0), -1)])

        all_in_features = torch.cat((features_object, bbox, word_embedding, sentiment), dim=2)
        all_in_features = self.ca(all_in_features) * all_in_features
        all_in_features = self.sa(all_in_features) * all_in_features

        all_in_features = all_in_features.sum(dim=1)
        all_in_features = all_in_features.view(all_in_features.shape[0], -1)

        out = self.fc1(all_in_features)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.d1(out)
        out = out + embrace_features
        return out


class Objects_netv3(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_features_object, num_bbox, num_word_embedding, N, num_sentiment, num_body, embracement_size,
                 num_output_sentiment):
        super(Objects_netv3, self).__init__()
        self.num_features_object = num_features_object
        self.num_bbox = num_bbox
        self.num_word_embedding = num_word_embedding
        self.num_sentiment = num_sentiment
        self.num_body = num_body
        self.num_output_sentiment = num_output_sentiment
        self.N = N

        self.mlp_object = MLP_2layer(1024, 256)
        self.mlp_sentiment = MLP_2layer(4096, 256)
        self.mlp_body = MLP_2layer(2048, 256)

        self.fc1 = nn.Linear(
            (256 + self.num_bbox + self.num_word_embedding + 256 + 256 + 3),
            embracement_size)
        self.bn1 = nn.BatchNorm1d(embracement_size)
        self.d1 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.ca = ChannelAttention(10)
        self.sa = SpatialAttention()

    def forward(self, features_object, bbox, word_embedding, sentiment, body, output_sentiment):
        list_sentiment = torch.Tensor().cuda()
        list_object = torch.Tensor().cuda()
        list_body = torch.Tensor().cuda()
        body = body.view(body.size(0), -1)
        body = body.unsqueeze(dim=1).repeat(1, self.N, 1)

        for i in range(sentiment.size(1)):
            out_s = sentiment[:, i, :].view(sentiment.size(0), -1)
            out_s = self.mlp_sentiment(out_s)
            out_s = out_s.unsqueeze(dim=1)
            list_sentiment = torch.cat((list_sentiment, out_s), dim=1)

        for i in range(features_object.size(1)):
            out_o = features_object[:, i, :].view(features_object.size(0), -1)
            out_o = self.mlp_object(out_o)
            out_o = out_o.unsqueeze(dim=1)
            list_object = torch.cat((list_object, out_o), dim=1)

        for i in range(body.size(1)):
            out_b = body[:, i, :].view(body.size(0), -1)
            out_b = self.mlp_body(out_b)
            out_b = out_b.unsqueeze(dim=1)
            list_body = torch.cat((list_body, out_b), dim=1)

        # import pdb; pdb.set_trace()

        all_in_features = torch.cat((list_object, bbox, word_embedding, list_sentiment, list_body, output_sentiment),
                                    dim=2)
        out_weight_impact = self.ca(all_in_features)
        all_in_features = out_weight_impact * all_in_features

        all_in_features = self.sa(all_in_features) * all_in_features

        all_in_features = all_in_features.sum(dim=1)
        all_in_features = all_in_features.view(all_in_features.shape[0], -1)

        out = self.fc1(all_in_features)
        out = self.bn1(out)
        out = self.relu(out)
        # import pdb; pdb.set_trace()
        return out, out_weight_impact, output_sentiment


class Emotic_v8(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_hook, cont_vad=False, embracement_size=128,
                 num_box=4):
        super(Emotic_v8, self).__init__()

        self.model_context, self.model_body = prep_models(model_dir='proj/debug_exp/models',
                                                          context_model='resnet50')
        self.model_context = nn.Sequential(*(list(self.model_context.children())[:-1]))
        self.model_body = nn.Sequential(*(list(self.model_body.children())[:-1]))

        self.model_hook = Objects_netv3(1024, num_box, 50, N=10, num_sentiment=4096, num_body=2048,
                                        embracement_size=embracement_size, num_output_sentiment=3)

        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((num_context_features + num_body_features), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.6)
        self.fc_cat = nn.Linear((256 + num_hook), 26)

        self.relu = nn.ReLU()
        self.cont_vad = cont_vad
        if cont_vad:
            self.fc_cont = nn.Linear((256 + num_hook), 3)

        self.fc3 = nn.Linear(self.num_context_features, 26)
        self.fc4 = nn.Linear(self.num_body_features, 26)
        self.fc5 = nn.Linear(num_hook, 26)

    def forward(self, images_context, images_body, features_object, bbox, word_embedding, sentiment_object,
                sentiment_output):

        x_context = self.model_context(images_context)
        x_body = self.model_body(images_body)
        hook_features, out_weight_impact, output_sentiment = self.model_hook(features_object, bbox, word_embedding,
                                                                             sentiment_object, x_body, sentiment_output)

        context_features = x_context.view(x_context.size(0), -1)
        body_features = x_body.view(x_body.size(0), -1)

        fuse_features = torch.cat((context_features, body_features), 1)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)

        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        out_context = self.fc3(context_features)
        out_body = self.fc4(body_features)
        out_hook = self.fc5(hook_features)

        cat_out = self.fc_cat(torch.cat((fuse_out, hook_features), 1))

        if self.cont_vad:
            cont_out = self.fc_cont(fuse_out)
            return cat_out, cont_out, out_context, out_body, out_hook
        else:
            return cat_out, out_context, out_body, out_hook, out_weight_impact, output_sentiment


class Emotic_v9(nn.Module):
    def __init__(self, ):
        super(Emotic_v9, self).__init__()
        self.model_object = prepare_model_relation()
        self.model_context, self.model_body = prep_models(context_model='resnet50')
        self.model_context = nn.Sequential(*(list(self.model_context.children())[:-1]))
        self.model_body = nn.Sequential(*(list(self.model_body.children())[:-1]))
        self.mlp1 = MLP_2layer(in_features=2048 * 3, out_features=26, p_drop=0.6)

        # self.fc = nn.Linear(in_features=256, out_features=26)

    def forward(self, x_context, x_body, x_objects):
        # import pdb; pdb.set_trace()
        out_o = self.model_object(x_objects)
        out_c = self.model_context(x_context)
        out_b = self.model_body(x_body)
        out_o = out_o.view(out_o.size(0), -1)
        out_c = out_o.view(out_c.size(0), -1)
        out_b = out_o.view(out_b.size(0), -1)

        out = self.mlp1(torch.cat((out_o, out_b, out_c), dim=1))

        return out


class Emotic_face(nn.Module):
    def __init__(self):
        super(Emotic_face, self).__init__()
        model_face = load_face_model()
        self.model_face = nn.Sequential(*(list(model_face.children())[:-1]))
        self.mpl = MLP_1layer(in_features=512, out_features=26)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        out = self.model_face(x)
        out = self.mpl(out)
        return out


class Emotic_v10(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_hook, cont_vad=False, embracement_size=128,
                 num_box=4, num_face=64):
        super(Emotic_v10, self).__init__()

        self.model_context, self.model_body = prep_models(model_dir='proj/debug_exp/models',
                                                          context_model='resnet50')
        self.model_context = nn.Sequential(*(list(self.model_context.children())[:-1]))
        self.model_body = nn.Sequential(*(list(self.model_body.children())[:-1]))

        self.model_hook = Objects_netv2(1024, num_box, 50, N=10, num_sentiment=4096, num_body=2048,
                                        embracement_size=embracement_size)

        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((num_context_features + num_body_features), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.6)
        self.fc_cat = nn.Linear((256 + num_hook + num_face), 26)

        self.relu = nn.ReLU()
        self.cont_vad = cont_vad
        if cont_vad:
            self.fc_cont = nn.Linear((256 + num_hook), 3)

        self.fc3 = nn.Linear(self.num_context_features, 26)
        self.fc4 = nn.Linear(self.num_body_features, 26)
        self.fc5 = nn.Linear(num_hook, 26)

        model_face = load_face_model()
        self.model_face = nn.Sequential(*(list(model_face.children())[:-1]))
        self.mpl = MLP_1layer(in_features=512, out_features=num_face)

    def forward(self, images_context, images_body, features_object, bbox, word_embedding, sentiment_object, x_face):

        x_context = self.model_context(images_context)
        x_body = self.model_body(images_body)
        hook_features = self.model_hook(features_object, bbox, word_embedding, sentiment_object, x_body)

        context_features = x_context.view(x_context.size(0), -1)
        body_features = x_body.view(x_body.size(0), -1)

        fuse_features = torch.cat((context_features, body_features), 1)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)

        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        out_context = self.fc3(context_features)
        out_body = self.fc4(body_features)
        out_hook = self.fc5(hook_features)

        face_features = self.model_face(x_face)
        face_features = self.mpl(face_features)
        cat_out = self.fc_cat(torch.cat((fuse_out, hook_features, face_features), 1))

        if self.cont_vad:
            cont_out = self.fc_cont(fuse_out)
            return cat_out, cont_out, out_context, out_body, out_hook
        else:
            return cat_out, out_context, out_body, out_hook


class Objects_netv4(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_features_object, num_bbox, num_word_embedding, N, num_sentiment, embracement_size):
        super(Objects_netv4, self).__init__()
        self.num_features_object = num_features_object
        self.num_bbox = num_bbox
        self.num_word_embedding = num_word_embedding
        self.num_sentiment = num_sentiment
        self.N = N

        self.mlp_object = MLP_2layer(1024, 128)
        self.mlp_sentiment = MLP_2layer(4096, 128)

        self.fc1 = nn.Linear(
            (128 + self.num_bbox + self.num_word_embedding + 128),
            embracement_size)
        self.bn1 = nn.BatchNorm1d(embracement_size)
        self.d1 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.ca = ChannelAttention(10)
        self.sa = SpatialAttention()

    def forward(self, features_object, bbox, word_embedding, sentiment):
        list_sentiment = torch.Tensor().cuda()
        list_object = torch.Tensor().cuda()

        for i in range(sentiment.size(1)):
            out_s = sentiment[:, i, :].view(sentiment.size(0), -1)
            out_s = self.mlp_sentiment(out_s)
            out_s = out_s.unsqueeze(dim=1)
            list_sentiment = torch.cat((list_sentiment, out_s), dim=1)

        for i in range(features_object.size(1)):
            out_o = features_object[:, i, :].view(features_object.size(0), -1)
            out_o = self.mlp_object(out_o)
            out_o = out_o.unsqueeze(dim=1)
            list_object = torch.cat((list_object, out_o), dim=1)

        # import pdb; pdb.set_trace()

        all_in_features = torch.cat((list_object, bbox, word_embedding, list_sentiment), dim=2)
        all_in_features = self.ca(all_in_features) * all_in_features
        all_in_features = self.sa(all_in_features) * all_in_features

        all_in_features = all_in_features.sum(dim=1)
        all_in_features = all_in_features.view(all_in_features.shape[0], -1)

        out = self.fc1(all_in_features)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.d1(out)

        return out


class Emotic_v11(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_hook, cont_vad=False, embracement_size=128,
                 num_box=4):
        super(Emotic_v11, self).__init__()

        self.model_context, self.model_body = load_model_pair()
        # self.model_context = nn.Sequential(*(list(self.model_context.children())[:-1]))
        # self.model_body = nn.Sequential(*(list(self.model_body.children())[:-1]))

        self.model_hook = Objects_netv4(1024, num_box, 300, N=10, num_sentiment=4096,
                                        embracement_size=embracement_size)

        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((num_context_features + num_body_features), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.6)
        self.fc_cat = nn.Linear((256 + num_hook), 26)

        self.relu = nn.ReLU()
        self.cont_vad = cont_vad
        if cont_vad:
            self.fc_cont = nn.Linear((256 + num_hook), 3)

        self.fc3 = nn.Linear(self.num_context_features, 26)
        self.fc4 = nn.Linear(self.num_body_features, 26)
        self.fc5 = nn.Linear(num_hook, 26)

    def forward(self, images_context, images_body, features_object, bbox, word_embedding, sentiment_object):

        x_context = self.model_context(images_context)
        x_body = self.model_body(images_body)
        hook_features = self.model_hook(features_object, bbox, word_embedding, sentiment_object)

        context_features = x_context.view(x_context.size(0), -1)
        body_features = x_body.view(x_body.size(0), -1)

        fuse_features = torch.cat((context_features, body_features), 1)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)

        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        out_context = self.fc3(context_features)
        out_body = self.fc4(body_features)
        out_hook = self.fc5(hook_features)

        cat_out = self.fc_cat(torch.cat((fuse_out, hook_features), 1))

        if self.cont_vad:
            cont_out = self.fc_cont(fuse_out)
            return cat_out, cont_out, out_context, out_body, out_hook
        else:
            return cat_out, out_context, out_body, out_hook


class Emotic_v12(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_hook, cont_vad=False,
                 num_box=5408, fold=0, weight=False):
        super(Emotic_v12, self).__init__()
        self.weight = weight
        self.model_context, self.model_body = load_model_pair(fold)
        self.model_hook = Objects_netv6(1024, num_box, 300, N=10, num_sentiment=4096, weight=weight)

        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((num_context_features + num_body_features), 256, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.6)
        self.fc_cat = nn.Linear((256 + num_hook), 26)

        self.relu = nn.ReLU()
        self.cont_vad = cont_vad
        if cont_vad:
            self.fc_cont = nn.Linear((256 + num_hook), 3)

        self.fc3 = nn.Linear(self.num_context_features, 26)
        self.fc4 = nn.Linear(self.num_body_features, 26)
        self.fc5 = nn.Linear(num_hook, 26)

    def forward(self, images_context, images_body, features_object, bbox, word_embedding, sentiment_object):

        x_context = self.model_context(images_context)
        x_body = self.model_body(images_body)
        hook_features, impact_weight = self.model_hook(features_object, bbox, word_embedding, sentiment_object)

        context_features = x_context.view(x_context.size(0), -1)
        body_features = x_body.view(x_body.size(0), -1)

        fuse_features = torch.cat((context_features, body_features), 1)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)

        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        out_context = self.fc3(context_features)
        out_body = self.fc4(body_features)
        out_hook = self.fc5(hook_features)

        cat_out = self.fc_cat(torch.cat((fuse_out, hook_features), 1))

        if self.weight:
            return cat_out, impact_weight, out_context, out_body, out_hook
        else:
            return cat_out, out_context, out_body, out_hook


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Objects_netv5(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_features_object, num_bbox, num_word_embedding, N, num_sentiment):
        super(Objects_netv5, self).__init__()
        self.num_features_object = num_features_object
        self.num_bbox = num_bbox
        self.num_word_embedding = num_word_embedding
        self.num_sentiment = num_sentiment
        self.N = N

        self.sp_to_head = nn.Sequential(
            nn.Conv2d(2, 64, 5),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, 5),
            nn.MaxPool2d(2),
            Flatten(),
        )
        self.tail_head = MLP_2layer(self.num_bbox + self.num_features_object + self.num_word_embedding + num_sentiment,
                                    128)

        self.ca = ChannelAttention(10)
        self.sa = SpatialAttention()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features_object, bbox, word_embedding, sentiment):

        list_bbox = torch.Tensor().cuda()

        for i in range(bbox.size(1)):
            out_box = bbox[:, i].squeeze(dim=1)
            out_box = self.sp_to_head(out_box)
            out_box = out_box.unsqueeze(dim=1)
            list_bbox = torch.cat((list_bbox, out_box), dim=1)
        # import pdb;
        # pdb.set_trace()
        all_in_features = torch.cat((features_object, list_bbox, word_embedding, sentiment), dim=2)

        all_in_features = self.ca(all_in_features) * all_in_features
        all_in_features = self.sa(all_in_features) * all_in_features

        all_in_features = all_in_features.sum(dim=1)
        all_in_features = all_in_features.view(all_in_features.shape[0], -1)

        all_in_features = self.tail_head(all_in_features)

        return all_in_features


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            out = self.relu(out)
        else:
            out += residual
            out = self.relu(out)
        return out


class Objects_netv6(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_features_object, num_bbox, num_word_embedding, N, num_sentiment, weight=False):
        super(Objects_netv6, self).__init__()
        self.num_features_object = num_features_object
        self.num_bbox = num_bbox
        self.num_word_embedding = num_word_embedding
        self.num_sentiment = num_sentiment
        self.N = N
        self.weight = weight

        self.mlp_object = MLP_2layer(self.num_features_object + self.num_word_embedding, 256)
        self.mlp_sentiment = MLP_2layer(self.num_sentiment, 256)
        self.mlp_box = MLP_2layer(self.num_bbox, 256)
        self.sp_to_head = nn.Sequential(
            nn.Conv2d(2, 64, 5),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, 5),
            nn.MaxPool2d(2),
            Flatten(),
        )
        self.sp_new = nn.Sequential(
            BasicBlock(2, 16, downsample=True),
            nn.MaxPool2d(2),
            BasicBlock(16, 16),
            nn.MaxPool2d(2),
            Flatten(),
        )

        self.tail_head = MLP_1layer(256 * 3, 256)
        self.ca = ChannelAttention(10)
        self.sa = SpatialAttention()
        # self.sk = SKConv(features=10)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features_object, bbox, word_embedding, sentiment, weight_area):
        list_sentiment = torch.Tensor().cuda()
        list_object = torch.Tensor().cuda()
        list_bbox = torch.Tensor().cuda()

        for i in range(sentiment.size(1)):
            out_s = sentiment[:, i, :].view(sentiment.size(0), -1)
            out_s = self.mlp_sentiment(out_s)
            out_s = out_s.unsqueeze(dim=1)
            list_sentiment = torch.cat((list_sentiment, out_s), dim=1)

        for i in range(features_object.size(1)):
            out_o = features_object[:, i, :].view(features_object.size(0), -1)
            out_w = word_embedding[:, i, :].view(word_embedding.size(0), -1)
            out_o = self.mlp_object(torch.cat((out_o, out_w), dim=1))
            out_o = out_o.unsqueeze(dim=1)
            list_object = torch.cat((list_object, out_o), dim=1)

        for i in range(bbox.size(1)):
            out_box = bbox[:, i].squeeze(dim=1)
            out_box = self.sp_new(out_box)
            out_box = self.mlp_box(out_box)
            out_box = out_box.unsqueeze(dim=1)
            list_bbox = torch.cat((list_bbox, out_box), dim=1)

        all_in_features = torch.cat((list_sentiment, list_object, list_bbox), dim=2)

        out_weight_impact, att = self.ca(all_in_features, weight_area)
        all_in_features = out_weight_impact * all_in_features
        all_in_features = self.sa(all_in_features) * all_in_features

        all_in_features = all_in_features.sum(dim=1)
        all_in_features = all_in_features.view(all_in_features.shape[0], -1)

        all_in_features = self.tail_head(all_in_features)
        if self.weight:
            return all_in_features, att
        else:
            return all_in_features


class Emotic_v13(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_hook):
        super(Emotic_v13, self).__init__()

        self.model_context, self.model_body = load_model_pair()
        self.model_hook = self.model_context

        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.num_hook = num_hook
        self.fc1 = nn.Linear((num_context_features + num_body_features + num_hook), 256, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.6)
        self.fc_cat = nn.Linear(256, 26)
        self.relu = nn.ReLU()

    def forward(self, images_context, images_body, image_object):
        x_context = self.model_context(images_context)
        x_body = self.model_body(images_body)
        x_object = self.model_hook(image_object)

        context_features = x_context.view(x_context.size(0), -1)
        body_features = x_body.view(x_body.size(0), -1)
        object_features = x_object.view(x_object.size(0), -1)

        fuse_features = torch.cat((context_features, body_features, object_features), 1)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)

        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)
        cat_out = self.fc_cat(fuse_out)

        return cat_out


class Emotic_v14(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_hook, cont_vad=False,
                 num_box=5408, fold=0):
        super(Emotic_v14, self).__init__()

        self.model_context, self.model_body = load_model_pair(fold)
        self.model_hook = Objects_netv6(1024, num_box, 300, N=10, num_sentiment=4096)

        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.tail_context = MLP_1layer(in_features=num_context_features, out_features=256)
        self.tail_body = MLP_1layer(in_features=num_body_features, out_features=256)

        self.fc_cat = nn.Linear((256 * 3), 26)

        self.relu = nn.ReLU()
        self.cont_vad = cont_vad
        if cont_vad:
            self.fc_cont = nn.Linear((256 * 3), 3)

        self.fc3 = nn.Linear(256, 26)
        self.fc4 = nn.Linear(256, 26)
        self.fc5 = nn.Linear(num_hook, 26)

    def forward(self, images_context, images_body, features_object, bbox, word_embedding, sentiment_object):
        x_context = self.model_context(images_context)
        x_body = self.model_body(images_body)
        hook_features = self.model_hook(features_object, bbox, word_embedding, sentiment_object)

        context_features = x_context.view(x_context.size(0), -1)
        body_features = x_body.view(x_body.size(0), -1)

        context_features = self.tail_context(context_features)
        body_features = self.tail_body(body_features)

        fuse_features = torch.cat((context_features, body_features, hook_features), 1)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)

        out_context = self.fc3(context_features)
        out_body = self.fc4(body_features)
        out_hook = self.fc5(hook_features)

        cat_out = self.fc_cat(fuse_features)

        return cat_out, out_context, out_body, out_hook


# class Emotic_v15(nn.Module):
#     ''' Emotic Model'''
#
#     def __init__(self, num_context_features, num_body_features, num_hook, cont_vad=False,
#                  num_box=5408, fold=0, num_face=512):
#         super(Emotic_v15, self).__init__()
#
#         self.model_context, self.model_body = load_model_pair(fold)
#         self.model_hook = Objects_netv6(1024, num_box, 300, N=10, num_sentiment=4096)
#
#         self.model_face = load_face_model()
#         self.model_face = nn.Sequential(*(list(self.model_face.children())[:-1]))
#
#         self.num_context_features = num_context_features
#         self.num_body_features = num_body_features
#
#         self.tail_context = MLP_1layer(in_features=num_context_features, out_features=256)
#         self.tail_body = MLP_1layer(in_features=num_body_features+num_face, out_features=256)
#
#         self.fc_cat = nn.Linear((256 * 3), 26)
#
#         self.relu = nn.ReLU()
#         self.cont_vad = cont_vad
#         if cont_vad:
#             self.fc_cont = nn.Linear((256 * 3), 3)
#
#         self.fc3 = nn.Linear(256, 26)
#         self.fc4 = nn.Linear(256, 26)
#         self.fc5 = nn.Linear(num_hook, 26)
#
#     def forward(self, images_context, images_body, features_object, bbox, word_embedding, sentiment_object, image_face):
#         x_context = self.model_context(images_context)
#         x_body = self.model_body(images_body)
#         hook_features = self.model_hook(features_object, bbox, word_embedding, sentiment_object)
#         x_face = self.model_face(image_face)
#
#         context_features = x_context.view(x_context.size(0), -1)
#         body_features = x_body.view(x_body.size(0), -1)
#         x_face = x_face.view(x_face.size(0), -1)
#
#         context_features = self.tail_context(context_features)
#         body_features = self.tail_body(torch.cat((body_features, x_face), dim=1))
#
#         fuse_features = torch.cat((context_features, body_features, hook_features), 1)
#         fuse_features = fuse_features.view(fuse_features.size(0), -1)
#
#         out_context = self.fc3(context_features)
#         out_body = self.fc4(body_features)
#         out_hook = self.fc5(hook_features)
#
#         cat_out = self.fc_cat(fuse_features)
#
#         return cat_out, out_context, out_body, out_hook


class Emotic_v15(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_hook, cont_vad=False,
                 num_box=5408, fold=0, weight=False, num_face=512):
        super(Emotic_v15, self).__init__()
        self.weight = weight
        self.model_context, self.model_body = load_model_pair(fold)
        self.model_hook = Objects_netv6(1024, num_box, 300, N=10, num_sentiment=4096, weight=weight)
        self.model_face = load_face_model()
        self.model_face = nn.Sequential(*(list(self.model_face.children())[:-1]))

        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((num_context_features + num_body_features + num_face), 256, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.6)
        self.fc_cat = nn.Linear((256 + num_hook), 26)

        self.relu = nn.ReLU()
        self.cont_vad = cont_vad
        if cont_vad:
            self.fc_cont = nn.Linear((256 + num_hook), 3)

        self.fc3 = nn.Linear(self.num_context_features, 26)
        self.fc4 = nn.Linear(self.num_body_features, 26)
        self.fc5 = nn.Linear(num_hook, 26)

    def forward(self, images_context, images_body, features_object, bbox, word_embedding, sentiment_object, image_face,
                weight_area):

        x_context = self.model_context(images_context)
        x_body = self.model_body(images_body)
        hook_features, impact_weight = self.model_hook(features_object, bbox, word_embedding, sentiment_object,
                                                       weight_area)
        x_face = self.model_face(image_face)

        context_features = x_context.view(x_context.size(0), -1)
        body_features = x_body.view(x_body.size(0), -1)
        x_face = x_face.view(x_face.size(0), -1)
        # import pdb; pdb.set_trace()
        fuse_features = torch.cat((context_features, body_features, x_face), 1)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)

        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        out_context = self.fc3(context_features)
        out_body = self.fc4(body_features)
        out_hook = self.fc5(hook_features)

        cat_out = self.fc_cat(torch.cat((fuse_out, hook_features), 1))

        if self.weight:
            return cat_out, impact_weight, out_context, out_body, out_hook
        else:
            return cat_out, out_context, out_body, out_hook


class AttentionModel(nn.Module):
    def __init__(self, hidden_layer=380):
        super(AttentionModel, self).__init__()

        self.attn_hidden_layer = hidden_layer
        self.net = nn.Sequential(nn.Conv2d(2048, self.attn_hidden_layer, kernel_size=1),
                                 nn.Conv2d(self.attn_hidden_layer, 1, kernel_size=1))

    def forward(self, x):
        attn_mask = self.net(x)  # Shape BS 1x7x7
        attn_mask = attn_mask.view(attn_mask.size(0), -1)
        attn_mask = nn.Softmax(dim=1)(attn_mask)
        attn_mask = attn_mask.view(attn_mask.size(0), 1, x.size(2), x.size(3))
        x_attn = x * attn_mask
        x = x + x_attn
        return x, attn_mask


class Emotic_v16(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_hook,
                 num_box=5408, fold=0, weight=False, num_face=512, n=10):
        super(Emotic_v16, self).__init__()
        self.weight = weight

        self.model_context, self.model_body = load_model_pair(fold)
        self.model_context = nn.Sequential(*list(self.model_context.children())[:-1])

        self.attention_module = AttentionModel()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.model_hook = Objects_netv6(1024, num_box, 300, N=n, num_sentiment=4096, weight=weight)
        self.model_face = load_face_model()
        self.model_face = nn.Sequential(*(list(self.model_face.children())[:-1]))

        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((num_context_features + num_body_features + num_face), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.6)
        self.fc_cat = nn.Linear((256 + num_hook), 26)
        self.relu = nn.ReLU()

    def forward(self, images_context, images_body, features_object, bbox, word_embedding, sentiment_object, image_face,
                weight_area):
        # import pdb; pdb.set_trace()
        x_context = self.model_context(images_context)

        x_context, att_mask = self.attention_module(x_context)
        x_context = self.global_pooling(x_context)
        x_body = self.model_body(images_body)
        hook_features, impact_weight = self.model_hook(features_object, bbox, word_embedding, sentiment_object,
                                                       weight_area)
        x_face = self.model_face(image_face)

        context_features = x_context.view(x_context.size(0), -1)
        body_features = x_body.view(x_body.size(0), -1)
        x_face = x_face.view(x_face.size(0), -1)

        fuse_features = torch.cat((context_features, body_features, x_face), 1)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)
        # import pdb; pdb.set_trace()
        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        cat_out = self.fc_cat(torch.cat((fuse_out, hook_features), 1))

        if self.weight:
            return cat_out, impact_weight, att_mask
        else:
            return cat_out, att_mask


class Emotic_stream1(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_hook,
                 num_box=5408, fold=0, weight=False, num_face=512):
        super(Emotic_stream1, self).__init__()
        self.weight = weight
        self.model_context, self.model_body = load_model_pair(fold)
        self.model_face = load_face_model()
        self.model_face = nn.Sequential(*(list(self.model_face.children())[:-1]))

        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((num_body_features + num_face), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.6)
        self.fc_cat = nn.Linear(256, 26)
        self.relu = nn.ReLU()

    def forward(self, images_body, image_face):
        # import pdb; pdb.set_trace()
        x_body = self.model_body(images_body)
        x_face = self.model_face(image_face)

        body_features = x_body.view(x_body.size(0), -1)
        x_face = x_face.view(x_face.size(0), -1)

        fuse_features = torch.cat((body_features, x_face), 1)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)

        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        cat_out = self.fc_cat(fuse_out)
        return cat_out


class Emotic_stream2(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_hook,
                 num_box=5408, fold=0, weight=False, num_face=512):
        super(Emotic_stream2, self).__init__()
        self.weight = weight

        self.model_context, self.model_body = load_model_pair(fold)
        self.model_context = nn.Sequential(*list(self.model_context.children())[:-1])

        self.attention_module = AttentionModel()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.num_context_features = num_context_features

        self.fc1 = nn.Linear(num_context_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.6)
        self.fc_cat = nn.Linear(256, 26)
        self.relu = nn.ReLU()

    def forward(self, images_context):
        # import pdb; pdb.set_trace()
        x_context = self.model_context(images_context)

        x_context, att_mask = self.attention_module(x_context)
        x_context = self.global_pooling(x_context)

        context_features = x_context.view(x_context.size(0), -1)
        fuse_out = self.fc1(context_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        cat_out = self.fc_cat(fuse_out)

        return cat_out, att_mask


class Emotic_stream3(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_hook,
                 num_box=5408, fold=0, weight=False, num_face=512):
        super(Emotic_stream3, self).__init__()
        self.weight = weight

        self.model_hook = Objects_netv6(1024, num_box, 300, N=10, num_sentiment=4096, weight=weight)
        self.fc_cat = nn.Linear(num_hook, 26)
        self.relu = nn.ReLU()

    def forward(self, features_object, bbox, word_embedding, sentiment_object, weight_area):
        hook_features, impact_weight = self.model_hook(features_object, bbox, word_embedding, sentiment_object,
                                                       weight_area)
        cat_out = self.fc_cat(hook_features)

        return cat_out


class Objects_netv7(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_features_object, num_bbox, num_word_embedding, N, num_sentiment, weight=False, mode_list=2):
        super(Objects_netv7, self).__init__()
        self.num_features_object = num_features_object
        self.num_bbox = num_bbox
        self.num_word_embedding = num_word_embedding
        self.num_sentiment = num_sentiment
        self.N = N
        self.weight = weight

        self.mlp_object = MLP_2layer(self.num_features_object + self.num_word_embedding, 256)
        self.mlp_sentiment = MLP_2layer(self.num_sentiment, 256)
        self.mlp_box = MLP_2layer(self.num_bbox, 256)
        self.sp_to_head = nn.Sequential(
            nn.Conv2d(2, 64, 5),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, 5),
            nn.MaxPool2d(2),
            Flatten(),
        )
        self.sp_new = nn.Sequential(
            BasicBlock(2, 16, downsample=True),
            nn.MaxPool2d(2),
            BasicBlock(16, 16),
            nn.MaxPool2d(2),
            Flatten(),
        )
        if mode_list == 2:
            self.tail_head = MLP_1layer(256 * 3, 256)
        else:
            self.tail_head = MLP_1layer(256 * 2, 256)

        self.ca = ChannelAttention(N)
        self.sa = SpatialAttention()
        # self.sk = SKConv(features=10)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features_object, bbox, word_embedding, sentiment, weight_area, mode_list):
        list_sentiment = torch.Tensor().cuda()
        list_object = torch.Tensor().cuda()
        list_bbox = torch.Tensor().cuda()

        for i in range(features_object.size(1)):
            out_o = features_object[:, i, :].view(features_object.size(0), -1)
            out_w = word_embedding[:, i, :].view(word_embedding.size(0), -1)
            out_o = self.mlp_object(torch.cat((out_o, out_w), dim=1))
            out_o = out_o.unsqueeze(dim=1)
            list_object = torch.cat((list_object, out_o), dim=1)

        for i in range(bbox.size(1)):
            out_box = bbox[:, i].squeeze(dim=1)
            out_box = self.sp_new(out_box)
            out_box = self.mlp_box(out_box)
            out_box = out_box.unsqueeze(dim=1)
            list_bbox = torch.cat((list_bbox, out_box), dim=1)

        for i in range(sentiment.size(1)):
            out_s = sentiment[:, i, :].view(sentiment.size(0), -1)
            out_s = self.mlp_sentiment(out_s)
            out_s = out_s.unsqueeze(dim=1)
            list_sentiment = torch.cat((list_sentiment, out_s), dim=1)

        if mode_list == 0:
            all_in_features = torch.cat((list_object, list_bbox), dim=2)
        elif mode_list == 1:
            all_in_features = torch.cat((list_sentiment, list_object), dim=2)
        else:
            all_in_features = torch.cat((list_sentiment, list_object, list_bbox), dim=2)

        out_weight_impact, att = self.ca(all_in_features, weight_area)
        all_in_features = out_weight_impact * all_in_features
        all_in_features = self.sa(all_in_features) * all_in_features

        all_in_features = all_in_features.sum(dim=1)
        all_in_features = all_in_features.view(all_in_features.shape[0], -1)

        all_in_features = self.tail_head(all_in_features)
        if self.weight:
            return all_in_features, att
        else:
            return all_in_features



class Emotic_v17(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_hook,
                 num_box=5408, fold=0, weight=False, num_face=512, n=10, mode_list=2):
        super(Emotic_v17, self).__init__()
        self.weight = weight
        self.mode_list = mode_list
        # self.model_context, self.model_body = prep_models(model_dir='proj/debug_exp/models',
        #                                                   context_model='resnet50',
        #                                                   file_name='model_best')
        # self.model_context = nn.Sequential(*(list(self.model_context.children())[:-2]))
        # self.model_body = nn.Sequential(*list(self.model_body.children())[:-1])
        self.model_context, self.model_body = load_model_pair(fold)
        self.model_context = nn.Sequential(*list(self.model_context.children())[:-1])

        self.attention_module = AttentionModel()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.model_hook = Objects_netv7(1024, num_box, 300, N=n, num_sentiment=4096, weight=weight, mode_list=mode_list)
        self.model_face = load_face_model()
        self.model_face = nn.Sequential(*(list(self.model_face.children())[:-1]))

        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((num_context_features + num_body_features + num_face), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.5)
        self.fc_cat = nn.Linear((256 + num_hook), 26)
        self.relu = nn.ReLU()

    def forward(self, images_context, images_body, features_object, bbox, word_embedding, sentiment_object, image_face,
                weight_area):
        # import pdb; pdb.set_trace()
        x_context = self.model_context(images_context)

        x_context, att_mask = self.attention_module(x_context)
        x_context = self.global_pooling(x_context)
        x_body = self.model_body(images_body)
        hook_features, impact_weight = self.model_hook(features_object, bbox, word_embedding, sentiment_object,
                                                       weight_area, self.mode_list)
        x_face = self.model_face(image_face)

        context_features = x_context.view(x_context.size(0), -1)
        body_features = x_body.view(x_body.size(0), -1)
        x_face = x_face.view(x_face.size(0), -1)
        # import pdb; pdb.set_trace()
        fuse_features = torch.cat((context_features, body_features, x_face), 1)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)

        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        cat_out = self.fc_cat(torch.cat((fuse_out, hook_features), 1))

        if self.weight:
            return cat_out, impact_weight, att_mask
        else:
            return cat_out, att_mask


class CaerNet(nn.Module):
    def __init__(self, fold=4, num_context_features=2048, num_body_features=2048):
        super(CaerNet, self).__init__()
        self.model_context, self.model_body = load_model_pair(fold)

        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((num_context_features + num_body_features), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.5)
        self.fc_cat = nn.Linear(256, 7)
        self.relu = nn.ReLU()

    def forward(self, images_context, images_body):
        x_context = self.model_context(images_context)
        x_body = self.model_body(images_body)

        context_features = x_context.view(x_context.size(0), -1)
        body_features = x_body.view(x_body.size(0), -1)

        fuse_features = torch.cat((context_features, body_features), 1)

        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        cat_out = self.fc_cat(fuse_out)
        return cat_out


class CaerNet_v2(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_hook,
                 num_box=5408, fold=0, weight=False, num_face=512, n=10):
        super(CaerNet_v2, self).__init__()
        self.weight = weight

        self.model_context, self.model_body = load_model_pair(fold)
        # self.model_context = nn.Sequential(*list(self.model_context.children())[:-1])

        self.attention_module = AttentionModel()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.model_hook = Objects_netv6(1024, num_box, 300, N=n, num_sentiment=4096, weight=weight)
        self.model_face = load_face_model()
        self.model_face = nn.Sequential(*(list(self.model_face.children())[:-1]))

        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((num_context_features + num_body_features + num_face), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.5)
        self.fc_cat = nn.Linear((256 + num_hook), 7)
        self.relu = nn.ReLU()

    def forward(self, images_context, images_body, features_object, bbox, word_embedding, sentiment_object, image_face,
                weight_area):
        # import pdb; pdb.set_trace()
        x_context = self.model_context(images_context)

        x_context, att_mask = self.attention_module(x_context)
        x_context = self.global_pooling(x_context)
        x_body = self.model_body(images_body)
        hook_features, impact_weight = self.model_hook(features_object, bbox, word_embedding, sentiment_object,
                                                       weight_area)
        x_face = self.model_face(image_face)

        context_features = x_context.view(x_context.size(0), -1)
        body_features = x_body.view(x_body.size(0), -1)
        x_face = x_face.view(x_face.size(0), -1)

        fuse_features = torch.cat((context_features, body_features, x_face), 1)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)

        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        cat_out = self.fc_cat(torch.cat((fuse_out, hook_features), 1))

        if self.weight:
            return cat_out, impact_weight, att_mask
        else:
            return cat_out, att_mask


class CaerNet_v3(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_hook,
                 num_box=5408, fold=0, weight=False, num_face=512, n=10, mode_list=2):
        super(CaerNet_v3, self).__init__()
        self.weight = weight
        self.mode_list = mode_list

        self.model_context, self.model_body = load_model_pair(fold, data='caer')
        # self.model_context = nn.Sequential(*list(self.model_context.children())[:-1])

        self.attention_module = AttentionModel()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.model_hook = Objects_netv7(1024, num_box, 300, N=n, num_sentiment=4096, weight=weight, mode_list=mode_list)
        self.model_face = load_face_model()
        self.model_face = nn.Sequential(*(list(self.model_face.children())[:-1]))

        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((num_context_features + num_body_features + num_face), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.5)
        self.fc_cat = nn.Linear((256 + num_hook), 7)
        self.relu = nn.ReLU()

    def forward(self, images_context, images_body, features_object, bbox, word_embedding, sentiment_object, image_face,
                weight_area):
        # import pdb; pdb.set_trace()
        x_context = self.model_context(images_context)

        x_context, att_mask = self.attention_module(x_context)
        x_context = self.global_pooling(x_context)
        x_body = self.model_body(images_body)
        hook_features, impact_weight = self.model_hook(features_object, bbox, word_embedding, sentiment_object,
                                                       weight_area, self.mode_list)
        x_face = self.model_face(image_face)

        context_features = x_context.view(x_context.size(0), -1)
        body_features = x_body.view(x_body.size(0), -1)
        x_face = x_face.view(x_face.size(0), -1)
        # import pdb; pdb.set_trace()
        fuse_features = torch.cat((context_features, body_features, x_face), 1)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)

        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        cat_out = self.fc_cat(torch.cat((fuse_out, hook_features), 1))

        if self.weight:
            return cat_out, impact_weight, att_mask
        else:
            return cat_out, att_mask


class Caer_stream1(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_hook,
                 num_box=5408, fold=0, weight=False, num_face=512):
        super(Caer_stream1, self).__init__()
        self.weight = weight
        self.model_context, self.model_body = load_model_pair(fold, data='caer')
        self.model_face = load_face_model()
        self.model_face = nn.Sequential(*(list(self.model_face.children())[:-1]))

        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((num_body_features + num_face), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.6)
        self.fc_cat = nn.Linear(256, 7)
        self.relu = nn.ReLU()

    def forward(self, images_body, image_face):
        # import pdb; pdb.set_trace()
        x_body = self.model_body(images_body)
        x_face = self.model_face(image_face)

        body_features = x_body.view(x_body.size(0), -1)
        x_face = x_face.view(x_face.size(0), -1)

        fuse_features = torch.cat((body_features, x_face), 1)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)

        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        cat_out = self.fc_cat(fuse_out)
        return cat_out


class Caer_stream2(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_hook,
                 num_box=5408, fold=0, weight=False, num_face=512):
        super(Caer_stream2, self).__init__()
        self.weight = weight

        self.model_context, self.model_body = load_model_pair(fold, data='caer')
        # self.model_context = nn.Sequential(*list(self.model_context.children())[:-1])

        self.attention_module = AttentionModel()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.num_context_features = num_context_features

        self.fc1 = nn.Linear(num_context_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.6)
        self.fc_cat = nn.Linear(256, 7)
        self.relu = nn.ReLU()

    def forward(self, images_context):
        # import pdb; pdb.set_trace()
        x_context = self.model_context(images_context)

        x_context, att_mask = self.attention_module(x_context)
        x_context = self.global_pooling(x_context)

        context_features = x_context.view(x_context.size(0), -1)
        fuse_out = self.fc1(context_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        cat_out = self.fc_cat(fuse_out)

        return cat_out, att_mask


class Caer_stream3(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_hook,
                 num_box=5408, fold=0, weight=False, num_face=512):
        super(Caer_stream3, self).__init__()
        self.weight = weight

        self.model_hook = Objects_netv6(1024, num_box, 300, N=10, num_sentiment=4096, weight=weight)
        self.fc_cat = nn.Linear(num_hook, 7)
        self.relu = nn.ReLU()

    def forward(self, features_object, bbox, word_embedding, sentiment_object, weight_area):
        hook_features, impact_weight = self.model_hook(features_object, bbox, word_embedding, sentiment_object,
                                                       weight_area)
        cat_out = self.fc_cat(hook_features)

        return cat_out