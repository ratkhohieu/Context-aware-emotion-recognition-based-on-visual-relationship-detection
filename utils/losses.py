import os
import random
from abc import ABC

import numpy as np
import torch
import torch.nn as nn


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def seed_everything(seed=1):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class RMSELoss(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


def CCC_metric(x, y):
    """

    :param x: input
    :param y: target
    :return:
    """

    y = y.view(-1)
    x = x.view(-1)
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))))
    x_m = torch.mean(x)
    y_m = torch.mean(y)
    x_s = torch.std(x)
    y_s = torch.std(y)
    ccc = 2 * rho * x_s * y_s / (torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
    return ccc


def CCC_loss(pred, label):
    ccc = 0
    # import pdb; pdb.set_trace()
    for i in range(pred.shape[1]):
        ccc += CCC_metric(pred[:, i], label[:, i])
    return 1 - ccc / 3.0


class MSE_loss(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, label):
        loss = 0
        for i in range(pred.shape[1]):
            if i != 2:
                loss += self.mse(pred[:, i], label[:, i])
            else:
                loss += 2 * self.mse(pred[:, i], label[:, i])
        return loss


def new_loss(out_put, r=None, weight=1.0):
    if r is None:
        r = [0.65, 0.95, 0.74]
        # import pdb; pdb.set_trace()
    loss = r[0] * 1 / (torch.mean(abs(out_put[:, 0] - out_put[:, 1]))) + \
           r[1] * 1 / (torch.mean(abs(out_put[:, 1] - out_put[:, 2]))) + \
           r[2] * torch.mean(abs(out_put[:, 2] - out_put[:, 0]))
    return torch.mean(loss) * weight


def multi_gpu(model):
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    return model


def weight_loss_pseudo(epoch, num_epoch_1=1, num_epoch_2=10, max_weight=0.2):
    if epoch < num_epoch_1:
        return 0.0
    elif epoch > num_epoch_2:
        return max_weight
    else:
        return ((epoch - num_epoch_1) / (num_epoch_2 - num_epoch_1)) * max_weight


def loss_distance(out_put, labels):
    # import pdb;
    # pdb.set_trace()
    # criterion = nn.L1Loss()

    # for i in range(out_put.shape[1]):
    l1 = ((out_put[:, 0] - labels[:, 0]) - (out_put[:, 1] - labels[:, 1])) ** 2
    l2 = ((out_put[:, 1] - labels[:, 1]) - (out_put[:, 2] - labels[:, 2])) ** 2
    l3 = ((out_put[:, 2] - labels[:, 2]) - (out_put[:, 0] - labels[:, 0])) ** 2
    loss = torch.sum(torch.tensor([torch.mean(l1), torch.mean(l2), torch.mean(l3)]))
    return loss


class GAIN_re_loss:
    # TODO : weight for multiply
    def __init__(
            self,
            input_key: str = "targets",
            output_cls_key: str = "logits",
            output_am_key: str = "logits_am",
    ):
        self.input_key = input_key
        self.output_cls_key = output_cls_key
        self.output_am_key = output_am_key

    def _compute_loss(self, output, label, criterion):
        outputs_cls = output[self.output_cls_key]
        outputs_am = output[self.output_am_key]
        input = label[self.input_key]
        outputs_cls = outputs_cls.sigmoid()
        loss = criterion(outputs_cls, input) * 0.8
        loss_am = F.sigmoid(outputs_am)
        loss_am = loss_am.sum() / loss_am.size(0)
        loss += loss_am * 0.2
        return loss


class GAIN_cl_loss:
    def __init__(
            self,
            input_key: str = "targets",
            output_cls_key: str = "logits",
            output_am_key: str = "logits_am",
    ):
        self.input_key = input_key
        self.output_cls_key = output_cls_key
        self.output_am_key = output_am_key
        self.soft_mask_criterion = nn.BCEWithLogitsLoss()

    def _compute_loss(self, output, label, criterion, heatmap, mark):
        outputs_cls = output[self.output_cls_key]
        outputs_am = output[self.output_am_key]
        input = label[self.input_key]
        loss_cl = criterion(outputs_cls, input)
        loss_am = F.softmax(outputs_am)
        loss_am, _ = loss_am.max(dim=1)
        loss_am = loss_am.sum() / loss_am.size(0)

        loss_mask = self.soft_mask_criterion(heatmap, mark)
        loss = loss_cl * 0.5 + loss_am * 0.2 + loss_mask * 0.3
        return loss


class DiscreteLoss(nn.Module):
    ''' Class to measure loss between categorical emotion predictions and labels.'''

    def __init__(self, weight_type='mean', device=torch.device('cpu')):
        super(DiscreteLoss, self).__init__()
        self.weight_type = weight_type
        self.device = device
        if self.weight_type == 'mean':
            self.weights = torch.ones((1, 26)) / 26.0
            self.weights = self.weights.to(self.device)
        elif self.weight_type == 'static':
            self.weights = torch.FloatTensor([0.1435, 0.1870, 0.1692, 0.1165, 0.1949, 0.1204, 0.1728, 0.1372, 0.1620,
                                              0.1540, 0.1987, 0.1057, 0.1482, 0.1192, 0.1590, 0.1929, 0.1158, 0.1907,
                                              0.1345, 0.1307, 0.1665, 0.1698, 0.1797, 0.1657, 0.1520,
                                              0.1537]).unsqueeze(0)
            self.weights = self.weights.to(self.device)

    def forward(self, pred, target):
        if self.weight_type == 'dynamic':
            self.weights = self.prepare_dynamic_weights(target)
            self.weights = self.weights.to(self.device)
        loss = (((pred - target) ** 2) * self.weights)
        return loss.sum()

    def prepare_dynamic_weights(self, target):
        target_stats = torch.sum(target, dim=0).float().unsqueeze(dim=0).cpu()
        weights = torch.zeros((1, 26))
        weights[target_stats != 0] = 1.0 / torch.log(target_stats[target_stats != 0].data + 1.2)
        weights[target_stats == 0] = 0.0001
        return weights


class ContinuousLoss_L2(nn.Module):
    ''' Class to measure loss between continuous emotion dimension predictions and labels. Using l2 loss as base. '''

    def __init__(self, margin=1):
        super(ContinuousLoss_L2, self).__init__()
        self.margin = margin

    def forward(self, pred, target):
        labs = torch.abs(pred - target)
        loss = labs ** 2
        loss[(labs < self.margin)] = 0.0
        return loss.sum()


class ContinuousLoss_SL1(nn.Module):
    ''' Class to measure loss between continuous emotion dimension predictions and labels. Using smooth l1 loss as base. '''

    def __init__(self, margin=1):
        super(ContinuousLoss_SL1, self).__init__()
        self.margin = margin

    def forward(self, pred, target):
        labs = torch.abs(pred - target)
        loss = 0.5 * (labs ** 2)
        loss[(labs > self.margin)] = labs[(labs > self.margin)] - 0.5
        return loss.sum()


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


class CB_loss(nn.Module):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """

    def __init__(self, device):
        super().__init__()
        self.no_of_classes = 26
        self.beta = 0.9999
        self.gamma = 2.0
        self.loss_type = "focal"
        self.samples_per_cls = torch.tensor([2685, 491, 983, 11701, 573, 9397, 862, 3413, 1791,
                                             2090, 445, 21539, 2547, 9827, 1198, 576, 11372, 490,
                                             3380, 5312, 948, 709, 652, 1153, 2018, 1538])
        self.device = device

    def forward(self, logits, labels):
        import pdb;
        pdb.set_trace()
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.no_of_classes

        labels_one_hot = F.one_hot(labels, self.no_of_classes).float().to(self.device)

        weights = torch.tensor(weights).float().to(self.device)
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)

        weights = weights.repeat(1, self.no_of_classes)

        if self.loss_type == "focal":
            cb_loss = focal_loss(labels_one_hot, logits, weights, self.gamma)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "softmax":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        return cb_loss


import torch
from torch.optim import Optimizer


class Adas(Optimizer):
    """
    Introduction:
        For the mathematical part see https://github.com/YanaiEliyahu/AdasOptimizer,
        the `Theory` section contains the major innovation,
        and then `How ADAS works` contains more low level details that are still somewhat related to the theory.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr: float > 0. Initial learning rate that is per feature/input (e.g. dense layer with N inputs and M outputs, will have N learning rates).
        lr2: float >= 0.  lr's Initial learning rate. (just ~1-2 per layer, additonal one because of bias)
        lr3: float >= 0. lr2's fixed learning rate. (global)
        beta_1: 0 < float < 1. Preferably close to 1. Second moments decay factor to update lr and lr2 weights.
        beta_2: 0 < float < 1. Preferably close to 1. 1/(1 - beta_2) steps back in time that `lr`s will be optimized for, larger dataset might require more nines.
        beta_3: 0 < float < 1. Preferably close to 1. Same as beta_2, but for `lr2`s.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
    """

    def __init__(self, params,
                 lr=0.001, lr2=.005, lr3=.0005,
                 beta_1=0.999, beta_2=0.999, beta_3=0.9999,
                 epsilon=1e-8, **kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid lr: {}".format(lr))
        if not 0.0 <= lr2:
            raise ValueError("Invalid lr2: {}".format(lr))
        if not 0.0 <= lr3:
            raise ValueError("Invalid lr3: {}".format(lr))
        if not 0.0 <= epsilon:
            raise ValueError("Invalid epsilon value: {}".format(epsilon))
        if not 0.0 <= beta_1 < 1.0:
            raise ValueError("Invalid beta_1 parameter: {}".format(beta_1))
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError("Invalid beta_2 parameter: {}".format(beta_2))
        if not 0.0 <= beta_3 < 1.0:
            raise ValueError("Invalid beta_3 parameter: {}".format(beta_3))
        defaults = dict(lr=lr, lr2=lr2, lr3=lr3, beta_1=beta_1, beta_2=beta_2, beta_3=beta_3, epsilon=epsilon)
        self._varn = None
        self._is_create_slots = None
        self._curr_var = None
        self._lr = lr
        self._lr2 = lr2
        self._lr3 = lr3
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._beta_3 = beta_3
        self._epsilon = epsilon
        super(Adas, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adas, self).__setstate__(state)

    @torch.no_grad()
    def _add(self, x, y):
        x.add_(y)
        return x

    @torch.no_grad()
    # TODO: fix variables' names being too convoluted in _derivatives_normalizer and _get_updates_universal_impl
    def _derivatives_normalizer(self, derivative, beta):
        steps = self._make_variable(0, (), derivative.dtype)
        self._add(steps, 1)
        factor = (1. - (self._beta_1 ** steps)).sqrt()
        m = self._make_variable(0, derivative.shape, derivative.dtype)
        moments = self._make_variable(0, derivative.shape, derivative.dtype)
        m.mul_(self._beta_1).add_((1 - self._beta_1) * derivative * derivative)
        np_t = derivative * factor / (m.sqrt() + self._epsilon)
        # the third returned value should be called when the moments is finally unused, so it's updated
        return (moments, np_t, lambda: moments.mul_(beta).add_((1 - beta) * np_t))

    def _make_variable(self, value, shape, dtype):
        self._varn += 1
        name = 'unnamed_variable' + str(self._varn)
        if self._is_create_slots:
            self.state[self._curr_var][name] = torch.full(size=shape, fill_value=value, dtype=dtype,
                                                          device=self._curr_var.device)
        return self.state[self._curr_var][name]

    @torch.no_grad()
    def _get_updates_universal_impl(self, grad, param):
        lr = self._make_variable(value=self._lr, shape=param.shape[1:], dtype=param.dtype)
        moment, deriv, f = self._derivatives_normalizer(grad, self._beta_3)
        param.add_(- torch.unsqueeze(lr, 0) * deriv)
        lr_deriv = torch.sum(moment * grad, 0)
        f()
        master_lr = self._make_variable(self._lr2, (), dtype=torch.float32)
        m2, d2, f = self._derivatives_normalizer(lr_deriv, self._beta_2)
        self._add(lr, master_lr * lr * d2)
        master_lr_deriv2 = torch.sum(m2 * lr_deriv)
        f()
        m3, d3, f = self._derivatives_normalizer(master_lr_deriv2, 0.)
        self._add(master_lr, self._lr3 * master_lr * d3)
        f()

    @torch.no_grad()
    def _get_updates_universal(self, param, grad, is_create_slots):
        self._curr_var = param
        self._is_create_slots = is_create_slots
        self._varn = 0
        return self._get_updates_universal_impl(grad, self._curr_var.data)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adas does not support sparse gradients')
                self._get_updates_universal(p, grad, len(self.state[p]) == 0)
        return loss


def multitask_selection_loss(pred, label):
    return torch.sum(label * (torch.pow(pred, 1 / 2) * torch.log(pred + 0.0001)))


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


import torch
import torch.nn.functional as F
import torch.nn as nn


class BCELoss(nn.Module):
    def __init__(self, eps=1e-7, if_mean=True):
        super(BCELoss, self).__init__()
        self.eps = eps
        self.if_mean = if_mean

    def forward(self, inputs, target):
        logit = inputs.clamp(self.eps, 1. - self.eps)
        loss = -(target.float() * torch.log(logit) +
                 (1 - target.float()) * torch.log(1 - logit))
        if self.if_mean:
            return loss.mean()
        else:
            return loss


class DependentLoss(nn.Module):
    '''
    Attributes:
        alpha: a metric to indicate the global probability
        binary_loss: the binary classification loss for each class

    Functions:
        forward:
            attr:
                inputs: the sigmoid probability with shape (batch_size, n_class)
                target: the label with shape (batch_size, n_class)
            return:
                count_loss: the dependent loss for each class
                count_p: the dependent probability for each class
    '''

    def __init__(self, alpha=None):
        super(DependentLoss, self).__init__()
        self.alpha = alpha
        self.binary_loss = BCELoss(if_mean=False)

    def forward(self, inputs, target):
        n_class = inputs.size(1)
        batch_size = inputs.size(0)
        count_loss = 0
        count_p = []

        if self.alpha is not None:
            for class_index in range(n_class):
                cur_p = []
                for condition_index in range(n_class):
                    alpha_condition_batch = self.alpha[condition_index, class_index] * inputs[:, condition_index]
                    cur_p.append(alpha_condition_batch)
                cur_p = torch.stack(cur_p, 1).sum(1) / n_class
                count_p.append(cur_p)
            count_p = torch.stack(count_p, 1)
        else:
            count_p = inputs

        for class_index in range(n_class):
            cur_loss = self.binary_loss(count_p[:, class_index],
                                        target[:, class_index])
            count_loss += cur_loss.mean()
        return count_loss


class MultiLabelLoss(nn.Module):
    """
    Weighted BCELoss. This loss was used for comparation.
    reference
    @inproceedings{segthor_tao2019,
    author = {Tao He, Jixiang Guo, Jianyong Wang, Xiuyuan Xu, Zhang Yi},
    title = {Multi-task Learning for the Segmentation of Thoracic Organs at Risk in CT images},
    booktile = {Proceedings of the 2019 Challenge on Segmentation of THoracic
    Organs at Risk in CT Images (SegTHOR2019) },
    volume = {2349},
    year = {2019},
    }
    Args:
        alpha: the weight for current class (alpha in the paper)
    Funs:
        forward: the forward computing of bceloss
            Returns:
            count_loss: the loss
            inputs: the probability for each class

    """

    def __init__(self, alpha=None):
        super(MultiLabelLoss, self).__init__()
        self.alpha = alpha
        self.binary_loss = BCELoss(if_mean=False)

    def forward(self, inputs, target):
        n_class = inputs.size(1)
        count_loss = 0
        for class_index in range(n_class):
            cur_loss = self.binary_loss(inputs[:, class_index],
                                        target[:, class_index])
            count_loss += cur_loss.mean()
        if self.alpha is not None:
            count_loss = count_loss * self.alpha
        return count_loss, inputs


class SoftDiceLoss(nn.Module):
    """
    The Dice Loss function
    """

    def __init__(self, smooth=1e-6):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, probs, labels):
        numerator = 2 * torch.sum(labels * probs, 2)
        denominator = torch.sum(labels + probs ** 2, 2) + self.smooth
        return 1 - torch.mean(numerator / denominator)


class CombinedLoss(nn.Module):
    """
    The combined loss for multi-task learning.
    if if_closs is True, the multi-task learning is used; otherwise, the dice loss is used.
    if alpha=None, the c_loss_fun is the weighted BCELoss; otherwise, the c_loss_fun is the DependentLoss.
    Args:
        alpha: the weight
        if_closs: the flag whether use multi-task learning
        s_loss_fun: the segmentation loss (SoftDiceLoss)
        c_loss_fun: the multi-label classification loss (DependentLoss or MultiLabelLoss)
    Functions:
        Args:
            s_logit: network output for segmentation
            c_logit: network output for classification
            s_label:

    """

    def __init__(self, alpha=None, if_closs=1):
        super(CombinedLoss, self).__init__()
        self.closs_flag = if_closs
        self.s_loss_fun = SoftDiceLoss()
        if alpha is not None:
            self.c_loss_fun = DependentLoss(alpha)
        else:
            self.c_loss_fun = MultiLabelLoss()

    def forward(self, s_logit, c_logit, s_label, c_label):
        probs = F.softmax(s_logit, 1)
        batch_size, n_class = probs.size(0), probs.size(1)
        labels = s_label.view(batch_size, n_class, -1).float()
        probs = probs.view(batch_size, n_class, -1)
        s_loss = self.s_loss_fun(probs, labels)
        c_loss, c_p = self.c_loss_fun(c_logit, c_label)
        total_loss = s_loss + self.closs_flag * c_loss
        return total_loss, c_loss, s_loss, c_p
