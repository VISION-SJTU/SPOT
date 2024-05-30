import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import math


def build_opt_lr_ocean(settings, model, current_epoch=0):
    # Fix all backbone at first
    for param in model.features.features.parameters():
        param.requires_grad = False
    for m in model.features.features.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    if current_epoch >= settings.unfreeze_epoch - 1:
        if len(settings.trainable_layers) > 0:
            # Specific trainable layers
            for layer in settings.trainable_layers:
                for param in getattr(model.features.features, layer).parameters():
                    param.requires_grad = True
                for m in getattr(model.features.features, layer).modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.train()
        else:
            # Train all backbone layers
            for param in model.features.features.parameters():
                param.requires_grad = True
            for m in model.features.features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
    else:
        for param in model.features.features.parameters():
            param.requires_grad = False
        for m in model.features.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.features.features.parameters()),
                          'lr': settings.opt_params['backbone_ratio'] * settings.opt_params['base_lr']}]
    try:
        trainable_params += [{'params': model.neck.parameters(),
                              'lr': settings.opt_params['base_lr']}]
    except:
        pass

    trainable_params += [{'params': model.connect_model.parameters(),
                          'lr': settings.opt_params['base_lr']}]

    try:
        trainable_params += [{'params': model.align_head.parameters(),
                              'lr': settings.opt_params['base_lr']}]
    except:
        pass

    # Print trainable parameter (first check)
    print('==========first check trainable==========')
    for param in trainable_params:
        print(param)

    optimizer = torch.optim.SGD(trainable_params,
                                momentum=settings.opt_params['momentum'],
                                weight_decay=settings.opt_params['weight_decay'])

    lr_scheduler = build_lr_scheduler(optimizer, settings)
    lr_scheduler.step(current_epoch)
    return optimizer, lr_scheduler


def load_pretrain(model, pretrained_path, print_unuse=True, gpus=None):
    print('load pretrained model from {}'.format(pretrained_path))

    if gpus is not None:
        torch.cuda.set_device(gpus[0])
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
        pretrained_dict = remove_prefix(pretrained_dict, 'feature_extractor.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
        pretrained_dict = remove_prefix(pretrained_dict, 'feature_extractor.')

    check_keys(model, pretrained_dict, print_unuse=print_unuse)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def check_keys(model, pretrained_state_dict, print_unuse=True):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = list(ckpt_keys - model_keys)
    missing_keys = list(model_keys - ckpt_keys)

    # Remove num_batches_tracked
    for k in sorted(missing_keys):
        if 'num_batches_tracked' in k:
            missing_keys.remove(k)

    print('missing keys:{}'.format(missing_keys))
    if print_unuse:
        print('unused checkpoint keys:{}'.format(unused_pretrained_keys))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    """
        Old style model is stored with all names of parameters share common prefix 'module.'
    """
    print('remove prefix \'{}\''.format(prefix))
    function = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {function(key): value for key, value in state_dict.items()}


def _build_lr_scheduler(optimizer, settings_lr, epoch_each_lr_step):
    LRs = {'log': LogScheduler, 'step': StepScheduler}
    return LRs[settings_lr['style']](optimizer, settings_lr, epoch_each_lr_step)


def _build_warm_up_scheduler(optimizer, settings_lr, epoch_each_lr_step):
    warmup_epoch = settings_lr[0]['epoch_step']
    sc1 = _build_lr_scheduler(optimizer, settings_lr[0], epoch_each_lr_step)
    sc2 = _build_lr_scheduler(optimizer, settings_lr[1], epoch_each_lr_step)
    return WarmUPScheduler(optimizer, sc1, sc2, warmup_epoch)


def build_lr_scheduler(optimizer, settings):
    epoch_each_lr_step = settings.epoch_each_lr_step
    if len(settings.opt_params['optimizers']) >= 2:
        return _build_warm_up_scheduler(optimizer, settings.opt_params['optimizers'], epoch_each_lr_step)
    else:
        return _build_lr_scheduler(optimizer, settings.opt_params['optimizers'][0], epoch_each_lr_step)


class LRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        if 'lr_spaces' not in self.__dict__:
            raise Exception('lr_spaces must be set in "LRSchduler"')
        super(LRScheduler, self).__init__(optimizer, last_epoch)

    def get_cur_lr(self):
        return self.lr_spaces[self.last_epoch]

    def get_lr(self):
        epoch = self.last_epoch
        return [self.lr_spaces[epoch] * pg['initial_lr'] / self.start_lr
                for pg in self.optimizer.param_groups]

    def __repr__(self):
        return "({}) lr spaces: \n{}".format(self.__class__.__name__,
                                             self.lr_spaces)


class LogScheduler(LRScheduler):
    def __init__(self, optimizer, settings_lr, epoch_each_lr_step, last_epoch=-1):

        end_lr = settings_lr['end_lr'] if 'end_lr' in settings_lr else None
        start_lr = settings_lr['start_lr'] if 'start_lr' in settings_lr else None
        epochs = settings_lr['epoch_step'] if 'epoch_step' in settings_lr else 45

        self.start_lr = start_lr
        self.end_lr = end_lr
        self.epochs = epochs
        lr_spaces_ori = np.logspace(math.log10(start_lr),
                                    math.log10(end_lr), epochs)
        self.lr_spaces = np.repeat(lr_spaces_ori, epoch_each_lr_step)

        super(LogScheduler, self).__init__(optimizer, last_epoch)


class StepScheduler(LRScheduler):
    def __init__(self, optimizer, settings_lr, epoch_each_lr_step, last_epoch=-1):
        end_lr = settings_lr['end_lr'] if 'end_lr' in settings_lr else None
        start_lr = settings_lr['start_lr'] if 'start_lr' in settings_lr else None
        mult = settings_lr['mult'] if 'mult' in settings_lr else 0.1
        epochs = settings_lr['epoch_step'] if 'epoch_step' in settings_lr else 5
        step = settings_lr['step'] if 'step' in settings_lr else 1
        if end_lr is not None:
            if start_lr is None:
                start_lr = end_lr / (mult ** (epochs // step))
            else:  # for warm up policy
                mult = math.pow(end_lr/start_lr, 1. / (epochs // step))
        self.start_lr = start_lr
        lr_spaces_ori = self.start_lr * (mult**(np.arange(epochs) // step))
        self.lr_spaces = np.repeat(lr_spaces_ori, epoch_each_lr_step)
        self.mult = mult
        self._step = step

        super(StepScheduler, self).__init__(optimizer, last_epoch)


class WarmUPScheduler(LRScheduler):
    def __init__(self, optimizer, warmup, normal, last_epoch=-1):
        warmup = warmup.lr_spaces
        normal = normal.lr_spaces
        self.lr_spaces = np.concatenate([warmup, normal])
        self.start_lr = normal[0]

        super(WarmUPScheduler, self).__init__(optimizer, last_epoch)