'''
Helper functions.
'''
import numpy as np
import os
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


def check_path(path):
    '''
    Checks if a given path exists. If not, create the directory.
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    return None


def get_device():
    '''
    Checks if GPU is available to be used. If not, CPU is used.
    '''
    # return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def onehot_encode(tensor, num_classes, device):
    '''
    Encodes the given tensor into one-hot vectors.
    '''
    return torch.eye(num_classes).to(device).index_select(dim=0, index=tensor.to(device))


def accuracy_calc(predictions, labels):
    '''
    Calculates prediction accuracy.
    '''
    # print(predictions.size())
    # print(labels.size())
    num_data = labels.size()[0]
    correct_pred = torch.sum(predictions == labels)
    accuracy = (correct_pred.item()*100)/num_data

    return accuracy

def get_learning_rate(optimizer):
    '''
    Returns the current LR from the optimizer module.
    '''

    for params in optimizer.param_groups:
        return params['lr']

def get_best_accuracy(accuracy_list):
    '''
    Returns the best accuracy from the list of accuracies.
    '''
    best_accuracy = max(accuracy_list)

    return best_accuracy

def compute_conv(input_height, input_width, kernel_size, stride, padding, filters):
    """
    Computes the output shape of a standard convolution operation.
    
    If padding is "same", compute the padding needed to keep the output shape same as input shape.
    """
    if padding == "same":
        # Calculate the padding needed to keep output shape same as input shape
        out_height = (input_height + stride - 1) // stride
        out_width = (input_width + stride - 1) // stride
        padding_h = max(0, (out_height - 1) * stride + kernel_size - input_height)
        padding_w = max(0, (out_width - 1) * stride + kernel_size - input_width)
        padding = max(padding_h, padding_w) // 2
    else:
        out_height = ((input_height + 2 * padding - kernel_size) // stride) + 1
        out_width = ((input_width + 2 * padding - kernel_size) // stride) + 1
        

    return out_height, out_width, filters

def reshape_tensor(x,y):
    
    x_4D = x.view(x.size()[0], x.size()[1]*x.size()[2], x.size()[3], x.size()[4])
    y_5D = y.view(y.size()[0], x.size()[1], x.size()[2],y.size()[-1], y.size()[-2])
    
    return x_4D, y_5D

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model, model_path,epoch_idx,test_epoch_accuracy)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model, model_path,epoch_idx,test_epoch_accuracy)
            self.counter = 0


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    在optimizer中会设置一个基础学习率base lr,
    当multiplier>1时,预热机制会在total_epoch内把学习率从base lr逐渐增加到multiplier*base lr,再接着开始正常的scheduler
    当multiplier==1.0时,预热机制会在total_epoch内把学习率从0逐渐增加到base lr,再接着开始正常的scheduler
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler and (not self.finished):
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                # !这是很关键的一个环节，需要直接返回新的base-lr
            return [base_lr for base_lr in self.after_scheduler.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        # print('warmuping...')
        if self.last_epoch <= self.total_epoch:
            warmup_lr=None
            if self.multiplier == 1.0:
                warmup_lr = [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
            else:
                warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics,epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)





            
            


