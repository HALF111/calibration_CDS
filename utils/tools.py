import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import json

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args, scheduler=None):
    if args.model == 'ETSformer':
        for param_group in optimizer.param_groups:
            if param_group['name'] == 'smoothing': continue
            elif param_group['name'] == 'damping': continue
            else: learning_rate = args.learning_rate

            if args.lradj == 'exponential':
                lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
            elif args.lradj == 'schedule':
                lr_adjust = {
                    2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
                    10: 5e-7, 15: 1e-7, 20: 5e-8
                }
            elif args.lradj == 'cos':
                lr_adjust = {epoch: learning_rate * 0.5 * (1. + math.cos(math.pi * epoch / args.train_epochs))}
            elif args.lradj == 'cos_with_warmup':
                if epoch <= args.warmup_epochs:
                    lr = args.min_lr + (learning_rate - args.min_lr) * (epoch / (args.warmup_epochs + 1))
                else:
                    curr_epoch = epoch - args.warmup_epochs
                    total_epochs = args.train_epochs - args.warmup_epochs
                    lr = learning_rate * 0.5 * (1. + math.cos(math.pi * curr_epoch / total_epochs))
                lr_adjust = {epoch: lr}
            elif args.lradj == 'exponential_with_warmup':
                if epoch <= args.warmup_epochs:
                    lr = args.min_lr + (learning_rate - args.min_lr) * (epoch / (args.warmup_epochs + 1))
                else:
                    curr_epoch = epoch - args.warmup_epochs
                    lr = learning_rate * (0.5 ** ((curr_epoch - 1) // 1))
                lr_adjust = {epoch: lr}
            else:
                raise NotImplementedError
    elif args.model == 'Crossformer':
        if args.lradj=='type1':
            lr_adjust = {2: args.learning_rate * 0.5 ** 1, 4: args.learning_rate * 0.5 ** 2,
                         6: args.learning_rate * 0.5 ** 3, 8: args.learning_rate * 0.5 ** 4,
                         10: args.learning_rate * 0.5 ** 5}
        elif args.lradj=='type2':
            lr_adjust = {5: args.learning_rate * 0.5 ** 1, 10: args.learning_rate * 0.5 ** 2,
                         15: args.learning_rate * 0.5 ** 3, 20: args.learning_rate * 0.5 ** 4,
                         25: args.learning_rate * 0.5 ** 5}
    elif args.model == "PatchTST":
        if args.lradj == 'type1':
            lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        elif args.lradj == 'type2':
            lr_adjust = {
                2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
                10: 5e-7, 15: 1e-7, 20: 5e-8
            }
        elif args.lradj == 'type3':
            lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
        elif args.lradj == 'constant':
            lr_adjust = {epoch: args.learning_rate}
        elif args.lradj == '3':
            lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
        elif args.lradj == '4':
            lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
        elif args.lradj == '5':
            lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
        elif args.lradj == '6':
            lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
        elif args.lradj == 'TST':
            lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    else:
        # lr = args.learning_rate * (0.2 ** (epoch // 2))
        if args.lradj == 'type1':
            lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        elif args.lradj == 'type2':
            lr_adjust = {
                2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
                10: 5e-7, 15: 1e-7, 20: 5e-8
            }
        elif args.lradj =='type3':
            lr_adjust = {epoch: args.learning_rate}
        elif args.lradj == 'type4':
            lr_adjust = {epoch: args.learning_rate * (0.9 ** ((epoch - 1) // 1))}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


# Crossformer
def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)
    return args

def string_split(str_for_split):
    str_no_space = str_for_split.replace(' ', '')
    str_split = str_no_space.split(',')
    value_list = [eval(x) for x in str_split]

    return value_list
