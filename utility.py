import os
import time
import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import shutil

class Timer():
    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v

class Averager():
    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v

def compute_num_params(model, text=False):
    num_params = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if num_params >= 1e6:
            return '{:.1f}M'.format(num_params / 1e6)
        else:
            return '{:.1f}K'.format(num_params / 1e3)
    else:
        return num_params

class checkpoint():
    def __init__(self, load='', save='', test_only=False):
        self.ok = True
        self.train_log = torch.Tensor()
        self.val_log = torch.Tensor()

        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        if load == '':
            if not save:
                save = now
            self.dir = os.path.join('experiment', save)
        else:
            self.dir = os.path.join('experiment', load)
            if os.path.exists(self.dir) and not test_only:
                self.train_log = torch.load(self.get_path('loss_log.pt'))
                self.val_log = torch.load(self.get_path('rmse_log.pt'))
                print('Continue from epoch {}...'.format(len(self.train_log)))
        
        print('experiment directory is {}'.format(self.dir))
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        os.makedirs(self.get_path('results'), exist_ok=True)

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def add_train_log(self, log):
        self.train_log = torch.cat([self.train_log, log])

    def add_val_log(self, log):
        self.val_log = torch.cat([self.val_log, log])

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.optimizer.save(self.dir)
        torch.save(self.train_log, os.path.join(self.dir, 'loss_log.pt'))
        torch.save(self.val_log, os.path.join(self.dir, 'rmse_log.pt'))

    def save_results(self, recon, imgname, recon_size, scale):
        save_dir = os.path.join(self.dir, 'recon'+str(max(recon_size))+'_x'+str(max(scale)))
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, imgname[0])
        np.save(filename, recon)

def calc_rmse(img1, img2):
    mse = ((img1 - img2) ** 2).mean([-2, -1])
    return torch.sqrt(mse).mean().cpu()

def normalize(img1, mean, std):
    img1 = (img1 - mean)/std
    return img1

def denormalize(img, mean, std):
    img1 = img * std + mean
    return img1

def make_optimizer(optim_spec, target):
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': optim_spec['lr'], 'weight_decay': optim_spec['weight_decay']}

    if optim_spec['name'] == 'SGD':
        optimizer_class = optim.SGD
    elif optim_spec['name'] == 'ADAM':
        optimizer_class = optim.Adam
    elif optim_spec['name'] == 'RMSprop':
        optimizer_class = optim.RMSprop
    elif optim_spec['name'] == 'RADAM':
        optimizer_class = optim.RAdam

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)
        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')
        
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    return optimizer