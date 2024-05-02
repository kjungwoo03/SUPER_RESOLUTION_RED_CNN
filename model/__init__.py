import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.utils.model_zoo
from model import common
import numpy as np

class Model(nn.Module):
    def __init__(self, config, ckp):
        super(Model, self).__init__()
        print('Making model... {}'.format(config['model']['name']))

        self.cpu = config['cpu']
        self.device = torch.device('cpu' if config['cpu'] else 'cuda')
        self.save_models = config['save_models']
        module = import_module('model.' + config['model']['name'].lower())
        self.model = module.make_model(config).to(self.device)
        
        self.load(
            ckp.get_path('model'),
            resume=config['resume'],
            cpu=config['cpu']
        )
        
    def forward(self, ldct):
        if self.training:
            return self.model(ldct)
        else:
            return self.model(ldct)

    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, 'model_latest.pt')]
        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(os.path.join(apath, 'model_{}.pt'.format(epoch)))
        for i in save_dirs:
            torch.save(self.model.state_dict(), i)

    def load(self, apath, resume=-1, cpu=False):
        load_from = None
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        if resume == 0:
            print('Load the model from {}'.format(apath))
            load_from = torch.load(
                os.path.join(apath, 'model_latest.pt'),
                **kwargs)
        elif resume == 1:
            print('Load the model from {}'.format(os.path.join(apath, 'model_best.pt')))
            load_from = torch.load(
                os.path.join(apath, 'model_best.pt'),
                **kwargs)
        elif resume == 2:
            load_from = torch.load(
                os.path.join(apath, 'model_{}.pt'.format(resume)),
                **kwargs)
        if load_from:
            self.model.load_state_dict(load_from, strict=False)