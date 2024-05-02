import os
import math
import utility
import torch.nn.functional as F
import torch.nn as nn
import torch
from tqdm import tqdm
import time
import vessl
import numpy as np
import random
from model import common
from torch.optim.lr_scheduler import MultiStepLR

class Trainer():
    def __init__(self, config, loader, my_model, ckp, load):
        self.config = config
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.loss = nn.L1Loss()
        torch.set_float32_matmul_precision('high')
        self.model = torch.compile(my_model)
        # self.model = my_model
        self.optimizer = utility.make_optimizer(config['optimizer'], self.model)
        
        if load != '':
            self.optimizer.load(ckp.dir)
            # load at VESSL experiment
            for i in range(len(ckp.train_log)):
                vessl.log(step=i, payload={'train_loss': ckp.train_log[i].item()})
            for j in range(len(ckp.val_log)):
                vessl.log(step=j, payload={'val_rmse': ckp.val_log[j].item()})
        
        # loss를 특정 epoch에서 변화시키는데, 그 타이밍을 정해주는 코드
        self.scheduler = MultiStepLR(self.optimizer, 
                                    milestones=config['optimizer']['milestones'], 
                                    gamma=config['optimizer']['gamma'],
                                    last_epoch=len(ckp.train_log)-1)

        print('Training Paitent ID:{}'.format(self.config['dataset']['train']['patient_ID_train']))
        print('Validation Paitent ID:{}'.format(self.config['dataset']['valid']['patient_ID_valid']))
        print('total number of parameter is {}'.format(sum(p.numel() for p in self.model.parameters())))

    def train(self):
        epoch = self.scheduler.last_epoch
        self.ckp.add_train_log(torch.zeros(1))
        learning_rate = self.scheduler.get_last_lr()[0]
        self.model.train()
        train_loss = utility.Averager()
        timer = utility.Timer()
        for batch, (ldct, ndct, loc_name) in enumerate(tqdm(self.loader_train)):
            ldct, ndct = self.prepare(ldct, ndct)
            
            ldct = utility.normalize(ldct - 1024, 500, 500) # calc mean and std of ndct training dataset
            ndct = utility.normalize(ndct - 1024, 500, 500)
            self.optimizer.zero_grad()

            denoised = self.model(ldct)
            loss = self.loss(denoised, ndct)
            loss.backward()
            self.optimizer.step()
            
            train_loss.add(loss.item())

        vessl.log(step=epoch, payload={'train_loss': train_loss.item(), 'train_time': timer.t(), 'learning_rate': learning_rate})
        self.ckp.train_log[-1] = train_loss.item()
        self.scheduler.step()

    def eval(self):
        epoch = self.scheduler.last_epoch
        if epoch % self.config['test_every'] == 0:
            self.ckp.add_val_log(torch.zeros(1))
            self.model.eval()
            timer = utility.Timer()
            
            with torch.no_grad():
                for i, (ldct, ndct, loc_name) in enumerate(self.loader_test):
                    ldct, ndct = self.prepare(ldct, ndct)

                    ldct = utility.normalize(ldct - 1024, 500, 500)

                    denoised = self.model(ldct) # (batch, x*y, ch)
                    denoised = utility.denormalize(denoised, 500, 500)
                    self.ckp.val_log[-1] += utility.calc_rmse(denoised, ndct-1024).mean() / len(self.loader_test)

                best = self.ckp.val_log.min(0) # best[0] is the minimum value, best[1] is the index of the minimum value
                vessl.log(step=epoch//self.config['test_every']-1, payload={'val_rmse': self.ckp.val_log[-1], 'val_time': timer.t()})

            self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch // self.config['test_every']))

    def prepare(self, *args):
        device = torch.device('cpu' if self.config['cpu'] else 'cuda')
        def _prepare(tensor):
            return tensor.to(device)
        return [_prepare(a) for a in args]

    def terminate(self):
        epoch = self.scheduler.last_epoch
        return epoch >= self.config['epochs']
