import torch
import utility
import data
import model
from trainer import Trainer
import warnings
import vessl
import yaml
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='train/train_edsr-liif') # required=True
    parser.add_argument('--save', type=str, default='test')
    parser.add_argument('--load', type=str, default='')
    args = parser.parse_args()

    with open(os.path.join('configs', args.config+'.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded, config_path: {}'.format(os.path.join('configs', args.config+'.yaml')))
 
    torch.manual_seed(config['seed'])
    checkpoint = utility.checkpoint(args.load, args.save)

    # vessl.configure(organization_name='yonsei-medisys', project_name='CRET')
    # vessl.init(message=args.save)
    
    if checkpoint.ok:
        loader = data.Data(config)
        _model = model.Model(config, checkpoint) 
        t = Trainer(config, loader, _model, checkpoint, args.load)
        while not t.terminate():
            t.train()
            t.eval()
    
if __name__ == '__main__':
    main()


