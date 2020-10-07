# -*- coding: utf-8 -*-

### basic modules
import numpy as np
import time, pickle, os, sys, json, PIL, tempfile, warnings, importlib, math, copy, shutil, setproctitle
# import seaborn as sns
# import matplotlib.pyplot as plt

### torch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn.functional as F
from torch import autograd

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import utils, data_load, BCP

if __name__ == "__main__":
    args = utils.argparser(data='mnist',epsilon=1.58,epsilon_infty=0.1,epochs=60,rampup=20,wd_list=[21,30,40],warmup=1,opt_iter=10)
    print(args)
    print('saving file to {}'.format(args.prefix))
    setproctitle.setproctitle(args.prefix)
    train_log = open(args.prefix + "_train.log", "w")
    test_log = open(args.prefix + "_test.log", "w")
    train_loader, _ = data_load.data_loaders(args.data, args.batch_size, args.augmentation, args.normalization, args.drop_last, args.shuffle)
    _, test_loader = data_load.data_loaders(args.data, args.test_batch_size, args.augmentation, args.normalization, args.drop_last, args.shuffle)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    for X,y in train_loader: 
        break    
    
    best_err = 1
    err = 1
    sampler_indices = []
    model = [utils.select_model(args.data, args.model)]
    print(model[-1])
    if args.opt == 'adam': 
        opt = optim.Adam(model[-1].parameters(), lr=args.lr)
    elif args.opt == 'sgd': 
        opt = optim.SGD(model[-1].parameters(), lr=args.lr, 
                        momentum=args.momentum,
                        weight_decay=args.weight_decay) 
    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler =='multistep':
        lr_scheduler = MultiStepLR(opt, milestones=args.wd_list, gamma=args.gamma)
    
    eps_schedule = np.linspace(args.starting_epsilon, 
                               args.epsilon_train,                                
                               args.schedule_length)
    
    kappa_schedule = np.linspace(args.starting_kappa, 
                               args.kappa,                                
                               args.kappa_schedule_length)
    u_list = None
    for t in range(args.epochs):    
            
        if t < args.warmup:
            epsilon = 0
        elif args.warmup <= t < args.warmup+len(eps_schedule) and args.starting_epsilon is not None: 
            epsilon = float(eps_schedule[t-args.warmup])
        else:
            epsilon = args.epsilon
            
        if t < args.warmup:
            kappa = 1
        elif args.warmup <= t < args.warmup+len(kappa_schedule):
            kappa = float(kappa_schedule[t-args.warmup])
        else:
            kappa = args.kappa
        print('%.f th epoch: epsilon: %.3f, kappa: %.3f, lr: %.5f'%(t,epsilon,kappa,opt.state_dict()['param_groups'][0]['lr']))
            
        if t < args.warmup:
            utils.train(train_loader, model[-1], opt, t, train_log, args.verbose)
            _ = utils.evaluate(test_loader, model[-1], t, test_log, args.verbose)
        elif args.method == 'BCP' and args.warmup <= t:
            st = time.time()
            u_list = BCP.train_BCP(train_loader, model[-1], opt, epsilon, kappa, t, train_log, args.verbose, args, u_list)
            print('Taken', time.time()-st, 's/epoch')
                
            err = BCP.evaluate_BCP(test_loader, model[-1], args.epsilon, t, test_log, args.verbose, args, u_list)
            print("Error: %.4f, Best Error: %.4f'%(err.data.cpu().item(), best_err))
            
        if args.lr_scheduler == 'step': 
            lr_scheduler.step(epoch=max(t-len(eps_schedule)-args.warmup, 0))
        elif args.lr_scheduler =='multistep':
            lr_scheduler.step(epoch=t)   
            
        if err < best_err and args.save: 
            print('Best Error Found! %.3f'%err)
            best_err = err
            torch.save({
                'state_dict' : [m.state_dict() for m in model], 
                'err' : best_err,
                'epoch' : t,
                'sampler_indices' : sampler_indices
                }, args.prefix + "_best.pth")

        torch.save({ 
            'state_dict': [m.state_dict() for m in model],
            'err' : err,
            'epoch' : t,
            'sampler_indices' : sampler_indices
            }, args.prefix + "_checkpoint.pth")  
        
    args.sniter = 10000
    args.opt_max = 20
    args.print = True
    
    aa = torch.load(args.prefix + "_best.pth")['state_dict'][0]
    model_eval = utils.select_model(args.data, args.model)
    model_eval.load_state_dict(aa)
    print('std testing ...')
    std_err = utils.evaluate(test_loader, model_eval, t, test_log, args.verbose)
    print('pgd testing ...')
    pgd_err = utils.evaluate_pgd(test_loader, model_eval, args)
    print('verification testing ...')
    last_err = BCP.evaluate_BCP(test_loader, model_eval, args.epsilon, t, test_log, args.verbose, args, u_list)
    print('Best model evaluation:', last_err.item())
    print('Best model evaluation:', std_err.item(), pgd_err.item(), last_err.item())
