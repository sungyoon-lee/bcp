# -*- coding: utf-8 -*-

### basic modules
import numpy as np
import time, pickle, os, sys, json, PIL, tempfile, warnings, importlib, math, copy, shutil, setproctitle
import seaborn as sns
import matplotlib.pyplot as plt

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
    args = utils.argparser(data='mnist',epsilon=1.58,epsilon_infty=0.1,epochs=60,rampup=20,wd_list=[21,30,40])
    print(args)
    setproctitle.setproctitle(args.prefix)
    train_log = open(args.prefix + "_train.log", "w")
    test_log = open(args.prefix + "_test.log", "w")
    
    _, test_loader = data_load.data_loaders(args.data, args.test_batch_size, args.normalization, args.augmentation, args.drop_last, args.shuffle)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

        
    args.sniter = 1000000
    args.opt_max = 10000
    args.print = True
    t = 100
    
    aa = torch.load(args.test_pth)['state_dict'][0]
    model_eval = utils.select_model(args.data, args.model)
    model_eval.load_state_dict(aa)
    print('std testing ...')
    std_err = utils.evaluate(test_loader, model_eval, t, test_log, args.verbose)
    print('pgd testing ...')
    pgd_err = utils.evaluate_pgd(test_loader, model_eval, args)
    print('verification testing ...')
    if args.method=='BCP':
        last_err = BCP.evaluate_BCP(test_loader, model_eval, args.epsilon, t, test_log, args.verbose, args, None)
#     if args.method=='SR':
#         last_err = SR.evaluate_BCP(test_loader, model_eval, args.epsilon, t, test_log, args.verbose, args, u_list)
    elif args.method=='IBP':
        last_err = IBP.evaluate_IBP(test_loader, model_eval, args.epsilon, t, test_log, args.verbose, args)
    print('Best model evaluation:', std_err.item(), pgd_err.item(), last_err.item())