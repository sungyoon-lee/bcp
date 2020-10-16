# -*- coding: utf-8 -*-

### basic modules
import numpy as np
import time, pickle, os, sys, json, PIL, tempfile, warnings, importlib, math, copy, shutil, setproctitle

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
    args = utils.argparser()
    print(args)
    setproctitle.setproctitle(args.prefix)
    test_log = open(args.prefix + "_test.log", "w")
    
    _, test_loader = data_load.data_loaders(args.data, args.test_batch_size, args.normalization, args.augmentation, args.drop_last, args.shuffle)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.print = False
    t = 100
    
    aa = torch.load(args.test_pth)['state_dict'][0]
    model_eval = utils.select_model(args.data, args.model)
    model_eval.load_state_dict(aa)
    print('verification testing ...')
    ver_errs = []
    
    eps_list =[0,4/255,8/255,12/255,16/255,20/255,24/255,28/255,32/255,36/255,40/255,44/255,48/255,52/255,56/255,60/255,64/255,68/255,72/255,76/255,80/255,88/255,96/255,104/255,112/255,120/255,128/255,136/255,142/255]
    for eps in eps_list:
        print('eps_eval:',eps)
        if args.method=='BCP':
            last_err = BCP.evaluate_BCP(test_loader, model_eval, eps, t, test_log, args.verbose, args, None)
            ver_errs.append(last_err.data.cpu().item())
            print(last_err.data.cpu().item())
        print('-'*100)
    print(ver_errs)
