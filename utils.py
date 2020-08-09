# -*- coding: utf-8 -*-

### basic modules
import numpy as np
import time, pickle, os, sys, json, PIL, tempfile, warnings, importlib, math, copy, shutil
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

import argparse

### default cifar10
def argparser(data='cifar10', model='large', batch_size=50, epochs=100, seed=0, verbose=200, lr=0.0003, 
              rampup=10,
              epsilon=36/255, epsilon_infty=8/255, epsilon_train=36/255, epsilon_train_infty=8/255, starting_epsilon=0.001, 
              opt='adam', momentum=0.9, weight_decay=5e-4,
              gamma=0.1, opt_iter=1, sniter=1,
              starting_kappa=0.99, kappa=0,
              warmup=3, wd_list=[51,70,90], niter=100): 

    parser = argparse.ArgumentParser()
    
    # main settings
    parser.add_argument('--rampup', type=int, default=rampup) ## rampup
    parser.add_argument('--warmup', type=int, default=warmup)
    parser.add_argument('--sniter', type=int, default=sniter) ###
    parser.add_argument('--opt_iter', type=int, default=opt_iter) 
    parser.add_argument('--linfty', default=False) 
    parser.add_argument('--save', default=True) 
    parser.add_argument('--test_pth', default=None)
    parser.add_argument('--print', default=False)
    parser.add_argument('--bce', default=False) 

    # optimizer settings
    parser.add_argument('--opt', default=opt)
    parser.add_argument('--momentum', type=float, default=momentum)
    parser.add_argument('--weight_decay', type=float, default=weight_decay)
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=gamma)
    parser.add_argument("--wd_list", nargs='*', type=int, default=wd_list)
    parser.add_argument("--lr_scheduler", default='multistep')
    
    # test settings during training
    parser.add_argument('--test_sniter', type=int, default=1000) 
    parser.add_argument('--test_opt_iter', type=int, default=10) 

    # pgd settings
    parser.add_argument("--epsilon_pgd", type=float, default=epsilon)
    parser.add_argument("--alpha", type=float, default=epsilon/4)
    parser.add_argument("--niter", type=float, default=niter)
    
    # epsilon settings
    parser.add_argument("--epsilon", type=float, default=epsilon)
    parser.add_argument("--epsilon_infty", type=float, default=epsilon_infty)
    parser.add_argument("--epsilon_train", type=float, default=None)
    parser.add_argument("--epsilon_train_infty", type=float, default=None)
    parser.add_argument("--starting_epsilon", type=float, default=starting_epsilon)
    parser.add_argument('--schedule_length', type=int, default=rampup) ## rampup
    
    # kappa settings
    parser.add_argument("--kappa", type=float, default=kappa)
    parser.add_argument("--starting_kappa", type=float, default=starting_kappa)
    parser.add_argument('--kappa_schedule_length', type=int, default=rampup) ## rampup

    # model arguments
    parser.add_argument('--model', default=model)
    parser.add_argument('--model_factor', type=int, default=8)
    parser.add_argument('--method', default='BCP')
    parser.add_argument('--resnet_N', type=int, default=1)
    parser.add_argument('--resnet_factor', type=int, default=1)


    # other arguments
    parser.add_argument('--prefix')
    parser.add_argument('--data', default=data)
    parser.add_argument('--real_time', action='store_true')
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--verbose', type=int, default=verbose)
    parser.add_argument('--cuda_ids', type=int, default=0)
    
    # loader arguments
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--test_batch_size', type=int, default=batch_size)
    parser.add_argument('--normalization', default=False)
    parser.add_argument('--augmentation', default=True)
    parser.add_argument('--drop_last', default=False)
    parser.add_argument('--shuffle', default=True)

    
    args = parser.parse_args()
    if args.rampup:
        args.schedule_length = args.rampup
        args.kappa_schedule_length = args.rampup 
    if args.epsilon_train is None:
        args.epsilon_train = args.epsilon 
    if args.epsilon_train_infty is None:
        args.epsilon_train_infty = args.epsilon_infty 
    if args.linfty:
        args.epsilon = args.epsilon_infty
        args.epsilon_train = args.epsilon_train_infty
        
        
    if args.starting_epsilon is None:
        args.starting_epsilon = args.epsilon
    if args.prefix: 
        if args.model is not None: 
            args.prefix += '_'+args.model

        if args.method is not None: 
            args.prefix += '_'+args.method

        banned = ['verbose', 'prefix',
                  'resume', 'baseline', 'eval', 
                  'method', 'model', 'cuda_ids', 'load', 'real_time', 
                  'test_batch_size', 'augmentation','batch_size','drop_last','normalization','print','save','step_size','epsilon','gamma','linfty','lr_scheduler','seed','shuffle','starting_epsilon','kappa','kappa_schedule_length','test_sniter','test_opt_iter', 'niter','epsilon_pgd','alpha','schedule_length','epsilon_infty','epsilon_train_infty','test_pth']
        if args.method == 'baseline':
            banned += ['epsilon', 'starting_epsilon', 'schedule_length', 
                       'l1_test', 'l1_train', 'm', 'l1_proj']

        # Ignore these parameters for filename since we never change them
        banned += ['momentum', 'weight_decay']


        # if not using a model that uses model_factor, 
        # ignore model_factor
        if args.model not in ['wide', 'deep']: 
            banned += ['model_factor']

        # if args.model != 'resnet': 
        banned += ['resnet_N', 'resnet_factor']

        for arg in sorted(vars(args)): 
            if arg not in banned and getattr(args,arg) is not None: 
                args.prefix += '_' + arg + '_' +str(getattr(args, arg))
#         args.prefix += '_cifar10'

        if args.schedule_length > args.epochs: 
            raise ValueError('Schedule length for epsilon ({}) is greater than '
                             'number of epochs ({})'.format(args.schedule_length, args.epochs))
    else: 
        args.prefix = 'temporary'

    if args.cuda_ids is not None: 
        print('Setting CUDA_VISIBLE_DEVICES to {}'.format(args.cuda_ids))
#         os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids
        torch.cuda.set_device(args.cuda_ids)


    return args



def select_model(data, m): 
    if data=='mnist':
        if m == 'large': ### Wong et al. large
            model = mnist_model_large().cuda()
        elif m == 'large2': ### Wong et al. large
            model = mnist_model_large2().cuda()
        else: ### Wong et al. small
            model = mnist_model().cuda() 
    elif data=='cifar10':
        if m == 'large':  ### Wong et al. large
            model = cifar_model_large().cuda()
        elif m == 'c5f2':  ### IBP
            model = c5f2().cuda()
        elif m == 'c6f2':  ### IBP
            model = c6f2().cuda()
        else: ### Wong et al. small
            model = cifar_model().cuda()
    elif data=='tinyimagenet':
        model = tinyimagenet().cuda()
        
        
    return model


def mnist_model(): 
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def mnist_model_large(): 
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*7*7,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    return model


def mnist_model_large2(): 
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(1024,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    return model


def cifar_model(): 
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model

def cifar_model_large(): 
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*8*8,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model


def c5f2():
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, stride=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=2),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model


def c6f2():
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(3136,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model


def tinyimagenet():
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2),
        nn.ReLU(),
        
        nn.Conv2d(64, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 4, stride=2),
        nn.ReLU(),
        
        nn.Conv2d(128, 256, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, 4, stride=2),
        nn.ReLU(),
        
        Flatten(),
        nn.Linear(9216,256),
        nn.ReLU(),
        nn.Linear(256,200)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model



############################## Flatten / one_hot

class Flatten(nn.Module): ## =nn.Flatten()
    def forward(self, x):
        return x.view(x.size()[0], -1)

    
def one_hot(batch,depth=10):
    ones = torch.eye(depth).cuda()
    return ones.index_select(0,batch)


##############################





def train(loader, model, opt, epoch, log, verbose):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    
    model.train()

    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda()
        data_time.update(time.time() - end)
 
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.max(1)[1] != y).float().sum()  / X.size(0)
        
        loss = ce
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
        errors.update(err.item(), X.size(0))
        
        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()


        print(epoch, i, ce.item(), err.item(), file=log)
        if verbose and (i==0 or (i+1) % verbose == 0): 
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.3f} ({errors.avg:.3f})'.format(
                   epoch, i+1, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, errors=errors))
        log.flush()
              
              

def evaluate(loader, model, epoch, log, verbose):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    model.eval()

    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda()
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        # print to logfile
        print(epoch, i, ce.item(), err.item(), file=log)

        # measure accuracy and record loss
        losses.update(ce.data, X.size(0))
        errors.update(err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose and (i==0 or (i+1) % verbose == 0): 
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {error.val:.3f} ({error.avg:.3f})'.format(
                      i+1, len(loader), batch_time=batch_time, loss=losses,
                      error=errors))
        log.flush()

    print(' * Error {error.avg:.3f}'
          .format(error=errors))
    return errors.avg


def pgd_l2(model_eval, X, y, epsilon=36/255, niters=100, alpha=9/255):   
    EPS = 1e-24
    X_pgd = Variable(X.data, requires_grad=True)
    
    for i in range(niters): 
        opt = optim.Adam([X_pgd], lr=1.)
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model_eval(X_pgd), y)
        loss.backward()
        grad = 1e10*X_pgd.grad.data
                
        grad_norm = grad.view(grad.shape[0],-1).norm(2, dim=-1, keepdim=True)
        grad_norm = grad_norm.view(grad_norm.shape[0],grad_norm.shape[1],1,1)
                    
        eta = alpha*grad/(grad_norm+EPS)
        eta_norm = eta.view(eta.shape[0],-1).norm(2,dim=-1)
         
        
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = X_pgd.data-X.data                           
        mask = eta.view(eta.shape[0], -1).norm(2, dim=1) <= epsilon
        
        scaling_factor = eta.view(eta.shape[0],-1).norm(2,dim=-1)+EPS
        scaling_factor[mask] = epsilon
        
        eta *= epsilon / (scaling_factor.view(-1, 1, 1, 1)) 

        X_pgd = torch.clamp(X.data + eta, 0, 1)
        X_pgd = Variable(X_pgd.data, requires_grad=True)          
   
    return X_pgd.data
              

def pgd(model_eval, X, y, epsilon=8/255, niters=100, alpha=2/255): 
    X_pgd = Variable(X.data, requires_grad=True)
    for i in range(niters): 
        opt = optim.Adam([X_pgd], lr=1.)
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model_eval(X_pgd), y)
        loss.backward()
        eta = alpha*X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        
        X_pgd = torch.clamp(X.data + eta, 0, 1)
        X_pgd = Variable(X_pgd, requires_grad=True)          
       
    return X_pgd.data

def evaluate_pgd(loader, model, args):
    losses = AverageMeter()
    errors = AverageMeter()

    model.eval()

    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda()
        if args.linfty:
            X_pgd = pgd(model, X, y, args.epsilon, args.niter, args.alpha)
        else:
            X_pgd = pgd_l2(model, X, y, args.epsilon, args.niter, args.alpha)
            
        out = model(Variable(X_pgd))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)
        losses.update(ce.data, X.size(0))
        errors.update(err, X.size(0))
    print(' * Error {error.avg:.3f}'
          .format(error=errors))
    return errors.avg



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
    
def test(net_eval, test_data_loader, imagenet=0):
    st = time.time()
    n_test = len(test_data_loader.dataset)
    
    err = 0
    n_done = 0
    for j, (batch_images, batch_labels) in enumerate(test_data_loader):

        X = Variable(batch_images.cuda())
        Y = Variable(batch_labels.cuda())
        out = net_eval(X)
        
        err += (out.max(1)[1].data != (batch_labels-imagenet).cuda()).float().sum()
        
        b_size = len(Y)
        n_done += b_size
        acc = 100*(1-err/n_done)
        if j % 10 == 0:
            print('%.2f %%'%(100*(n_done/n_test)), end='\r')

    print('test accuracy: %.3f%%'%(acc))
    
    
def test_topk(net_eval, test_data_loader, k=5, imagenet=1):
    st = time.time()
    n_test = len(test_data_loader.dataset)
    
    err = 0
    n_done = 0
    res = 0
    for j, (batch_images, batch_labels) in enumerate(test_data_loader):

        X = Variable(batch_images.cuda())
        Y = Variable(batch_labels.cuda())
        out = net_eval(X)
        
        
        b_size = len(Y)
        n_done += b_size
        
        _,pred= out.topk(max((k,)),1,True,True)        
        aa = (batch_labels-imagenet).view(-1, 1).expand_as(pred).cuda()
        correct = pred.eq(aa)

        for kk in (k,):
            correct_k = correct[:,:kk].view(-1).float().sum(0)
            res += correct_k# (correct_k.mul_(100.0 / b_size))
            
        
        if j % 10 == 0:
            print('%.2f %%'%(100*(n_done/n_test)), end='\r')

    print('test accuracy: %.3f%%'%(100*res/n_done))
    
    
def translation_relu_tight_cut(W, save_mu, r, i_mu, i_r, args, show=False):
    EPS = 1e-12
    

    ## W: c, h
    c, h = W.size()
    b = save_mu.size()[0]

    all_one = torch.ones(1,c,c,1).cuda()
    eye = torch.eye(c,c).view(1,c,c,1).cuda()
    diag_zero = ((all_one-eye)==1).repeat(1,1,1,1) # b,c,c,1

#     WW = W.repeat(c,1,1) # 10,10(j:0~9),h
#     WWW = W.view(c,1,-1).repeat(1,c,1) # 10,1,h -> 10(i:0~9),10,h
    wi_wj = (W.view(c,1,-1).repeat(1,c,1) -W.repeat(c,1,1)).unsqueeze(0) # 1,10(i),10(j),h
    wi_wj_rep = wi_wj.repeat(b,1,1,1)

    # Use EPS to avoid an error in gradient computation.
    w_norm = (wi_wj+EPS).norm(2,dim=-1,keepdim=True) # 1,10,10,1

    r_rep = r.view(b,1,1,1) # 1,1,1,1

#     wi_wj1 =  r.view(b,1,1,1)*wi_wj/(w_norm+EPS) # b,10,10,1
    wi_wj1 =  r_rep*F.normalize(wi_wj, p=2, dim=-1) # b,10,10,1

    norm = diag_zero*(r_rep+EPS)+EPS
#     norm = diag_zero*(wi_wj1.norm(2,-1,keepdim=True)+EPS) # 1,10,10,1
    #norm = diag_zero*wi_wj1.norm(2,-1,keepdim=True) # 1,10,10,1

    mu_rep = save_mu.view(b,1,1,h) # b,1,1,h                    
    i_mu_rep = i_mu.view(b,1,1,h) # b,1,1,h
    i_r_rep = i_r.view(b,1,1,h) # b,1,1,h

    ### mu as a zero point
    ub = i_mu_rep + i_r_rep - mu_rep # b,1,1,h
    lb = i_mu_rep - i_r_rep - mu_rep # b,1,1,h
    
    if ub.min()<0 or lb.max()>0:
        ub[ub<0]=0        
        lb[lb>0]=0


    opt_iter = 0
    opt_iter_max = args.opt_iter_max

    while (opt_iter == 0 or ((diag_zero*wi_wj1<lb)+(diag_zero*wi_wj1>ub)).sum()>0 and opt_iter < opt_iter_max):
        opt_iter +=1

        before_clipped = diag_zero*wi_wj1*(wi_wj1<=lb).type(torch.float) + diag_zero*wi_wj1*(wi_wj1>=ub).type(torch.float)
        after_clipped = diag_zero*lb*(wi_wj1<=lb).type(torch.float) + diag_zero*ub*(wi_wj1>=ub).type(torch.float)

        original_norm = (diag_zero*(before_clipped+EPS)).norm(2,dim=-1,keepdim=True)
        clipped_norm = (diag_zero*(after_clipped+EPS)).norm(2,dim=-1,keepdim=True)

        r_before2 = torch.clamp(norm.abs()**2-original_norm.abs()**2, min=0) #######
        r_after2 = torch.clamp(norm.abs()**2-clipped_norm.abs()**2, min=0)

        r_before = (r_before2+EPS+eye).sqrt()
        r_after = (r_after2+EPS+eye).sqrt()
#             raise ValueError("NAN value")

        ratio = r_after/(r_before+EPS)

        in_sample = ((wi_wj1>lb)*(wi_wj1<ub)).type(torch.float) 
        enlarged_part = ratio*diag_zero*wi_wj1*in_sample 


        new_wi_wj = (after_clipped + enlarged_part)

        if show:
            print('iter', opt_iter, end=' / ')
            print('out ratio: %.8f'%(((diag_zero*wi_wj1<lb)+(diag_zero*wi_wj1>ub)).sum().float()/(b*c*c*h)), end=' -> ')
            print('%.8f'%(((new_wi_wj<lb)+(new_wi_wj>ub)).sum().float()/(b*c*c*h)), end=' / ')

        inner_prod  = (wi_wj_rep*new_wi_wj).sum(-1)
        
        if show:
            LMT_ip = (wi_wj_rep.view(-1,h)*wi_wj1.view(-1,h)).sum(-1).view(b,c,c)
            print('t_BCP/t_LMT: %.4f'%((diag_zero*inner_prod.view(b,c,c,1)).sum()/(diag_zero*LMT_ip.view(b,c,c,1)).sum()))

        wi_wj1 = new_wi_wj
        
    return inner_prod#inner_prod_cut ## b,10,10


def translation_IBP(W, ibp_mu, ibp_r): ################# bottle neck W, mu, net_ibp.ibp_mu, prev_ibp_r
#     print(mu.shape)
#     print(ibp_mu.shape)
#     print(ibp_r.shape)
    EPS = 1e-24
    c, h = W.size()
    b = ibp_r.size()[0]
    WW = W.repeat(c,1,1)
    WWW = W.view(c,1,-1).repeat(1,c,1)
    
    wi_wj = (WWW-WW).unsqueeze(0) # 1,10,10,h
    wi_wj_rep = wi_wj.repeat(b,1,1,1) # b,10,10,h
    
    u_rep = (ibp_r).view(b,1,1,-1).repeat(1,c,c,1) # b,10,10,h
    l_rep = (-ibp_r).view(b,1,1,-1).repeat(1,c,c,1) # b,10,10,h
    
#     new_wi_wj =  # b,10,10,h
    
    inner_prod  = ((wi_wj_rep>=0)*wi_wj_rep*u_rep).sum(-1)+((wi_wj_rep<0)*wi_wj_rep*l_rep).sum(-1)
    
    return inner_prod # b,10,10


def translation_LMT(W, save_mu, r): ################# bottle neck
    EPS = 1e-24
    c, h = W.size()
    b = save_mu.size()[0]
    WW = W.repeat(c,1,1)
    WWW = W.view(c,1,-1).repeat(1,c,1)
    
    wi_wj = (WWW-WW).unsqueeze(0) # 1,10,10,h
    w_norm = (wi_wj.view(-1,h).norm(2,dim=-1)).view(1,c,c,1)+EPS # 1,10,10,1

    r_rep = r.view(b,1,1,1).repeat(1,c,c,1) # b,10,10,1
    
    new_wi_wj = r_rep*wi_wj/w_norm #wi_wj1
    
    wi_wj_rep = wi_wj.repeat(b,1,1,1)
    inner_prod  = (wi_wj_rep.view(-1,h)*new_wi_wj.view(-1,h)).sum(-1).view(b,c,c)
    
    return inner_prod

def translate(mu, translation, onehot_y):
    b, c = onehot_y.size()
    # mu : b, 10
    # translation : b, 10, 10
    # onehot_y: b, 10
    
    delta = translation.bmm(onehot_y.unsqueeze(2)).view(b,c) 
    
    # b,10,10 x b,10,1 -> b,10,1 -> b,10
    # delta: b,10
    translated_logit = mu+((delta)*(1-onehot_y))
    return translated_logit    



def B2bdry(B, mu, onehot_y):
    b, c = onehot_y.size()
    EPS = 1e-24
    # mu : b, 10
    # translation : b, 10, 10
    # onehot_y: b, 10
        
    # b,10 x b,10 --> b,1
    y_mu = (onehot_y*mu).sum(1).view(-1,1)
    # b,10,10 x b,10,1 -> b,10,1 -> b,10
    # delta: b,10
    
    translated_logit = B *( (B+(y_mu-B)*(B>y_mu)) /(B+EPS)).detach()
    return translated_logit 


def robust_test_translate_relu_tight_cut(net_ibp, eps, test_data_loader, W, u_list, uu_list, sn_iter, args): 
    st = time.time()
    n_test = len(test_data_loader.dataset)
    
    err = 0
    n_done = 0
    for j, (batch_images, batch_labels) in enumerate(test_data_loader):

        X = Variable(batch_images.cuda())
        Y = Variable(batch_labels.cuda())
        onehot_y = one_hot(Y, args.n_class)   
        
        out = net_ibp.clf_model(X)
        mu, r, i_mu, i_r, W, _, _ = net_ibp(X, eps, False, u_list, uu_list, sn_iter)
        A = translation_relu_tight_cut(W, net_ibp.prev_mu, r, i_mu, i_r, args, show=True)
        worst_logit = translate(mu, A, onehot_y)
        
        err += (worst_logit.max(1)[1].data != batch_labels.cuda()).float().sum()
        
        b_size = len(Y)        
        n_done += b_size
        acc = 100*(1-err/n_done)
        if j % 10 == 0:
            print('%.2f %%'%(100*(n_done/n_test)), end='\r')

    print('verified accuracy: %.3f%%'%(acc))    

    
def robust_test_translate(net_ibp, eps, test_data_loader, W, u_list, uu_list, sn_iter, args):  #### args 
    st = time.time()
    n_test = len(test_data_loader.dataset)
    tot_err = 0
    n_done = 0
    for j, (batch_images, batch_labels) in enumerate(test_data_loader):

        X = Variable(batch_images.cuda())
        Y = Variable(batch_labels.cuda())
        b_size = len(Y)        
        onehot_y = one_hot(Y, args.n_class)   
        
        out = net_ibp.clf_model(X)
        mu, r, _, _, W, _, _ = net_ibp(X, eps, False, u_list, uu_list, sn_iter)
        
        B = translation_LMT(W, net_ibp.prev_mu, r)
        worst_logit_translate = translate(mu, B, onehot_y)
        
        err = (worst_logit_translate.max(1)[1].data != batch_labels.cuda()).float().sum()
        tot_err += err
        n_done += b_size
        if j % 10 == 0:
            print('%.2f %%'%(100*(n_done/n_test)), end='\r')

    print('verified accuracy: %.3f%%'%(100*(1-tot_err/n_done)))
    
    
def robust_test_IBP(net_ibp, eps, test_data_loader, W, args):  #### args 
    st = time.time()
    n_test = len(test_data_loader.dataset)
    tot_err = 0
    n_done = 0
    for j, (batch_images, batch_labels) in enumerate(test_data_loader):

        X = Variable(batch_images.cuda())
        Y = Variable(batch_labels.cuda())
        b_size = len(Y)        
        onehot_y = one_hot(Y, args.n_class)   
        
        out = net_ibp.clf_model(X)
        a = net_ibp(X, eps, False)
        _, _, prev_ibp_mu, prev_ibp_r, W, _, _ = net_ibp(X, eps, False)
        
        B = translation_IBP(W, prev_ibp_mu, prev_ibp_r)# models.translation_IBP(W, net_ibp.prev_mu, prev_ibp_mu, prev_ibp_r)
        worst_logit_translate = translate(net_ibp.ibp_mu, B, onehot_y) ##########
        
        err = (worst_logit_translate.max(1)[1].data != batch_labels.cuda()).float().sum()
        tot_err += err
        n_done += b_size
        if j % 10 == 0:
            print('%.2f %%'%(100*(n_done/n_test)), end='\r')

    print('verified accuracy: %.3f%%'%(100*(1-tot_err/n_done)))