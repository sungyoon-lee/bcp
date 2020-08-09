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
import utils

DEBUG = False



def train_IBP(loader, model, opt, epsilon, kappa, epoch, log, verbose, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()
#     another_robust_losses = AverageMeter()
#     another_robust_errors = AverageMeter()
    
    model.train()

    end = time.time()
    net_ibp = net_IBP(model, args)
        
    for i, (X,y) in enumerate(loader):
        epsilon1 = epsilon+i/len(loader)*(args.epsilon-args.starting_epsilon)/args.schedule_length
        kappa1 = kappa+i/len(loader)*(args.kappa-args.starting_kappa)/args.schedule_length
        X,y = X.cuda(), y.cuda()
        data_time.update(time.time() - end)
 
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.max(1)[1] != y).float().sum()  / X.size(0)

        ibp_loss, ibp_err = IBP_loss(net_ibp, epsilon1, Variable(X), Variable(y), bce=args.bce)
        loss = kappa1*ce + (1-kappa1)*ibp_loss
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
        errors.update(err.item(), X.size(0))
        robust_losses.update(ibp_loss.detach().item(), X.size(0))
        robust_errors.update(ibp_err, X.size(0))
        
#         another_robust_losses.update(another_loss.detach().item(), X.size(0))
#         another_robust_errors.update(another_err, X.size(0))
        
        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()


        print(epoch, i, ibp_loss.detach().item(), 
                ibp_err.item(), ce.item(), err.item(), file=log)
              
        if verbose and (i==0 or (i+1) % verbose == 0): 
            endline = '\n' if (i==0 or (i+1) % verbose == 0) else '\r'
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Eps {eps:.3f}\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'L {loss.val:.4f} ({loss.avg:.4f})//'
                  'RL {rloss.val:.4f} ({rloss.avg:.4f})\t'
                  'E {errors.val:.3f} ({errors.avg:.3f})//'
                  'RE {rerrors.val:.3f} ({rerrors.avg:.3f})'.format(
                   epoch, i+1, len(loader), batch_time=batch_time, eps=epsilon1,
                   data_time=data_time, loss=losses, errors=errors, 
                   rloss = robust_losses, rerrors = robust_errors), end=endline)
        log.flush()

        del X, y, out, ce, err, ibp_loss, ibp_err, loss
        if DEBUG and i ==10: 
            break
    print('')
    torch.cuda.empty_cache()
    

def evaluate_IBP(loader, model, epsilon, epoch, log, verbose, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()
#     another_robust_losses = AverageMeter()
#     another_robust_errors = AverageMeter()

    model.eval()

    end = time.time()

    torch.set_grad_enabled(False)
    net_ibp = net_IBP(model, args)
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)


        ibp_loss, ibp_err = IBP_loss(net_ibp, epsilon, Variable(X), Variable(y), show=args.print)

        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.max(1)[1] != y).float().sum()  / X.size(0)

        # _,pgd_err = _pgd(model, Variable(X), Variable(y), epsilon)

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
        errors.update(err, X.size(0))
        robust_losses.update(ibp_loss.detach().item(), X.size(0))
        robust_errors.update(ibp_err, X.size(0))
        
#         another_robust_losses.update(another_loss.detach().item(), X.size(0))
#         another_robust_errors.update(another_err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        print(epoch, i, ibp_loss.item(), ibp_err.item(), ce.item(), err.item(),
           file=log)
        if verbose and not args.print: 
            # print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err)
            endline = '\n' if ((i+1) % verbose == 0 or i==0) else '\r'
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'L {loss.val:.4f} ({loss.avg:.4f})//'
                  'RL {rloss.val:.3f} ({rloss.avg:.3f})\t'
                  'E {error.val:.3f} ({error.avg:.3f})//'
                  'RE {rerrors.val:.3f} ({rerrors.avg:.3f})'.format(
                      i+1, len(loader), batch_time=batch_time, 
                      loss=losses, error=errors, rloss = robust_losses, 
                      rerrors = robust_errors), end=endline)
        log.flush()

        del X, y, ibp_loss, out, ce
        if DEBUG and i ==10: 
            break
    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()
    print('')
    print(' * Robust error {rerror.avg:.3f}\t'
          'Error {error.avg:.3f}'
          .format(rerror=robust_errors, error=errors))
    return robust_errors.avg


def IBP_loss(net_ibp, epsilon, X, y, show=False, bce=False):
#     mu, mu_prev, r_prev, 
    ibp_mu, ibp_mu_prev, ibp_r_prev, W = net_ibp(X, epsilon)
    onehot_y = utils.one_hot(y)
    ibp_t = ibp_translation(ibp_mu_prev, ibp_r_prev, W) #####
    ibp_logit = translation(ibp_mu, ibp_t, onehot_y)
    if bce:                        
        adv_probs = F.softmax(ibp_logit, dim=1)
        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:,-1]==y, tmp1[:,-2], tmp1[:,-1])
        ibp_loss = nn.CrossEntropyLoss()(ibp_logit,y) + F.nll_loss(torch.log(1.0001-adv_probs+1e-12), new_y)
    else:
        ibp_loss = nn.CrossEntropyLoss()(ibp_logit, y)
    
    
    ibp_err = (ibp_logit.max(1)[1].data != y.cuda()).float().sum()/ X.size(0)
    
    return ibp_loss, ibp_err   


class net_IBP(nn.Module):
    def __init__(self, model, args):
        super(net_IBP,self).__init__()
        self.model = model
        self.args = args
        
    def forward(self, x, epsilon=None):
        args = self.args
        eps = epsilon
        
        mu = x.clone() 
        self.b =  x.size()[0] # batch_size
        
        ibp_r = eps*torch.ones(x.size()).cuda() # b,3,32,32
            
        depth = len(list(self.model.children())) # number of main layers
        ibp_mu = mu
            
        for i, layer in enumerate(self.model.children()):
            if (i+1)==depth:
                ibp_mu_prev = ibp_mu
                ibp_r_prev = ibp_r
                ibp_mu = self._linear(ibp_mu_prev, layer.weight.abs(), None)
                
                return ibp_mu, ibp_mu_prev, ibp_r_prev, layer.weight
            
            if isinstance(layer, nn.Conv2d):
                ibp_mu, ibp_r = self.ibp_conv2d(layer, ibp_mu, ibp_r)
            elif isinstance(layer, nn.Linear):
                ibp_mu, ibp_r = self.ibp_linear(layer, ibp_mu, ibp_r)
            elif isinstance(layer, nn.ReLU):
                ibp_mu, ibp_r = self.ibp_relu(layer, ibp_mu, ibp_r)
            elif layer.__class__.__name__=='Flatten': #isinstance(layer, nn.Flatten):
                ibp_mu, ibp_r = self.ibp_flatten(layer, ibp_mu, ibp_r)

    def ibp_conv2d(self, layer, ibp_mu, ibp_r):
        ibp_mu = self._conv2d(ibp_mu, layer.weight, layer.bias, layer.stride, padding=layer.padding)
        ibp_r = self._conv2d(ibp_r, torch.abs(layer.weight), None, layer.stride, padding=layer.padding)        
                
        return ibp_mu, ibp_r
    
    def ibp_linear(self, layer, ibp_mu, ibp_r):
        
        ibp_mu = self._linear(ibp_mu, layer.weight, layer.bias)
        ibp_r = self._linear(ibp_r, torch.abs(layer.weight),None)
        
        return ibp_mu, ibp_r
    
    def ibp_relu(self, layer,ibp_mu, ibp_r):  
        ibp_ub = ibp_mu+ibp_r
        ibp_lb = ibp_mu-ibp_r
        ibp_ub = self._relu(ibp_ub)
        ibp_lb = self._relu(ibp_lb)
        ibp_mu = (ibp_ub+ibp_lb)/2
        ibp_r = (ibp_ub-ibp_lb)/2

        return ibp_mu, ibp_r
    
    def ibp_flatten(self, layer, ibp_mu, ibp_r):
        ibp_mu = self._flatten(ibp_mu)
        ibp_r = self._flatten(ibp_r)
        
        return ibp_mu, ibp_r
    
    def _conv2d(self,x,w,b,stride=1,padding=0):
        return F.conv2d(x, w,bias=b, stride=stride, padding=padding)
    
    def _linear(self,x,w,b):
        return F.linear(x,w,b)
    
    def _relu(self,x):
        return F.relu(x)
    
    def _flatten(self,x):
        return x.view(x.size()[0],-1)
    
    def _tanh(self,x):
        return torch.tanh(x)
    
    def _maxpool2d(self,x,kernel_size=2):
        return F.max_pool2d(x,kernel_size=kernel_size)

    def _avgpool2d(self, x, kernel_size, stride, padding=0):
        return F.avg_pool2d(x,kernel_size=kernel_size, stride=stride, padding=padding)
    
    def _batchnorm2d(self, x, mean, var, weight, bias, bn_eps):
        mean = mean.repeat(x.size()[0],1,x.size()[2], x.size()[3])
        var = var.repeat(x.size()[0],1,x.size()[2], x.size()[3])
        weight = weight.view(1,-1,1,1)
        bias = bias.view(1,-1,1,1)
        weight = weight.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        bias = bias.repeat(x.size()[0], 1,x.size()[2], x.size()[3])
        
        multiplier = torch.rsqrt(var+bn_eps)*weight ### rsqrt=1/sqrt
        b = -multiplier*mean + bias
        y = multiplier*x
        y+=b
        
        return y
    
def translation(mu, translation, onehot_y):
    b, c = onehot_y.size()
    # mu : b, 10
    # translation : b, 10, 10
    # onehot_y: b, 10
    
    delta = translation.bmm(onehot_y.unsqueeze(2)).view(b,c) 
    
    # b,10,10 x b,10,1 -> b,10,1 -> b,10
    # delta: b,10
    translated_logit = mu+((delta)*(1-onehot_y))
    return translated_logit
    
#     another_translation = another_translation(mu_prev, r_prev, W)
#     another_logit = translation(mu, another_translation, onehot_y)
#     another_loss = nn.CrossEntropyLoss()(another_logit, y)    

def ibp_translation(ibp_mu_prev, ibp_r_prev, W): ################# bottle neck
    EPS = 1e-24
    c, h = W.size()
    b = ibp_mu_prev.size()[0]
    WW = W.repeat(c,1,1)
    WWW = W.view(c,1,-1).repeat(1,c,1)
    
    wi_wj = (WWW-WW).unsqueeze(0) # 1,10,10,h
    wi_wj_rep = wi_wj.repeat(b,1,1,1)

    u_rep = ibp_r_prev.view(b,1,1,-1).repeat(1,c,c,1) # b,10,10,1
    l_rep = -ibp_r_prev.view(b,1,1,-1).repeat(1,c,c,1) # b,10,10,1
    
   
    inner_prod  = ((wi_wj_rep>=0).type(torch.float)*wi_wj_rep*u_rep).sum(-1)+((wi_wj_rep<0).type(torch.float)*wi_wj_rep*l_rep).sum(-1)
    
    return inner_prod


def _conv2d(x,w,b,stride=1,padding=0):
    return F.conv2d(x, w, bias=b, stride=stride, padding=padding)


def _conv_trans2d(x,w,stride=1, padding=0, output_padding = 0):
    return F.conv_transpose2d(x, w,stride=stride, padding=padding, output_padding=output_padding)


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
    