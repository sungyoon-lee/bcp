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
import utils

DEBUG = False

def train_BCP(loader, model, opt, epsilon, kappa, epoch, log, verbose, args, u_list):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()
    another_robust_losses = AverageMeter()
    another_robust_errors = AverageMeter()
    
    model.train()

    end = time.time()
    net_bcp = net_BCP(model, args)
        
    for i, (X,y) in enumerate(loader):
        epsilon1 = epsilon+i/len(loader)*(args.epsilon-args.starting_epsilon)/args.schedule_length
        kappa1 = kappa+i/len(loader)*(args.kappa-args.starting_kappa)/args.schedule_length
        X,y = X.cuda(), y.cuda()
        data_time.update(time.time() - end)
 
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.max(1)[1] != y).float().sum()  / X.size(0)

        bcp_loss, another_loss, bcp_err, another_err, u_list = BCP_loss(net_bcp, epsilon1, 
                                                                 Variable(X), Variable(y), u_list, args.sniter, args.opt_iter, bce=args.bce, linfty=args.linfty)
        loss = kappa1*ce + (1-kappa1)*bcp_loss
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
        errors.update(err.item(), X.size(0))
        robust_losses.update(bcp_loss.detach().item(), X.size(0))
        robust_errors.update(bcp_err, X.size(0))
        
        another_robust_losses.update(another_loss.detach().item(), X.size(0))
        another_robust_errors.update(another_err, X.size(0))
        
        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()


        print(epoch, i, bcp_loss.detach().item(), 
                bcp_err.item(), ce.item(), err.item(), file=log)
              
        if verbose and (i==0 or (i+1) % verbose == 0): 
            endline = '\n' if (i==0 or (i+1) % verbose == 0) else '\r'
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Eps {eps:.3f}\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'L {loss.val:.4f} ({loss.avg:.4f})//'
                  'RL {rloss.val:.4f} ({rloss.avg:.4f})//'
                  'RL1 {rloss1.val:.4f} ({rloss1.avg:.4f})\t'
                  'E {errors.val:.3f} ({errors.avg:.3f})//'
                  'RE {rerrors.val:.3f} ({rerrors.avg:.3f})//'
                  'RE1 {rerrors1.val:.3f} ({rerrors1.avg:.3f})'.format(
                   epoch, i+1, len(loader), batch_time=batch_time, eps=epsilon1,
                   data_time=data_time, loss=losses, errors=errors, 
                   rloss = robust_losses, rerrors = robust_errors, rloss1 = another_robust_losses, rerrors1 = another_robust_errors), end=endline)
        log.flush()

        del X, y, out, ce, err, bcp_loss, another_loss, bcp_err, another_err, loss
        if DEBUG and i ==10: 
            break
    print('')
    torch.cuda.empty_cache()
    return u_list
    

def evaluate_BCP(loader, model, epsilon, epoch, log, verbose, args, u_list):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()
    another_robust_losses = AverageMeter()
    another_robust_errors = AverageMeter()

    model.eval()

    end = time.time()

    torch.set_grad_enabled(False)
    net_bcp = net_BCP(model, args)
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)


        bcp_loss, another_loss, bcp_err, another_err, u_list = BCP_loss(net_bcp, epsilon, 
                                                                 Variable(X), Variable(y), u_list, args.test_sniter, args.test_opt_iter, show=args.print, linfty=args.linfty)

        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.max(1)[1] != y).float().sum()  / X.size(0)

        # _,pgd_err = _pgd(model, Variable(X), Variable(y), epsilon)

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
        errors.update(err, X.size(0))
        robust_losses.update(bcp_loss.detach().item(), X.size(0))
        robust_errors.update(bcp_err, X.size(0))
        
        another_robust_losses.update(another_loss.detach().item(), X.size(0))
        another_robust_errors.update(another_err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        print(epoch, i, bcp_loss.item(), bcp_err.item(), ce.item(), err.item(),
           file=log)
        if verbose and not args.print: 
            # print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err)
            endline = '\n' if ((i+1) % verbose == 0 or i==0) else '\r'
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'L {loss.val:.4f} ({loss.avg:.4f})//'
                  'RL {rloss.val:.3f} ({rloss.avg:.3f})//'
                  'RL1 {rloss1.val:.3f} ({rloss1.avg:.3f})\t'
                  'E {error.val:.3f} ({error.avg:.3f})//'
                  'RE {rerrors.val:.3f} ({rerrors.avg:.3f})//'
                  'RE1 {rerrors1.val:.3f} ({rerrors1.avg:.3f})'.format(
                      i+1, len(loader), batch_time=batch_time, 
                      loss=losses, error=errors, rloss = robust_losses, 
                      rerrors = robust_errors, rloss1 = another_robust_losses, 
                      rerrors1 = another_robust_errors), end=endline)
        log.flush()

        del X, y, bcp_loss, out, ce
        if DEBUG and i ==10: 
            break
    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()
    print('')
    print(' * Robust error {rerror.avg:.3f}\t'
          'Error {error.avg:.3f}'
          .format(rerror=robust_errors, error=errors))
    return robust_errors.avg

def BCP_loss(net_bcp, epsilon, X, y, u_list=None, sniter=1, opt_iter=1, show=False, bce=False, linfty=False):
    mu, mu_prev, r_prev, ibp_mu, ibp_mu_prev, ibp_r_prev, W, u_list = net_bcp(X, epsilon, u_list, sniter)
    onehot_y = utils.one_hot(y)
    bcp_translation = BCP_translation(mu_prev, r_prev, ibp_mu_prev, ibp_r_prev, W, opt_iter, show)
    bcp_translation[bcp_translation!=bcp_translation]=0
    bcp_logit = translation(mu, bcp_translation, onehot_y)
    if bce:                        
        adv_probs = F.softmax(bcp_logit, dim=1)
        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:,-1]==y, tmp1[:,-2], tmp1[:,-1])
        bcp_loss = nn.CrossEntropyLoss()(bcp_logit,y) + F.nll_loss(torch.log(1.0001-adv_probs+1e-12), new_y)
    else:
        bcp_loss = nn.CrossEntropyLoss()(bcp_logit, y)
    
    
    bcp_err = (bcp_logit.max(1)[1].data != y.cuda()).float().sum()/ X.size(0)
    if linfty:
        another_t = ibp_translation(ibp_mu_prev, ibp_r_prev, W) #####
        another_logit = translation(ibp_mu, another_t, onehot_y)
    else:
        another_t = lmt_translation(mu_prev, r_prev, W)
        another_logit = translation(mu, another_t, onehot_y)
    another_loss = nn.CrossEntropyLoss()(another_logit, y)    
    another_err = (another_logit.max(1)[1].data != y.cuda()).float().sum()/ X.size(0)
    
    return bcp_loss, another_loss, bcp_err, another_err, u_list
    
class net_BCP(nn.Module):
    def __init__(self, model, args):
        super(net_BCP,self).__init__()
        self.model = model
        self.args = args
        
    def forward(self, x, epsilon=None, u_list=None, sniter=1):
        args = self.args
        eps = epsilon
        self.sniter = sniter
        
        mu = x.clone() 
        self.b =  x.size()[0] # batch_size
        
        r = eps*torch.ones(x.size()[0]).cuda() # b, 
        ibp_r = eps*torch.ones(x.size()).cuda() # b,3,32,32
        if args.linfty:
            r *= np.sqrt(x.reshape(x.size(0),-1).size(1)) ### 55.4256
            
        depth = len(list(self.model.children())) # number of main layers
        ibp_mu = mu
            
        if u_list is None:
            u_list = [None]*depth
        self.u_list = u_list
        for i, layer in enumerate(self.model.children()):
            if (i+1)==depth:
                mu_prev = mu
                r_prev = r
                ibp_mu_prev = ibp_mu
                ibp_r_prev = ibp_r
                mu = self._linear(mu, layer.weight, layer.bias)
                ibp_mu = self._linear(ibp_mu_prev, layer.weight.abs(), None)
                
                return mu, mu_prev, r_prev, ibp_mu, ibp_mu_prev, ibp_r_prev, layer.weight, u_list
            
            if isinstance(layer, nn.Conv2d):
                mu, r, ibp_mu, ibp_r = self.bcp_conv2d(layer, mu, r, ibp_mu, ibp_r, i)
            elif isinstance(layer, nn.Linear):
                mu, r, ibp_mu, ibp_r = self.bcp_linear(layer, mu, r, ibp_mu, ibp_r, i)
            elif isinstance(layer, nn.ReLU):
                mu, r, ibp_mu, ibp_r = self.bcp_relu(layer, mu, r, ibp_mu, ibp_r, i)
            elif layer.__class__.__name__=='Flatten': #isinstance(layer, nn.Flatten):
                mu, r, ibp_mu, ibp_r = self.bcp_flatten(layer, mu, r, ibp_mu, ibp_r, i)

    def bcp_conv2d(self, layer, mu, r, ibp_mu, ibp_r, i):
        ibp_mu = self._conv2d(ibp_mu, layer.weight, layer.bias, layer.stride, padding=layer.padding)
        ibp_r = self._conv2d(ibp_r, torch.abs(layer.weight), None, layer.stride, padding=layer.padding)
        ibp_ub = (ibp_mu+ibp_r)
        ibp_lb = (ibp_mu-ibp_r) 
        
        mu_after = self._conv2d(mu, layer.weight, layer.bias, layer.stride, layer.padding)

        u = self.u_list[i]
        p, u = power_iteration_conv_evl(mu, layer, self.sniter, u) 
        self.u_list[i] = u.data                

        W = layer.weight

        ibp_mu1 = mu_after  
        ibp_p1 = W.view(W.size()[0],-1).norm(2,dim=-1)
        ibp_r1 = r.view(-1,1,1,1)*torch.ones_like(mu_after)
        ibp_r1 = ibp_r1*ibp_p1.view(1,-1,1,1)
        ibp_ub1 = ibp_mu1+ibp_r1
        ibp_lb1 = ibp_mu1-ibp_r1
        ibp_ub = torch.min(ibp_ub, ibp_ub1)
        ibp_lb = torch.max(ibp_lb, ibp_lb1)
        ibp_mu = (ibp_ub+ibp_lb)/2
        ibp_r = (ibp_ub-ibp_lb)/2  

        r = r*p 
        mu = mu_after
                
        return mu, r, ibp_mu, ibp_r
    
    def bcp_linear(self, layer, mu, r, ibp_mu, ibp_r, i):
        
        ibp_mu = self._linear(ibp_mu, layer.weight, layer.bias)
        ibp_r = self._linear(ibp_r, torch.abs(layer.weight),None)
        ibp_ub = (ibp_mu+ibp_r)
        ibp_lb = (ibp_mu-ibp_r)
        
        mu_after = self._linear(mu, layer.weight, layer.bias)
        W = layer.weight ### h(-1),h(-2)

        u = self.u_list[i]
        p, u = power_iteration_evl(W, self.sniter, u)
        self.u_list[i] = u.data

        ibp_mu1 = mu_after ### b, h(-1)
        ibp_r1 = r.view(self.b,1)*W.norm(2,dim=-1).view(1,-1).repeat(self.b,1)

        ibp_ub1 = ibp_mu1+ibp_r1
        ibp_lb1 = ibp_mu1-ibp_r1

        ibp_ub = torch.min(ibp_ub, ibp_ub1)
        ibp_lb = torch.max(ibp_lb, ibp_lb1)
        ibp_mu = (ibp_ub+ibp_lb)/2
        ibp_r = (ibp_ub-ibp_lb)/2

        r = r*p 
        mu = mu_after
        return mu, r, ibp_mu, ibp_r
    
    def bcp_relu(self, layer, mu, r, ibp_mu, ibp_r, i):  
        ibp_ub = ibp_mu+ibp_r
        ibp_lb = ibp_mu-ibp_r
        ibp_ub = self._relu(ibp_ub)
        ibp_lb = self._relu(ibp_lb)
        ibp_mu = (ibp_ub+ibp_lb)/2
        ibp_r = (ibp_ub-ibp_lb)/2

        mu_size = []
        for j in range(len(mu.size())):
            if j<1:
                continue
            mu_size.append(1)        

        ibp_ub1 = self._relu(mu+r.view(-1,*mu_size))
        ibp_lb1 = self._relu(mu-r.view(-1,*mu_size))
        ibp_ub = torch.min(ibp_ub, ibp_ub1)
        ibp_lb = torch.max(ibp_lb, ibp_lb1)
        ibp_mu = (ibp_ub+ibp_lb)/2
        ibp_r = (ibp_ub-ibp_lb)/2
        
        mu = self._relu(mu)        
        r = r
        self.u_list[i] = 1
        return mu, r, ibp_mu, ibp_r
    
    def bcp_flatten(self, layer, mu, r, ibp_mu, ibp_r, i):
        ibp_mu = self._flatten(ibp_mu)
        ibp_r = self._flatten(ibp_r)
        mu = self._flatten(mu)
        r = r
        self.u_list[i] = 1
        return mu, r, ibp_mu, ibp_r
    
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
    

def BCP_translation(mu_prev, r_prev, ibp_mu_prev, ibp_r_prev, W, opt_iter, show):
    EPS = 1e-12
    
    ## W: c, h
    c, h = W.size()
    b = mu_prev.size()[0]

    all_one = torch.ones(1,c,c,1).cuda()
    eye = torch.eye(c,c).view(1,c,c,1).cuda()
    diag_zero = ((all_one-eye)==1).type(torch.cuda.FloatTensor).repeat(1,1,1,1) # b,c,c,1
    
    wi_wj = (W.view(c,1,-1).repeat(1,c,1) -W.repeat(c,1,1)).unsqueeze(0) # 1,10(i),10(j),h
    wi_wj_rep = wi_wj.repeat(b,1,1,1)

    # Use EPS to avoid an error in gradient computation.
    w_norm = (wi_wj+EPS).norm(2,dim=-1,keepdim=True) # 1,10,10,1

    r_rep = r_prev.view(b,1,1,1) # 1,1,1,1

    wi_wj1 =  r_rep*F.normalize(wi_wj, p=2, dim=-1) # b,10,10,1

    norm = diag_zero*(r_rep+EPS)+EPS

    mu_rep = mu_prev.view(b,1,1,h) # b,1,1,h                    
    i_mu_rep = ibp_mu_prev.view(b,1,1,h) # b,1,1,h
    i_r_rep = ibp_r_prev.view(b,1,1,h) # b,1,1,h

    ### mu as a zero point
    ub = i_mu_rep + i_r_rep - mu_rep # b,1,1,h
    lb = i_mu_rep - i_r_rep - mu_rep # b,1,1,h
    
    if ub.min()<0 or lb.max()>0:
        ub[ub<0]=0        
        lb[lb>0]=0


    iteration = 0

    while (iteration == 0 or ((diag_zero*wi_wj1<lb)+(diag_zero*wi_wj1>ub)).sum()>0 and iteration < opt_iter):
        iteration +=1
        before_clipped = diag_zero*wi_wj1*(wi_wj1<=lb).type(torch.float) + diag_zero*wi_wj1*(wi_wj1>=ub).type(torch.float)
        after_clipped = diag_zero*lb*(wi_wj1<=lb).type(torch.float) + diag_zero*ub*(wi_wj1>=ub).type(torch.float)

        original_norm = (diag_zero*(before_clipped+EPS)).norm(2,dim=-1,keepdim=True)
        clipped_norm = (diag_zero*(after_clipped+EPS)).norm(2,dim=-1,keepdim=True)

        r_before2 = torch.clamp(norm.abs()**2-original_norm.abs()**2, min=0)
        r_after2 = torch.clamp(norm.abs()**2-clipped_norm.abs()**2, min=0)

        r_before = (r_before2+EPS+eye).sqrt()
        r_after = (r_after2+EPS+eye).sqrt()

        ratio = r_after/(r_before+EPS)

        in_sample = ((wi_wj1>lb)*(wi_wj1<ub)).type(torch.float) 
        enlarged_part = ratio*diag_zero*wi_wj1*in_sample 

        if (r_before!=r_before).sum():
            print('NaN')
            print(opt_iter)
            print(inner_prod)
            return inner_prod
        
        new_wi_wj = (after_clipped + enlarged_part)
        if torch.isnan(new_wi_wj).sum()>0:
            print('NanNan')
            print(opt_iter)
            return inner_prod

        inner_prod  = (wi_wj_rep*new_wi_wj).sum(-1)

        if show:
            print('iter', iteration, end=' / ')
            print('out ratio: %.8f'%(((diag_zero*wi_wj1<lb)+(diag_zero*wi_wj1>ub)).sum().float()/(b*c*c*h)), end=' -> ')
            print('%.8f'%(((new_wi_wj<lb)+(new_wi_wj>ub)).sum().float()/(b*c*c*h)), end=' / ')
            LMT_ip = (wi_wj_rep.view(-1,h)*wi_wj1.view(-1,h)).sum(-1).view(b,c,c)
            print('t_BCP/t_LMT: %.4f'%((diag_zero*inner_prod.view(b,c,c,1)).sum()/(diag_zero*LMT_ip.view(b,c,c,1)).sum()))

        wi_wj1 = new_wi_wj

    return inner_prod

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


def lmt_translation(mu_prev, r_prev, W): ################# bottle neck
    EPS = 1e-24
    c, h = W.size()
    b = mu_prev.size()[0]
    WW = W.repeat(c,1,1)
    WWW = W.view(c,1,-1).repeat(1,c,1)
    
    wi_wj = (WWW-WW).unsqueeze(0) # 1,10,10,h
    w_norm = (wi_wj.view(-1,h).norm(2,dim=-1)).view(1,c,c,1)+EPS # 1,10,10,1

    r_rep = r_prev.view(b,1,1,1).repeat(1,c,c,1) # b,10,10,1
    
    new_wi_wj = r_rep*wi_wj/w_norm #wi_wj1
    
    wi_wj_rep = wi_wj.repeat(b,1,1,1)
    inner_prod  = (wi_wj_rep.view(-1,h)*new_wi_wj.view(-1,h)).sum(-1).view(b,c,c)
    
    return inner_prod

def _conv2d(x,w,b,stride=1,padding=0):
    return F.conv2d(x, w, bias=b, stride=stride, padding=padding)


def _conv_trans2d(x,w,stride=1, padding=0, output_padding = 0):
    return F.conv_transpose2d(x, w,stride=stride, padding=padding, output_padding=output_padding)

    
def power_iteration_evl(A, num_simulations, u=None):
    if u is None:
        u = torch.randn((A.size()[0],1)).cuda()
        
    B = A.t()
#     with torch.no_grad():
    for i in range(num_simulations):
        u1 = B.mm(u)
        u1_norm = u1.norm(2)
        v = u1 / u1_norm
        u_tmp = u

        v1 = A.mm(v)
        v1_norm = v1.norm(2)
        u = v1 / v1_norm

        if (u-u_tmp).norm(2)<1e-3 or (i+1)==num_simulations: ### https://www.cs.cornell.edu/~bindel/class/cs6210-f16/lec/2016-10-17.pdf
            break
        
    out = u.t().mm(A).mm(v)[0][0]

    return out, u


def power_iteration_conv_evl(mu, layer, num_simulations, u=None):
    EPS = 1e-24
    output_padding = 0
    if u is None:
        u = torch.randn((1,*mu.size()[1:])).cuda()
        
    W = layer.weight
    if layer.bias is not None:
        b=torch.zeros_like(layer.bias)
    else:
        b = None
#     with torch.no_grad():
    for i in range(num_simulations):
        u1 = _conv2d(u, W, b, stride=layer.stride, padding=layer.padding)
        u1_norm = u1.norm(2)
        v = u1 / (u1_norm+EPS)
        u_tmp = u

        v1 = _conv_trans2d(v, W, stride=layer.stride, padding=layer.padding, output_padding = output_padding)
        #  When the output size of conv_trans differs from the expected one.
        if v1.shape != u.shape:
            output_padding = 1
            v1 = _conv_trans2d(v, W, stride=layer.stride, padding=layer.padding, output_padding = output_padding)            
        v1_norm = v1.norm(2)
        u = v1 / (v1_norm+EPS)

        if (u-u_tmp).norm(2)<1e-3 or (i+1)==num_simulations:
            break
        
    out = (v*(_conv2d(u, W, b, stride=layer.stride, padding=layer.padding))).view(v.size()[0],-1).sum(1)[0]
    return out, u



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
    
