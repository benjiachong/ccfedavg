#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import copy

def log_batchnorm_record(net, i, logger, global_step):
    for module_prefix, module in net.named_modules():
        classname = module.__class__.__name__
        if classname.find('BatchNorm') != -1:
            for n, p in module.state_dict().items():
                try:
                    logger.add_scalar(module_prefix+n +'/mean' + str(i), p.mean(), global_step)
                    logger.add_scalar(module_prefix + n + '/var' + str(i), p.var(), global_step)
                except:
                    pass


def log_batchnorm_record_one(net, i, logger, step):
    for module_prefix, module in net.named_modules():
        classname = module.__class__.__name__
        if classname.find('BatchNorm') != -1:
            for n, p in module.state_dict().items():
                try:
                    logger.add_scalar(module_prefix+n +'/' + str(i) + '-0', p[0], step)
                    logger.add_scalar(module_prefix + n + '/' + str(i)+ '-1', p[1], step)
                except:
                    pass


def log_batchnorm_record2(net, wlist, logger, global_step):
    for module_prefix, module in net.named_modules():
        classname = module.__class__.__name__
        if classname.find('BatchNorm') != -1:
            for n, p in wlist[0].items():
                if module_prefix in n:
                    mean = p.detach().clone()
                    var = p.detach().clone() ** 2
                    for w in wlist[1:]:
                        mean = mean + w[n]
                        var = var + w[n]**2
                    try:
                        mean = mean/len(wlist)
                        var = var/len(wlist) - mean **2
                        logger.add_scalar(n +'/Evar', var.mean(), global_step)
                    except:
                        pass

def batchnorm_layer_k(net):
    prelist = []
    for module_prefix, module in net.named_modules():
        classname = module.__class__.__name__
        if classname.find('BatchNorm') != -1:
            prelist.extend([module_prefix+'.'+k for k in module.state_dict().keys()])
    return prelist


def is_in(lst, str):
    for e in lst:
        if e in str:
            return True

    return False


class EWC(object):
    def __init__(self, model: nn.Module, dataloader: object, device:str):
        self.model = model
        self.dataloader = dataloader
        self.device = device

        self.params = {n: p for n, p in self.model.named_parameters()}  # if p.requires_grad

    def cal_para(self):

        self._means = {}
        self._precision_matrices = self._diag_fisher()

        #for n, p in deepcopy(self.params).items():
        #    self._means[n] = variable(p.data)
        for n, p in self.params.items():
            self._means[n] = p.clone().detach()

    def set_para(self, weights, precision_matrices):
        self._means = {}
        for n,p in weights:
            self._means[n] = p.clone().detach()
        #self._gradient_matrices_list = gradient_matrices
        self._precision_matrices_list = precision_matrices

    def _diag_fisher(self):
        #gradient_matrices = {}
        precision_matrices = {}
        #for n, p in deepcopy(self.params).items():
        #    p.data.zero_()
        #    precision_matrices[n] = variable(p.data)
        #    gradient_matrices[n] = torch.zeros_like(p)
        for n, p in self.params.items():
            #gradient_matrices[n] = torch.zeros_like(p)
            precision_matrices[n] = torch.zeros_like(p)


        self.model.eval()
        '''
        for input in self.dataset:
            self.model.zero_grad()
            input = variable(input)
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)
        '''
        criterion = nn.CrossEntropyLoss().to(self.device)
        count = 0
        for images, labels in self.dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            output = self.model(images)
            loss = criterion(output, labels)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n] += p.grad.detach() ** 2
                #gradient_matrices[n] += p.grad.detach()
            count += len(labels)


        #gradient_matrices = {n: p / count for n, p in gradient_matrices.items()}
        precision_matrices = {n: p / count for n, p in precision_matrices.items()}
        #return gradient_matrices, precision_matrices
        return precision_matrices

    def penalty1(self, model: nn.Module, count: int):
        loss = torch.tensor(0.0)
        for x in self._gradient_matrices_list:
            loss_one = torch.tensor(0.0)
            for n, p in model.named_parameters():
                _loss = count * x[n] * (p - self._means[n])
                loss_one = loss_one + max(torch.tensor(0.0), _loss.sum())
            loss = loss + loss_one
        return loss

    def penalty2(self, model: nn.Module, count: int):
        loss = torch.tensor(0.0)
        for n, p in model.named_parameters():
            for x in self._gradient_matrices_list:
                _loss = count * x[n] * (p - self._means[n]) ** 2
                loss = loss + _loss.sum()
        return loss

    def penalty0(self, model: nn.Module, count: int):
        loss = torch.tensor(0.0)
        for n, p in model.named_parameters():
            _loss = count * (p - self._means[n]) ** 2
            loss = loss + _loss.sum()
        return loss


def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]


def relu_evidence(y):
    return F.relu(y)

def kl_divergence(alpha, num_classes, device=None):
    beta = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - \
        torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                        keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1,
                   keepdim=True) + lnB + lnB_uni
    return kl

def loglikelihood_loss(y, alpha, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum(
        (y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(torch.tensor(
        1.0, dtype=torch.float32), torch.tensor(epoch_num / annealing_step, dtype=torch.float32))

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * \
        kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(torch.tensor(
        1.0, dtype=torch.float32), torch.tensor(epoch_num / annealing_step, dtype=torch.float32))

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * \
        kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div



def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(mse_loss(target, alpha, epoch_num,
                               num_classes, annealing_step, device=device))
    return loss


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(edl_loss(torch.log, target, alpha,
                               epoch_num, num_classes, annealing_step, device))
    return loss


def edl_digamma_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(edl_loss(torch.digamma, target, alpha,
                               epoch_num, num_classes, annealing_step, device))
    return loss

def cal_gpu(module):
    '''
    给定模块，返回所在的device
    :param module:
    :return:
    '''
    if isinstance(module, torch.nn.DataParallel):
        module = module.module
    for submodule in module.children():
        if hasattr(submodule, "_parameters"):
            parameters = submodule._parameters
            if "weight" in parameters:
                return parameters["weight"].device


def model_minus(w1, w2):
    minus = {}
    for n, p in w1.items():
        minus[n] = (p - w2[n])
    return minus

def model_add(w1, w2):
    add = {}
    for n, p in w1.items():
        add[n] = (p + w2[n])
    return add


def model_avg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        if w_avg[k].numel() != 1:
            w_avg[k] = torch.div(w_avg[k], len(w))
        else:
            w_avg[k] = w_avg[k] // len(w)
        #w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def model_distance(w1, w2):
    dis = torch.tensor(0.0)
    for n,p in w1.items():
        if ('running' not in n) and ('track' not in n):
            dis1 = (w1[n] - w2[n]) ** 2
            dis = dis + dis1.sum()
    #print("dis {:.4f}".format(dis.sqrt().cpu().numpy()), end='')
    return dis.sqrt().cpu().numpy()



#cos
def vector_cos(w1, w2):
    numerator = torch.tensor(0.0)
    denominator1 = torch.tensor(0.0)
    denominator2 = torch.tensor(0.0)
    for n, p in w1.items():
        numerator = numerator + (p*w2[n]).sum()
        denominator1 = denominator1 + (p*p).sum()
        denominator2 = denominator2 + (w2[n]*w2[n]).sum()
    # return print(" cos {:.4f}".format(numerator/(denominator1.sqrt()*denominator2.sqrt())))
    return (numerator/(denominator1.sqrt()*denominator2.sqrt())).cpu().numpy()