#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np
import itertools

def FedAvg(w):
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


def FedWeightedAvg(w, p):
    p1 = np.array(p)/p[0]
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] = w_avg[k] + w[i][k] * p1[i]
        if w_avg[k].numel() != 1:
            w_avg[k] = torch.div(w_avg[k], sum(p1))
        else:
            w_avg[k] = w_avg[k] // sum(p1)
        #w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def FedEWCAvg(w, fisher):
    fisher_sum = copy.deepcopy(fisher[0])
    fisher_rate = np.array([0.0] * len(w))
    for k in fisher_sum.keys():
        for i in range(0, len(fisher)):
            if i != 0:
                fisher_sum[k] = fisher_sum[k] + fisher[i][k]
            # 记录每个client fisher量
            fisher_rate[i] += fisher[i][k].sum().cpu().numpy()

        for i in range(0, len(fisher)):
            fisher[i][k] = fisher[i][k]/fisher_sum[k]
            #若fisher_sum为0，则会导致除数为0，结果为nan。此时退化为FedAvg结果
            fisher[i][k] = torch.where(torch.isnan(fisher[i][k]), torch.full_like(fisher[i][k], 1/len(fisher)), fisher[i][k])

    fisher_rate = fisher_rate / fisher_rate[0]

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(0, len(w)):
            if i == 0:
                if k in fisher_sum.keys():
                    w_avg[k] = w_avg[k] * fisher[i][k]
            else:
                if k in fisher_sum.keys():
                    w_avg[k] = w_avg[k] + w[i][k] * fisher[i][k]
                else:
                    w_avg[k] += w[i][k]

        if w_avg[k].numel() != 1:
            if k in fisher_sum.keys():
                pass
            else:
                w_avg[k] = torch.div(w_avg[k], len(w))
        else:
            w_avg[k] = w_avg[k] // len(w)

    return w_avg, fisher_rate
'''

def FedEWCAvg(w, fisher):
    w_avg = copy.deepcopy(w[0])
    fisher_sum = copy.deepcopy(fisher[0])
    fisher_rate = np.array([0.0] * len(w))
    for k in w_avg.keys():
        for i in range(0, len(w)):
            if i == 0:
                if k in fisher_sum.keys():
                    w_avg[k] = w_avg[k] * fisher[i][k]
            else:
                if k in fisher_sum.keys():
                    w_avg[k] = w_avg[k] + w[i][k] * fisher[i][k]
                    fisher_sum[k] = fisher_sum[k] + fisher[i][k]
                else:
                    w_avg[k] += w[i][k]

            fisher_rate[i] += fisher[i][k].sum().cpu().numpy()

        if w_avg[k].numel() != 1:
            if k in fisher_sum.keys():
                w_avg[k] = torch.div(w_avg[k], fisher_sum[k])
            else:
                w_avg[k] = torch.div(w_avg[k], len(w))
        else:
            w_avg[k] = w_avg[k] // len(w)

    fisher_rate = fisher_rate / fisher_rate[0]
    return w_avg, fisher_rate
'''



def FedAvg_special_for_normbatch(w, batchlayer_klist):
    w_avg = copy.deepcopy(w[0])
    #记录running_mean的平方和
    temp_dic = {}
    running_var = []
    for k in w_avg.keys():
        if k not in batchlayer_klist:
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            if w_avg[k].numel() != 1:
                w_avg[k] = torch.div(w_avg[k], len(w))
            else:
                w_avg[k] = w_avg[k] // len(w)
        else:
            if 'running_mean' in k:
                temp_dic[k] = w_avg[k].clone()
                temp_dic[k] = temp_dic[k]**2
                for i in range(1, len(w)):
                    w_avg[k] += w[i][k]
                    temp_dic[k] = temp_dic[k] + w[i][k]**2
                w_avg[k] = torch.div(w_avg[k], len(w))
                temp_dic[k] = torch.div(temp_dic[k], len(w)) - w_avg[k]**2
            elif 'global_' in k:
                local_k = k.replace('global_', 'local_')
                w_avg[k] = w[0][local_k]
                for i in range(1, len(w)):
                    w_avg[k] += w[i][local_k]
            elif 'local_' in k:
                pass
            else:
                for i in range(1, len(w)):
                    w_avg[k] += w[i][k]
                if w_avg[k].numel() != 1:
                    w_avg[k] = torch.div(w_avg[k], len(w))
                else:
                    w_avg[k] = w_avg[k] // len(w)

                if 'running_var' in k:
                    running_var.append(k)

    '''
    for k in running_var:
        meank = k.split('running_var')[0]+'running_mean'
        assert meank in temp_dic.keys()
        w_avg[k] = w_avg[k] + temp_dic[meank]
    '''
    return w_avg


def FedAvgMask(w, offset_locals, current_epoch):
    w_avg = copy.deepcopy(w[0])
    changemask = {}
    masksum = {}
    #rate = 0.1
    for k in w_avg.keys():
        #masksum记录每个位置对所有worker统计，等于原始值的数量
        #th = torch.quantile(torch.abs(offset_locals[0][k].reshape(-1)), rate).cpu().numpy().min()
        #masksum[k] = (torch.isclose(offset_locals[0][k], torch.zeros_like(offset_locals[0][k]), atol=th)).int()
        masksum[k] = (torch.isclose(offset_locals[0][k], torch.zeros_like(offset_locals[0][k]), atol=1e-06)).int()
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
            #th = torch.quantile(torch.abs (offset_locals[i][k].reshape(-1)), rate).cpu().numpy().min()
            #masksum[k] = (torch.isclose(offset_locals[i][k], torch.zeros_like(offset_locals[i][k]), atol=th)).int()
            masksum[k] += (torch.isclose(offset_locals[i][k], torch.zeros_like(offset_locals[i][k]), atol=1e-06)).int()
        if w_avg[k].numel() != 1:
            w_avg[k] = torch.div(w_avg[k], len(w))
        else:
            w_avg[k] = w_avg[k] // len(w)
        #若大于一半的worker某个参数没有更新,则下次允许的方向反过来
        changemask[k] = (masksum[k] > (len(w)/2.0 - 1))
        #gmask[k] = torch.bitwise_xor(gmask[k], changemask[k])
        #w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg, changemask


def server_opt(w_glob_old, w_1order_glob, server_momentum, momentum=0, optrate=1):
    if momentum > 0:
        for k, v in w_1order_glob.items():
            if k in server_momentum:
                server_momentum[k] = momentum * server_momentum[k] + (1 - momentum) * w_1order_glob[k]
            else:
                server_momentum[k] = w_1order_glob[k]
    else:
        server_momentum = w_1order_glob

    rate = optrate
    w_glob = {}
    for k, v in w_glob_old.items():
        w_glob[k] = v + rate * server_momentum[k]

    return w_glob


def model_distance(w1, w2):
    dis = torch.tensor(0.0)
    for n,p in w1.items():
        if ('running' not in n) and ('track' not in n):
            dis1 = (w1[n] - w2[n]) ** 2
            dis = dis + dis1.sum()
    return dis.sqrt().cpu().numpy()

def fedavgdis(points):
    center = FedAvg(points)
    dis = sum([model_distance(p, center) for p in points])
    print('fedavg_dis: %f' %(dis))
    return

def BestCandidate(result_lists):
    ''':arg: result_lists：[[...],[...], ...]
        每个元素仍然为列表。
        每个元素中选出一个，得到一组距离最近的点作为本轮聚合参数
    '''
    smallest_dis = 1000000.0
    best_points = []
    for points in itertools.product(*result_lists):
        center = FedAvg(points)
        dis = sum([model_distance(p, center) for p in points])
        print('dis: %f'%(dis))
        if dis < smallest_dis:
            smallest_dis = dis
            best_points = points
    print('smallest_dis: %f'%(smallest_dis))
    return best_points