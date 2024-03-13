#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from models.Nets import fix_bn
import copy
import utils.util as util

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label




class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, worker_id=0, logger=None):
        self.args = args
        if args.uncertainty == 1:
            self.loss_func = util.edl_digamma_loss
        elif args.uncertainty == 2:
            self.loss_func = util.edl_log_loss
        elif args.uncertainty == 3:
            self.loss_func = util.edl_mse_loss
        else:
            self.loss_func = torch.nn.CrossEntropyLoss()  # nn.BCEWithLogitsLoss()  torch.nn.MSELoss() #

        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.dataset = dataset
        self.idxs = idxs
        self.worker_id = worker_id
        self.train_iter = self.ldr_train.__iter__()
        #记录当前状态
        self.epoch = 0
        self.step = 0
        self.local_state_dict = {}
        self.logger = logger
        self.offset = {}
        self.local_list = [] #['layer_input'] #['classifier'] #['fc3'] #['conv1', 'b1', 're1', 'fc3']   #['conv1', 'b1', 're1']
        self.move_vector = {}
        self._means= {} #init_model
        self.last_model = {}
        self.gamma = 1.0

    def get_next_batch(self):
        try:
            batch_data = self.train_iter.next()
        except:
            #每个epoch重新划分训练集和测试集（这里测试集实际应该是验证集）
            self.ldr_train = DataLoader(DatasetSplit(self.dataset, self.idxs), batch_size=self.args.local_bs, shuffle=True)
            self.train_iter = self.ldr_train.__iter__()
            batch_data = self.train_iter.next()
            self.epoch += 1
        self.step += 1

        return batch_data

    def update_local_model(self, global_model):
        global_model.update(self.local_state_dict)
        return global_model

    def penalty0(self, model: nn.Module, count: float):
        loss = torch.tensor(0.0)
        for n, p in model.named_parameters():
            _loss = count * (p - self._means[n]) ** 2
            loss = loss + _loss.sum()
        return loss

    def l1reg(self, model: nn.Module, count: float):
        '''
        L1 loss
        :param model:
        :param count:
        :return:
        '''
        regularization_loss = torch.tensor(0.0, device=util.cal_gpu(model))
        for n, p in model.named_parameters():
            regularization_loss += torch.sum(abs(p))
        return regularization_loss*count


    def penalty1(self, net, global_batchnorm_parameters, count):
        loss = torch.tensor(0.0)
        for module_prefix, module in net.named_modules():
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for n, p in module.named_parameters():
                    _loss = count * torch.abs(p - global_batchnorm_parameters[module_prefix][n]) ** 2
                    loss = loss + _loss.sum()
        return loss

    def penalty2(self, model: nn.Module, count: float):
        loss = torch.tensor(0.0)
        for n, p in model.named_parameters():
            p1 = p.reshape(p.size()[0], -1)
            p2 = torch.mm(p1.T, p1)
            p3 = torch.norm(p2 - torch.eye(p2.size()[0]).to(p2.device))**2
            loss = loss + count * p3
        return loss

    def penalty3(self, model: nn.Module, count: float):
        '''
        对指定层加约束项
        :param model:
        :param count:
        :return:
        '''
        loss = torch.tensor(0.0)
        for n, p in model.named_parameters():
            if util.is_in(self.local_list, n):
                _loss = count * (p - self._means[n]) ** 2
                loss = loss + _loss.sum()
        return loss


    def penalty4(self, model: nn.Module, count: float):
        loss = torch.tensor(0.0)
        for n, p in model.named_parameters():
            zeros = torch.zeros_like(p)
            dis = p - self._means[n]
            # w0所有正值
            w0 = dis>0
            #同为True或False，则为False
            w1 = torch.bitwise_xor(w0, self.mask[n])
            #dis1提取符号与mask指定相反的
            dis1 = dis.masked_fill(w1, torch.tensor(0,device=w1.device))
            dis2 = dis.masked_fill(~w1, torch.tensor(0,device=w1.device))

            #_loss = count*2 * (dis1) ** 2 + count/2 * (dis2) ** 2
            _loss = count * (dis1) ** 2 #+ count / 2 * (dis2) ** 2
            loss = loss + _loss.sum()
        return loss


    def penalty4_1(self, model: nn.Module, count: float):
        '''
        mask转换为方向向量，惩罚为与方向向量的夹角
        :param model:
        :param count:
        :return:
        '''
        c = torch.tensor(0.0)
        a = torch.tensor(0.0)
        for n, p in model.named_parameters():
            if n in self.direction.keys():
                a_v = p - self._means[n]
                _a = a_v ** 2
                _c = (a_v - self.direction[n]) ** 2
                a = a + _a.sum()
                c = c + _c.sum()

        b = self.direction_len
        if b > 0:
            trilist = [torch.sqrt(a).detach().cpu().numpy(), torch.sqrt(b).detach().cpu().numpy(),
                       torch.sqrt(c).detach().cpu().numpy()]
            if trilist[0] + trilist[1] < trilist[2]:
                print('a={:.6f}, b={:.6f}, c={:.6f}.'.format(trilist[0], trilist[1], trilist[2]))
            e = torch.tensor(1e-4)
            loss = count * (1 - (a + b - c) / (2 * torch.sqrt((a * b + e))))
        else:
            loss = torch.tensor(0.0)
        return loss


    def penalty5_1(self, model: nn.Module, w_1order_glob:dict, b: torch.tensor, count: float):
        #loss = torch.tensor(0.0)
        c = torch.tensor(0.0)
        a = torch.tensor(0.0)
        for n, p in model.named_parameters():
            if n in w_1order_glob.keys():
                _c = (p - w_1order_glob[n]) ** 2
                _a = p ** 2
                a = a + _a.sum()
                c = c + _c.sum()

        if b>0:
            e = torch.tensor(1e-4)
            loss = count * (1-(a+b-c)/(2*torch.sqrt((a*b+e))))
        else:
            loss = torch.tensor(0.0)
        return loss


    def penalty5(self, model: nn.Module, w_1order_glob:dict, b: torch.tensor, count: float):
        #loss = torch.tensor(0.0)
        c = torch.tensor(0.0)
        a = torch.tensor(0.0)
        for n, p in model.named_parameters():
            if n in w_1order_glob.keys():
                a_v = p - self._means[n]
                _a =  a_v ** 2
                _c = (a_v - w_1order_glob[n]) ** 2
                a = a + _a.sum()
                c = c + _c.sum()

        if b>0:
            trilist = [torch.sqrt(a).detach().cpu().numpy(), torch.sqrt(b).detach().cpu().numpy(), torch.sqrt(c).detach().cpu().numpy()]
            if trilist[0]  + trilist[1] < trilist[2]:
                print('a={:.6f}, b={:.6f}, c={:.6f}.'.format(trilist[0], trilist[1], trilist[2]))
            e = torch.tensor(1e-4)
            loss = count * (1-(a+b-c)/(2*torch.sqrt((a*b+e))))
        else:
            loss = torch.tensor(0.0)
        return loss

    def penalty6(self, model: nn.Module, w_1order_glob_norm:dict, count: float):
        loss = torch.tensor(0.0)

        for n, p in model.named_parameters():
            if n in w_1order_glob_norm.keys():
                _loss = w_1order_glob_norm[n] * p
                loss = loss + count * _loss.sum()

        return loss

    def get_batchnorm_parameters(self, net):
        dic = {}
        for module_prefix, module in net.named_modules():
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                dic[module_prefix] = copy.deepcopy(module.state_dict())
        return dic


    def set_para(self, weights):
        self._means = {}
        for n,p in weights.items():
            self._means[n] = p.clone().detach()

    def set_move_vector(self, weights):
        self.move_vector = {}
        for n, p in weights.items():
            self.move_vector[n] = p.clone().detach()

    def set_rangen(self, seed):
        '''
        初始化随机数生成器
        :return:
        '''
        torch.manual_seed(seed)

    def opt_step(self, optimizer, net, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        group = optimizer.param_groups[0]
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        for name, p in net.named_parameters():
            if p.grad is None:
                continue
            d_p = p.grad.data
            if weight_decay != 0:
                d_p.add_(weight_decay, p.data)
            if momentum != 0:
                param_state = optimizer.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = d_p.clone()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf

            # 对于mask为False的tensor位置，留正值；为True的位置，留负值
            self.offset[name].add_(d_p, alpha=-group['lr'])
            zeros = torch.zeros_like(d_p)
            # w0保留应该取负值的部分
            w0 = self.offset[name].masked_fill(~self.mask[name], torch.tensor(0, device=d_p.device))
            w0 = torch.where(w0 < 0, w0, zeros)
            # w1保留应该取正值的部分
            w1 = self.offset[name].masked_fill(self.mask[name], torch.tensor(0, device=d_p.device))
            w1 = torch.where(w1 > 0, w1, zeros)

            self.offset[name] = w0+w1
            p.data = self.offset[name] + self._means[name]

        return loss

    def set_mask(self, weights):
        self.mask = {}
        for n,p in weights:
            self.mask[n] = (torch.rand(p.size()) < 0.5).to(p.device)

    def set_offset(self, weights):
        self.offset = {}
        for n,p in weights:
            self.mask[n] = (torch.rand(p.size()) < 0.5).to(p.device)


    def update_mask(self, change_mask):
        for n, p in self.mask.items():
            self.mask[n] = torch.bitwise_xor(p, change_mask[n])

    def update_direction(self):
        self.direction = {}
        b = torch.tensor(0.0)
        for n, p in self.mask.items():
            self.direction[n] = self.mask[n].float()
            _b = self.direction[n] ** 2
            b = b + _b.sum()
        self.direction_len = b

    def inverse_mask(self):
        for n, p in self.mask.items():
            self.mask[n] = ~self.mask[n]

    def train(self, net, step_in_round, global_round, lr, args, change_mask=None, w_1order_glob={}, train_flag=True):
        net.train()

        #保存初始模型
        self.set_para(net.state_dict())

        b = torch.tensor(0.0)
        for k, v in net.named_parameters():
            if k in w_1order_glob.keys():
                _b = w_1order_glob[k] ** 2
                b = b + _b.sum()

        if (args.method == 4 or args.method == 8) and train_flag == False and self.move_vector:
            #flag为True，正常训练，
            #flag为False 直接求出
            #method=8返回模型并不使用，而是占位，若无法使用方法8则退化使用方法4
            return util.model_add(net.state_dict(), self.move_vector), 0, 0, self.offset

        if (args.method == 5 or args.method == 6) and train_flag == False and self.last_model:
            #flag为True，正常训练，
            #flag为False 直接使用上次模型
            # method=6返回模型并不使用，而是占位
            return self.last_model, 0, 0, self.offset


        if args.method == 7 and train_flag == False:
            # false则不参与训练
            return {}, 0, 0, {}

        # train and update
        #冻结部分网络层
        if global_round > -1:
           for name, param in net.named_parameters():
               if util.is_in(self.local_list, name):
                   param.requires_grad = False

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr) #, momentum=self.args.momentum)

        correct = 0
        total = 0
        batch_loss = []
        loss1 = torch.tensor(0.0)


        for iter_step in range(step_in_round):
            images, targets = self.get_next_batch()
            images = images.to(self.args.device)
            targets = targets.to(self.args.device)
            labels = targets
            #改为多标签分类
            #labels = torch.zeros(len(targets), 10, device=self.args.device).scatter_(1, targets.reshape((-1,1)), 1)


            net.zero_grad()

            log_probs = net(images)
            loss = self.loss_func(log_probs, labels) #+ self.l1reg(net, 1e-3)#+ nn.CrossEntropyLoss()(log_probs, targets)
            _, predicted = log_probs.max(1)

            if args.alg == 1:
                loss1 = self.penalty0(net, args.imp0)
                loss = loss + loss1

            elif args.alg == 2:
                loss1 = self.penalty5(net, w_1order_glob, b, args.imp0)
                loss = loss + loss1


            loss.backward()

            optimizer.step()
            #if args.method != 5:
            #    optimizer.step()
            #else:
            #    self.opt_step(optimizer, net)

            correct += predicted.eq(targets).sum().item()
            total += labels.size(0)
            batch_loss.append(loss.item())


        print('Worker: {}: Update Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Loss1: {:.6f}, acc: {:.2f}%({}/{})'.format(self.worker_id,
            self.epoch, self.get_past_batch(), self.get_step_per_epoch(),
                  100. * self.get_past_batch() / self.get_step_per_epoch(), loss.item(), loss1.item(), 100. * correct / total, correct, total))


        # if args.selfbn == 1:
        #     #保留不需要同步的参数
        #     #self.local_state_dict = {k: v for k, v in net.state_dict().items() if util.is_in(self.local_list, k)}
        #     self.local_state_dict = {k: v for k, v in net.state_dict().items()}
        #     #self.local_state_dict = {k: v for k, v in net.state_dict().items() if 'fc3' in k}
        #     pass
        # if args.method == 3:
        #     self.local_state_dict = {k: v for k, v in net.state_dict().items()}
        # elif args.method == 4:
        #     #保留最后一层
        #     self.local_state_dict = {k: v for k, v in net.state_dict().items() if util.is_in(self.local_list, k)}
        # #util.log_batchnorm_record(net, self.worker_id, self.logger, self.get_step())

        if args.method == 4 or args.method == 8:
            # 记住本轮训练，最终模型与初始模型的移动向量
            self.set_move_vector(util.model_minus(net.state_dict(), self._means))
        if args.method == 5 or args.method == 6:
            # 记住本轮训练的模型
            self.last_model = net.state_dict()

        return net.state_dict(), sum(batch_loss) / len(batch_loss), correct / total, self.offset

    def get_step_per_epoch(self):
        '''
        获取当前epoch中第几个step
        :return:
        '''
        return len(self.ldr_train)

    def get_epoch(self):
        '''
        获取当前所处的epoch
        :return:
        '''
        return self.epoch

    def get_step(self):
        '''
        获取当前总的step数量
        :return:
        '''
        return self.step

    def get_past_batch(self):
        '''
        get the used batch in the current epoch
        :return:
        '''
        return  self.step - self.epoch*len(self.ldr_train)

    def get_remain_batch(self):
        '''
        get the remain batch in the current epoch
        :return:
        '''
        return len(self.ldr_train)-self.get_past_batch()


