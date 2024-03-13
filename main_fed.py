#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torchfusion.datasets as fdatasets
import torch

from utils.sampling import mnist_iid, mnist_noniid, mnist_noniid_multi, mnist_noniid_multi2, mnist_noniid_multi3, cifar_iid, cifar_noniid, cifar_noniid2, cifar_noniid_multi,cifar_noniid_multi2,cifar_noniid_multi3
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, MLP2, CNNMnist, CNNCifar, convert_bn_model_to_gn, fix_bn, print_bn
import models.Nets as Nets
import models.Fed as Fed
from models.test import test_img, test_ensemble
from models.vgg import VGG
from models.lenet import LeNet
import models.resnet as resnet
import models.resnet_gn as resnet_gn
import models.densenet as densenet
import utils.util as util
import random

import utils.fmnist as fmnist

from tensorboardX import SummaryWriter
import utils.summary as summary

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def change_lr(lr, current_epoch):
    if current_epoch >= 60:
        return 0.01
    elif current_epoch >= 100:
        return 0.001
    else:
        return lr

def model_minus(w1, w2):
    minus = {}
    for n, p in w1.items():
        minus[n] = (p - w2[n])
    return minus

def model_multiply(w1, a):
    mpy = {}
    for n, p in w1.items():
        mpy[n] = a*p
    return mpy

def model_distance(w1, w2):
    dis = torch.tensor(0.0)
    for n,p in w1.items():
        if ('running' not in n) and ('track' not in n):
            dis1 = (w1[n] - w2[n]) ** 2
            dis = dis + dis1.sum()
    return dis.sqrt().cpu().numpy()

def vector_cos(w1, w2):
    numerator = torch.tensor(0.0)
    denominator1 = torch.tensor(0.0)
    denominator2 = torch.tensor(0.0)
    for n, p in w1.items():
        numerator = numerator + (p*w2[n]).sum()
        denominator1 = denominator1 + (p*p).sum().sqrt()
        denominator2 = denominator2 + (w2[n]*w2[n]).sum().sqrt()
    return numerator/(denominator1*denominator2)

if __name__ == '__main__':
    setup_seed(111)
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    logger = SummaryWriter(args.summary_path)

    # 1.使用print打印
    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))  # str, arg_type

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=False, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=False, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fmnist':
        #_train = fmnist.load_mnist('../data/fmnist/', kind='train')
        #dataset_test = fmnist.load_mnist('../data/fmnist/', kind='t10k')
        trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = fdatasets.FashionMNIST('../data/fmnist/',train=True, transform=trans_fmnist, download=False)
        dataset_test = fdatasets.FashionMNIST('../data/fmnist/', train=False, transform=trans_fmnist, download=False)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:

            if args.num_users > 10:
                dict_users = mnist_noniid_multi(dataset_train, args.num_users)
                #dict_users = mnist_noniid_multi2(dataset_train, args.num_users)
                #dict_users = mnist_noniid_multi3(dataset_train, args.num_users)
            else:
                dict_users = mnist_noniid(dataset_train, args.num_users, iidpart=args.iidpart)

    elif args.dataset == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        #trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/', train=True, download=False, transform=transform_train)
        dataset_test = datasets.CIFAR10('../data/', train=False, download=False, transform=transform_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            if args.num_users > 10:
                dict_users = cifar_noniid_multi(dataset_train, args.num_users)
                #dict_users = cifar_noniid_multi2(dataset_train, args.num_users)
                #dict_users = cifar_noniid_multi3(dataset_train, args.num_users)
            else:
                # dict_users = cifar_noniid2(dataset_train, args.num_users)
                dict_users = cifar_noniid(dataset_train, args.num_users, args.iidpart)
    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])
        dataset_train = datasets.CIFAR100('../data/', train=True, download=False, transform=transform_train)
        dataset_test = datasets.CIFAR100('../data/', train=False, download=False, transform=transform_test)

        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            if args.num_users > 10:
                dict_users = cifar_noniid_multi(dataset_train, args.num_users)
            else:
                dict_users = cifar_noniid(dataset_train, args.num_users, iidpart = args.iidpart)

    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=50, dim_out=args.num_classes).to(args.device)
    elif args.model == 'mlp2':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP2(dim_in=len_in, dim_hidden=50, dim_out=args.num_classes).to(args.device)
    elif args.model == 'bmlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = Nets.MLP_BenchMark(dim_in=len_in, dim_hidden=50, dim_out=args.num_classes).to(args.device) #default为200
    elif args.model == 'wmlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = Nets.MLP_Wider(dim_in=len_in, dim_hidden=50, dim_out=args.num_classes, multi_ratio=args.multi_ratio).to(args.device)
    elif args.model == 'dmlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = Nets.MLP_Deeper(dim_in=len_in, dim_hidden=50, dim_out=args.num_classes, multi_ratio=args.multi_ratio).to(args.device)
    else:
        if args.dataset == 'cifar':
            if args.model == 'cnn':
                net_glob = CNNCifar(args=args).to(args.device)
            elif args.model == 'vgg16':
                net_glob = VGG('VGG16', momentum=0.1, m=args.num_users).to(args.device)
            elif args.model == 'resnet18':
                #net_glob = resnet.ResNet18().to(args.device)
                net_glob = resnet_gn.resnet18(group_norm=16).to(args.device)
            elif args.model == 'densenet':
                net_glob = densenet.densenet_cifar().to(args.device)
            elif args.model == 'lenet':
                net_glob = LeNet(momentum=0.1, m=args.num_users, selfbn=args.selfbn, device=args.device).to(args.device)
        elif args.dataset == 'mnist' or args.dataset == 'fmnist':
            if args.model == 'cnn':
                net_glob = CNNMnist(args=args).to(args.device)
        elif args.dataset == 'cifar100':
            if args.model == 'vgg16':
                from models.cifar100.vgg import vgg16
                net_glob = vgg16().to(args.device)
            elif args.model == 'resnet18':
                #net_glob = resnet.ResNet18().to(args.device)
                net_glob = resnet_gn.resnet18(num_classes=100, group_norm=16).to(args.device)

    #net_glob = convert_bn_model_to_gn(net_glob)
    #batchlayer_klist = util.batchnorm_layer_k(net_glob)
    #print(net_glob)
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        summary.summary(net_glob, input_size=(1, 28, 28), device=args.device)
    if args.dataset == 'cifar' or args.dataset == 'cifar100':
        summary.summary(net_glob, input_size=(3, 32, 32), device=args.device)

    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    w_1order_glob = {}
    w_1order_glob_old = {}
    w_locals_traj = {}
    w_locals = []
    w_locals_old = []
    fisher_locals = []
    result_lists = [[] for i in range(args.num_users)]
    idxs_users_old = []
    # if args.all_clients:
    if args.frac == 1:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
        fisher_locals = [w_glob for i in range(args.num_users)]

    clients = []
    for idx in range(args.num_users):
        clients.append(LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], worker_id=idx, logger=logger))

    change_mask = {}
    server_momentum = {}
    #if args.method == 5 or args.method == 6:
    #    for k, v in w_glob.items():
    #        change_mask[k] = torch.zeros_like(v, dtype=bool)

    current_epoch = 0
    global_round = 0
    step_in_round = args.step_in_round
    lr = args.lr
    # while current_epoch < args.epochs:
    while global_round < args.global_round:
        #测试前面用strategy 3 后面用strategy 2是否可行
        # if global_round > 100 and args.method == 4:
        #    args.method = 5
    #for iter in range(args.epochs):
        #net_glob.apply(print_bn)

        #lr = change_lr(lr, current_epoch)

        net_glob.train()
        loss_locals = []
        acc_locals = []
        dis_locals = []
        offset_locals = []
        w_locals = []
        idxs_users = []
        #run_flag 标记最终client是否运行本地迭代
        run_flag = [False] * args.num_users
        # if not args.all_clients:
        if args.frac < 1:
            fisher_locals = []
        # if not args.all_clients:
        if args.frac < 1:
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            for i in idxs_users:
                run_flag[i] = True
            # 造一个有偏选择
            # m = max(int(args.frac * args.num_users - 10), 1)
            # idxs_users = np.random.choice(range(10, args.num_users), m, replace=False)
            # idxs_users = np.concatenate((idxs_users, np.array(range(10))), axis=0)
            if args.method == 6 or args.method == 8:
                #idxs_users_old和剩余的里面分别random取，保证有client两轮都参与
                if len(idxs_users_old) > 0:
                    num1 = int(np.ceil(args.frac *m))
                    idxs_users1 = np.random.choice(idxs_users_old, num1, replace=False)
                    idxs_users2 = np.random.choice(np.array(list((set(range(args.num_users)) - set(idxs_users_old)))), m-num1, replace=False)
                    idxs_users = np.append(idxs_users1, idxs_users2)
        else:
            idxs_users = np.array(range(args.num_users))
            run_flag = [True] * args.num_users


        #1/4 device: 1 , 1/2, 1/4 ,  1/8   随机
        if global_round > 0:
            if args.beta == 4:
                train_flag_list = (np.random.rand(args.num_users // 4) < 1 / 8).tolist()
                train_flag_list.extend((np.random.rand(args.num_users // 4) < 1 / 4).tolist())
                train_flag_list.extend((np.random.rand(args.num_users // 4) < 1 / 2).tolist())
                train_flag_list.extend([True] * (args.num_users - 3 * (args.num_users // 4)))
            elif args.beta == 3:
                train_flag_list = (np.random.rand(args.num_users // 3) < 1 / 4).tolist()
                train_flag_list.extend((np.random.rand(args.num_users // 3) < 1 / 2).tolist())
                train_flag_list.extend([True] * (args.num_users - 2 * (args.num_users // 3)))
            elif args.beta == 2:
                train_flag_list = (np.random.rand(args.num_users // 2) < 1 / 2).tolist()
                train_flag_list.extend([True] * (args.num_users - args.num_users // 2))
            elif args.beta == 1:
                train_flag_list = [True] * args.num_users
        else:
            train_flag_list = [True] * args.num_users
        print('train_flag_list (round ' + str(global_round) + '): ', train_flag_list)


        #1/4 devices: 1/8, 1/4, 1/2, 1    轮询
        # train_flag_list = [False] * args.num_users
        # if global_round % 8 == 0:
        #     for i in range(0,args.num_users//4):
        #         train_flag_list[i] = True
        # if global_round % 4 == 0:
        #     for i in range(args.num_users//4, 2*args.num_users // 4):
        #         train_flag_list[i] = True
        # if global_round % 2 == 0:
        #     for i in range(2*args.num_users // 4, 3*args.num_users//4):
        #         train_flag_list[i] = True
        # for i in range(3*args.num_users//4, args.num_users):
        #     train_flag_list[i] = True

        capability_list = [1.0] * args.num_users
        # 下面配置为fednova设置，使得每轮迭代数量不同
        #fednova 在各client数据量相同的情况下，最终即每个client模型平均。只是每个client迭代次数可以不同
        for i in range(0,args.num_users//4):
            capability_list[i] = 0.125
        for i in range(args.num_users//4, 2*args.num_users // 4):
            capability_list[i] = 0.25
        for i in range(2*args.num_users // 4, 3*args.num_users//4):
            capability_list[i] = 0.5
        for i in range(3*args.num_users//4, args.num_users):
            capability_list[i] = 1.0


    #根据W和M确定
        #if global_round > 0:
        #    train_flag_list = (np.random.rand(int(args.num_users*args.M)) < 1.0 / args.W).tolist()
        #    train_flag_list.extend([True] * (args.num_users - int(args.num_users*args.M)))
        #else:
        #    train_flag_list = [True] * args.num_users
        #print('train_flag_list (round ' + str(global_round) + '): ', train_flag_list)

        # #轮询
        # train_flag_list = [False] * args.num_users
        # if global_round % args.W == 0:
        #     train_flag_list = [True] * args.num_users
        # print('train_flag_list (round ' + str(global_round) + '): ', train_flag_list)

    #for idx in range(args.num_users):
        #    train_flag = train_flag_list[idx] & run_flag[idx]
        for idx in idxs_users:
            local = clients[idx]

            if args.method == 7 and args.alg == 4:
                #当超过可执行的轮数后（认为dropout），则将train_flag置为False
                #该方法下train_flag只受是否dropput影响
                if capability_list[idx]* args.global_round < global_round:
                    train_flag = False
                else:
                    train_flag = True
            else:
                train_flag = train_flag_list[idx]

            if args.alg == 2:
                step_in_round1 = int(step_in_round * capability_list[idx])
            else:
                step_in_round1 = step_in_round
            w, loss, acc, offset = local.train(net=copy.deepcopy(net_glob).to(args.device), step_in_round=step_in_round1,
                                       global_round=global_round, lr=lr, args=args, change_mask=change_mask, w_1order_glob=w_1order_glob, train_flag=train_flag)
            if w:
                #if args.alg == 2: #fednova
                    #model_multiply(copy.deepcopy(w)-copy.deepcopy(net_glob), 1.0/int(step_in_round*capability_list[idx]))
                #else:
                w_locals.append(copy.deepcopy(w))
            if loss != 0:
                loss_locals.append(copy.deepcopy(loss))
            if acc != 0:
                acc_locals.append(acc)
            if offset:
                offset_locals.append(offset)
            # suppose all the workers have the same num of batches
            current_epoch = local.get_epoch()
            past_batch_in_epoch = local.get_past_batch()
            step_in_epoch = local.get_step_per_epoch()
            global_step = local.get_step()

        if (args.method == 6 or args.method == 8) and global_round > 0:
            # 利用其他模型的轨迹，拟合指定模型轨迹
            m_dic = []
            # 该段代码只能针对全参与
            # for i in range(len(train_flag_list)):
            #     if train_flag_list[i] == True:
            #         m_dic.append(util.model_minus(w_locals[i], w_locals_old[i]))
            # avg_m = util.model_avg(m_dic)
            # for i in range(len(train_flag_list)):
            #     if train_flag_list[i] == False:
            #         w_locals[i] = util.model_add(w_locals_old[i], avg_m)

            # 对部分参与也适用
            for i in idxs_users:
                #本轮和上轮都被选中，且本轮flag为True
                if train_flag_list[i] == True:
                    #查找上次是否也存在
                    if i in idxs_users_old:
                        m_dic.append(util.model_minus(w_locals[np.argwhere(idxs_users==i)[0][0]], w_locals_old[np.argwhere(idxs_users_old==i)[0][0]]))
            if len(m_dic) > 0:
                #如果能够得到平均move，则更新false的位置。否则保留原始结果（对method 6，保留之前的结果，对method 8，采用method 4的结果）
                avg_m = util.model_avg(m_dic)
                for i in idxs_users:
                    if train_flag_list[i] == False:
                        w_locals[np.argwhere(idxs_users==i)[0][0]] = util.model_add(w_locals_old[np.argwhere(idxs_users_old==i)[0][0]], avg_m)
        # update global weights
        #if args.selfbn == 1:
            # 对normbatch进行特殊处理
        #    w_glob = FedAvg_special_for_normbatch(w_locals, batchlayer_klist)
        #else:


        #records
        #util.log_batchnorm_record(net_glob, 'global', logger, global_step)
        #for i, w in enumerate(w_locals):
        #    util.log_batchnorm_record(w, i, logger, global_step)
        #util.log_batchnorm_record2(net_glob, w_locals, logger, global_step)

        global_round += 1
        step_in_round = args.step_in_round

        # print loss
        if len(loss_locals) > 0:
            loss_avg = sum(loss_locals) / len(loss_locals)
        else:
            loss_avg = 0
        if len(acc_locals) > 0:
            acc_avg = sum(acc_locals) / len(acc_locals)
        else:
            acc_avg = 0
        print('Round {:3d}, Average loss {:.3f}'.format(global_round, loss_avg))
        #loss_train.append(loss_avg)
        record_step = global_round * step_in_round
        # if args.all_clients:
        #该记录在method4，5，6，7都不对，先注释掉
        #if args.frac == 1:
        #    record_step = global_step


        # 计算模型与模型0的距离
        # for j in range(1, len(w_locals)):
        #     dis = model_distance(w_locals[0], w_locals[j])
        #     cos = (dis_locals[0] ** 2 + dis_locals[j] ** 2 - dis ** 2) / (2 * dis_locals[0] * dis_locals[j])
        #     print('model 0 <--dis--> model {}: {:.6f}, cos={:.6f}'.format(j, dis, cos))
        #     if j == 1:
        #         logger.add_scalar('model0<cos>model1', cos, record_step)
        #         logger.add_scalar('model0<dis>model1', dis, record_step)


        #logger.add_scalar('loss/train', loss_locals[-1], record_step)
        #logger.add_scalar('acc/train', 100 * acc_locals[-1], record_step)

        # copy weight to net_glob
        w_glob_old = copy.deepcopy(w_glob)
        if len(w_locals)>0:
            #如果本轮所有client都不训练，则全局模型不变
            w_glob = Fed.FedAvg(w_locals)
        w_1order_glob = model_minus(w_glob, w_glob_old)

        if len(w_1order_glob_old) > 0:
            print('last_step<----cos--->this step: {:.6f}'.format(vector_cos(w_1order_glob_old, w_1order_glob)))

        #_traj[global_round] = [copy.deepcopy(w) for w in w_locals]
        w_locals_old = w_locals
        idxs_users_old = idxs_users
        #w_glob = Fed.server_opt(w_glob_old, w_1order_glob, server_momentum, momentum=args.server_mom, optrate=args.optrate)

        global_move = model_distance(w_glob_old, w_glob)
        print('last_model <--dis--> model: {:.6f}'.format(global_move))
        logger.add_scalar('global_move', global_move, record_step)
        net_glob.load_state_dict(w_glob)

        if True:#(args.step_in_round * global_round) % 100 == 0:
            # testing
            net_glob.eval()
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print("epoch {} [{:.0f}] Testing accuracy: {:.2f}".format(current_epoch, past_batch_in_epoch*1.0 / step_in_epoch, acc_test))

            logger.add_scalar('loss/test', loss_test, record_step)
            logger.add_scalar('acc/test', acc_test, record_step)

        #记录每轮每个client的模型
        w_locals_traj[global_round] = w_locals

    #import pickle
    #with open('w_locals_traj2_r100', 'wb') as fw:
    #    pickle.dump(w_locals_traj, fw)
