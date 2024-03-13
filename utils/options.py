#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=1000, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--multi_ratio', type=int, default=2, help="extend mlp")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--iid', type=bool, default=False, help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    parser.add_argument('--iidpart', type=float, default=0.1, help="the fraction of clients: C")

    parser.add_argument('--gpu', type=int, default=3, help="GPU ID, -1 for CPU")
    # parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--global_round', type=int, default=100, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    # parser.add_argument('--all_clients', type=bool, default=True, help='aggregation over all clients')
    parser.add_argument('--beta', type=float, default=4, help="the heterogeneity of computational resources")
    parser.add_argument('--selfbn', type=int, default=0, help='0: batchnorm 1:modified batchnorm 2:frn')
    parser.add_argument('--alg', type=int, default=0, help='0: fedavg 1:fedprox 2:fedcos 3:fednova 4：dropout')
    parser.add_argument('--method', type=int, default=0,
                        help='''0: fedavg, 
                             4:本地模型使用上轮位移推出本轮位移 5:部分模型本地不更新直接用上轮结果(strategy2) 
                             6:本地模型利用其他模型平均轨迹推出本模型轨迹位置
                             7:部分模型不参与(strategy1), 8:方法4与方法6结合''')
    parser.add_argument('--W', type=int, default=1, help="skip round number: W")
    parser.add_argument('--M', type=float, default=0.0, help="number of users with inefficient resources: M")
    parser.add_argument('--imp0', type=float, default=0.02, help='fedprox importance')
    parser.add_argument('--imp1', type=float, default=1, help='fedbatch importance')
    parser.add_argument('--multiple_agg', type=int, default=3, help='how many points each client returns')
    parser.add_argument('--step_in_round_init', type=int, default=200, help='step num in first round')
    parser.add_argument('--step_in_round', type=int, default=200, help='step num in each round')
    parser.add_argument('--server_mom', type=float, default=0, help='server momentum')
    parser.add_argument('--optrate', type=float, default=1, help='FedOpt rate')
    #parser.add_argument('--summary_path', default='logs/mnist/cnn/noniid(4worker-60e-128bs-0.1lr-FedEWCAvg)',
    parser.add_argument('--summary_path', default='logs/cifar10/cnn/noniid(5worker-1frac-100e-128bs-200r-0.01lr',
                        type=str,
                        help='model saved path')
    parser.add_argument('--uncertainty', type=int, default=0, help='0: none, 1:digamma 2:log  3:mse')
    args = parser.parse_args()
    return args
