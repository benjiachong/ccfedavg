# CC-FedAvg

compare cc-fedavg with stratgy 1, strategy 2, FedNova and FedAvg.

**settings about cc-fedavg with stratgy 1, strategy 2 and FedAvg:**

--alg:
0

--method 
4:cc-fedavg
5:(strategy2) 
7:(strategy1)
0:Fedavg

**settings about FedNova:**

--alg:
2

--method:
0

**other important hyper-parameters:**

--beta: control the number of clients with different computational resources.

For the situation with two different parts of clients (one is normal, the other is with inadequate resources),
we can use W and M to do more elaborate experiments (the code conflicts with the implementation of "beta" hyper-parameter, so the corresponding code is commented by default).

--W:skip round number between each normal round

--M:number of users with inefficient resources


## Requirements
python>=3.6  
pytorch>=0.4

## Run


For example:
> python -u  main_fed.py --model mlp --global_round 400 --dataset fmnist --iidpart 0 --step_in_round 20 --local_bs 32 --frac 0.1 --num_users 100 --num_classes 10 --lr 0.01 --gpu 1 --beta 4 --method 4

> python -u  main_fed.py --model cnn --global_round 400 --dataset cifar --iidpart 0 --step_in_round 300 --local_bs 64 --frac 1 --num_users 8 --num_classes 10 --lr 0.01 --gpu 0 --method 7
>
>python -u  main_fed.py --model cnn --global_round 400 --dataset cifar --iidpart 0 --step_in_round 100 --local_bs 64 --frac 1 --num_users 8 --num_classes 10 --lr 0.01 --gpu 7 --method 0 --alg=2
>

##Cite

@article{zhang2023cc,
  title={Cc-fedavg: Computationally customized federated averaging},
  author={Zhang, Hao and Wu, Tingting and Cheng, Siyao and Liu, Jie},
  journal={IEEE Internet of Things Journal},
  year={2024},
  volume={11},
  number={3},
  pages={4826-4841},
  publisher={IEEE}
  doi={10.1109/JIOT.2023.3300080}}
}