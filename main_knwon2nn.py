import torch
import torch.nn.functional as F

import argparse
from data.datasets import input_dataset
from hoc import *
import time
import random
import argparse
import numpy as np
import pickle



# Options ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--G', type=int, default=50, help='num of rounds (parameter G in Algorithm 1)')
parser.add_argument('--max_iter', type=int, default=1500, help='num of iterations to get a T')
parser.add_argument("--local", default=False, action='store_true')
parser.add_argument("--data_path", default='./data/test.csv')




def get_T_global_min(args, record, max_step = 501, T0 = None, p0 = None, lr = 0.1, NumTest = 50, all_point_cnt = 15000):

    KINDS = args.num_classes
    all_point_cnt = np.min((all_point_cnt,int(len(record)*0.9)))
    p_estimate = [[] for _ in range(3)]
    p_estimate[0] = torch.zeros(KINDS)
    p_estimate[1] = torch.zeros(KINDS, KINDS)
    p_estimate[2] = torch.zeros(KINDS, KINDS, KINDS)
    for idx in range(NumTest):
        print(idx, flush=True)
        cnt_y_3 = count_y_known2nn(KINDS, record, all_point_cnt)
        for i in range(3):
            cnt_y_3[i] /= all_point_cnt
            p_estimate[i] = p_estimate[i] + cnt_y_3[i] if idx != 0 else cnt_y_3[i]

    for j in range(3):
        p_estimate[j] = p_estimate[j] / NumTest

    args.device = set_device()
    loss_min, E_calc, P_calc, T_init = calc_func(KINDS, p_estimate, False, args.device, max_step, T0, p0, lr = lr)

    E_calc = E_calc.cpu().numpy()
    T_init = T_init.cpu().numpy()
    return E_calc, T_init

def error(T, T_true):
    error = np.sum(np.abs(T-T_true)) / np.sum(np.abs(T_true))
    return error

if __name__ == "__main__":

    # Setup ------------------------------------------------------------------------

    config = parser.parse_args()
    config.device = set_device()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    raw_data = np.genfromtxt(config.data_path, delimiter=',')[1:]
    remove_nan = np.isnan(raw_data).sum(1).astype(bool)
    my_data = raw_data[~remove_nan].astype(int)

    label_classes = np.unique(my_data.reshape(-1))
    print(f'Current label classes: {label_classes}')
    if np.min(label_classes) > 0:
        label_classes = np.min(label_classes)
        label_classes = label_classes.astype(int)
        print(f'Reset counting from 0')
        print(f'Current label classes: {label_classes}')
        
    config.num_classes = len(label_classes)
    
    # minimal implementation of HOC (an example)
    new_estimate_T, _ = get_T_global_min(config, my_data, max_step=config.max_iter, lr = 0.1, NumTest = config.G)
    print(f'\n\n-----------------------------------------')
    print(f'Estimation finished!')
    
    np.set_printoptions(precision=1)
    print(f'The estimated T (*100) is \n{new_estimate_T*100}')
    # The following code can print the error (matrix L11 norm) when the true T is given
    # estimate_error_2 = error(True_T, new_estimate_T)
    # print('---------New Estimate error: {:.6f}'.format(estimate_error_2))


