import torch
import torchvision
import torch.nn.functional as F

import argparse
from data.cifar import CIFAR10
from data.datasets import input_dataset
from hoc import *
import time
import random
import argparse
import numpy as np
import pickle



# Options ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--pre_type", type=str, default='cifar')  # image, cifar
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--noise_type', type = str, default='manual')#manual
parser.add_argument('--dataset', type = str, help = 'cifar10, cifar100', default = 'cifar100')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--G', type=int, default=50, help='num of rounds (parameter G in Algorithm 1)')
parser.add_argument('--max_iter', type=int, default=1500, help='num of iterations to get a T')
parser.add_argument("--local", default=False, action='store_true')
parser.add_argument('--loss', type = str, help = 'ce, fw', default = 'fw')
parser.add_argument('--label_file_path', type = str, help = './data/IDN_0.6_C10.pt', default = './data/IDN_0.6_C10.pt') 

global GLOBAL_T_REAL
GLOBAL_T_REAL = []




def forward_loss(output, target, trans_mat, index = None):
    # l_{forward}(y, h(x)) = l_{ce}(y, h(x) @ T)
    outputs = F.softmax(output, dim=1)
    if index is None:
        outputs = outputs @ trans_mat
    else:
        outputs1 = outputs.view(outputs.shape[0] * outputs.shape[1],-1).repeat(1,outputs.shape[1])
        T1 = trans_mat[index].view(outputs.shape[0] * trans_mat.shape[1], trans_mat.shape[2])
        outputs = torch.sum((outputs1 * T1).view(outputs.shape[0],trans_mat.shape[1],trans_mat.shape[2]),1)

    outputs = torch.log(outputs)
    #loss = CE(outputs, target)
    loss = F.cross_entropy(outputs,target)

    return loss

def set_model_min(config):
    # use resnet18 (pretrained with CIFAR-10). Only for the minimum implementation of HOC
    if config.pre_type == 'cifar':
        model = res_cifar.resnet18(pretrained=True)
    else:
        RuntimeError('Undefined pretrained model.')
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config.num_classes)
    model.to(config.device)
    return model


def get_T_global_min(args, record, max_step = 501, T0 = None, p0 = None, lr = 0.1, NumTest = 50, all_point_cnt = 15000):
    total_len = sum([len(a) for a in record])
    origin_trans = torch.zeros(total_len, record[0][0]['feature'].shape[0])
    origin_label = torch.zeros(total_len).long()
    cnt, lb = 0, 0
    for item in record:
        for i in item:
            origin_trans[cnt] = i['feature']
            origin_label[cnt] = lb
            cnt += 1
        lb += 1
    data_set = {'feature': origin_trans, 'noisy_label': origin_label}

    # Build Feature Clusters --------------------------------------
    KINDS = args.num_classes
    # NumTest = 50
    # all_point_cnt = 15000


    p_estimate = [[] for _ in range(3)]
    p_estimate[0] = torch.zeros(KINDS)
    p_estimate[1] = torch.zeros(KINDS, KINDS)
    p_estimate[2] = torch.zeros(KINDS, KINDS, KINDS)
    p_estimate_rec = torch.zeros(NumTest, 3)
    for idx in range(NumTest):
        print(idx, flush=True)
        # global
        sample = np.random.choice(range(data_set['feature'].shape[0]), all_point_cnt, replace=False)
        # final_feat, noisy_label = get_feat_clusters(data_set, sample)
        final_feat = data_set['feature'][sample]
        noisy_label = data_set['noisy_label'][sample]
        cnt_y_3 = count_y(KINDS, final_feat, noisy_label, all_point_cnt)
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

    
    # load dataset
    train_dataset,test_dataset,num_classes,num_training_samples, num_testing_samples = input_dataset(config.dataset,config.noise_type,config.noise_rate, transform=False, noise_file = config.label_file_path)
    config.num_classes = num_classes
    config.num_training_samples = num_training_samples
    config.num_testing_samples = num_testing_samples

    model_pre = set_model_min(config)


    train_dataloader_EF = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=128,
                                                    shuffle=True,
                                                    num_workers=2,
                                                    drop_last=False)
    model_pre.eval()
    record = [[] for _ in range(config.num_classes)]

    for i_batch, (feature, label, index) in enumerate(train_dataloader_EF):
        feature = feature.to(config.device)
        label = label.to(config.device)
        extracted_feature, _ = model_pre(feature)
        for i in range(extracted_feature.shape[0]):
            record[label[i]].append({'feature': extracted_feature[i].detach().cpu(), 'index': index[i]})

    # minimal implementation of HOC (an example)
    new_estimate_T, _ = get_T_global_min(config, record, max_step=config.num_iter, lr = 0.1, NumTest = config.G)
    print(f'\n\n-----------------------------------------')
    print(f'Estimation finished!')
    print(f'The estimated T is \n{np.round(np.array(new_estimate_T),3)}')
    # The following code can print the error (matrix L11 norm) when the true T is given
    # estimate_error_2 = error(True_T, new_estimate_T)
    # print('---------New Estimate error: {:.6f}'.format(estimate_error_2))


