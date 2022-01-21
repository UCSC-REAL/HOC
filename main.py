import torch
import torchvision
import torch.nn.functional as F

import argparse
from data.cifar import CIFAR10, CIFAR100
# from data.mnist import MNIST
from data.datasets import input_dataset
from hoc import *
import time
import random
import argparse
import numpy as np
import pickle

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))



# Options ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--pre_type", type=str, default='image')  # image, cifar
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--noise_type', type = str, default='manual')#manual
parser.add_argument('--dataset', type = str, help = 'cifar10, cifar100', default = 'cifar100')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument("--local", default=False, action='store_true')
parser.add_argument('--loss', type = str, help = 'ce, fw', default = 'fw')
parser.add_argument('--label_file_path', type = str, help = './data/IDN_0.6_C10.pt', default = 'NA') 

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



if __name__ == "__main__":

    # Setup ------------------------------------------------------------------------

    config = parser.parse_args()
    config.device = set_device()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    config.numLocal = 250
    config.numNoisyGroup = 150
    config.local = False
    
    # load dataset
    train_dataset_EF,test_dataset,num_classes,num_training_samples, num_testing_samples = input_dataset(config.dataset,'clean', config.noise_rate, transform=False) # will load noise file later (in get_T_HOC)
    config.num_classes = num_classes
    config.num_training_samples = num_training_samples
    config.num_testing_samples = num_testing_samples

    config.P = [1.0/num_classes] * num_classes  # Distribution of 10 clusters
    config.T = build_T(config.num_classes)  # [i][j]: The probability changing from i to j
    model_pre = set_model_pre(config)


    train_dataloader_EF = torch.utils.data.DataLoader(train_dataset_EF,
                                                    batch_size=128,
                                                    shuffle=True,
                                                    num_workers=2,
                                                    drop_last=False)
    # save the 512-dim feature as a dataset
    T_est, T_init, _ = get_T_HOC(config, model_pre, train_dataloader_EF, -1, max_step=301 if num_classes==100 else 1501, lr = 0.1)


    # the above procedures are initializing T using pretrained models

    train_dataset,test_dataset,num_classes,num_training_samples, num_testing_samples = input_dataset(config.dataset,config.noise_type,config.noise_rate, transform=True, noise_file = config.label_file_path)
    train_dataloader, test_loader = [], []
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=128,
                                                    shuffle=True,
                                                    num_workers=2,
                                                    drop_last=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size = 64,
                                    num_workers=2,
                                    shuffle=False)

    model, optimizer = set_model_train(config)

    # model training
    print(f'Begin training', flush= True)
    step_sz_EM = 20
    if config.loss == 'fw':
        alpha_plan = [0.1]*step_sz_EM + [0.01]*step_sz_EM + [0.1]*step_sz_EM + [0.01]*step_sz_EM + [0.1]*60 + [0.01]*60 + [0.001]*60
    else:
        alpha_plan = [0.1]*50 + [0.01]*50 + [0.01]*20
    if config.dataset == "cifar100":
        step_sz_EM = 30
        alpha_plan = [0.1]*step_sz_EM + [0.01]*step_sz_EM + [0.1]*step_sz_EM + [0.01]*step_sz_EM + [0.1]*step_sz_EM + [0.01]*step_sz_EM + [0.001]*step_sz_EM
    criterion = torch.nn.CrossEntropyLoss()
    # print(f'The true T is {np.round(config.T,3)}')
    # the following code is testing the performance of 512-dim feature

    record_t_loss = [[] for _ in range(2)]
    for epoch in range(len(alpha_plan)):
        adjust_learning_rate(optimizer, epoch, alpha_plan)

        # training
        model.train()
        acc = 0.0
        cnt = 0.0
        if config.loss == 'fw' and epoch == 0:
            print(f'Use T {np.round(np.array(T_est)*100,1)} in forward loss-correction')
        t1 = time.time()

        for i_batch, (feature, label, index) in enumerate(train_dataloader):

            cnt += index.size(0)#index.shape[0]
            optimizer.zero_grad()
            feature = feature.to(config.device)
            label = label.to(config.device)
            _, y_pred = model(feature)        
            
            if config.loss == 'fw':
                if config.local:
                    loss = forward_loss(y_pred,label,torch.tensor(T_local).float().to(config.device), map_index_T[index])
                else:
                    loss = forward_loss(y_pred,label,torch.tensor(T_est).float().to(config.device))
                    
            elif config.loss == 'fw_real':
                loss = forward_loss(y_pred,label,torch.tensor(config.T).float().to(config.device))
            else: # ce 
                loss = criterion(y_pred, label)


            outputs = F.softmax(y_pred, dim=1)
            _, pred = torch.max(outputs.data, 1)
            pred = pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            acc += (pred == label).sum()
            loss.backward()
            optimizer.step()

        t2 = time.time()
        print(f"time in an epoch: {t2-t1}(s)", flush=True)
        train_acc = float(acc)/float(cnt) if cnt != 0 else -1


        # evaluation
        model.eval()
        acc = 0.0
        cnt = 0.0
        for i_batch, (feature, label, index) in enumerate(test_loader):
            cnt += index.shape[0]
            feature = feature.to(config.device)
            label = label.to(config.device)
            _, y_pred = model(feature)
            outputs = F.softmax(y_pred, dim=1)
            _, pred = torch.max(outputs.data, 1)
            acc += (pred.cpu() == label.cpu()).sum()
        test_acc = float(acc)/float(cnt) if cnt != 0 else -1
        print(f'Epoch {epoch}: train acc = {train_acc}, test acc = {test_acc}', flush = True)

        if config.loss == 'fw':
            if epoch+1 == step_sz_EM*4:
                if num_classes == 100: 
                    config.local = False
                    T_est, T_init, T_err = get_T_HOC(config, model, train_dataloader_EF, -1, max_step=101 if num_classes==100 else 1501, T0 = torch.tensor(T_init).float(), lr = 0.1)
                    print(f'This version only reports HOC global for CIFAR-100')
                    print(f'Update global T with the extracted feature at epoch {epoch}')
                    model, optimizer = set_model_train(config)
                    record_t_loss[0].append({'epoch': epoch, 'T_err': T_err, 'local': config.local})
                else:
                    config.local = True
                    T_local, map_index_T, T_err = get_T_HOC(config, model, train_dataloader_EF, -1)
                    print(f'Update local T with the extracted feature at epoch {epoch}')
                    model, optimizer = set_model_train(config)
                    record_t_loss[1].append(T_err)
                # torch.cuda.empty_cache()
            if epoch+1 == step_sz_EM*2:
                config.local = False
                T_est, T_init, T_err = get_T_HOC(config, model, train_dataloader_EF, -1, max_step=101 if num_classes==100 else 1501, T0 = torch.tensor(T_init).float(), lr = 0.1)
                print(f'Update global T with the extracted feature at epoch {epoch}')
                model, optimizer = set_model_train(config)
                record_t_loss[0].append({'epoch': epoch, 'T_err': T_err, 'local': config.local})
                # torch.cuda.empty_cache()

    torch.save(record_t_loss, f'./out/{config.dataset}_{config.loss}_final.pt')

