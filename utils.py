from scipy.optimize import leastsq
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from demo1_real import *
import numpy as np
import torch.nn.functional as F
from multiprocessing import Pool
import time

import torch
import random
import math



# numLocal = config.numLocal 
# numNoisyGroup = config.numNoisyGroup

smp = torch.nn.Softmax(dim=0)
smt = torch.nn.Softmax(dim=1)


def distCosine(x, y):
    """
    :param x: m x k array
    :param y: n x k array
    :return: m x n array
    """
    xx = np.sum(x ** 2, axis=1) ** 0.5
    x = x / xx[:, np.newaxis]
    yy = np.sum(y ** 2, axis=1) ** 0.5
    y = y / yy[:, np.newaxis]
    dist = 1 - np.dot(x, y.transpose())  # 1 - cosine distance
    return dist

def check_T_torch(KINDS, clean_label, noisy_label):
    T_real = np.zeros((KINDS,KINDS))
    for i in range(clean_label.shape[0]):
        T_real[clean_label[i]][noisy_label[i]] += 1
    P_real = [sum(T_real[i])*1.0 for i in range(KINDS)] # random selection
    for i in range(KINDS):
        if P_real[i]>0:
            T_real[i] /= P_real[i]
    P_real = np.array(P_real)/sum(P_real)
    print(f'Check: P = {P_real},\n T = \n{np.round(T_real,3)}')
    return T_real, P_real






def extract_sub_dataset_local_c100(origin_trans, center_idx = None, numLocal = 10000):
    feat_cord0 = origin_trans[center_idx]
    dist_all = torch.norm(feat_cord0.view(1,-1) - origin_trans, dim=1)
    dist_s, idx = torch.sort(dist_all)
    idx_sel = idx[:numLocal].detach().cpu().tolist()

    return idx_sel

def extract_sub_dataset_local(origin_trans, center_idx = None, numLocal = 250):
    feat_cord0 = origin_trans[center_idx]
    dist_all = torch.norm(feat_cord0.view(1,-1) - origin_trans, dim=1)
    dist_s, idx = torch.sort(dist_all)
    idx_sel = idx[:numLocal].detach().cpu().tolist()

    return idx_sel


def extract_sub_dataset(sub_cluster_each, origin, sub_clean_dataset_name, sub_noisy_dataset_name = None):
    for i in range(len(sub_cluster_each)):   #KINDS
        random.shuffle(origin[i])
        origin[i] = origin[i][:sub_cluster_each[i]]
        
        for ori in origin[i]:
            ori['label'] = i

    total_len = sum([len(a) for a in origin])

    origin_trans = torch.zeros(total_len,origin[0][0]['feature'].shape[0])
    origin_label = torch.zeros(total_len).long()
    origin_index = torch.zeros(total_len).long()
    cnt = 0
    for item in origin:
        for i in item:
            origin_trans[cnt] = i['feature']
            origin_label[cnt] = i['label']
            origin_index[cnt] = i['index']
            cnt += 1
    torch.save({'feature': origin_trans, 'clean_label': origin_label, 'index': origin_index},f'{sub_clean_dataset_name}')
    origin_dataset = torch.load(f'{sub_clean_dataset_name}')
    origin_dataset['noisy_label'] = origin_dataset['clean_label'].clone()
    torch.save(origin_dataset, f'{sub_noisy_dataset_name}')


def add_noise_dataset(KINDS, sub_clean_dataset_name, sub_noisy_dataset_name, cluster_cnt, sub_cluster_each, label_list, T):
    origin_dataset = torch.load(f'{sub_noisy_dataset_name}')

    T_real = np.zeros((KINDS,KINDS))
    for i in range(sum(sub_cluster_each)):
        origin_dataset['noisy_label'][i] = torch.tensor(np.random.choice(label_list,1,p=T[origin_dataset['clean_label'][i]])).long()
        T_real[origin_dataset['clean_label'][i]][origin_dataset['noisy_label'][i]] += 1
    P_real = [sum(T_real[i])*1.0 for i in range(KINDS)] # random selection
    for i in range(KINDS):
        if P_real[i]>0:
            T_real[i] /= P_real[i]
    P_real = np.array(P_real)/sum(P_real)
    torch.save(origin_dataset, f'{sub_noisy_dataset_name}')


def add_noise_dataset_local(KINDS, sub_noisy_dataset_name, cluster_cnt, sub_cluster_each, label_list, T, idx_sel):
    origin_dataset = torch.load(f'{sub_noisy_dataset_name}')

    T_real = np.zeros((KINDS,KINDS))
    for i in range(len(idx_sel)):
        origin_dataset['noisy_label'][idx_sel[i]] = torch.tensor(np.random.choice(label_list,1,p=T[origin_dataset['clean_label'][idx_sel[i]]])).long()
        T_real[origin_dataset['clean_label'][idx_sel[i]]][origin_dataset['noisy_label'][idx_sel[i]]] += 1
    P_real = [sum(T_real[i])*1.0 for i in range(KINDS)] # random selection
    for i in range(KINDS):
        if P_real[i]>0:
            T_real[i] /= P_real[i]
    P_real = np.array(P_real)/sum(P_real)
    torch.save(origin_dataset, f'{sub_noisy_dataset_name}')
    return P_real, T_real


def get_feat_clusters(origin, sample):
    final_feat = origin['feature'][sample]
    noisy_label = origin['noisy_label'][sample]
    return final_feat, noisy_label


def get_feat_clusters_local(subdataset_name, sample):
    origin = torch.load(f'{subdataset_name}', map_location=torch.device('cpu'))
    final_feat = origin['feature'][sample]
    noisy_label = origin['noisy_label'][sample]

    return 0, final_feat, noisy_label



def count_real(KINDS, T, P, mode, _device = 'cpu'):
    # time1 = time.time()
    P = P.reshape((KINDS, 1))
    p_real = [[] for _ in range(3)]

    p_real[0] = torch.mm(T.transpose(0, 1), P).transpose(0, 1)
    # p_real[2] = torch.zeros((KINDS, KINDS, KINDS)).to(_device)
    p_real[2] = torch.zeros((KINDS, KINDS, KINDS))

    temp33 = torch.tensor([])
    for i in range(KINDS):
        Ti = torch.cat((T[:, i:], T[:, :i]), 1)
        temp2 = torch.mm((T * Ti).transpose(0, 1), P)
        p_real[1] = torch.cat([p_real[1], temp2], 1) if i != 0 else temp2

        for j in range(KINDS):
            Tj = torch.cat((T[:, j:], T[:, :j]), 1)
            temp3 = torch.mm((T * Ti * Tj).transpose(0, 1), P)
            temp33 = torch.cat([temp33, temp3], 1) if j != 0 else temp3
        # adjust the order of the output (N*N*N), keeping consistent with p_estimate
        t3 = []
        for p3 in range(KINDS):
            t3 = torch.cat((temp33[p3, KINDS - p3:], temp33[p3, :KINDS - p3]))
            temp33[p3] = t3
        if mode == -1:
            for r in range(KINDS):
                p_real[2][r][(i+r+KINDS)%KINDS] = temp33[r]
        else:
            p_real[2][mode][(i + mode + KINDS) % KINDS] = temp33[mode]


    temp = []       # adjust the order of the output (N*N), keeping consistent with p_estimate
    for p1 in range(KINDS):
        temp = torch.cat((p_real[1][p1, KINDS-p1:], p_real[1][p1, :KINDS-p1]))
        p_real[1][p1] = temp
    return p_real


def build_T(cluster):
    T = [[0 for _ in range(cluster)] for _ in range(cluster)]
    for i in range(cluster):
        rand_sum = 0
        for j in range(cluster):
            if i != j:
                rand = round(random.uniform(0.01, 0.07), 3)
                rand_sum += rand
                T[i][j] = rand
        T[i][i] = 1 - rand_sum
    return T



def build_T_local(cluster, center_class):
    T = [[0 for _ in range(cluster)] for _ in range(cluster)]
    zero_class = np.random.choice(range(cluster),int(np.sqrt(cluster)), replace = False)
    for i in range(cluster):
        rand_sum = 0
        for j in range(cluster):
            if i != j:
                rand = round(random.uniform(0.15, 0.25), 3) * (j in zero_class) if i == center_class else round(random.uniform(0.01, 0.07), 3)
                rand_sum += rand
                T[i][j] = rand
        T[i][i] = 1 - rand_sum
    # print(torch.tensor(T))
    # exit()
    return T



def check_T(KINDS, noisy_label, point_each_cluster):
    temp_error_matrix = [[0 for _ in range(KINDS)] for _ in range(KINDS)]
    cnt_point = 0
    for cnt in range(len(point_each_cluster)):
        for label in noisy_label[cnt_point : point_each_cluster[cnt]+cnt_point]:
            temp_error_matrix[cnt][label] = temp_error_matrix[cnt][label] + 1
        cnt_point += point_each_cluster[cnt]
    for i in range(KINDS):
        temp_sum = sum(temp_error_matrix[i])
        for j in range(KINDS):
            temp_error_matrix[i][j] = round(temp_error_matrix[i][j]/temp_sum, 3)
    print(f'Check_Error_Rate = \n{np.array(temp_error_matrix)}')





def select_next_idx(selected_idx, idx_sel):
    # selected_idx[idx_sel[:int(numLocal*0.7)]] = -1
    selected_idx[idx_sel] = -1
    if selected_idx[selected_idx > -1].size(0) > 0:
        next_select_idx = random.choice(selected_idx[selected_idx > 0])      # select one from the remaining part
        return next_select_idx, selected_idx
    else:
        return random.randint(0, 49999), selected_idx


def adjust_learning_rate(optimizer, epoch,alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_device():
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")
    print(f'Current device is {_device}', flush=True)
    return _device

def set_model_pre(config):
    # use resnet50 for ImageNet pretrain (PyTorch official pre-trained model)
    if config.pre_type == 'image':
        model = res_image.resnet50(pretrained=True)
    else:
        RuntimeError('Undefined pretrained model.')
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config.num_classes)
    model.to(config.device)
    return model




def set_model_train(config):
    model = res_cifar_new.ResNet34(num_classes = config.num_classes)
    model.to(config.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    return model, optimizer



def init_feature_set(config, model_pre, train_dataloader, rnd):
    c1m_cluster_each = [0 for _ in range(config.num_classes)]
    # save the 512-dim feature as a dataset
    model_pre.eval()
    record = [[] for _ in range(config.num_classes)]

    for i_batch, (feature, label, index) in enumerate(train_dataloader):
        feature = feature.to(config.device)
        label = label.to(config.device)
        extracted_feature, _ = model_pre(feature)
        for i in range(extracted_feature.shape[0]):
            record[label[i]].append({'feature': extracted_feature[i].detach().cpu(), 'index': index[i]})

    path = f'./data/{config.pre_type}_{config.dataset}_{config.label_file_path[7:-3]}.pt'
    return path, record, c1m_cluster_each


def build_dataset_informal(config, record, c1m_cluster_each):
    # estimate T
    original_clean_dataset_name = config.path
    sub_clean_dataset_name = f'{config.path[:-3]}_clean.pt'
    sub_noisy_dataset_name = f'{config.path[:-3]}_noisy.pt'
    # if ~config.build_feat:
    #     return sub_clean_dataset_name, sub_noisy_dataset_name
    # Build Dataset -----------------------------------------------
    label_list = [i for i in range(config.num_classes)]  # label-type: 0, 1, 2 ... 9
    P = config.P

    sub_cluster_each = [int(50000/config.num_classes)] * config.num_classes
    extract_sub_dataset(sub_cluster_each, record, sub_clean_dataset_name,
                        sub_noisy_dataset_name)

    if config.label_file_path is 'NA':
        RuntimeError('Cannot load noisy labels.')
    else:
        # load noise file. re-format
        origin_dataset = torch.load(f'{sub_noisy_dataset_name}')
        noise_label = torch.load(config.label_file_path)

        T_real = np.zeros((config.num_classes,config.num_classes))
        for i in range(sum(sub_cluster_each)):
            idx = origin_dataset['index'][i]
            assert origin_dataset['clean_label'][i] == noise_label['clean_label_train'][idx]
            origin_dataset['noisy_label'][i] = noise_label['noise_label_train'][idx].long()
            T_real[origin_dataset['clean_label'][i]][origin_dataset['noisy_label'][i]] += 1
        P_real = [sum(T_real[i])*1.0 for i in range(config.num_classes)] # random selection
        for i in range(config.num_classes):
            if P_real[i]>0:
                T_real[i] /= P_real[i]
        P_real = np.array(P_real)/sum(P_real)
        config.T = T_real
        torch.save(origin_dataset, f'{sub_noisy_dataset_name}')


    return sub_clean_dataset_name, sub_noisy_dataset_name