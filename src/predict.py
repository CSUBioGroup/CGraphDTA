import pickle
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as dglfn
import einops
from loguru import logger
from collections import deque
from tqdm.auto import tqdm
from IPython.display import clear_output
import time
from optim.lookahead import Lookahead
from optim.radam import RAdam
from dataset import SepDataset, collate_fn
from model import ModelNew
import metrics

def timeSince(since):
    now = time.time()
    s = now - since
    return now, s

def test(model, valid_loader_protein, criterion,device):
    model.eval()
    outputs = []
    targets = []
    name_list =[]
    tbar = tqdm(valid_loader_protein, total=len(valid_loader_protein))
    for i, data in enumerate(tbar):
        data0 = [i.to(device) for i in data[0]]
        ga, gr, aff = data0         
        idnames = data[1]
        name_l = []
        for name in idnames:
            name_l.append(name)
        name_list.append(name_l)
        with torch.no_grad():
            y_pred = model(ga,gr).squeeze()
        y_true = aff.float().squeeze()
        assert y_pred.shape == y_true.shape
        outputs.append(y_pred.cpu().detach().numpy().reshape(-1))
        targets.append(y_true.cpu().detach().numpy().reshape(-1))
    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)
    name_list = np.concatenate(name_list).reshape(-1)
        
    evaluation = {
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),}
    
    return evaluation,targets,outputs,name_list

def main():
    
    graphs16 = dgl.load_graphs('../data/all_in_one_graph_retrainedligand_test2016.bin')[0]
    labels16 = pd.read_csv('../data/labels_test2016.csv')
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")      
    ####test16
    test16_dataset = SepDataset([graphs16[i] for i in range(len(graphs16))], [labels16.id[i] for i in range(len(labels16))], [labels16.affinity[i] for i in range(len(labels16))], ['a_conn','r_conn'])      
    test16_loader = DataLoader(test16_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn)
   
    flag = 'predict' 
    model_path = '../models/model.pth'    ### the best model
#     print(f'结果将保存在 以 {flag}开头的csv文件中')                
    model = ModelNew()
    checkpoint = torch.load(model_path,map_location=device)
    # print(state_dict)
    model.load_state_dict(checkpoint['model'])
    model.to(device)        
    optimizer_inner = RAdam(model.parameters(), lr=7e-5, weight_decay=1e-4)
    optimizer = Lookahead(optimizer_inner, k=5, alpha=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=40, eta_min=1e-6)
    criterion = torch.nn.MSELoss()    
        
    p_f = 'test2016'
    logger.info('%s_%s.csv'%(flag,p_f))
    evoluation,targets,outputs,names = test(model, test16_loader, criterion,device)
    a = pd.DataFrame()
    a=a.assign(pdbid=names,predicted=outputs,real=targets,set=p_f)
    a.to_csv(f'../results/{flag}_{p_f}.csv')  
    logger.info(evoluation)

if __name__ == "__main__":
    main()