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

def init_logger(log_file):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def timeSince(since):
    now = time.time()
    s = now - since
    return now, s

def train(model, train_loader_protein, criterion, optimizer, epoch,device):
    model.train()
    tbar = tqdm(train_loader_protein, total=len(train_loader_protein))
    losses = []
    t = time.time()
    
    for i, data in enumerate(tbar):
        data0 = [i.to(device) for i in data[0]]
        ga, gr, aff = data0  
        y_pred = model(ga,gr).squeeze()
        y_true = aff.float().squeeze()      
        assert y_pred.shape == y_true.shape
        loss = criterion(y_pred,y_true).cuda()
        optimizer.zero_grad()          
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)        
        optimizer.step()     
        losses.append(loss.item())
        tbar.set_description(f'epoch {epoch+1} loss {np.mean(losses[-10:]):.4f} grad {grad_norm:.4f}')

    m_losses=np.mean(losses)
    return m_losses      

def valid(model, valid_loader_protein, criterion,device):
    model.eval()
    losses = []
    outputs = []
    targets = []
    tbar = tqdm(valid_loader_protein, total=len(valid_loader_protein))
    for i, data in enumerate(tbar):
        data0 = [i.to(device) for i in data[0]]
        ga, gr, aff = data0        

        with torch.no_grad():
            y_pred = model(ga,gr).squeeze()
        y_true = aff.float().squeeze()
        assert y_pred.shape == y_true.shape
        loss = criterion(y_pred,y_true).cuda()
        losses.append(loss.item())
        outputs.append(y_pred.cpu().detach().numpy().reshape(-1))
        targets.append(y_true.cpu().detach().numpy().reshape(-1))
    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)
        
    evaluation = {
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),}
    ml=np.mean(losses)      
    return ml, evaluation

def main():
    
    seed = np.random.randint(2023000000, 2023012399) ##random
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    F=open(r'../data/train_valte13688_0.35_0.05.pkl','rb')
    content=pickle.load(F)
    graphs = dgl.load_graphs('../data/all_in_one_graph_retrainedligand_13688.bin')[0]
    labels = pd.read_csv('../data/labels_13688.csv')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    protein_train = content[0]
    protein_valid = content[1]
    protein_test = content[2]
    train_dataset_protein = SepDataset([graphs[i] for i in protein_train],[labels.id[i] for i in protein_train], [labels.affinity[i] for i in protein_train], ['a_conn','r_conn'])
    valid_dataset_protein = SepDataset([graphs[i] for i in protein_valid],[labels.id[i] for i in protein_valid], [labels.affinity[i] for i in protein_valid], ['a_conn','r_conn'])
    test_dataset_protein = SepDataset([graphs[i] for i in protein_test],[labels.id[i] for i in protein_test], [labels.affinity[i] for i in protein_test], ['a_conn','r_conn'])   
    train_loader_protein = DataLoader(train_dataset_protein, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate_fn,pin_memory=False,drop_last=False,)
    valid_loader_protein = DataLoader(valid_dataset_protein, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader_protein = DataLoader(test_dataset_protein, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_fn)    
    model = ModelNew()
    model = model.to(device)    
#         optimizer = torch.optim.AdamW(model.parameters(), 1.2e-4, weight_decay=1e-6)   ### (model.parameters(), 1e-3, weight_decay=1e-5)
    optimizer_inner = RAdam(model.parameters(), lr=7e-5, weight_decay=1e-4)
    optimizer = Lookahead(optimizer_inner, k=5, alpha=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=40, eta_min=1e-6)
    criterion = torch.nn.MSELoss()    
    n_epoch = 120
    print_freq = 10
    best_R = 0
    ll = -1
    for epoch in range(n_epoch):
        logger.info('EPOCH: {} train'.format(epoch+1))
        ll = train(model, train_loader_protein, criterion, optimizer, epoch,device)
        if epoch%1==0:
            logger.info('EPOCH: {} valid'.format(epoch+1))
            l,evaluation = valid(model, valid_loader_protein, criterion,device)
            l_t, evaluation_ = valid(model, test_loader_protein, criterion,device)
            logger.info('TrainLoss:%0.4f,ValidLoss:%0.4f\n'%(ll,l))
            if evaluation_['CORR']>best_R:
                best_R= evaluation_['CORR']
                torch.save({'model': model.state_dict()}, '../models/modelxin.pth') 
        scheduler.step()
if __name__ == "__main__":
    main()