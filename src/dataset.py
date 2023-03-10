from torch.utils.data import Dataset
import numpy as np
import dgl
import torch

 
class WholeDataset(Dataset):             
    def __init__(self, graphs,labels, affinities):
        self.graphs = graphs
        self.labels = labels
        self.affinities = affinities
        self.length = len(labels)

    def __getitem__(self, index: int):
        return (self.graphs[index], self.affinities[index])

    def get_name(self, index):
        return self.labels[index]

    def __len__(self) -> int:
        return self.length

    
class SepDataset(WholeDataset):                      
    def __init__(self, graphs, labels, affinities, edge_types):
        super().__init__(graphs,labels, affinities)
        self.edge_types = edge_types
        
    def __getitem__(self, index: int):
        g = self.graphs[index]
        sub_gs = [g.edge_type_subgraph((i,)) for i in self.edge_types]
        return (*sub_gs, self.affinities[index],self.labels[index])


def collate_fn(data): 

    *gs, labels,idlist = zip(*data)

    labels = torch.tensor(labels)

    gs = [dgl.batch(i) for i in gs]

    return gs+ [labels.float()], idlist 