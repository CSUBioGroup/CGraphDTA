import dgl
import dgl.function as dglfn
import dgl.nn.pytorch as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math  

class ModelNew(nn.Module):

    def __init__(self):
        super().__init__()
        self.a_init = nn.Linear(382, 128)
        self.a_init1 = nn.Linear(128, 64)
        self.a_init2 = nn.Linear(64, 128)
        self.b_init = nn.Linear(12, 8)    
        self.r_init_1 = nn.Linear(54,64)
        self.r_init_2 = nn.Linear(64,128)
        self.normp = nn.LayerNorm(128)
        self.norml = nn.LayerNorm(128)
        self.normp1 = nn.LayerNorm(128)
        self.norml1 = nn.LayerNorm(128)
        self.scale = math.sqrt(0.5)

        # ligand
        self.A = nn.Linear(128, 128)
        self.B = nn.Linear(128*2,128)
        self.E = nn.Linear(128,128*2)
        self.a_conv1 = SConv1(128, 8, 128, 2)    
        self.a_conv2 = SConv1(128, 8, 128, 2) 
        self.a_conv3 = SConv1(128, 8, 128, 3)
        
        self.encoder_l = Encoder(
            protein_dim=64,  ###64
            hid_dim=64,   ###64
            n_layers=3,   ###3
            kernel_size=7,  ###7
            dropout=0.1,    ###0.2
        )
        # protein
        self.encoder_p = Encoder(
            protein_dim=64,  ###64
            hid_dim=64,   ###64
            n_layers=3,   ###3
            kernel_size=7,  ###7
            dropout=0.1,    ###0.2
        )
        self.C = nn.Linear(128, 128)
        self.D = nn.Linear(128*2,128)
        self.F = nn.Linear(128,128*2)
        self.r_conv1 = PConv1(128, 1, 128, 2)
        self.r_conv2 = PConv1(128, 1, 128, 2)  
        self.r_conv3 = PConv1(128, 1, 128, 2) 

        self.classifier = nn.Sequential(
            nn.Linear(128*2 + 128*2, 1024), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        self.sum_pool = dglnn.SumPooling()
        self.mean_pool = dglnn.AvgPooling()

    def forward(self, ga, gr): 
        device = torch.device("cuda:0")
        ga = ga.to('cuda:0')
        gr = gr.to('cuda:0')
        va_init = self.a_init1(F.relu(self.a_init(ga.ndata['feat'])))
        va_init = self.encoder_l(va_init)
        va_init = self.a_init2(va_init)
        ea = self.b_init(ga.edata['feat'])  
  
        vr_init = F.relu(self.r_init_1(gr.ndata['feat'])) 
        vr_init = self.encoder_p(vr_init)
        vr_init = self.r_init_2(vr_init)      
        er = gr.edata['weight'].reshape(-1).unsqueeze(1)
        # ligand
        va = F.leaky_relu(self.A(va_init), 0.1)  
        sa = self.sum_pool(ga, va)
        va, sa = self.a_conv1(ga, va, ea, sa)
        va, sa = self.a_conv2(ga, (va+va_init)*self.scale, ea, sa)
        fa = torch.cat((self.mean_pool(ga, va), sa), dim=-1)
        fa = self.B(fa)
        fa = fa + self.mean_pool(ga,va_init)
        fa = self.E(self.norml1(fa))
        # protein
        vr = F.leaky_relu(self.C(vr_init), 0.1)   
        sr = self.sum_pool(gr, vr)
        vr, sr = self.r_conv1(gr, vr, er, sr)
        vr, sr = self.r_conv2(gr, (vr+vr_init)*self.scale, er, sr)
        fr = torch.cat((self.mean_pool(gr, vr), sr), dim=-1)
        fr = self.D(fr)
        fr = fr + self.mean_pool(gr,vr_init)
        fr = self.F(self.normp1(fr))
        # predict
        f = torch.cat((fa, fr), dim=-1)  
        y = self.classifier(f)

        return y

class Encoder(nn.Module):
    """feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout): 
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"  ### 奇数
        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.scale = math.sqrt(0.5)
        self.convs = nn.ModuleList([
            nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2),nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size-2, padding=(kernel_size - 3) // 2),nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size-4, padding=(kernel_size - 5) // 2)]) 
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.gn = nn.GroupNorm(8, hid_dim * 2) 
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, protein):
        protein= torch.unsqueeze(protein,dim=0)
        conv_input = self.fc(protein)
        conv_input = conv_input.permute(0, 2, 1)
        for i, conv in enumerate(self.convs):
            conved = conv(conv_input)
            conved = F.glu(conved, dim=1)  
            conved = (conved + conv_input) * self.scale
            conv_input = conved
        conved = conved.permute(0, 2, 1)
        conved= torch.squeeze(conved,dim=0)
        conved = self.ln(conved)
        return conved
      
class SConv1(nn.Module):
    def __init__(self, v_dim, e_dim, h_dim, k_head): 
        super().__init__()
        self.A = nn.Linear(v_dim, h_dim)
        self.m2s = nn.ModuleList([SConv1.Helper(v_dim, h_dim) for _ in range(k_head)])

        self.B = nn.Linear(h_dim * k_head, h_dim)
        self.C = nn.Linear(v_dim, h_dim)
        self.D = nn.Linear(e_dim + v_dim, h_dim)   
        self.E = nn.Linear(h_dim + v_dim, h_dim)
        self.K = nn.Linear(e_dim, h_dim) 

        self.gate_update_m = SConv1.GateUpdate(h_dim)
        self.gate_update_s = SConv1.GateUpdate(h_dim)

    def __msg_func(self, edges):
        v = edges.src['v']
        e = edges.data['e']
        return {'ve': F.leaky_relu(self.K(e) * v,0.1)}   

    def forward(self, g, v, e, s): 
        s2s = torch.tanh(self.A(s)) 

        m2s = torch.cat([layer(g, v, s) for layer in self.m2s],dim=1) 
        m2s = torch.tanh(self.B(m2s))
        ###  main to super
        s2m = torch.tanh(self.C(s))  # g,h
        s2m = dgl.broadcast_nodes(g, s2m)  # N,h
        ### suepr to main
        with g.local_scope():
            g.ndata['v'] = v
            g.edata['e'] = e
            g.update_all(self.__msg_func, dglfn.sum('ve', 'sve'))
            svev = torch.cat((g.ndata['sve'], v),dim=1)   
        m2m = F.leaky_relu(self.E(svev), 0.1 )
        vv = self.gate_update_m(m2m, s2m, v)  
        ss = self.gate_update_s(s2s, m2s, s) 

        return vv, ss

    class Helper(nn.Module):
        def __init__(self, v_dim, h_dim):
            super().__init__()

            self.A = nn.Linear(v_dim, h_dim)
            self.B = nn.Linear(v_dim, h_dim)  
            self.C = nn.Linear(h_dim, 1)     
            self.D = nn.Linear(v_dim, h_dim)  

        def forward(self, g, v, s):
            d_node = torch.tanh(self.A(v))  
            d_super = torch.tanh(self.B(s))  
            d_super = dgl.broadcast_nodes(g, d_super) 
            a = self.C(d_node * d_super).reshape(-1) 
            with g.local_scope():
                g.ndata['a'] = a
                a = dgl.softmax_nodes(g, 'a')
                g.ndata['h'] = self.D(v) * a.unsqueeze(1)              
                main2super_i = dgl.sum_nodes(g, 'h')

            return main2super_i 

    class GateUpdate(nn.Module):

        def __init__(self, h_dim):
            super().__init__()
            self.A = nn.Linear(h_dim, h_dim)
            self.B = nn.Linear(h_dim, h_dim)
            self.gru = nn.GRUCell(h_dim, h_dim)

        def forward(self, a, b, c):
            z = torch.sigmoid(self.A(a) + self.B(b))
            h = z * b + (1 - z) * a
            cc = self.gru(c, h)
            return cc

class PConv1(nn.Module):
    def __init__(self, v_dim, e_dim, h_dim, k_head):
        super().__init__()

        self.A = nn.Linear(v_dim, h_dim)
        self.m2s = nn.ModuleList([SConv1.Helper(v_dim, h_dim) for _ in range(k_head)])
        self.B = nn.Linear(h_dim * k_head, h_dim)
        self.C = nn.Linear(v_dim, h_dim)
        self.D = nn.Linear(e_dim + v_dim, h_dim)   
        self.E = nn.Linear(h_dim + v_dim, h_dim)
        self.K = nn.Linear(e_dim, h_dim)  
        self.gate_update_m = SConv1.GateUpdate(h_dim)
        self.gate_update_s = SConv1.GateUpdate(h_dim)

    def __msg_func(self, edges):
        v = edges.src['v']
        e = edges.data['e']
        
        return {'ve': F.leaky_relu(self.K(e) * v,0.1)} 

    def forward(self, g, v, e, s):   
        s2s = torch.tanh(self.A(s))  
        m2s = torch.cat([layer(g, v, s) for layer in self.m2s],dim=1)  
        m2s = torch.tanh(self.B(m2s))
        
        s2m = torch.tanh(self.C(s))  
        s2m = dgl.broadcast_nodes(g, s2m)
        with g.local_scope():
            g.ndata['v'] = v
            g.edata['e'] = e
            g.update_all(self.__msg_func, dglfn.sum('ve', 'sve'))
            svev = torch.cat((g.ndata['sve'], v),dim=1)  
        m2m = F.leaky_relu(self.E(svev), 0.1 ) 
        vv = self.gate_update_m(m2m, s2m, v)   
        ss = self.gate_update_s(s2s, m2s, s) 

        return vv, ss

    class Helper(nn.Module):
        def __init__(self, v_dim, h_dim):
            super().__init__()

            self.A = nn.Linear(v_dim, h_dim)  
            self.B = nn.Linear(v_dim, h_dim) 
            self.C = nn.Linear(h_dim, 1)   
            self.D = nn.Linear(v_dim, h_dim) 
        def forward(self, g, v, s):
            d_node = torch.tanh(self.A(v))
            d_super = torch.tanh(self.B(s))  
            d_super = dgl.broadcast_nodes(g, d_super) 
            a = self.C(d_node * d_super).reshape(-1)  # N
            with g.local_scope():
                g.ndata['a'] = a
                a = dgl.softmax_nodes(g, 'a')
                g.ndata['h'] = self.D(v) * a.unsqueeze(1) 
                main2super_i = dgl.sum_nodes(g, 'h')  

            return main2super_i

    class GateUpdate(nn.Module):
        def __init__(self, h_dim):
            super().__init__()
            self.A = nn.Linear(h_dim, h_dim)
            self.B = nn.Linear(h_dim, h_dim)
            self.gru = nn.GRUCell(h_dim, h_dim) 

        def forward(self, a, b, c):
            z = torch.sigmoid(self.A(a) + self.B(b))
            h = z * b + (1 - z) * a
            cc = self.gru(c, h)
            
            return cc
    