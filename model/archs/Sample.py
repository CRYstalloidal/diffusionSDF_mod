import torch
import torch.nn as nn
import numpy as np
from knn_cuda import KNN

from model.archs.Attention import Attention

class Sample(nn.Module):


    def pool_local(self,indices, c):
        bs,h = c.size(0),c.size(2)  #（b,n,h）
        point_num = indices.size(1)
        for i in range(bs):
            if(i == 0):
                result = c[i].index_select(0,indices[i].flatten()).reshape(1,point_num,self.k,h)
            else:
                result = torch.cat([result,c[i].index_select(0,indices[i].flatten()).reshape(1,point_num,self.k,h)],0)
        return result #（b,n,k,h）

    def __init__(self,n_heads=8,k=20,latent_dim=128):
        super().__init__()
        self.k=k
        # self.n_heads = n_heads
        # self.attentions = nn.ModuleList([
        #         Attention(original_dim=latent_dim,out_dim=latent_dim)	 for i in range(n_heads)
        # ])
        self.linear = nn.Linear(k*latent_dim,latent_dim)
        # self.actvn = nn.ReLU()
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=8,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.context_pre_k = nn.Linear(7,latent_dim)
        self.context_pre_b = nn.Linear(7,latent_dim)

        self.knn = KNN(k=k, transpose_mode=True)

    def forward(self,x,pc,query):
        #x (b,n,h) query (b,n_q,3) pc(b,n_x,3)
        bs,n_q,h = query.size(0),query.size(1),x.size(2)
        
        dis,indices = self.knn(pc, query)
        dis = dis.unsqueeze(-1)
        pooled_points = self.pool_local(indices=indices,c=pc)
        context = torch.cat([dis,pooled_points,query.unsqueeze(2).repeat(1,1,self.k,1)],dim=-1)

        pooled = self.pool_local(indices, x) * self.context_pre_k(context) + self.context_pre_b(context)
        out = self.transformer_encoder(pooled.reshape(-1,self.k,h))
        # for i in range(self.n_heads):
        #     net = self.attentions[i](pooled,context)#(b,n_q,h)
        #     if(i == 0):
        #         out = net
        #     else :
        #         out = torch.cat((out,net),-1)
        #     #print(1,out)
        # out = self.actvn(out)
        out = self.linear(out.reshape(bs,n_q,-1))
        #print(1,out)
        return out