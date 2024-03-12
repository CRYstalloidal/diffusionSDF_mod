import torch
import torch.nn as nn
import torch.nn.functional as F
from ..resnet_block import ResnetBlockFC
from ..Attention import Attention
from knn_cuda import KNN

class AttentionPointnet(nn.Module):

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, 
                 n_blocks=6,EmbedDimk=128,EmbedDimv=128,k=20):
        super().__init__()
        self.n_blocks = n_blocks

        self.fc_pos = nn.Linear(dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.attentions = nn.ModuleList([
            Attention(original_dim=hidden_dim,out_dim=hidden_dim)	 for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.k = k
        self.knn = KNN(k=k, transpose_mode=True)



    def forward(self, p):#, query2):
        batch_size, T, D = p.size()

        dis,indices = self.knn(p, p)
        dis = dis.unsqueeze(-1)
        pooled_points = self.pool_local(indices=indices,c=p)
        context = torch.cat([dis,pooled_points,p.unsqueeze(2).repeat(1,1,self.k,1)],dim=-1)


        last_net = 0
        net = self.fc_pos(p)


        for i in range(self.n_blocks):
            pooled = self.pool_local(indices=indices,c=net)
            #print(pooled.shape)
            pooled = self.attentions[i](pooled,context)
            net = torch.cat([net, pooled], dim=2)
            net = self.blocks[i](net)
            #print(net)
            net += last_net
            last_net = net
            

        c = self.fc_c(net)
        return c



    def pool_local(self,indices, c):
        bs,point_num,h = c.size(0),c.size(1),c.size(2)  #（b,n,h）
        result = torch.empty(bs,point_num,self.k,h).cuda()
        for i in range(bs):
            if(i == 0):
                result = c[i].index_select(0,indices[i].flatten()).reshape(1,point_num,self.k,h)
            else:
                result = torch.cat([result,c[i].index_select(0,indices[i].flatten()).reshape(1,point_num,self.k,h)],0)
        return result #（b,n,k,h）