import torch
import torch.nn as nn
import numpy as np

class Attention(nn.Module):
    def __init__(self,original_dim=128,Embed_Dimk=128,Embed_Dimv=128,out_dim=128,context_dim=7):
        super().__init__()

        self.Embed_Dimk =Embed_Dimk
        self.Embed_Dimv =Embed_Dimv
        self.out_dim = out_dim
        
        self.context_pre_k = nn.Linear(context_dim,original_dim)
        self.context_pre_b = nn.Linear(context_dim,original_dim)

        self.querylinear = nn.Linear(original_dim,Embed_Dimk)
        self.keylinear = nn.Linear(original_dim,Embed_Dimk)
        self.valuelinear = nn.Linear(original_dim,Embed_Dimv)
        self.softmax = nn.Softmax(dim = -1)
        self.linear = nn.Linear(Embed_Dimv,out_dim)
        self.actvn = nn.ReLU()
        self.lnv = nn.LayerNorm(Embed_Dimv)
    def forward(self,x,context):
        #x (b,n,k,h)
        bs,point_num,k,h = x.size(0),x.size(1),x.size(2),x.size(3)

        context_k = self.context_pre_k(context)
        context_b = self.context_pre_b(context)
        x = context_k*x+context_b

        query=self.querylinear(x)
        key=self.keylinear(x)
        value=self.valuelinear(x)
        #query = self.lnk(query)
        key = key.permute(0,1,3,2)
        #value = self.lnv(value)
        attn_matrix = torch.matmul(query, key)/ np.sqrt(self.Embed_Dimk) 
        attn_matrix = self.softmax(attn_matrix)
        out = torch.matmul(attn_matrix,value)
        out = self.actvn(self.lnv(out))
        out = self.linear(out)

        result = torch.max(out,dim=2)[0]

        return result