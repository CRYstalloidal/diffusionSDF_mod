#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.neighbors import NearestNeighbors

import numpy as np
import math

import os 
from pathlib import Path
import time 

from model import base_pl
from model.archs.encoders.attentionPointnet import AttentionPointnet
from model.archs.encoders.dgcnn import DGCNN
from model.archs.decoders.sdfDecoder import SdfDecoder
from model.archs.Sample import Sample

from utils import mesh, evaluate

class PointSDF(base_pl.Model):
    def __init__(self, specs, dataloaders):
        super().__init__(specs)
        
        self.k = self.specs ["k"]

        encoder_specs = self.specs["EncoderSpecs"]
        self.latent_size = encoder_specs["latent_size"]

        decoder_specs = self.specs["DecoderSpecs"]
        self.decoder_hidden_dim = decoder_specs["hidden_dim"]
        lr_specs = self.specs["LearningRate"]
        self.lr_init = lr_specs["init"]
        self.lr_step = lr_specs["step_size"]
        self.lr_gamma = lr_specs["gamma"]

        
        self.dataloaders = dataloaders


        self.build_model()


    def build_model(self):
        #self.encoder = AttentionPointnet(k=self.k,c_dim=self.latent_size)
        self.encoder = DGCNN(k=self.k,output_channels=self.latent_size)
        
        self.sampler = Sample(k=self.k,latent_dim=self.latent_size)
        
        self.decoder = SdfDecoder(self.latent_size, self.decoder_hidden_dim,)


    def configure_optimizers(self):
    
        optimizer = torch.optim.Adam(self.parameters(), self.lr_init)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer, self.lr_step, self.lr_gamma)

        return [optimizer], [lr_scheduler]



    def training_step(self, x, batch_idx):


        context = x['context']
        
        context_pc = context['point_cloud']
        context_xyz = context['sdf_xyz']
        context_gt = context['gt_sdf']



        fea = self.encoder(context_pc)#(b,n,h)
        feat_sum = self.sampler(fea,context_pc,context_xyz)
        #print(feat_sum)
        decoder_input = torch.cat([feat_sum,context_xyz],dim=-1)
        pred_sdf = self.decoder(decoder_input)
        l1_loss = self.labeled_loss(pred_sdf,context_gt)
        #print(l1_loss)

        loss_dict =  {
                        "lab":l1_loss,

                    }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)
        

        return l1_loss
        
        

    def labeled_loss(self, pred_sdf, gt_sdf):

        l1_loss = nn.L1Loss()(pred_sdf.squeeze(), gt_sdf.squeeze())
            
        return l1_loss 

    def unlabeled_loss(self, pred_pt, gt_pt):
        
        return nn.MSELoss()(pred_pt, gt_pt)

    def forward(self, pc, query):
        fea = self.encoder(pc)#(b,n,h)

        feat_sum = self.sampler(fea,pc,query)
        decoder_input = torch.cat([feat_sum,query],dim=-1)
        pred_sdf = self.decoder(decoder_input)


        return pred_sdf
    
    def get_unlab_offset(self, query_xyz, query_gt_pt, pred_sdf):
        dir_vec = F.normalize(query_xyz - query_gt_pt, dim=-1)

        # different for batch size=1 and batch_size >1
        # TODO: combine, shouldn't need this condition
        if query_xyz.shape[0] ==1:
            pred_sdf = pred_sdf.unsqueeze(0)
            neg_idx = torch.where(pred_sdf.squeeze()<0)[0]
            pos_idx = torch.where(pred_sdf.squeeze()>=0)[0]

            neg_pred = query_xyz[:,neg_idx] + dir_vec[:, neg_idx] * pred_sdf[:,neg_idx]
            pos_pred = query_xyz[:,pos_idx] - dir_vec[:, pos_idx] * pred_sdf[:,pos_idx]

            pred_pt = torch.cat((neg_pred, pos_pred), dim=1)                                                                  
            query_gt_pt = torch.cat((query_gt_pt[:,neg_idx], query_gt_pt[:,pos_idx]), dim=1)
        
        else:
            # splits into a tuple of two tensors; one tensor for each dimension; then can use as index
            neg_idx = pred_sdf.squeeze()<0
            neg_idx = neg_idx.nonzero().split(1, dim=1) 

            pos_idx = pred_sdf.squeeze()>=0
            pos_idx = pos_idx.nonzero().split(1, dim=1)

            # based on sign of sdf value, need to direct in different direction
            # indexing in this way results in an extra dimension that should be squeezed
            neg_pred = query_xyz[neg_idx].squeeze(1) + dir_vec[neg_idx].squeeze(1) * pred_sdf[neg_idx].squeeze(1)
            pos_pred = query_xyz[pos_idx].squeeze(1) - dir_vec[pos_idx].squeeze(1) * pred_sdf[pos_idx].squeeze(1)

            # for batch size 4, query_per_batch 16384, 
            # dimension 4,16384,3 -> 4*16384, 3
            pred_pt = torch.cat((neg_pred, pos_pred), dim=0) # batches are combined
            query_gt_pt = torch.cat((query_gt_pt[neg_idx].squeeze(1), query_gt_pt[pos_idx].squeeze(1)), dim=0)

        return pred_pt, query_gt_pt

    def train_dataloader(self):
        return self.dataloaders



    def reconstruct(self, model, test_data, eval_dir, testopt=True, sampled_points=15000):
        recon_samplesize_param = 256
        recon_batch = 20000

        gt_pc = test_data['point_cloud'].float()
        #print("gt pc shape: ",gt_pc.shape)
        sampled_pc = gt_pc[:,torch.randperm(gt_pc.shape[1])[0:15000]]
        #print("sampled pc shape: ",sampled_pc.shape)

        if testopt:
            start_time = time.time()
            model = self.fast_opt(model, sampled_pc, num_iterations=800)

        model.eval() 
        

        with torch.no_grad():
            Path(eval_dir).mkdir(parents=True, exist_ok=True)
            mesh_filename = os.path.join(eval_dir, "reconstruct") #ply extension added in mesh.py
            evaluate_filename = os.path.join("/".join(eval_dir.split("/")[:-2]), "evaluate.csv")
            
            mesh_name = test_data["mesh_name"]

            levelset = 0.005 if testopt else 0.0
            mesh.create_mesh(model, sampled_pc, mesh_filename, recon_samplesize_param, recon_batch, level_set=levelset,device=self.specs["Device"])
            try:
                evaluate.main(gt_pc, mesh_filename, evaluate_filename, mesh_name) # chamfer distance
            except Exception as e:
                print(e)

    def fast_opt(self, model, full_pc, num_iterations=200):

        num_iterations = num_iterations
        xyz_full, gt_pt_full = self.fast_preprocess(full_pc)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        print("performing refinement on input point cloud...")
        #print("shapes: ", full_pc.shape, xyz_full.shape)
        for e in range(num_iterations):
            samp_idx = torch.randperm(xyz_full.shape[1])[0:3000]
            xyz = xyz_full[ :,samp_idx ].to(self.specs["Device"])
            gt_pt = gt_pt_full[ :,samp_idx ].to(self.specs["Device"])
            pc = full_pc[:,torch.randperm(full_pc.shape[1])[0:3000]].to(self.specs["Device"])


            fea = self.encoder(pc)#(b,n,h)
            feat_sum = self.sampler(fea,pc,xyz)
            decoder_input = torch.cat([feat_sum,xyz],dim=-1)
            pred_sdf = self.decoder(decoder_input).unsqueeze(-1)

            fea = self.encoder(pc)#(b,n,h)
            feat_sum = self.sampler(fea,pc,pc)
            decoder_input = torch.cat([feat_sum,pc],dim=-1)
            pc_pred = self.decoder(decoder_input)


            pred_pt, gt_pt = model.get_unlab_offset(xyz, gt_pt, pred_sdf)

            # loss of pt offset and loss of L1
            unlabeled_loss = nn.L1Loss()(pred_pt, gt_pt)
            # using pc to supervise query as well
            pc_l1 = nn.L1Loss()(pc_pred, torch.zeros_like(pc_pred))

            loss = unlabeled_loss + 0.01*pc_l1
            loss.requires_grad_(True)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return model


    def fast_preprocess(self, pc):
        pc = pc.squeeze()
        pc_size = pc.shape[0]
        query_per_point=20

        def gen_grid(start, end, num):
            x = np.linspace(start,end,num=num)
            y = np.linspace(start,end,num=num)
            z = np.linspace(start,end,num=num)
            g = np.meshgrid(x,y,z)
            positions = np.vstack(map(np.ravel, g))
            return positions.swapaxes(0,1)

        dot5 = gen_grid(-0.5,0.5, 70) 
        dot10 = gen_grid(-1.0, 1.0, 50)
        grid = np.concatenate((dot5,dot10))
        grid = torch.from_numpy(grid).float()
        grid = grid[ torch.randperm(grid.shape[0])[0:30000] ]

        total_size = pc_size*query_per_point + grid.shape[0]

        xyz = torch.empty(size=(total_size,3))
        gt_pt = torch.empty(size=(total_size,3))

        # sample xyz
        dists = torch.cdist(pc, pc)
        std, _ = torch.topk(dists, 50, dim=-1, largest=False)
        std = std[:,-1].unsqueeze(-1)

        count = 0
        for idx, p in enumerate(pc):
            # query locations from p
            q_loc = torch.normal(mean=0.0, std=std[idx].item(),
                                 size=(query_per_point, 3))

            # query locations in space
            q = p + q_loc
            xyz[count:count+query_per_point] = q
            count += query_per_point

    
        xyz[pc_size*query_per_point:] = grid

        # nearest neighbor
        dists = torch.cdist(xyz, pc)
        _, min_idx = torch.min(dists, dim=-1) 
        gt_pt = pc[min_idx]
        return xyz.unsqueeze(0), gt_pt.unsqueeze(0)