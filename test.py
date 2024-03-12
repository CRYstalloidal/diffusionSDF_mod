#!/usr/bin/env python3

import torch
import torch.utils.data 
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


import os
import json
import time
from tqdm import tqdm

# remember to add paths in model/__init__.py for new models
from model import *



def main():
    
    test_dataset = init_dataset("acronym", specs)
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False,
    )


    model = init_model(specs["Model"], specs, len(test_dataset))
    
    if args.resume is not None:
        ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
    else:
        resume = None  
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    

    with tqdm(test_dataloader, unit="batch") as pbar:
        for idx, data in enumerate(pbar):
            pbar.set_description(f"Testing file {idx}")

            if args.outdir is None:
                eval_dir = os.path.join(args.exp_dir, args.resume, data['mesh_name'][0])
            else:
                eval_dir = os.path.join(args.outdir, args.resume, data['mesh_name'][0])
            model.reconstruct(model, data, eval_dir,testopt=False)



def init_model(model, specs, num_objects):
    return PointSDF(specs, None).to(specs["Device"])

def init_dataset(dataset, specs):

    test_split_file = specs["TestSplit"]
    with open(test_split_file, "r") as f:
        test_split = json.load(f)

    if dataset == "acronym":
        from dataloader.test_loader import TestAcronymDataset
        return TestAcronymDataset(specs["DataSource"], test_split, 16000, 
                                pc_size=specs.get("ReconPCsize",30000)) # for calculating chamfer distance

    
if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train DeepSDF and shapecode")
    arg_parser.add_argument(
        "--exp_dir", "-e",
        required=True,
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    arg_parser.add_argument(
        "--resume", "-r",
        default=None,
        help="continue from previous saved logs, integer value or 'last'",
    )

    arg_parser.add_argument(
        "--outdir", "-o",
        default=None
    )


    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))
    print(specs["Description"][0])

    main()
