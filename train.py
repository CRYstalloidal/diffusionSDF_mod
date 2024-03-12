#!/usr/bin/env python3

import torch
import torch.utils.data 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

import os
import json
import time

# add paths in model/__init__.py for new models
from model import *


def main():

    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    train_dataset = init_dataset(specs["TrainData"], specs)



    lab_set = train_dataset
    lab_dataloader = torch.utils.data.DataLoader(
        lab_set,
        batch_size=args.batch_size,
        num_workers= 8 if args.workers is None else args.workers,
    drop_last=True,
            shuffle=True
    )
    dataloaders = {"context":lab_dataloader}
    
    data_len = len(train_dataset)
    print("Training on {} objects...".format(data_len))

    model = init_model(specs["Model"], specs, data_len, dataloaders)

    max_epochs = specs["NumEpochs"]
    log_frequency = specs["LogFrequency"]

    
    if args.resume is not None:
        ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
    else:
        resume = None  

    callbacks = []

    callback = ModelCheckpoint(
        dirpath=args.exp_dir, filename='{epoch}',
        save_top_k=-1, save_last=True, every_n_epochs=log_frequency)

    callbacks.append(callback)

    
    trainer = pl.Trainer(accelerator='gpu', devices=[0,2], precision=16, max_epochs=max_epochs, 
                        callbacks=callbacks, gradient_clip_val=0.5)
    trainer.fit(model=model, ckpt_path=resume) 



def init_model(model, specs, num_objects, dataloaders):
    return PointSDF(specs, dataloaders)

def init_dataset(dataset, specs):
        from dataloader.labeled_ds import LabeledDS
        labeled_train = specs["LabeledTrainSplit"]
        with open(labeled_train, "r") as f:
            labeled_train_split = json.load(f)


        return LabeledDS(
            specs["DataSource"], labeled_train_split, 
            samples_per_mesh=specs["LabSamplesPerMesh"], pc_size=specs["LabPCsize"]
            )


    
if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser()
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
        "--batch_size", "-b",
        default=1, type=int
    )

    arg_parser.add_argument(
        "--workers", "-w",
        default=None, type=int
    )


    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))
    print(specs["Description"][0])

    main()
