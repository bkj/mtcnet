#!/usr/bin/env python

"""
    classifier.py
    
    * Super simple implementation of matching networks
    * Only works for 1-shot ATM
    * No rotation
    * Not sure data split is the same as in original paper
"""

import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from time import time

import torch
from torch.nn import functional as F

from torchvision import transforms

from helpers import OmniglotTaskWrapper, MTCNet, \
    precompute_batches, RotOmniglot

np.random.seed(789)
torch.backends.cudnn.benchmark = True
torch.manual_seed(123)
torch.cuda.manual_seed(456)

# --
# Network

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-classes', type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()
    
    # --
    # IO
    
    stats = {
        "mean" : (0.07793742418289185,),
        "std"  : (0.2154727578163147,)
    }
    
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1 - x), # Make sparse
        transforms.Normalize(**stats),
    ])
    
    back_dataset = RotOmniglot(root='./data', background=True, transform=transform)
    test_dataset = RotOmniglot(root='./data', background=False, transform=transform)
    
    back_loader = torch.utils.data.DataLoader(
        back_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = precompute_batches(
        wrapper=OmniglotTaskWrapper(test_dataset), 
        num_epochs=1,
        num_steps=args.num_epochs,
        batch_size=20 * args.batch_size,
        num_classes=args.num_classes,
        num_shots=1
    )

    # --
    # Run

    num_classes = len(set(list(zip(*back_dataset._flat_character_images))[1]))

    net = MTCNet(num_classes=num_classes).cuda()
    opt = torch.optim.Adam(net.parameters())

    train_loss = []
    train_pred = []
    epoch_train_obs = len(back_dataset)
    
    for epoch in range(args.num_epochs):
        t = time()
        
        # Train
        _ = net.train()
        for X, y in tqdm(back_loader):
            X, y = X.cuda(), y.cuda()
            
            opt.zero_grad()
            out  = net.cls_forward(X)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            
            train_pred += list((out.argmax(dim=-1) == y.squeeze()).cpu().numpy())
            train_loss.append(float(loss))
        
        # Eval
        _ = net.eval()
        q, X, _ = next(test_loader)
        sim  = net.mtc_forward(q, X)
        test_acc = list((sim.argmax(dim=-1) == 0).cpu().numpy())
        
        # Log
        print(json.dumps({
            "train_loss" : np.mean(train_loss[-epoch_train_obs:]),
            "train_acc"  : np.mean(train_pred[-epoch_train_obs:]),
            "test_acc"   : np.mean(test_acc),
            "elapsed"    : time() - t,
        }))
        sys.stdout.flush()


