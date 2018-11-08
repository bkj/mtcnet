#!/usr/bin/env python

"""
    main.py
    
    * Super simple implementation of matching networks
    * Only works for 1-shot ATM
    * No rotation
    * Not sure data split is the same as in original paper
"""

import sys
import json
import argparse
import numpy as np
from time import time

import torch
from torch.nn import functional as F

from torchvision import transforms
from torchvision.datasets import Omniglot

from helpers import OmniglotTaskWrapper, MTCNet, precompute_batches

np.random.seed(789)
torch.backends.cudnn.benchmark = True
torch.manual_seed(123)
torch.cuda.manual_seed(456)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--num-steps', type=int, default=100)
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

    back_dataset = Omniglot(root='./data', background=True, transform=transform)
    test_dataset = Omniglot(root='./data', background=False, transform=transform)

    back_wrapper = OmniglotTaskWrapper(back_dataset)
    test_wrapper = OmniglotTaskWrapper(test_dataset)

    back_loader  = precompute_batches(
        wrapper=back_wrapper,
        num_epochs=args.num_epochs,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        num_shots=1
    )

    test_loader  = precompute_batches(
        wrapper=test_wrapper, 
        num_epochs=1,
        num_steps=args.num_epochs,
        batch_size=10 * args.batch_size,
        num_classes=args.num_classes,
        num_shots=1
    )

    # --
    # Run

    net = MTCNet().cuda()
    opt = torch.optim.Adam(net.parameters())

    train_loss = []
    train_correct = []
    epoch_train_obs = args.num_steps * args.batch_size
    
    for epoch in range(args.num_epochs):
        t = time()
        
        # Train
        _ = net.train()
        for step in range(args.num_steps):
            opt.zero_grad()
            q, X, y = next(back_loader)
            sim  = net.mtc_forward(q, X)
            loss = F.cross_entropy(sim, y)
            loss.backward()
            opt.step()
            
            train_correct += list((sim.argmax(dim=-1) == 0).cpu().numpy())
            train_loss.append(float(loss))
        
        # Eval
        _ = net.eval()
        q, X, _ = next(test_loader)
        sim  = net.mtc_forward(q, X)
        test_correct = list((sim.argmax(dim=-1) == 0).cpu().numpy())
        
        # Log
        print(json.dumps({
            "train_loss" : np.mean(train_loss[-epoch_train_obs:]),
            "train_acc"  : np.mean(train_correct[-epoch_train_obs:]),
            "test_acc"   : np.mean(test_correct),
            "elapsed"    : time() - t,
        }))
        sys.stdout.flush()


