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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import torch
from torch import nn
from torch.nn import functional as F

from torchvision import transforms
from torchvision.datasets import Omniglot

np.random.seed(789)
torch.backends.cudnn.benchmark = True
torch.manual_seed(123)
torch.cuda.manual_seed(456)

# --
# Data generators

class OmniglotTaskWrapper:
    def __init__(self, dataset):
        self.dataset  = dataset
        
        self.lookup = defaultdict(list)
        for idx,(f,lab) in enumerate(dataset._flat_character_images):
            self.lookup[lab].append(idx)
        
        self.classes = list(self.lookup.keys())
        
    def sample_task(self, num_classes, num_shots):
        classes = np.random.choice(self.classes, num_classes, replace=False)
        
        task = []
        for i, lab in enumerate(classes):
            rotation = np.random.choice((0, 1, 2, 3))
            is_target = int(i == 0)
            idxs = np.random.choice(self.lookup[lab], num_shots + is_target, replace=False)
            for idx in idxs:
                img = self.dataset[idx][0]
                img = self._rotate(img, rotation) 
                task.append(img)
        
        task = torch.cat(task).unsqueeze(1)
        
        return task[:1], task[1:]
    
    def _rotate(self, img, rotation):
        img = img.numpy()
        img = np.rot90(img, k=rotation, axes=(1, 2))
        img = np.ascontiguousarray(img)
        return torch.Tensor(img)
    
    def sample_tasks(self, batch_size, num_classes, num_shots):
        q, X = [], []
        for task in range(batch_size):
            q_, X_ = self.sample_task(num_classes=num_classes, num_shots=num_shots)
            q.append(q_)
            X.append(X_)
        
        q, X = torch.stack(q), torch.stack(X)
        y = torch.LongTensor([0] * batch_size)
        
        return q, X, y


_fn = None
def make_batches(wrapper, num_epochs, num_steps, batch_size, num_classes, num_shots, max_workers=16):
    """ Fancy function to parallelize batch creation """
    global _fn
    def _fn(seed):
        np.random.seed(111 * seed)
        out = []
        for _ in range(num_steps):
            out.append(wrapper.sample_tasks(
                batch_size=batch_size,
                num_classes=num_classes,
                num_shots=num_shots
            ))
        
        return out
            
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for batches in ex.map(_fn, range(num_epochs)):
            for q, X, y in batches:
                yield q.cuda(), X.cuda(), y.cuda()
# --
# Network

class Net(nn.Module):
    def __init__(self, in_channels=1, img_sz=28, hidden_dim=64):
        super().__init__()
        
        self.in_channels = in_channels
        self.img_sz      = img_sz
        self.hidden_dim  = hidden_dim
        
        self.layers = nn.Sequential(
            self._make_layer(in_channels=1, out_channels=self.hidden_dim),
            self._make_layer(in_channels=self.hidden_dim, out_channels=self.hidden_dim),
            self._make_layer(in_channels=self.hidden_dim, out_channels=self.hidden_dim),
            self._make_layer(in_channels=self.hidden_dim, out_channels=self.hidden_dim),
        )
    
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
    
    def _batch_forward(self, x):
        bs   = x.shape[0]
        nobs = x.shape[1]
        
        x = x.view(bs * nobs, self.in_channels, self.img_sz, self.img_sz)
        x = self.layers(x).squeeze()
        x = x.view(bs, nobs, self.hidden_dim)
        
        return x

    def forward(self, q, X):
        X = self._batch_forward(X)
        q = self._batch_forward(q)
        X = F.normalize(X, dim=-1)
        sim = torch.bmm(q, X.transpose(1, 2)).squeeze()
        return sim



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

    back_loader  = make_batches(
        wrapper=back_wrapper,
        num_epochs=args.num_epochs,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        num_shots=1
    )

    test_loader  = make_batches(
        wrapper=test_wrapper, 
        num_epochs=1,
        num_steps=args.num_epochs,
        batch_size=10 * args.batch_size,
        num_classes=args.num_classes,
        num_shots=1
    )

    # --
    # Run

    net = Net().cuda()
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
            sim  = net(q, X)
            loss = F.cross_entropy(sim, y)
            loss.backward()
            opt.step()
            
            train_correct += list((sim.argmax(dim=-1) == 0).cpu().numpy())
            train_loss.append(float(loss))
        
        # Eval
        _ = net.eval()
        q, X, _ = next(test_loader)
        sim  = net(q, X)
        test_correct = list((sim.argmax(dim=-1) == 0).cpu().numpy())
        
        # Log
        print(json.dumps({
            "train_loss" : np.mean(train_loss[-epoch_train_obs:]),
            "train_acc"  : np.mean(train_correct[-epoch_train_obs:]),
            "test_acc"   : np.mean(test_correct),
            "elapsed"    : time() - t,
        }))
        sys.stdout.flush()


