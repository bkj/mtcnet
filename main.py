#!/usr/bin/env python

"""
    main.py
    
    * Super simple implementation of matching networks
    * Only works for 1-shot ATM
    * Adds a linear layer to network
    * No rotation
    * Not sure data split is the same as in original paper
"""

import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from torchvision.datasets import Omniglot
from torchvision import transforms

from rsub import *
from matplotlib import pyplot as plt


class OmniglotTaskWrapper:
    def __init__(self, dataset, rotation=False):
        self.dataset  = dataset
        # self.rotation = rotation
        
        self.lookup = defaultdict(list)
        for idx,(f,lab) in enumerate(dataset._flat_character_images):
            self.lookup[lab].append(idx)
        
        self.classes = list(self.lookup.keys())
        
    def sample_task(self, num_classes, num_shots):
        classes = np.random.choice(self.classes, num_classes, replace=False)
        
        task = []
        for i, lab in enumerate(classes):
            is_target = int(i == 0)
            idxs = np.random.choice(self.lookup[lab], num_shots + is_target, replace=False)
            for idx in idxs:
                task.append(self.dataset[idx])
        
        task = list(zip(*task))[0]
        task = torch.cat(task).unsqueeze(1)
        
        return task[:1], task[1:]
    
    def sample_tasks(self, batch_size, num_classes, num_shots):
        # !! This should happen on a separate thread, I think 
        # Could probably implement a DataLoader that yields tasks
        q, X = [], []
        for task in range(batch_size):
            q_, X_ = self.sample_task(num_classes=num_classes, num_shots=num_shots)
            q.append(q_)
            X.append(X_)
        
        q, X = torch.stack(q).cuda(), torch.stack(X).cuda()
        y = torch.LongTensor([0] * batch_size).cuda()
        
        return q, X, y


class Net(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        
        self.in_channels = in_channels
        self.sz         = 28
        self.hidden_dim = 64
        
        self.layers = nn.Sequential(
            self._make_layer(in_channels=1, out_channels=self.hidden_dim),
            self._make_layer(in_channels=self.hidden_dim, out_channels=self.hidden_dim),
            self._make_layer(in_channels=self.hidden_dim, out_channels=self.hidden_dim),
            self._make_layer(in_channels=self.hidden_dim, out_channels=self.hidden_dim),
        )
        self.linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
    
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
        x = x.view(bs * nobs, self.in_channels, self.sz, self.sz)
        x = self.layers(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear(x) # often speeds up convergence
        x = F.normalize(x, dim=-1)
        x = x.view(bs, nobs, self.hidden_dim)
        return x
    
    def forward(self, q, X):
        X = self._batch_forward(X)
        q = self._batch_forward(q)
        
        sim = torch.bmm(q, X.transpose(2, 1))
        sim = sim.squeeze(1)
        return sim

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


# --
# Run

num_epochs  = 10
num_steps   = 100

batch_size  = 32
num_classes = 5

back_wrapper = OmniglotTaskWrapper(back_dataset)
test_wrapper = OmniglotTaskWrapper(test_dataset)

net = Net().cuda()
opt = torch.optim.Adam(net.parameters())

train_loss = []
train_pred = []

for epoch in range(num_epochs):
    
    _ = net.train()
    for step in tqdm(range(num_steps)):
        opt.zero_grad()
        q, X, y = back_wrapper.sample_tasks(batch_size=batch_size, num_classes=num_classes, num_shots=1)
        sim = net(q, X)
        loss = F.cross_entropy(sim, y)
        loss.backward()
        opt.step()
        
        train_pred += list((sim.argmax(dim=-1) == 0).cpu().numpy())
        train_loss.append(float(loss))
    
    _ = net.eval()
    q, X, _ = test_wrapper.sample_tasks(batch_size=10 * batch_size, num_classes=num_classes, num_shots=1)
    sim = net(q, X)
    test_pred = list((sim.argmax(dim=-1) == 0).cpu().numpy())
    
    print({
        "train_loss" : np.mean(train_loss[-100:]),
        "train_acc"  : np.mean(train_pred[-100:]),
        "test_acc"   : np.mean(test_pred),
    })
    _ = net.train()
    
    # _ = plt.plot(train_loss)
    # show_plot()


