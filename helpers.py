
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import torch
from torch import nn
from torch.nn import functional as F

from torchvision.datasets import Omniglot

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
def precompute_batches(wrapper, num_epochs, num_steps, batch_size, num_classes, num_shots, max_workers=16):
    """ Fancy function to parallelize batch creation """
    global _fn
    seed_ = (1 + np.random.choice(int(1e6)))
    def _fn(seed):
        np.random.seed(seed_ + seed)
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



class RotOmniglot(Omniglot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.num_classes = len(set(list(zip(*self._flat_character_images))[1]))
        
    def __getitem__(self, index):
        new_index = index % len(self._flat_character_images)
        rotation  = index // len(self._flat_character_images)
        
        img, character_class = super().__getitem__(new_index)
        
        img = img.numpy()
        img = np.rot90(img, k=rotation, axes=(1, 2))
        img = np.ascontiguousarray(img)
        img = torch.Tensor(img)

        new_class = (self.num_classes * rotation) + character_class
        
        return img, new_class

# --
# Networks

class MTCNet(nn.Module):
    def __init__(self, num_classes=0, in_channels=1, img_sz=28, hidden_dim=64):
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
        
        if num_classes > 0:
            self.num_classes = num_classes
            self.linear = nn.Linear(self.hidden_dim, self.num_classes)
    
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

    def mtc_forward(self, q, X):
        X = self._batch_forward(X)
        q = self._batch_forward(q)
        X = F.normalize(X, dim=-1) # !! This line may be missing from original implementation
        sim = torch.bmm(q, X.transpose(1, 2)).squeeze()
        return sim
    
    def cls_forward(self, x):
        x = self.layers(x).squeeze()
        x = self.linear(x)
        return x
