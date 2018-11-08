### mtcnet

#### Quickstart
```
conda create -n mtc_env python=3.6 pip -y
source activate mtc_env
conda install pytorch torchvision -c pytorch
```

#### Usage

See `run.sh` for usage.

#### Notes

The baseline classifier in `classifer.py` performs just as well as the matching network, contrary to the results reported in [Table 1 of the paper](https://arxiv.org/pdf/1606.04080.pdf). I'm trying to figure out an explanation for this.