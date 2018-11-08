#!/bin/bash

# run.sh

mkdir -p results

# 5-way / 1-shot (0.980 in paper)
python mtc.py | tee results/res-05.jl

# 20-way / 1-shot
python mtc.py | tee results/res-20.jl

