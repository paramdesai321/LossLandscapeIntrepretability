#!/bin/bash
jupyter nbconvert --to script train_resnet56.ipynb
PYTORCH_ENABLE_MPS_FALLBACK=1 python -u train_resnet56.py
