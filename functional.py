import torch
import _functions
# from modules import L2Norm, MultiBoxLoss

def l2norm(input, weight, n_channels=512, scale=20):
    f = L2Norm(n_channels,scale)
    return f(input, weight)
