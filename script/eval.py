import json
import os
import sys

from munch import Munch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import yaml

from data_generation.tasks import get_task_sampler
from data_generation.samplers import get_data_sampler

def interweave(xs_b,ys_b):
    bsize, points, dim = xs_b.shape
    ys_b_wide = torch.cat(
        (
            torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ys_b.view(bsize, points, 1),
        ),
        axis=2,
    )
    #import ipdb; ipdb.set_trace()
    zs = torch.stack((xs_b, ys_b_wide), dim=2)
    zs = zs.view(bsize, 2 * points, dim)
    return zs

def xy_concat(xs_b,ys_b):
    bsize, points, dim = xs_b.shape 
    zs = torch.cat([ys_b.view(bsize, points, 1),xs_b],-1)
    return zs

def eval_batch(model, n_dims,n_points=50,bsize=100000, eval_every_step = True, zero_pad = True, modal = 'interweave'):
    data_sampler = get_data_sampler("gaussian", n_dims=n_dims,zero_pad = zero_pad)
    task_sampler = get_task_sampler(
        "linear_regression",
        n_dims,
        bsize,
        zero_pad = zero_pad
    )
    task = task_sampler()
    xs = data_sampler.sample_xs(
            n_points,
            bsize
        )
    ys = task.evaluate(xs)
    if modal == 'interweave':
        zs = interweave(xs.cuda(),ys.cuda())
    elif modal == 'concat':
        zs = xy_concat(xs.cuda(),ys.cuda())
    else:
        print("Unknown modal")
        raise NotImplementedError
    with torch.no_grad():
        output, _ = model(zs, read_y=True)
        if eval_every_step:
            loss = output-ys.cuda()
            loss = torch.mean(loss.square(),0)
            total_loss = torch.mean(loss)
        else:
            loss = output[:,-1]-ys[:,-1].cuda()
            loss = torch.mean(loss.square(),0)
            total_loss = torch.mean(loss)



    return total_loss, loss
