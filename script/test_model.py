import os
from random import randint
import uuid
import datetime

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml
import random
import numpy as np

import sys
sys.path.append('/home/aiops/zhangfz/llm/pos_emb')

from data_generation.tasks import get_task_sampler
from data_generation.samplers import get_data_sampler
from script.schema import schema
from script.eval import eval_batch, interweave, xy_concat
from models.networks import SingleLayerTransformer, MultiLayerTransformer


model_para_add = '/home/aiops/zhangfz/llm/pos_emb/results/2024-05-08T07-20-58/model_99000.pt'
model_info = torch.load(model_para_add)
config_add = '/home/aiops/zhangfz/llm/pos_emb/results/2024-05-08T07-20-58/config.yaml'
with open(config_add, 'r', encoding='utf-8') as fin:
    args = yaml.load(fin, Loader=yaml.FullLoader)

if args['model']['zero_pad'] or args['model']['modal'] == 'concat':
    model = MultiLayerTransformer(n_dims = args['model']['n_dims']+1, layers = args['model']['layers'], pos_emb = args['model']['pos_emb'],causal = args['model']['causal'], output_mix = args['model']['output_mix'])
else:
    model = MultiLayerTransformer(n_dims = args['model']['n_dims'], layers = args['model']['layers'], pos_emb = args['model']['pos_emb'],causal = args['model']['causal'], output_mix = args['model']['output_mix'])


model.load_state_dict(model_info['model_state_dict'])
model.to('cuda')
total_loss, loss = eval_batch(model, n_dims = args['model']['n_dims'],n_points=args['model']['n_positions'], eval_every_step = args['model']['train_every_step'],zero_pad = args['model']['zero_pad'],modal = args['model']['modal'])
print(f"total loss is {total_loss}")
print(f"Every step losses are {loss}")