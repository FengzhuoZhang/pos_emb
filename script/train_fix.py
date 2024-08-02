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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

#import wandb

def seed_everything(TORCH_SEED):
	random.seed(TORCH_SEED)
	os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
	np.random.seed(TORCH_SEED)
	torch.manual_seed(TORCH_SEED)
	torch.cuda.manual_seed_all(TORCH_SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def train_step(model, xs, ys, optimizer, loss_func, modal,train_every_step):

    optimizer.zero_grad()
    if modal == 'concat':
        zs = xy_concat(xs,ys)
    elif modal == 'interweave':
        zs = interweave(xs,ys)
    else:
        print("Unknown modal")
        raise NotImplementedError
    #import ipdb; ipdb.set_trace()
    output, _ = model(zs, read_y=True)
    if train_every_step:
        loss = loss_func(output, ys)
    else:
        loss = loss_func(output[:,-1],ys[:,-1])
    loss.backward()
    #import ipdb; ipdb.set_trace()

    grad_dict = dict()
    for name, param in model.named_parameters():
        grad_dict[name] = param.grad
    #print(grad_dict.keys())
    #import ipdb; ipdb.set_trace()
    optimizer.step()
    return loss.detach().item(), output.detach()

def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds

def train(model,args):


    # import ipdb; ipdb.set_trace()
    # parameter_group = [
    #     {
    #         "param": [],
    #         "lr": args.training.learning_rate,
    #     },
    #     {
    #         "param": [],
    #         "lr": args.training.learning_rate * 0.
    #     }
    # ]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.95, patience=1000, verbose=True, min_lr=1e-5)
    #lr_scheduler = CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-4)

    n_points = args.model.n_positions
    n_dims = args.model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, zero_pad = args.model.zero_pad, n_dims=n_dims)
    task_sampler = get_task_sampler(
        "linear_regression",
        n_dims,
        bsize,
        zero_pad = args.model.zero_pad,
        num_tasks=None,
        **args.training.task_kwargs,
    )
    starting_step = 0
    pbar = tqdm(range(starting_step, args.training.train_steps))
    data_sampler_args = {}
    task_sampler_args = {}
    xs = data_sampler.sample_xs(
            n_points,
            bsize,
            **data_sampler_args,
        )
    task = task_sampler(**task_sampler_args)
    ys = task.evaluate(xs)


    loss_func = torch.nn.MSELoss()

    for i in pbar:
        loss, output = train_step(model, xs.cuda(), ys.cuda(), optimizer, loss_func, args.model.modal, args.model.train_every_step)
        lr_scheduler.step(loss)
        #lr_scheduler.step()
        if i % 1000 == 0:
            # print("\n".join([
            #     f"loss: {loss}",
            #     f"lk.bias: {model.blocks[0].lk.bias.norm()}",
            #     f"lq.bias: {model.blocks[0].lq.bias.norm()}",
            #     f"lv.bias: {model.blocks[0].lv.bias.norm()}",
            #     f"fc.bias: {model.blocks[0].fc.bias.norm()}",
            #     f"lr: {optimizer.param_groups[0]['lr']}",
            # ]))
            print("\n".join([f"loss: {loss}",
                             f"lr: {optimizer.param_groups[0]['lr']}"]))
            #import ipdb; ipdb.set_trace()
            if not args.test_run:
                training_state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "train_step": i,
                    "temp_loss": loss,
                }
                torch.save(training_state, os.path.join(args.out_dir, f"model_{i}.pt"))



def main(args):
    random_seed = random.randint(0,1000) #909
    print(random_seed)
    seed_everything(random_seed)
    if args.test_run:
        args.training.train_steps = 100000
    else:
        # wandb.init(
        #     dir=args.out_dir,
        #     project=args.wandb.project,
        #     entity=args.wandb.entity,
        #     config=args.__dict__,
        #     notes=args.wandb.notes,
        #     name=args.wandb.name,
        #     resume=True,
        # )
        seed_info = {
            "seed": random_seed,
            "simu": "fix_training_set"
        }
        torch.save(seed_info, os.path.join(args.out_dir, f"seed_info.pt"))

    if args.model.zero_pad or args.model.modal == 'concat':
        model = MultiLayerTransformer(n_dims = args.model.n_dims+1, layers = args.model.layers, pos_emb = args.model.pos_emb,causal = args.model.causal, output_mix = args.model.output_mix)
    else:
        model = MultiLayerTransformer(n_dims = args.model.n_dims, layers = args.model.layers, pos_emb = args.model.pos_emb,causal = args.model.causal, output_mix = args.model.output_mix)
    
    for name, param in model.named_parameters():
        mod_type = name.split('.')[-1]
        if mod_type == 'bias':
            torch.nn.init.constant_(param, 0)
    model.cuda()
    #import ipdb; ipdb.set_trace()
    #model = build_model(args.model)
    #model.train()
    #import ipdb; ipdb.set_trace()

    train(model, args)



if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    #import ipdb; ipdb.set_trace()

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            #run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)


