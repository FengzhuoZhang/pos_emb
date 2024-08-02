import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

import sys
sys.path.append('/home/aiops/zhangfz/llm/pos_emb')

from data_generation.tasks import get_task_sampler
from data_generation.samplers import get_data_sampler
from script.schema import schema
from script.eval import eval_batch, interweave, xy_concat
from models.networks import SingleLayerTransformer, MultiLayerTransformer

import wandb

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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)

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

    num_training_examples = None

    loss_func = torch.nn.MSELoss()

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]
        xs = data_sampler.sample_xs(
            n_points,
            bsize,
            **data_sampler_args,
        )
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)
        loss, output = train_step(model, xs.cuda(), ys.cuda(), optimizer, loss_func, args.model.modal, args.model.train_every_step)

        if i % 1000 == 0:
            total_loss, loss = eval_batch(model, n_dims, eval_every_step = args.model.train_every_step)
            print(f"total loss is {total_loss}")
            print(f"Every step losses are {loss}")

            # training_state = {
            #     "model_state_dict": model.state_dict(),
            #     "optimizer_state_dict": optimizer.state_dict(),
            #     "train_step": i,
            # }
            # torch.save(training_state, os.path.join(args.out_dir, f"model_{i}.pt"))



def main(args):
    if args.test_run:
        args.training.train_steps = 500001
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    if args.model.zero_pad or args.model.modal == 'concat':
        model = MultiLayerTransformer(n_dims = args.model.n_dims+1, layers = args.model.layers, pos_emb = args.model.pos_emb,causal = args.model.causal, output_mix = args.model.output_mix)
    else:
        model = MultiLayerTransformer(n_dims = args.model.n_dims, layers = args.model.layers, pos_emb = args.model.pos_emb,causal = args.model.causal, output_mix = args.model.output_mix)
    model.cuda()
    #import ipdb; ipdb.set_trace()
    #model = build_model(args.model)
    #model.train()

    train(model, args)



if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    #import ipdb; ipdb.set_trace()

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)


