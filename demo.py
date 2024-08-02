import torch
from models.networks import SingleLayerTransformer, MultiLayerTransformer, SinPosEmb, RoPosEmb
from data_generation.samplers import get_data_sampler
from data_generation.tasks import get_task_sampler

from models.rotary_embedding_torch import (
    RotaryEmbedding,
    apply_rotary_emb
)



# n_dims = 5
# pos_emb = ''

# model = SingleLayerTransformer(n_dims=n_dims, pos_emb=pos_emb, causal = True, output_mix= True)

# input = torch.ones(20,n_dims)
# output, logs = model(input,read_y=True)
# #print(output)

n_dims = 6
batch_size = 2
data_sampler = get_data_sampler("gaussian", n_dims)
#xs = data_sampler.sample_xs(b_size=batch_size, n_points=20)
xs = torch.randn(batch_size, 3, n_dims)
n_dims_add = n_dims
#import ipdb; ipdb.set_trace()
sin_emb = RoPosEmb(d_hid = n_dims_add)
print(xs)
x_p = sin_emb(xs)
print(x_p)


#task_sampler = get_task_sampler(
#     conf.training.task,
#     n_dims,
#     batch_size,
#     **conf.training.task_kwargs
# )

# task = task_sampler()
# xs = data_sampler.sample_xs(b_size=batch_size, n_points=conf.training.curriculum.points.end)
# ys = task.evaluate(xs)