from quinine import (
    tstring,
    tinteger,
    tfloat,
    tboolean,
    stdict,
    tdict,
    default,
    required,
    allowed,
    nullable,
)
from funcy import merge


POS_EMB_LIST = [
    "",
    "sin",
    "rope",
]
MODAL_LIST = [
    "concat",
    "interweave",
]

model_schema = {
    "n_positions": merge(tinteger, required),  # maximum context length
    "n_dims": merge(tinteger, required),  # latent dimension
    "layers": merge(tinteger, required),  # latent dimension
    "output_mix": merge(tboolean, default(True)),
    "causal": merge(tboolean, default(True)),
    "pos_emb": merge(tstring, allowed(POS_EMB_LIST)),
    "zero_pad": merge(tboolean, default(True)),
    "modal": merge(tstring, allowed(MODAL_LIST)),
    "train_every_step": merge(tboolean, required),
}

training_schema = {
    "task_kwargs": merge(tdict, required),
    "data": merge(tstring, allowed(["gaussian"])),
    "batch_size": merge(tinteger, default(64)),
    "learning_rate": merge(tfloat, default(3e-4)),
    "train_steps": merge(tinteger, default(1000)),
    "save_every_steps": merge(tinteger, default(1000)),  # how often to checkpoint
    "keep_every_steps": merge(tinteger, default(-1)),  # permanent checkpoints
    "resume_id": merge(tstring, nullable, default(None)),  # run uuid64
}

wandb_schema = {
    "project": merge(tstring, default("pos_emb")),
    "entity": merge(tstring, default("pos")),
    "notes": merge(tstring, default("")),
    "name": merge(tstring, nullable, default(None)),
    "log_every_steps": merge(tinteger, default(10)),
}

schema = {
    "out_dir": merge(tstring, required),
    "model": stdict(model_schema),
    "training": stdict(training_schema),
    "wandb": stdict(wandb_schema),
    "test_run": merge(tboolean, default(False)),
}
