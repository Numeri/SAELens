#!/usr/bin/env python
# coding: utf-8

import os

import torch
import wandb

from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

wandb.login()


def run_exp():
    wandb.init(project="sae_lens_tutorial")
    params = wandb.config
    total_training_steps = 60_000  # probably we should do more
    batch_size = 2048
    total_training_tokens = total_training_steps * batch_size

    lr_warm_up_steps = 0
    lr_decay_steps = total_training_steps // 5  # 20% of training
    l1_warm_up_steps = int(total_training_steps * (params["l1_warm_up_steps_percentage"] ** 3))  # 5% of training

    cfg = LanguageModelSAERunnerConfig(
        # Data Generating Function (Model + Training Distibuion)
        model_name="tiny-stories-1L-21M",  # our model (more options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html)
        hook_name="blocks.0.hook_mlp_out",  # A valid hook point (see more details here: https://neelnanda-io.github.io/TransformerLens/generated/demos/Main_Demo.html#Hook-Points)
        hook_layer=0,  # Only one layer in the model.
        d_in=1024,  # the width of the mlp output.
        dataset_path="apollo-research/roneneldan-TinyStories-tokenizer-gpt2",  # this is a tokenized language dataset on Huggingface for the Tiny Stories corpus.
        is_dataset_tokenized=True,
        streaming=True,  # we could pre-download the token dataset if it was small.
        # SAE Parameters
        mse_loss_normalization=None,  # We won't normalize the mse loss,
        expansion_factor=16,  # the width of the SAE. Larger will result in better stats but slower training.
        b_dec_init_method=params["b_dec_init_method"],  # The geometric median can be used to initialize the decoder weights.
        activation_fn=params["activation_fn"],
        apply_b_dec_to_input=False,  # We won't apply the decoder weights to the input.
        normalize_sae_decoder=False,
        scale_sparsity_penalty_by_decoder_norm=True,
        decoder_heuristic_init=params["decoder_heuristic_init"],
        init_encoder_as_decoder_transpose=params["init_encoder_as_decoder_transpose"],
        normalize_activations=params["normalize_activations"],
        # Training Parameters
        lr=params["lr"],  # lower the better, we'll go fairly high to speed up the tutorial.
        adam_beta1=0.9,  # adam params (default, but once upon a time we experimented with these.)
        adam_beta2=0.999,
        lr_scheduler_name="constant",  # constant learning rate with warmup. Could be better schedules out there.
        lr_warm_up_steps=lr_warm_up_steps,  # this can help avoid too many dead features initially.
        lr_decay_steps=lr_decay_steps,  # this will help us avoid overfitting.
        l1_coefficient=params["l1_coefficient"],  # will control how sparse the feature activations are
        l1_warm_up_steps=l1_warm_up_steps,  # this can help avoid too many dead features initially.
        lp_norm=params["lp_norm"],  # the L1 penalty (and not a Lp for p < 1)
        train_batch_size_tokens=batch_size,
        context_size=128,  # will control the lenght of the prompts we feed to the model. Larger is better but slower. so for the tutorial we'll use a short one.
        # Activation Store Parameters
        n_batches_in_buffer=16,  # controls how many activations we store / shuffle.
        act_store_device='cpu',
        training_tokens=total_training_tokens,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
        store_batch_size_prompts=16,
        # Resampling protocol
        use_ghost_grads=False,  # we don't use ghost grads anymore.
        feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
        dead_feature_window=1000,  # would effect resampling or ghost grads if we were using it.
        dead_feature_threshold=1e-4,  # would effect resampling or ghost grads if we were using it.
        # WANDB
        log_to_wandb=True,  # always use wandb unless you are just testing code.
        wandb_project="sae_lens_tutorial",
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=20,
        # Misc
        device=device,
        seed=42,
        n_checkpoints=0,
        checkpoint_path="checkpoints",
        dtype="bfloat16"
    )
    SAETrainingRunner(cfg).run()


sweep_config = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "metrics/l0"},
    "parameters": {
        "b_dec_init_method": {"values": ["zeros", "mean", "geometric_median"]},
        "activation_fn": {"values": ["relu", "tanh-relu"]},
        "decoder_heuristic_init": {"values": [True, False]},
        "init_encoder_as_decoder_transpose": {"values": [True, False]},
        "normalize_activations": {"values": ["none", "expected_average_only_in", "constant_norm_rescale"]},
        "lr": {"values": [5e-5, 1e-5, 5e-6, 1e-6, 5e-7]},
        "l1_coefficient": {"max": 10, "min": 1},
        "lp_norm": {"max": 3.0, "min": 0.5},
        "l1_warm_up_steps_percentage": {"max": 1.0, "min": 0.05 ** (1/3)}
    },
}

# sweep_id = wandb.sweep(sweep=sweep_config, project="sae_lens_tutorial")
sweep_id = "v7g9usy6"
wandb.agent(sweep_id=sweep_id, function=run_exp, count=300)
