# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
from functools import partial
from tqdm import tqdm
import wandb
import torch

from huggingface_hub import login
login()

print(f"Current huggingface cache dir: {os.environ['HF_HOME']}")

# HF classses
from transformers import LogitsProcessorList, DataCollatorWithPadding

# better bool flag type for argparse
from utils.submitit import str2bool

# some file i/o helpers
from utils.io import write_jsonlines, write_json

# watermarking functionality
from watermark_processor import WatermarkLogitsProcessor

# generation pipeline helpers
from utils.generation import (
    MAX_GENERATIONS,
    load_model,
    load_hf_dataset,
    check_input_lengths,
    check_output_lengths,
    tokenize_for_generation,
    generate,
    generate_embedding_pairs,
    load_semantic_model,
)


def main(args):
    ###########################################################################
    # Start logging
    ###########################################################################
    # storing slurm info to allow auditing logfiles later
    args.SLURM_JOB_ID = os.getenv("SLURM_JOB_ID")
    args.SLURM_ARRAY_JOB_ID = os.getenv("SLURM_ARRAY_JOB_ID")
    args.SLURM_ARRAY_TASK_ID = os.getenv("SLURM_ARRAY_TASK_ID")

    if args.wandb:
        # start a new wandb run to track this experiment, will send data to it later
        run = wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.run_name}",
            # track hyperparameters and run metadata
            config=args,
            tags=args.wandb_tags,
        )

    ###########################################################################
    # Create the output dir
    ###########################################################################
    print(f"Output dir for this run: {args.output_dir}")
    # notify if exists
    if os.path.exists(args.output_dir):
        print(f"Output dir for this run already exists!")
        print(f"Contents: {sorted(os.listdir(args.output_dir))}")
    else:
        # create the output dir where run artifacts are stored
        os.makedirs(args.output_dir)

    ###########################################################################
    # Load the dataset
    ###########################################################################
    # basic ops like shuffling and select are done in load fn
    dataset = load_hf_dataset(args)

    # import pdb ; pdb.set_trace()
    model, sem_tokenizer, device = load_model(args)
    # model.requires_grad(False)
    sem_model = model.get_decoder()
    # sem_model.requires_grad(False)
    # import pdb ; pdb.set_trace()

    generate_embedding_pairs_partial = partial(generate_embedding_pairs, sem_model=sem_model, tokenizer=sem_tokenizer, device=device, args=args,)

    sem_embedding_dataset = dataset.map(generate_embedding_pairs_partial, batched=False)

    def group_batch(examples):
        return {k: [v] for k, v in examples.items()}
    sem_embedding_batch_dataset = sem_embedding_dataset.map(group_batch, batched=True, batch_size=args.generation_batch_size)

    ###########################################################################
    # Main loop - actually executes the generation pipeline.
    # and accumulates the result rows in a list, assumes list is "small"-ish
    # and we aren't accumulating any tensors or other memory hogging artifacts
    ###########################################################################

    processed_examples = []
    # ds_iterator = iter(sem_embedding_batch_dataset)
    i = 0
    total_steps = 0
    pbar = tqdm(total=args.min_generations)
    import torch.optim as optim
    from utils.contrastive import CLModel, contrastive_train_batch, infoNCE_loss
    from torch.utils.data import DataLoader
    cl_mlp = CLModel(args.encoder_dim, feat_dim=args.cl_mlp_feat_dim).to(device)

    optimizer = optim.Adam(cl_mlp.parameters(), lr=args.cl_lr, weight_decay=1e-6)
    # optimizer = optim.SGD(cl_mlp.parameters(), lr=0.05, weight_decay=1e-4, momentum=0.9)

    if not os.path.exists(args.cl_savepath):
        os.makedirs(args.cl_savepath)

    for epoch_id in range(args.cl_epochs):
        ds_iterator = iter(sem_embedding_batch_dataset)
        while True:
            try:
                ex = next(ds_iterator)
                total_steps += 1
                # i += 1
                # pbar.update(1)
            except StopIteration:
                break

            # print(ex["sentence_embeddings"][0].shape)
            print("Generating training pair for one batch.")
            # import pdb; pdb.set_trace()
            pos_1 = torch.cat(ex["sentence_embeddings"], dim=1).squeeze(0)
            print(pos_1.shape)
            pos_2 = pos_1 + torch.normal(0, 0.01, pos_1.shape).to(device)
            loss, batch_size = contrastive_train_batch(cl_mlp, pos_1.detach(), pos_2.detach(), optimizer, temperature=0.5)
            emb_loss, _ = infoNCE_loss(pos_1, pos_2, temperature=0.5)
            print(total_steps, loss / batch_size, emb_loss / batch_size) # 2560
            
            if total_steps < 500:
                if total_steps % 100 == 0:
                    torch.save(cl_mlp.state_dict(), "{}/cl_model_01_{}.pt".format(args.cl_savepath, total_steps))
            else:
                if total_steps % 5000 == 0:
                    torch.save(cl_mlp.state_dict(), "{}/cl_model_01_{}.pt".format(args.cl_savepath, total_steps))
        
    pbar.close()
    return  # reload in separate script for metric measurement


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run watermarked huggingface LM generation pipeline"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-1.3b",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=True,
        help="Whether to run model in float16 precsion.",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="c4",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--json_data_dir",
        type=str,
        default="/mnt/home/renjie3/Documents/wmllm/data",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default="/egr/research-dselab/renjie3/renjie/LLM/cache",
    )
    parser.add_argument(
        "--cl_lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--cl_savepath",
        type=str,
        default="./results/cl_model",
    )
    parser.add_argument(
        "--cl_k",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--cl_epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--encoder_dim",
        type=int,
        default=2560,
    )
    parser.add_argument(
        "--cl_mlp_feat_dim",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--cl_mean_pooling",
        type=str2bool,
        default=False,
        help="Whether to stream the dataset from the web or download it locally.",
    )
    parser.add_argument(
        "--cl_pooling_method",
        type=str,
        default='mean',
        help="Whether to stream the dataset from the web or download it locally.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="realnewslike",
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="The split of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--stream_dataset",
        type=str2bool,
        default=True,
        help="Whether to stream the dataset from the web or download it locally.",
    )
    parser.add_argument(
        "--columns_to_remove",
        type=str,
        default=None,
        help="Comma separated list of columns to remove from the dataset before generation.",
    )
    parser.add_argument(
        "--shuffle_dataset",
        type=str2bool,
        default=False,
        help="Whether to shuffle the dataset before sampling.",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=1234,
        help="The seed to use for dataset shuffle op.",
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=10_000,
        help="The buffer size to use for dataset shuffle op - takes n rows first, then shuffles those indices",
    )
    parser.add_argument(
        "--prompt_id",
        type=int,
        default=0,
        help="If the dataset supports multiple instruction prompts, denotes which one to use. 0 is default/no prompt.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="The number of tokens to generate using the model, and the num tokens removed from real text sample",
    )
    parser.add_argument(
        "--min_prompt_tokens",
        type=int,
        default=50,  # 500
        help="The number of examples (first N) to process from the dataset.",
    )
    parser.add_argument(
        "--min_sample_tokens",
        type=int,
        default=0,
        help="The the minimum length of raw prompt samples to consider.",
    )
    parser.add_argument(
        "--limit_indices",
        type=int,
        default=None,
        help="The number of examples (first N) to pull from the dataset, if None, pull all, and then set this arg to the number of rows in the dataset.",
    )
    parser.add_argument(
        "--min_generations",
        type=int,
        default=500,
        help="The minimum number of valid generations according to the output check strat to sample.",
    )
    parser.add_argument(
        "--input_truncation_strategy",
        type=str,
        default="completion_length",
        choices=["no_truncation", "completion_length", "prompt_length"],
        help="The strategy to use when tokenizing and truncating raw inputs to make prompts.",
    )
    parser.add_argument(
        "--input_filtering_strategy",
        type=str,
        default="completion_length",
        choices=["no_filter", "completion_length", "prompt_length", "prompt_and_completion_length"],
        help="The strategy to use when tokenizing and truncating raw inputs to make prompts.",
    )
    parser.add_argument(
        "--output_filtering_strategy",
        type=str,
        default="no_filter",
        choices=["no_filter", "max_new_tokens"],
        help=(
            f"The strategy to use when filtering/skipping rows if the model didn't ",
            f"generate enough tokens to facilitate analysis.",
        ),
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=False,
        help=("Whether to perform sampling during generation. (non-greedy decoding)"),
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7,
        help="The temperature to use when generating using multinom sampling",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="The top k to use when generating using top_k version of multinom sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="The top p to use when generating using top_p version of sampling",
    )
    parser.add_argument(
        "--typical_p",
        type=float,
        default=1.0,
        help="The typical p to use when generating using typical decoding version of multinom sampling",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="The number of beams to use where '1' is no beam search.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=None,
        help="Seed for setting the torch rng prior to generation using any decoding scheme with randomness.",
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=4,
        help="The batch size to use for generation.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="The seeding procedure to use for the watermark.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="The ratio of tokens to put in the greenlist when splitting the vocabulary",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="The amount of bias (absolute) to add to the logits in the whitelist half of the vocabulary at every step",
    )
    parser.add_argument(
        "--store_spike_ents",
        type=str2bool,
        default=True,
        help=("Whether to store the spike entropies while generating with watermark processor. "),
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=False,
        help="Whether to log the generations to stdout.",
    )
    parser.add_argument(
        "--wandb",
        type=str2bool,
        default=False,
        help="Whether to log to wandb.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="lm-watermarking",
        help="The name of the wandb project.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="jwkirchenbauer",
        help="The wandb entity/user for the project.",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default="",
        help="The comma separated list of tags to add to the wandb run.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="The unique name for the run.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="The unique name for the run.",
    )
    parser.add_argument(
        "--overwrite",
        type=str2bool,
        default=False,
        help="Allow overwriting of old generation files at the same output location.",
    )
    args = parser.parse_args()

    ###########################################################################
    # Argument validation and conditional setting
    ###########################################################################
    # for removing some columns to save space
    args.columns_to_remove = args.columns_to_remove.split(",") if args.columns_to_remove else []

    # if decoding scheme is not sampling, then set generation seed to None
    # to avoid confusion and calling the torch rng unnecessarily
    args.generation_seed = args.generation_seed if args.use_sampling else None

    # -1 value for min_generations means no specified minimum
    # with the assumption that the
    if args.min_generations <= 0:
        args.min_generations = MAX_GENERATIONS
        print(
            f"Warning: min_generations is -1. A hardcoded value of {MAX_GENERATIONS} will be used to limit the generation loop."
        )

    if args.limit_indices is None:
        print("No limit_indices specified, pulling all examples from the dataset.")
    else:
        print(f"Limiting iteration to {args.limit_indices} examples from the dataset.")

    # split wandb tags
    if args.wandb_tags != "":
        args.wandb_tags = args.wandb_tags.split(",")
    else:
        args.wandb_tags = []

    main(args)
