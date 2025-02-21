import argparse
import json
import time
import os
import tqdm

import numpy as np

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from transformers import AutoTokenizer


def random_edit(
    data,
    args=None,
):

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.hf_cache_dir)
    kmeans_label = np.load(f"./results/kmean.npy") # size 50265
    # print(kmeans_label.shape)

    replace_lookup = []

    for i in range(len(kmeans_label)):
        label1 = np.where(kmeans_label == kmeans_label[i])[0]
        label2 = label1[label1 != i]
        replace_lookup.append(label2)

    # iterate over data and tokenize each instance
    w_wm_output_attacked = []
    for idx, dd in tqdm.tqdm(enumerate(data), total=len(data)):
        # tokenize prefix
        if "w_wm_output_attacked" not in dd:
            if args.no_wm_attack:
                if isinstance(dd["no_wm_output"], str):
                    input_gen = dd["no_wm_output"].strip()
                else:
                    input_gen = dd["no_wm_output"][0].strip()
            else:
                if isinstance(dd["w_wm_output"], str):
                    input_gen = dd["w_wm_output"].strip()
                else:
                    input_gen = dd["w_wm_output"][0].strip()
        
        input_tokens = tokenizer(input_gen)['input_ids']
        new_input_tokens = []
        # print(input_tokens)
        for token_id, token in enumerate(input_tokens):
            if token == tokenizer.bos_token or token == tokenizer.eos_token:
                new_input_tokens.append(token)
                continue
            rand_coin = np.random.rand()
            if rand_coin < args.random_edit_fraction:
                if rand_coin < args.random_edit_fraction * 0.5:
                    # remove it
                    continue
                else:
                    new_input_tokens.append(int(np.random.choice(replace_lookup[token])))
            else:
                new_input_tokens.append(token)

        # import pdb ; pdb.set_trace()

        output_text = tokenizer.decode(new_input_tokens)

        # print(output_text)
        # import pdb ; pdb.set_trace()

        w_wm_output_attacked.append(output_text.strip())

        # with open(output_file, "a") as f:
        #     f.write(json.dumps(dd) + "\n")
    # add w_wm_output_attacked to hf dataset object as a column
    data = data.add_column("w_wm_output_attacked", w_wm_output_attacked)

    # import pdb ; pdb.set_trace()

    return data
