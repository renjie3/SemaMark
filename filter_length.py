from utils.io import read_jsonlines, read_json, write_jsonlines
import numpy as np
import math

from transformers import AutoTokenizer

import argparse

parser = argparse.ArgumentParser(
        description="Run watermarked huggingface LM generation pipeline"
    )
parser.add_argument("--input_file", type=str, default=None,)

args = parser.parse_args()

gen_table_path = args.input_file

output_gen_table_path = gen_table_path.replace(".jsonl", "_filtered.jsonl")

gen_table_lst = [ex for ex in read_jsonlines(gen_table_path)]

print(len(gen_table_lst))

new_gen_table_lst = []

# tokenizer = AutoTokenizer.from_pretrained(
#         "facebook/opt-2.7b", padding_side="left", cache_dir="/egr/research-dselab/renjie3/renjie/LLM/cache"
#     )

# for ex in gen_table_lst:
#     tokens = tokenizer.encode(ex["w_wm_output_attacked"])
#     if len(tokens) < 10 or ex["w_wm_output_length"] < 10: # no_wm_output_length
#         continue
#     new_gen_table_lst.append(ex)

# write_jsonlines(new_gen_table_lst, output_gen_table_path)

for ex in gen_table_lst:
    if "w_wm_output_attacked_length" in ex:
        if ex["no_wm_output_length"] < 10 or ex["w_wm_output_length"] < 10 or ex["w_wm_output_attacked_length"] < 10: # no_wm_output_length w_wm_output_attacked_length
            continue
        new_gen_table_lst.append(ex)
    else:
        if ex["no_wm_output_length"] < 10 or ex["w_wm_output_length"] < 10: # no_wm_output_length w_wm_output_attacked_length
            continue
        new_gen_table_lst.append(ex)

write_jsonlines(new_gen_table_lst, output_gen_table_path)
