import argparse
import json
import nltk
import time
import os
import tqdm

from nltk.tokenize import sent_tokenize

from transformers import BartForConditionalGeneration, BartTokenizer
import torch

nltk.download("punkt")

def bart_paraphrase(
    data,
    args=None,
):
    model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase', cache_dir=args.hf_cache_dir)
    model = model.cuda()
    tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase', cache_dir=args.hf_cache_dir)
    # batch = tokenizer(input_sentence, return_tensors='pt')
    # generated_ids = model.generate(batch['input_ids'])
    # generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # iterate over data and tokenize each instance
    w_wm_output_attacked = []
    for idx, dd in tqdm.tqdm(enumerate(data), total=len(data)):
        # tokenize prefix
        if "w_wm_output_attacked" not in dd:
            # paraphrase_outputs = {}

            # print("check args.no_wm_attack", args.no_wm_attack)
            # import pdb; pdb.set_trace()
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

            final_input = tokenizer([input_gen], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}
            with torch.inference_mode():
                outputs = model.generate(
                    **final_input, max_length=300, min_length=175
                )
            output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

            # print(output_text)

            w_wm_output_attacked.append(output_text.strip())

        # with open(output_file, "a") as f:
        #     f.write(json.dumps(dd) + "\n")
    # add w_wm_output_attacked to hf dataset object as a column
    data = data.add_column("w_wm_output_attacked", w_wm_output_attacked)

    # import pdb ; pdb.set_trace()

    return data
