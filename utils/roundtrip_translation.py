import argparse
import json
import nltk
import time
import os
import tqdm

from nltk.tokenize import sent_tokenize

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from transformers import MarianMTModel, MarianTokenizer

nltk.download("punkt")


def roundtrip_translation_paraphrases(
    data,
    args=None,
):
    # opus-mt-en-ru
    tokenizer1 = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-en-{args.mid_lang}", cache_dir=args.hf_cache_dir)
    model1 = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-en-{args.mid_lang}", cache_dir=args.hf_cache_dir)
    tokenizer2 = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{args.mid_lang}-en", cache_dir=args.hf_cache_dir)
    model2 = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{args.mid_lang}-en", cache_dir=args.hf_cache_dir)

    model1.cuda()
    model1.eval()

    model2.cuda()
    model2.eval()

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

            final_input = tokenizer1([input_gen], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}
            with torch.inference_mode():
                outputs = model1.generate(
                    **final_input, do_sample=True, top_p=0.75, top_k=None, max_length=512
                )
            middle_outputs = tokenizer1.batch_decode(outputs, skip_special_tokens=True)

            final_input = tokenizer2(middle_outputs, return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}
            with torch.inference_mode():
                outputs = model2.generate(
                    **final_input, do_sample=True, top_p=0.75, top_k=None, max_length=512
                )
            output_text = tokenizer2.batch_decode(outputs, skip_special_tokens=True)[0]

            # print(output_text)

            w_wm_output_attacked.append(output_text.strip())

        # with open(output_file, "a") as f:
        #     f.write(json.dumps(dd) + "\n")
    # add w_wm_output_attacked to hf dataset object as a column
    data = data.add_column("w_wm_output_attacked", w_wm_output_attacked)

    # import pdb ; pdb.set_trace()

    return data
