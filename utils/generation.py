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

import torch
import torch.nn.functional as F

# HF classes

from datasets import load_dataset, IterableDataset

from .contrastive import CLModel

from torch import Tensor
from tokenizers import Tokenizer

from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModel,
    DataCollatorWithPadding,
)

from .data.lfqa import load_lfqa
from .data.essays import load_essays
from .data.wikitext import load_wikitext

MAX_GENERATIONS = int(10000)  # Hardcoded max length to avoid infinite loop


def load_model(args):
    """Load and return the model and tokenizer"""

    args.is_seq2seq_model = any(
        [(model_type in args.model_name_or_path) for model_type in ["t5", "T0"]]
    )
    args.is_decoder_only_model = any(
        [(model_type in args.model_name_or_path) for model_type in ["gpt", "opt", "bloom", "llama"]]
    )
    if args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, cache_dir=args.hf_cache_dir)
    elif args.is_decoder_only_model:
        if args.load_fp16:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, torch_dtype=torch.float16, device_map="auto", cache_dir=args.hf_cache_dir
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, cache_dir=args.hf_cache_dir, device_map="auto", use_auth_token=True) # use_auth_token=True
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.load_fp16:
            pass
        else:
            model = model.to(device)
    else:
        device = "cpu"
    # device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model.eval()

    if args.is_decoder_only_model:
        padding_side = "left"
    else:
        raise NotImplementedError(
            "Need to check how to handle padding for seq2seq models when calling generate"
        )

    if "llama" in args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(
            args.model_name_or_path, padding_side=padding_side, cache_dir=args.hf_cache_dir,
        )
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, padding_side=padding_side, cache_dir=args.hf_cache_dir
        )

    # # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, cache_dir="/egr/research-dselab/renjie3/renjie/LLM/cache").to(device)
    # # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir="/egr/research-dselab/renjie3/renjie/LLM/cache")

    # # gen_kwargs.update(output_hidden_states=True)
    # test_sample = ["Beginners BBQ Class Taking Place in Missoula! Do you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers. He will be teaching a beginner level class for everyone who wants to get better with their culinary skills. He will teach you everything you need to know to compete in a KCBS BBQ competition, including"]

    # test_input_ids = tokenizer(test_sample, return_tensors="pt").to(device)
    # # print(test_input_ids)
    # output = model.generate(**test_input_ids, max_length=1024)
    # print(len(output[0]))
    # ex = tokenizer.batch_decode(output)
    # print(ex)
    # import pdb; pdb.set_trace()

    args.model_max_length = model.config.max_position_embeddings

    # print(type(model))
    # import pdb ; pdb.set_trace() transformers.models.llama.modeling_llama.LlamaForCausalLM

    return model, tokenizer, device

def load_semantic_model(args):
    """Load and return the model and tokenizer"""

    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.load_fp16:
            pass
        else:
            model = model.to(device)
    else:
        device = "cpu"

    # #Mean Pooling - Take attention mask into account for correct averaging
    # def mean_pooling(model_output, attention_mask):
    #     token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    #     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    #     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', cache_dir=args.hf_cache_dir)
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', cache_dir=args.hf_cache_dir).to(device)

    # # Tokenize sentences
    # encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # # Compute token embeddings
    # with torch.no_grad():
    #     model_output = model(**encoded_input)

    # # Perform pooling
    # sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # # Normalize embeddings
    # sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return model, tokenizer, device

def load_mlp_model(args):
    """Load and return the model and tokenizer"""

    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.load_fp16:
            raise("cl_mlp is not emplemented with fp16")
        else:
            pass
    else:
        device = "cpu"

    if "llama" in args.cl_mlp_model_path or "6.7b" in args.cl_mlp_model_path:
        args.cl_encoder_dim = 4096
    else:
        args.cl_encoder_dim = 2560
    model = CLModel(encoder_dim=args.cl_encoder_dim)
    model.load_state_dict(torch.load(args.cl_mlp_model_path))
    model = model.to(device)
    model.eval()
    # print(args.cl_mlp_model_path)
    # print(model)
    # input("check")

    return model


def add_idx(example, idx):
    example.update({"idx": idx})
    return example


def load_hf_dataset(args):
    dataset_name, dataset_config_name = args.dataset_name, args.dataset_config_name

    if dataset_name == "lfqa":
        dataset = load_lfqa(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": False,
                "input_col_name": "prefix",
                "ref_output_col_name": "gold_completion",
            }
        )
        # other args set within the load_lfqa function
    elif dataset_name == "wikitext":
        dataset = load_wikitext(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": True,
                "input_col_name": "text",
                "ref_output_col_name": None,
            }
        )
        # other args set within the load_wikitext function
    elif dataset_name == "essays":
        dataset = load_essays(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": False,
                "input_col_name": "instructions",
                "ref_output_col_name": "essays",
            }
        )
    elif dataset_name == "cml_pile":
        subsets = [dataset_config_name]
        dataset = load_dataset(
            "./data/cml_pile.py",
            subsets=subsets,
            streaming=args.stream_dataset,
            split=None,
            ignore_verifications=True,
        )[args.dataset_split]
        args.__dict__.update(
            {
                "truncate_input_for_prompt": True,
                "input_col_name": "text",
                "ref_output_col_name": None,
            }
        )
    elif "json" in dataset_name:
        dataset = load_dataset(
            args.json_data_dir,
            data_files="c4-train.000*-of-00512.json",
            split=args.dataset_split,
            streaming=args.stream_dataset,
        )
        if "c4" in dataset_name:
            args.__dict__.update(
                {
                    "truncate_input_for_prompt": True,
                    "input_col_name": "text",
                    "ref_output_col_name": None,
                }
            )
            args.columns_to_remove = list(
                set(args.columns_to_remove + ["text", "timestamp", "url"])
            )
        elif "pile" in dataset_name:
            args.__dict__.update(
                {
                    "truncate_input_for_prompt": True,
                    "input_col_name": "text",
                    "ref_output_col_name": None,
                }
            )
            args.columns_to_remove = list(set(args.columns_to_remove + ["text", "meta"]))
        else:
            raise NotImplementedError(
                f"Dataset {dataset_name} not yet supported. Please add specs to load_hf_dataset function."
            )
    else:
        dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            split=args.dataset_split,
            streaming=args.stream_dataset,
        )
        if "c4" in dataset_name:
            args.__dict__.update(
                {
                    "truncate_input_for_prompt": True,
                    "input_col_name": "text",
                    "ref_output_col_name": None,
                }
            )
            args.columns_to_remove = list(
                set(args.columns_to_remove + ["text", "timestamp", "url"])
            )
        elif "pile" in dataset_name:
            args.__dict__.update(
                {
                    "truncate_input_for_prompt": True,
                    "input_col_name": "text",
                    "ref_output_col_name": None,
                }
            )
            args.columns_to_remove = list(set(args.columns_to_remove + ["text", "meta"]))
        else:
            raise NotImplementedError(
                f"Dataset {dataset_name} not yet supported. Please add specs to load_hf_dataset function."
            )

    # add index to each row of dataset
    indexed_dataset = dataset.map(add_idx, batched=False, with_indices=True)

    # shuffle the first shuffle_buffer_size rows of streaming dataset, or whole dataset if not streaming
    # and take/select only the first n rows of the dataset (which caps the total number of pipeline iters possible)
    if isinstance(indexed_dataset, IterableDataset):
        shuffled_dataset = (
            indexed_dataset.shuffle(seed=args.shuffle_seed, buffer_size=args.shuffle_buffer_size)
            if args.shuffle_dataset
            else indexed_dataset
        )
        limited_dataset = (
            shuffled_dataset.take(args.limit_indices)
            if args.limit_indices is not None
            else shuffled_dataset
        )
    else:
        shuffled_dataset = (
            indexed_dataset.shuffle(seed=args.shuffle_seed)
            if args.shuffle_dataset
            else indexed_dataset
        )
        limited_dataset = (
            shuffled_dataset.select(range(args.limit_indices))
            if args.limit_indices is not None
            else shuffled_dataset
        )

    if args.limit_indices is None:
        try:
            args.limit_indices = len(limited_dataset)
        except Exception as e:
            # can't infer length of dataset, probably because it's an IterableDataset
            pass
    return limited_dataset


def check_input_lengths(
    example,
    min_sample_len=0,
    min_prompt_len=0,
    min_completion_len=0,
    max_input_len=None,
    max_new_tokens=None,
):
    orig_sample_length = example["orig_sample_length"]
    prompt_length = example["prompt_length"]
    real_completion_length = example["baseline_completion_length"]

    if max_input_len is not None:
        assert (
            max_new_tokens is not None
        ), "need to specify max_new_tokens if max_input_length is specified"

    conds = all(
        [
            orig_sample_length >= min_sample_len,
            prompt_length >= min_prompt_len,
            real_completion_length >= min_completion_len,
            (
                ((prompt_length + max_new_tokens) <= max_input_len)
                if max_input_len is not None
                else True
            ),
        ]
    )
    return conds


def check_output_lengths(example, min_output_len=0):
    # FIXME, maybe should check baseline completion length too
    no_wm_output_len = example["no_wm_output_length"]
    w_wm_output_len = example["w_wm_output_length"]
    conds = all(
        [
            no_wm_output_len >= min_output_len,
            w_wm_output_len >= min_output_len,
        ]
    )
    return conds


def tokenize_and_truncate(
    example: dict,
    input_col_name: str = "text",
    completion_length: int = None,
    prompt_length: int = None,
    hf_model_name: str = None,
    tokenizer=None,
    truncate_left=False,
    model_max_length=None,
):
    """take hf dataset entry and preprocess it for completion by a model"""
    assert hf_model_name is not None, "need model name to know whether to adjust wrt special tokens"
    assert input_col_name in example, f"expects {input_col_name} field to be present"
    # tokenize
    inputs_ids = tokenizer(example[input_col_name], return_tensors="pt")["input_ids"]
    example.update({"untruncated_inputs": inputs_ids})

    if truncate_left:
        # truncate left
        inputs_ids = inputs_ids[:, -model_max_length:]
        if example["untruncated_inputs"].shape != inputs_ids.shape:
            print(
                "Input too long for model! ",
                "Left truncating under assumption that this is the prompt+output ",
                "to be fed to the *oracle* model",
            )
        example.update({"untruncated_inputs": inputs_ids})

    if (completion_length is not None) and (prompt_length is None):
        # leave at least one token as prefix # FIXME I think plus 1 since 0 is start tok
        slice_length = min(inputs_ids.shape[1] - 1, completion_length)
    elif (prompt_length is not None) and (completion_length is None):
        desired_comp_len = (inputs_ids.shape[1] - 1) - prompt_length
        slice_length = desired_comp_len if desired_comp_len > 0 else 0
    else:
        raise ValueError(
            (
                f"Can only tokenize and truncate based on either the desired prompt length or desired completion length,",
                f" but got completion_length:{completion_length},prompt_length:{prompt_length}",
            )
        )

    # truncate
    inputs_ids = inputs_ids[:, : inputs_ids.shape[1] - slice_length]
    # logic depending on special tokens for the model
    if "t5" in hf_model_name or "T0" in hf_model_name:
        inputs_ids[0, -1] = 1
    # else: pass
    example.update({"input_ids": inputs_ids})
    return example


def tokenize_only(
    example: dict,
    input_col_name: str = "text",
    ref_output_col_name: str = None,
    tokenize_ref_output: bool = False,
    hf_model_name: str = None,
    tokenizer=None,
    model_max_length=None,
):
    """take hf dataset entry and preprocess it for completion by a model
    (but don't truncate) where the dataset optionally has a secondary column
    that is the reference output to be scored against"""

    """take hf dataset entry and preprocess it for completion by a model"""
    assert hf_model_name is not None, "need model name to know whether to adjust wrt special tokens"
    assert input_col_name in example, f"expects {input_col_name} field to be present"
    if ref_output_col_name is not None:
        assert ref_output_col_name in example, f"expects {ref_output_col_name} field to be present"

    # tokenize input
    input_ids = tokenizer(
        example[input_col_name], return_tensors="pt", truncation=True, max_length=model_max_length
    )["input_ids"]

    example.update({"input_ids": input_ids})

    if tokenize_ref_output:
        # NOTE not sure this logic is useful/required
        if ref_output_col_name is not None:
            # tokenize ref output
            ref_output_ids = tokenizer(
                example[ref_output_col_name],
                return_tensors="pt",
                truncation=True,
                max_length=model_max_length,
            )["input_ids"]

        tokd_input_len, tokd_ref_output_length = input_ids.shape[1], ref_output_ids.shape[1]
        if tokd_input_len + tokd_ref_output_length > model_max_length:
            # truncate the ref output
            original_ref_output_len = tokd_ref_output_length
            ref_output_ids = ref_output_ids[:, : model_max_length - tokd_input_len]
            if original_ref_output_len != ref_output_ids.shape[1]:
                print(
                    "Right truncating output, input+ref output too long for model. "
                    "Note, since this is generation time truncating the reference doesn't affect anything really."
                )
        example.update({"ref_output_ids": ref_output_ids})

    # logic depending on special tokens for the model
    if "t5" in hf_model_name or "T0" in hf_model_name:
        raise NotImplementedError("T5 style model not yet supported")

    return example


def tokenize_for_generation(
    example: dict,
    max_new_tokens: int = None,
    min_prompt_tokens: int = None,
    hf_model_name: str = None,
    tokenizer: Tokenizer = None,
    args: dict = None,
):
    # preprocessing, generation & scoring
    assert isinstance(example, dict), "Expect no batch dimension currently!"

    if not args.truncate_input_for_prompt:
        tokenize_ref_output = True  # NOTE, note really sure how necessary this is
        # preprocess for model generation/completion
        example = tokenize_only(
            example,
            input_col_name=args.input_col_name,
            ref_output_col_name=args.ref_output_col_name,
            hf_model_name=hf_model_name,
            tokenizer=tokenizer,
            model_max_length=args.model_max_length,
            tokenize_ref_output=tokenize_ref_output,
        )
        # Parse the results of tokenization. Simple, since
        # the prompt and baseline completion are from the raw text
        re_decoded_input = example[args.input_col_name]
        decoded_baseline_completion = example[args.ref_output_col_name]
        prompt_len = example["input_ids"].shape[1]
        baseline_completion_len = example["ref_output_ids"].shape[1]
        full_sample_len = prompt_len + baseline_completion_len
        # for now, remove this here, since it's not used downstream
        example.pop("ref_output_ids")
    else:
        # preprocess for model generation/completion
        example = tokenize_and_truncate(
            example,
            completion_length=max_new_tokens,
            prompt_length=min_prompt_tokens,
            hf_model_name=hf_model_name,
            tokenizer=tokenizer,
        )
        # Logic to parse the results of tokenzation and splitting to
        # construct string versions of the prompt and baseline completion
        inputs = example["input_ids"]
        prompt_len = inputs.shape[1]
        # for isolating the "gold" baseline completion
        untruncated_inputs = example.pop("untruncated_inputs")
        full_sample_len = untruncated_inputs.shape[1]
        # decode the preprocessed input to store for audit
        re_decoded_input = tokenizer.batch_decode(inputs, skip_special_tokens=True)[0]
        # also decode the original suffix of the input for audit as the baseline
        baseline_completion_tokens = untruncated_inputs[:, inputs.shape[-1] :]
        decoded_baseline_completion = tokenizer.batch_decode(
            baseline_completion_tokens, skip_special_tokens=True
        )[0]
        baseline_completion_len = full_sample_len - prompt_len

    example.update(
        {
            "truncated_input": re_decoded_input,
            "baseline_completion": decoded_baseline_completion,
            "orig_sample_length": full_sample_len,
            "prompt_length": prompt_len,
            "baseline_completion_length": baseline_completion_len,
        }
    )
    return example


def collate_batch(input_ids: list, collator: DataCollatorWithPadding = None):
    """collate batch of input_ids into a padded batch of tensors"""
    assert (
        input_ids[0].shape[0] == 1 and input_ids[0].shape[1] > 0
    ), "expecting batch dimension of each tensor to be 1"
    # remove batch dimension for each tensor
    input_ids = [x.squeeze(0) for x in input_ids]
    return collator({"input_ids": input_ids})["input_ids"]


def generate(
    examples,
    data_collator=None,
    generate_without_watermark=None,
    generate_with_watermark=None,
    watermark_processor=None,
    tokenizer=None,
    device=None,
    args=None,
):
    # print(examples.keys())

    # test_sample = ["Beginners BBQ Class Taking Place in Missoula! Do you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers. He will be teaching a beginner level class for everyone who wants to get better with their culinary skills. He will teach you everything you need to know to compete in a KCBS BBQ competition, including"]

    # test_input_ids = tokenizer(test_sample, return_tensors="pt").to(device)
    # # print(test_input_ids)
    # output = generate_without_watermark(**test_input_ids)
    # print(output[0])
    # ex = tokenizer.batch_decode(output)
    # print(ex)
    # import pdb; pdb.set_trace()

    input_ids = collate_batch(input_ids=examples["input_ids"], collator=data_collator).to(device)
    # print(examples.keys())
    # print(examples['input_ids'])
    # print(examples['input_ids'].shape)
    # print(examples['truncated_input'][0])
    # print(examples['prompt_length'])

    # print(input_ids.shape)
    # print(input_ids[0].tolist())
    # # input("check")
    # import pdb; pdb.set_trace()
    

    with torch.no_grad():
        if args.generation_seed is not None:
            torch.manual_seed(args.generation_seed)
        output_without_watermark_output_dict = generate_without_watermark(input_ids=input_ids)
        output_without_watermark = output_without_watermark_output_dict.sequences

        # return output_without_watermark

        if args.generation_seed is not None:
            torch.manual_seed(args.generation_seed)
        output_with_watermark_output_dict = generate_with_watermark(input_ids=input_ids)
        output_with_watermark = output_with_watermark_output_dict.sequences
        # print(output_with_watermark_output_dict.keys())
        # print(type(output_with_watermark))
        # input("check print(type(output_with_watermark))")
        # import pdb; pdb.set_trace()
        # print("fixme")
        # output_with_watermark = output_with_watermark

    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:, input_ids.shape[-1] :]
        output_with_watermark = output_with_watermark[:, input_ids.shape[-1] :]

    decoded_output_without_watermark = tokenizer.batch_decode(
        output_without_watermark, skip_special_tokens=True
    )
    decoded_output_with_watermark = tokenizer.batch_decode(
        output_with_watermark, skip_special_tokens=True
    )

    # print(output_with_watermark_output_dict.wm_green.tolist())
    # import pdb; pdb.set_trace()
    examples.update(
        {
            "no_wm_output": decoded_output_without_watermark,
            "w_wm_output": decoded_output_with_watermark,
            "no_wm_output_length": (output_without_watermark != tokenizer.pad_token_id)
            .sum(dim=-1)
            .tolist(),
            "w_wm_output_length": (output_with_watermark != tokenizer.pad_token_id)
            .sum(dim=-1)
            .tolist(),
        }
    )

    if 'sem' in args.seeding_scheme:
        examples.update(
            {
                "w_wm_output_green": output_with_watermark_output_dict.wm_green.tolist(),
                "w_wm_output_seed": output_with_watermark_output_dict.wm_seed.tolist(),
                "generation_prompt": input_ids.tolist(),
            }
        )

    # import pdb; pdb.set_trace()

    if watermark_processor.spike_entropies is not None:
        examples["spike_entropies"] = watermark_processor._get_and_clear_stored_spike_ents()
        examples["spike_entropies"] = [
            ents[:num_toks]
            for ents, num_toks in zip(examples["spike_entropies"], examples["w_wm_output_length"])
        ]

    return examples

def generate_embedding_pairs(
    examples,
    sem_model=None,
    data_collator=None,
    watermark_processor=None,
    tokenizer=None,
    device=None,
    args=None,
):
    # input_ids = collate_batch(input_ids=examples["text"], collator=data_collator).to(device)
    # print(len(examples["text"]))
    # print(input_ids)
    # input("check")

    tokenized_input = tokenizer(examples["text"], return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
    # print(type(tokenized))
    # print(tokenized_input.keys())
    # print(tokenized_input["input_ids"].shape)
    # print(tokenized_input["token_type_ids"].shape)
    # input("check")

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Compute token embeddings
    with torch.no_grad():
        model_output = sem_model(**tokenized_input)

    # print(type(model_output))
    # print(model_output.keys())
    # print(type(model_output[0]))
    # print((model_output["past_key_values"]))
    # Finally I found which is the logits used to predict the next token https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/generation/utils.py#L3269 https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L956

    # print(model_output[0].shape)
    # print(tokenized_input)
    # import pdb; pdb.set_trace()

    selected_id = torch.randint(model_output[0].shape[1], (8,)) # Training on truncated sentences.

    mean_pooling_list = []

    if args.cl_mean_pooling:
        if args.cl_pooling_method == 'mean':
            for _id in selected_id:
                mean_pooling_list.append(torch.sum(model_output[0][:, :1+_id, :], dim=1) / (_id+1))
            sentence_embeddings = torch.stack(mean_pooling_list, dim=1)

        elif args.cl_pooling_method == 'former_k':
            for _id in selected_id:
                k = args.cl_k
                start_id = max(0, 1+_id - k)
                mean_pooling_list.append(torch.sum(model_output[0][:, start_id:1+_id, :], dim=1) / (_id+1 - start_id))
            sentence_embeddings = torch.stack(mean_pooling_list, dim=1)

        elif args.cl_pooling_method == 'weighted_former_k':
            for _id in selected_id:
                k = args.cl_k
                start_id = max(0, 1+_id - k)
                accumulated_emb = 0
                all_weight_count = 0
                for j in range(start_id, 1+_id):
                    weight = j - start_id + 1 + k // 2
                    all_weight_count += weight
                    accumulated_emb += model_output[0][:, j, :] * weight
                mean_pooling_list.append(accumulated_emb / all_weight_count)
            sentence_embeddings = torch.stack(mean_pooling_list, dim=1)

        else:
            raise("Wrong cl_pooling_method!")
    else:
        sentence_embeddings = model_output[0][:, selected_id, :]
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=-1)
    
    # print(torch.norm(sentence_embeddings, p=2, dim=1))
    # print(sentence_embeddings.shape)
    # import pdb; pdb.set_trace()

    # print(sentence_embeddings.shape)
    # input("check")

    examples.update(
        {
            "sentence_embeddings": sentence_embeddings.detach()
        }
    )

    # return {k: [v] for k, v in examples.items()}
    return examples

def generate_embedding_pairs_for_rebuttal(
    examples,
    sem_model=None,
    data_collator=None,
    watermark_processor=None,
    tokenizer=None,
    device=None,
    args=None,
):
    # input_ids = collate_batch(input_ids=examples["text"], collator=data_collator).to(device)
    # print(len(examples["text"]))
    # print(input_ids)
    # input("check")

    wm_tokenized_input = tokenizer(examples["w_wm_output"], return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
    nowm_tokenized_input = tokenizer(examples["no_wm_output"], return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
    # print(wm_tokenized_input.shape)
    # print(nowm_tokenized_input.shape)

    size = 8

    if wm_tokenized_input.input_ids.shape[1] < nowm_tokenized_input.input_ids.shape[1] - 2 or wm_tokenized_input.input_ids.shape[1] > nowm_tokenized_input.input_ids.shape[1] + 2:

        examples = {
                "wm_sentence_embeddings": torch.zeros(1, size, 2560).detach(),
                "nowm_sentence_embeddings": torch.zeros(1, size, 2560).detach(),
            }

        # print("check here")

        # return {k: [v] for k, v in examples.items()}
        return examples
    
    # Example tensor
    tensor = wm_tokenized_input.input_ids  # Replace with your tensor

    # Determine 5% of elements
    num_elements = tensor.size(1)
    num_to_remove = int(num_elements * 0.1) 
    # num_to_remove = 0

    # Create a mask with 5% False (elements to remove)
    mask = torch.ones(num_elements, dtype=torch.bool)
    mask_indices = torch.randperm(num_elements)[:num_to_remove]
    mask[mask_indices] = False

    # Create an index tensor
    index_tensor = torch.arange(num_elements)

    # Mark the indices of the removed elements as -1
    index_tensor[~mask] = -1

    # Apply the mask to the original tensor
    nowm_tokenized_input.input_ids = tensor[:, mask]
    nowm_tokenized_input.attention_mask = wm_tokenized_input.attention_mask[:, mask]

    # import pdb ; pdb.set_trace()

    # import pdb ; pdb.set_trace()
    
    # print(type(tokenized))
    # print(tokenized_input.keys())
    # print(tokenized_input["input_ids"].shape)
    # print(tokenized_input["token_type_ids"].shape)
    # input("check")

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # # Compute token embeddings
    # with torch.no_grad():
    #     model_output = sem_model(**tokenized_input)

    # print(type(model_output))
    # print(model_output.keys())
    # print(type(model_output[0]))
    # print((model_output["past_key_values"]))
    # Finally I found which is the logits used to predict the next token https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/generation/utils.py#L3269 https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L956

    # print(model_output[0].shape)
    # print(tokenized_input)
    # import pdb; pdb.set_trace()

    selected_id = torch.randint(0, 170, (size,)) # Training on truncated sentences.

    try: 
        # Compute token embeddings
        with torch.no_grad():
            model_output = sem_model(wm_tokenized_input.input_ids)
            wm_model_output = model_output[0].detach()

        # print(selected_id)

        mean_pooling_list = []

        if args.cl_mean_pooling:
            if args.cl_pooling_method == 'weighted_former_k':
                for _id in selected_id:
                    if index_tensor[_id] < 0:
                        continue
                    k = args.cl_k
                    start_id = max(0, 1+_id - k)
                    accumulated_emb = 0
                    all_weight_count = 0
                    for j in range(start_id, 1+_id):
                        weight = j - start_id + 1 + k // 2
                        all_weight_count += weight
                        accumulated_emb += model_output[0][:, j, :] * weight
                        # print(j, weight, accumulated_emb[0,0], model_output[0][0, j, 0], wm_model_output[0, j, 0])
                    mean_pooling_list.append(accumulated_emb / all_weight_count)
                sentence_embeddings = torch.stack(mean_pooling_list, dim=1)

            else:
                raise("Wrong cl_pooling_method!")
        wm_sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=-1)

        # Compute token embeddings
        with torch.no_grad():
            model_output = sem_model(nowm_tokenized_input.input_ids)
            nowm_model_output = model_output[0].detach()

        # print(selected_id)

        mean_pooling_list = []

        if args.cl_mean_pooling:
            if args.cl_pooling_method == 'weighted_former_k':
                for old_id in selected_id:
                    if index_tensor[old_id] < 0:
                        continue
                    _id = old_id - torch.sum(index_tensor[:old_id] < 0)
                    k = args.cl_k - torch.sum(index_tensor[old_id - args.cl_k:old_id] < 0)
                    start_id = max(0, 1+_id - k)
                    accumulated_emb = 0
                    all_weight_count = 0
                    for j in range(start_id, 1+_id):
                        # if index_tensor[j] < 0:
                        #     continue
                        weight = j - start_id + 1 + k // 2
                        all_weight_count += weight
                        accumulated_emb += model_output[0][:, j, :] * weight
                        # print(j, weight, accumulated_emb[0,0], model_output[0][0, j, 0], nowm_model_output[0, j, 0])
                    mean_pooling_list.append(accumulated_emb / all_weight_count)
                sentence_embeddings = torch.stack(mean_pooling_list, dim=1)

            else:
                raise("Wrong cl_pooling_method!")
        # else:
        #     sentence_embeddings = model_output[0][:, selected_id, :]
        nowm_sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=-1)
        
        # print(torch.norm(sentence_embeddings, p=2, dim=1))
        # print(sentence_embeddings.shape)
        # import pdb; pdb.set_trace()

        # print(sentence_embeddings.shape)
        # input("check")

        examples = {
                "wm_sentence_embeddings": wm_sentence_embeddings.detach(),
                "nowm_sentence_embeddings": nowm_sentence_embeddings.detach(),
            }
        # print(torch.sum(wm_sentence_embeddings.detach()))
        # print(torch.sum(nowm_sentence_embeddings.detach()))
        # import pdb ; pdb.set_trace()
        # examples = {
        #         "wm_sentence_embeddings": torch.zeros(1, 8, 2560).detach(),
        #         "nowm_sentence_embeddings": torch.zeros(1, 8, 2560).detach(),
        #     }
        # import pdb ; pdb.set_trace()
    except:
        # examples.update(
        #     {
        #         "wm_sentence_embeddings": torch.zeros(10, 10).detach(),
        #         "nowm_sentence_embeddings": torch.zeros(10, 10).detach(),
        #     }
        # )

        examples = {
                "wm_sentence_embeddings": torch.zeros(1, size, 2560).detach(),
                "nowm_sentence_embeddings": torch.zeros(1, size, 2560).detach(),
            }

        # return {k: [v] for k, v in examples.items()}
        return examples

    # return {k: [v] for k, v in examples.items()}
    return examples
