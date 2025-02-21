#!/bin/bash

# Script to run the generation, attack, and evaluation steps of the pipeline

# requires some OUTPUT_DIR to be set in the environment
# as well as a path to the hf format LLAMA model
# /egr/research-dselab/renjie3/renjie/LLM/cache test

# conda activate wmllm

OUTPUT_DIR="./results"

CL_MODEL_PATH="./results/cl_model_01_10000.pt"
CL_K=30
DISCRETE_VALUE=5

RUN_NAME=opt_2.7b_sem_kmeans_dipper_test_smallsamplesize

GENERATION_OUTPUT_DIR="$OUTPUT_DIR"/"$RUN_NAME"

BASE_PYTHON="This is a python where transformers is not modifed for evluation"

DEVICE=0

# HF_CACHE_DIR="/mnt/home/renjie3/.cache/huggingface"
HF_CACHE_DIR="/egr/research-dselab/renjie3/renjie/LLM/cache"
JSON_DATA_DIR="/egr/research-dselab/renjie3/renjie/LLM/watermark_LLM/data/test_data"

KMEANS_LABEL_FILE=fine_kmean

echo "Running generation pipeline with output dir: $GENERATION_OUTPUT_DIR"

# generation_pipeline.py semantic_train_pipeline

CUDA_VISIBLE_DEVICES=$DEVICE HF_HOME=$HF_CACHE_DIR TRANSFORMERS_CACHE=$HF_CACHE_DIR python generation_pipeline.py \
    --model_name facebook/opt-2.7b \
    --dataset_name=json_c4 \
    --dataset_config_name=realnewslike \
    --max_new_tokens=200 \
    --min_prompt_tokens=50 \
    --min_generations=200 \
    --input_truncation_strategy=completion_length \
    --input_filtering_strategy=prompt_and_completion_length \
    --output_filtering_strategy=max_new_tokens \
    --seeding_scheme=sem \
    --gamma=0.5 \
    --delta=2.0 \
    --run_name="$RUN_NAME"_gen \
    --wandb=False \
    --verbose=True \
    --generation_batch_size=1 \
    --stream_dataset=True \
    --load_fp16=False \
    --num_beams=1 \
    --use_sampling=True \
    --cl_mlp_model_path $CL_MODEL_PATH \
    --mean_pooling=True \
    --cl_pooling_method=weighted_former_k \
    --cl_discrete_value_number=$DISCRETE_VALUE \
    --cl_k=$CL_K \
    --flag_kmeans=False \
    --output_dir=$GENERATION_OUTPUT_DIR \
    --overwrite=True \
    --hf_cache_dir=$HF_CACHE_DIR \
    --kmeans_label_file=$KMEANS_LABEL_FILE \
    --json_data_dir=$JSON_DATA_DIR \
    --cl_discrete_value_number=$DISCRETE_VALUE

CUDA_VISIBLE_DEVICES=$DEVICE HF_HOME=$HF_CACHE_DIR OPENAI_API_KEY=`cat ../openai_key.txt` $BASE_PYTHON attack_pipeline.py \
    --attack_method=dipper \
    --run_name="$RUN_NAME"_dipper_attack \
    --wandb=False \
    --input_dir=$GENERATION_OUTPUT_DIR \
    --verbose=True \
    --overwrite_output_file=True

cp $GENERATION_OUTPUT_DIR/gen_table_attacked.jsonl $GENERATION_OUTPUT_DIR/gen_table_attacked_bakup.jsonl
$BASE_PYTHON filter_length.py --input_file $GENERATION_OUTPUT_DIR/gen_table_attacked.jsonl
cat $GENERATION_OUTPUT_DIR/gen_table_attacked_filtered.jsonl > $GENERATION_OUTPUT_DIR/gen_table_attacked.jsonl

CUDA_VISIBLE_DEVICES=$DEVICE HF_HOME=$HF_CACHE_DIR python evaluation_pipeline.py \
    --evaluation_metrics=z-score \
    --roc_test_stat=z_score \
    --run_name="$RUN_NAME"_eval \
    --input_dir=$GENERATION_OUTPUT_DIR \
    --output_dir="$GENERATION_OUTPUT_DIR"_dipper_eval \
    --cl_mlp_model_path $CL_MODEL_PATH \
    --cl_mean_pooling=True \
    --cl_pooling_method=weighted_former_k \
    --cl_discrete_value_number=$DISCRETE_VALUE \
    --cl_k=$CL_K \
    --flag_kmeans=False \
    --overwrite_output_file=True \
    --return_green_token_mask=False \
    --compute_scores_at_T=False \
    --hf_cache_dir=$HF_CACHE_DIR \
    --kmeans_label_file=$KMEANS_LABEL_FILE \
    --wandb=True \
    --wandb_entity=thu15renjie
    # --overwrite_args=True
    # --seeding_scheme=sem