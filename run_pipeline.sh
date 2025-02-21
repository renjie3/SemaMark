# Script to run the generation, attack, and evaluation steps of the pipeline

# requires some OUTPUT_DIR to be set in the environment
# as well as a path to the hf format LLAMA model
# /egr/research-dselab/renjie3/renjie/LLM/cache

OUTPUT_DIR="./results"

RUN_NAME=llama_N500_T200
LLAMA_PATH=No

GENERATION_OUTPUT_DIR="$OUTPUT_DIR"/"$RUN_NAME"
JSON_DATA_DIR="/egr/research-dselab/renjie3/renjie/LLM/watermark_LLM/data/train_data"


DEVICE="3"

echo "Running generation pipeline with output dir: $GENERATION_OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$DEVICE HF_HOME="/egr/research-dselab/renjie3/renjie/LLM/cache" python semantic_train_pipeline.py \
    --model_name=meta-llama/Llama-2-7b-hf \
    --dataset_name=json_c4 \
    --dataset_config_name=realnewslike \
    --max_new_tokens=200 \
    --min_prompt_tokens=50 \
    --min_generations=500 \
    --input_truncation_strategy=completion_length \
    --input_filtering_strategy=prompt_and_completion_length \
    --output_filtering_strategy=max_new_tokens \
    --seeding_scheme=selfhash \
    --gamma=0.25 \
    --delta=2.0 \
    --run_name="$RUN_NAME"_gen \
    --wandb=False \
    --cl_lr=1e-3 \
    --cl_mlp_feat_dim=2 \
    --cl_epochs=50 \
    --cl_savepath=./results/cl_model \
    --cl_mean_pooling=True \
    --cl_pooling_method=weighted_former_k \
    --cl_k=15 \
    --encoder_dim=4096 \
    --load_fp16=False \
    --verbose=True \
    --hf_cache_dir=${HF_HOME} \
    --generation_batch_size=16 \
    --stream_dataset=True \
    --json_data_dir=${JSON_DATA_DIR} \
    --output_dir=$GENERATION_OUTPUT_DIR 
