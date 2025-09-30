#!/bin/bash

# export TPU_PREFIX=llm-pruning-v6e
required_vars=(
    "BUCKET_NAME"
    "TPU_PREFIX"
)
for var in "${required_vars[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    echo "[ERROR] $var is not set"
    exit 1
  fi
done

export MODEL_NAME='llama3.1-4b-depth'
export NUM_STEPS=62500
export SEQ_LEN=8192
export BATCH_SIZE=2
export GRAD_ACCUM=4
export LR=2.e-4
export MIN_LR_RATIO=0.1
export WARMUP_RATIO=0.05
export ASYNC_CHECKPOINTING=false
export BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/model_ckpts/maxtext"
export DATA_FILES='/home/zephyr/gcs-bucket/datasets/dclm/llama3_64_array_record/*.array_record'

export RUN_NAME="${MODEL_NAME}_S250_seqlen_${SEQ_LEN}_bs_${BATCH_SIZE}_grad_accum_${GRAD_ACCUM}_lr_${LR}_min_lr_ratio_${MIN_LR_RATIO}_warmup_ratio_${WARMUP_RATIO}"

python -u multihost_runner_orig.py \
    --TPU_PREFIX=$TPU_PREFIX \
    --INTERNAL_IP=true \
    --COMMAND="
    export TPU_LOG_DIR=/home/zephyr/tpu_logs
    export WANDB_API_KEY='7d11bbca76b3081b6bd1efbbcf1572aab26c5d56'
    source ~/maxtext_env/bin/activate
    python3.10 -u -m MaxText.train MaxText/configs/base.yml \
        run_name=${RUN_NAME} \
        base_output_directory=${BASE_OUTPUT_DIRECTORY} \
        dataset_type=grain \
        grain_train_files=${DATA_FILES} \
        grain_file_type='arrayrecord' \
        grain_worker_count=1 \
        tokenize_train_data=False \
        tokenize_eval_data=False \
        max_target_length=${SEQ_LEN} \
        async_checkpointing=${ASYNC_CHECKPOINTING} \
        model_name=${MODEL_NAME} \
        steps=${NUM_STEPS} \
        per_device_batch_size=${BATCH_SIZE} \
        gradient_accumulation_steps=${GRAD_ACCUM} \
        learning_rate=${LR} \
        cosine_learning_rate_final_fraction=${MIN_LR_RATIO} \
        warmup_steps_fraction=${WARMUP_RATIO} \
        checkpoint_period=250 \
        checkpoint_max_to_keep=1 \
        use_wandb=False \
        wandb_project=llm_pruning \
        wandb_run_name=${RUN_NAME} \
        packing=false
    "