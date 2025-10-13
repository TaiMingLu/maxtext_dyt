#!/bin/bash
# Training script for Llama 3.1 1B with DyT modifications
# 
# Usage: 
#   export BUCKET_NAME="your-gcs-bucket"
#   bash train_llama1b_dyt.sh

set -e

# Check required environment variables
if [ -z "$BUCKET_NAME" ]; then
    echo "Error: BUCKET_NAME environment variable is not set"
    echo "Usage: export BUCKET_NAME='your-gcs-bucket' && bash train_llama1b_dyt.sh"
    exit 1
fi

# Model and training configuration
export MODEL_NAME='llama3.1-1b'
export NUM_STEPS=50000
export SEQ_LEN=2048
export BATCH_SIZE=8
export GRAD_ACCUM=1
export LR=1e-4
export MIN_LR_RATIO=0.1
export WARMUP_RATIO=0.05
export CHECKPOINT_PERIOD=1000

# DyT specific configuration
export NORM_TYPE='tanh'  # Options: 'rms', 'tanh', 'shifterf'
export ATTN_ALPHA_INIT=0.5
export FFN_ALPHA_INIT=0.5
export DEC_ALPHA_INIT=0.5
export SHIFT_INIT=0.0
export USE_SHARED_SCALE='True'

# Output directory
export BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/llama1b_dyt_experiments"

# Data configuration (using HuggingFace C4)
export DATASET_TYPE='hf'
export HF_PATH='allenai/c4'
export HF_DATA_DIR='en'

# Run name with DyT configuration
export RUN_NAME="${MODEL_NAME}_dyt_${NORM_TYPE}_steps${NUM_STEPS}_bs${BATCH_SIZE}_lr${LR}_attn${ATTN_ALPHA_INIT}_ffn${FFN_ALPHA_INIT}"

echo "=========================================="
echo "Training Llama 1B with DyT Modifications"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Norm Type: $NORM_TYPE"
echo "Alpha Init (Attn/FFN/Dec): $ATTN_ALPHA_INIT / $FFN_ALPHA_INIT / $DEC_ALPHA_INIT"
echo "Use Shared Scale: $USE_SHARED_SCALE"
echo "Training Steps: $NUM_STEPS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Output: $BASE_OUTPUT_DIRECTORY/$RUN_NAME"
echo "=========================================="

# Training command
python3 -m MaxText.train MaxText/configs/base.yml \
    run_name=${RUN_NAME} \
    model_name=${MODEL_NAME} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    steps=${NUM_STEPS} \
    per_device_batch_size=${BATCH_SIZE} \
    gradient_accumulation_steps=${GRAD_ACCUM} \
    max_target_length=${SEQ_LEN} \
    learning_rate=${LR} \
    cosine_learning_rate_final_fraction=${MIN_LR_RATIO} \
    warmup_steps_fraction=${WARMUP_RATIO} \
    checkpoint_period=${CHECKPOINT_PERIOD} \
    checkpoint_max_to_keep=5 \
    dataset_type=${DATASET_TYPE} \
    hf_path=${HF_PATH} \
    hf_data_dir=${HF_DATA_DIR} \
    tokenizer_path='meta-llama/Llama-3.1-8B' \
    vocab_size=128256 \
    enable_checkpointing=True \
    save_config_to_gcs=True \
    log_period=100 \
    use_wandb=False \
    packing=True \
    norm_type=${NORM_TYPE} \
    attn_alpha_init_value=${ATTN_ALPHA_INIT} \
    ffn_alpha_init_value=${FFN_ALPHA_INIT} \
    decoder_alpha_init_value=${DEC_ALPHA_INIT} \
    shift_init_value=${SHIFT_INIT} \
    use_shared_scale=${USE_SHARED_SCALE}

echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: $BASE_OUTPUT_DIRECTORY/$RUN_NAME"
echo "=========================================="

