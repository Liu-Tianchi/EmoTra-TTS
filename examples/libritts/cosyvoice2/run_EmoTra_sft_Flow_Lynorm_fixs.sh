#!/bin/bash
# EmoTra sft_Flow_Lynorm_fixs Pipeline — LayerNorm + Fixed Scale (Direction-Magnitude Decoupling)
# No V-series dependencies.
#
# Pipeline:
#   Stage 1: Check/extract neutral speaker embedding
#   Stage 2: Pre-compute LLM speech tokens + filter short/empty (min 100 tokens ≈ 4s)
#   Stage 3: Flow SFT training
#
# Architecture:
#   LLM: NOT loaded (only 12 VAD tensors extracted from sft_LLM checkpoint)
#   Flow: encoder + decoder_sft_Flow + flow_matching_sft_Flow + MLP + LayerNorm
#   VAD projection (3→896): frozen (loaded from LLM checkpoint)
#   Hidden reconstructor (896→1024): frozen (loaded from LLM checkpoint)
#   MLP vad_downsample (1024→256→ReLU→80): trainable (~280K params, last layer zero-init)
#   emo_layer_norm (80): trainable (160 params, locks magnitude ≈ √80 ≈ 8.94)
#   fixed_scale = 0.07: NOT trainable → effective_norm ≈ 0.626
#
# Strategy: mlp_layernorm_fixed_scale
#   All Flow params frozen, only MLP + LayerNorm trainable (~280K + 160).
#   spks_with_emo = spks_expanded + 0.07 * LayerNorm(MLP(vad))
#   Decoder is FROZEN — base weights preserved.

. ./path.sh || exit 1;

stage=1
stop_stage=3

# ========================================
# Logging
# ========================================
LOG_DIR="$(cd ../../../ && pwd)/training_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/sft_Flow_Lynorm_fixs_pipeline_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================"
echo "sft_Flow_Lynorm_fixs Pipeline Started at: $(date)"
echo "Method: MLP + LayerNorm + fixed scale (Direction-Magnitude Decoupling)"
echo "Log file: $LOG_FILE"
echo "========================================================"
echo ""

# ========================================
# Configuration
# ========================================

dataset_name=ash
# Source data directory (sft_LLM prepared data, WITHOUT llm_speech_token)
src_data_dir=data/transition_${dataset_name}_sft_LLM
# Output data directory (WITH pre-computed llm_speech_token)
flow_data_dir=data/transition_${dataset_name}_sft_Flow_Lynorm_fixs
pretrained_model_dir=../../../pretrained_models/CosyVoice2-0.5B

# Neutral speaker embedding: auto-derived from neutral_audio filename
neutral_audio="../../../data/EmoVoice-DB/audio/neutral/gpt4o_6212_neutral_ash.wav"
neutral_emb_name=$(basename "${neutral_audio}" .wav)
neutral_emb_path=../../../pretrained_models/${neutral_emb_name}_embedding.pt

# sft_LLM checkpoint (for VAD modules in training + LLM inference in Stage 2)
llm_checkpoint=exp/cosyvoice2_ContiEmoTra_sft_LLM/llm/torch_ddp/ash_sft_LLM_weight01_vadth035_vt5_20260310_114606/epoch_7_whole.pt

# Flow checkpoint (base for initialization)
flow_checkpoint=$pretrained_model_dir/flow.pt

# GPU settings
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=2048
dist_backend="nccl"
num_workers=4
prefetch=100
train_engine=torch_ddp

# ========================================
# Pre-flight checks
# ========================================
echo "========================================================"
echo "EmoTra sft_Flow_Lynorm_fixs"
echo "MLP + LayerNorm + fixed scale (Direction-Magnitude Decoupling)"
echo "========================================================"
echo ""
echo "Dataset: ${dataset_name}"
echo "Source data: ${src_data_dir}"
echo "Flow data (with LLM tokens): ${flow_data_dir}"
echo "LLM checkpoint: ${llm_checkpoint}"
echo "Flow checkpoint (base): ${flow_checkpoint}"
echo "Neutral embedding: ${neutral_emb_path}"
echo "Strategy: mlp_layernorm_fixed_scale (~280K + 160 trainable params)"
echo "Stages: ${stage} to ${stop_stage}"
echo ""

# Verify LLM checkpoint
if [ ! -f "$llm_checkpoint" ]; then
    echo "Error: sft_LLM checkpoint not found: $llm_checkpoint"
    echo "  Please train sft_LLM first using run_EmoTra_sft_LLM.sh"
    exit 1
fi

# Verify source data
if [ ! -f "${src_data_dir}/parquet/data.list" ]; then
    echo "Error: Source data not found: ${src_data_dir}/parquet/data.list"
    echo "  Run sft_LLM data preparation first."
    exit 1
fi

echo "All prerequisites verified"
echo ""


# ========================================
# Stage 1: Check/Extract Neutral Speaker Embedding
# ========================================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "========================================================"
    echo "Stage 1: Check/Extract Neutral Speaker Embedding"
    echo "========================================================"
    echo ""

    if [ -f "$neutral_emb_path" ]; then
        echo "Neutral embedding already exists: $neutral_emb_path"
        echo "Skipping extraction."
    else
        echo "Neutral embedding not found at: $neutral_emb_path"
        echo "Extracting from: $neutral_audio"

        if [ -f "$neutral_audio" ]; then
            echo "Extracting from audio: $neutral_audio"
            python ../../../tools/extract_neutral_embedding.py \
                --audio "$neutral_audio" \
                --onnx_path "$pretrained_model_dir/campplus.onnx" \
                --output "$neutral_emb_path"
        else
            echo "Error: Neutral audio not found: $neutral_audio"
            exit 1
        fi

        # Verify extraction succeeded
        if [ ! -f "$neutral_emb_path" ]; then
            echo "Error: Failed to extract neutral embedding."
            exit 1
        fi
    fi

    echo ""
    echo "Stage 1 Complete!"
    echo ""
fi


# ========================================
# Stage 2: Pre-compute LLM Speech Tokens
# ========================================
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "========================================================"
    echo "Stage 2: Pre-compute LLM Speech Tokens"
    echo "========================================================"
    echo ""

    output_parquet_dir="${flow_data_dir}/parquet"

    if [ -f "${output_parquet_dir}/data.list" ]; then
        echo "Pre-computed LLM tokens already exist: ${output_parquet_dir}/data.list"
        echo "Skipping. Delete ${output_parquet_dir} to re-compute."
    else
        echo "Running offline LLM inference to pre-compute speech tokens..."
        echo "  Input: ${src_data_dir}/parquet/data.list"
        echo "  Output: ${output_parquet_dir}/"
        echo "  LLM checkpoint: ${llm_checkpoint}"
        echo "  Using ${num_gpus} GPUs in parallel"
        echo ""

        mkdir -p "$output_parquet_dir"

        # Multi-GPU parallel pre-computation
        pids=()
        gpu_list=($(echo $CUDA_VISIBLE_DEVICES | tr ',' ' '))

        for rank in $(seq 0 $((num_gpus - 1))); do
            gpu_id=${gpu_list[$rank]}
            echo "  Launching rank $rank on GPU $gpu_id..."

            CUDA_VISIBLE_DEVICES=$gpu_id python ../../../tools/precompute_llm_tokens_sft_LLM.py \
                --config conf/cosyvoice2_EmoTra_sft_Flow_Lynorm_fixs.yaml \
                --llm_checkpoint "$llm_checkpoint" \
                --qwen_pretrain_path "$pretrained_model_dir/CosyVoice-BlankEN" \
                --input_data_list "${src_data_dir}/parquet/data.list" \
                --output_dir "$output_parquet_dir" \
                --gpu 0 \
                --rank $rank \
                --world_size $num_gpus \
                --min_token_len 100 &

            pids+=($!)
        done

        # Wait for all workers
        echo ""
        echo "  Waiting for all ${num_gpus} workers to finish..."
        failed=0
        for pid in "${pids[@]}"; do
            if ! wait $pid; then
                echo "  Error: Worker PID $pid failed!"
                failed=1
            fi
        done

        if [ $failed -eq 1 ]; then
            echo "Error: Some workers failed during LLM token pre-computation."
            exit 1
        fi

        # Merge partial data.list files
        if [ $num_gpus -gt 1 ]; then
            echo "  Merging partial data.list files..."
            cat ${output_parquet_dir}/data.list.rank* > ${output_parquet_dir}/data.list
            rm -f ${output_parquet_dir}/data.list.rank*
        fi

        echo ""
        total_parquets=$(wc -l < ${output_parquet_dir}/data.list)
        echo "  Total parquet files with LLM tokens: $total_parquets"
    fi

    echo ""
    echo "Stage 2 Complete!"
    echo ""
fi


# ========================================
# Stage 3: Flow SFT Training
# ========================================
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "========================================================"
    echo "Stage 3: sft_Flow_Lynorm_fixs Training"
    echo "========================================================"
    echo ""
    echo "Training Configuration:"
    echo "  LLM: NOT LOADED (only VAD modules from checkpoint)"
    echo "  Tokens: PRE-COMPUTED (from parquet)"
    echo "  Flow: MOSTLY FROZEN"
    echo "    - MLP vad_downsample (1024→256→ReLU→80): TRAINABLE (~280K params)"
    echo "    - emo_layer_norm(80): TRAINABLE (160 params)"
    echo "    - All other layers: FROZEN"
    echo "    - NO LoRA — decoder base weights preserved"
    echo "  fixed_scale = 0.07 → effective_norm ≈ 0.626"
    echo ""
    echo "Training engine: $train_engine"
    echo "GPUs: $num_gpus ($CUDA_VISIBLE_DEVICES)"
    echo ""

    flow_data_list="${flow_data_dir}/parquet/data.list"

    if [ ! -f "$flow_data_list" ]; then
        echo "Error: Data not found: $flow_data_list"
        echo "  Run Stage 2 first."
        exit 1
    fi

    # Update yaml neutral_emb_path to match current config
    echo "Updating neutral_emb_path in yaml to: $neutral_emb_path"
    sed -i "s|neutral_emb_path:.*|neutral_emb_path: ${neutral_emb_path}|" \
        conf/cosyvoice2_EmoTra_sft_Flow_Lynorm_fixs.yaml

    timestamp=$(date +"%Y%m%d_%H%M%S")
    run_name="sft_Flow_Lynorm_fixs_${dataset_name}_${timestamp}"

    echo "Run name: ${run_name}"
    echo ""

    # ========================================
    # Data Splitting (90% train, 10% validation)
    # ========================================
    echo "Preparing train/val split..."
    total_lines=$(wc -l < ${flow_data_list})
    train_lines=$((total_lines * 9 / 10))

    echo "  Total parquet files: $total_lines"
    echo "  Training: $train_lines (90%)"
    echo "  Validation: $((total_lines - train_lines)) (10%)"

    shuf ${flow_data_list} > data/shuffled_sft_Flow_Lynorm_fixs.list
    head -n $train_lines data/shuffled_sft_Flow_Lynorm_fixs.list > data/train_sft_Flow_Lynorm_fixs.data.list
    tail -n +$((train_lines + 1)) data/shuffled_sft_Flow_Lynorm_fixs.list > data/dev_sft_Flow_Lynorm_fixs.data.list
    rm data/shuffled_sft_Flow_Lynorm_fixs.list

    echo "  Data split complete"
    echo ""

    # ========================================
    # TensorBoard Info
    # ========================================
    echo "============================================================"
    echo "TensorBoard:"
    echo "  cd examples/libritts/cosyvoice2"
    echo "  tensorboard --logdir=tensorboard/cosyvoice2_EmoTra_sft_Flow_Lynorm_fixs --port=6015 --bind_all"
    echo "============================================================"
    echo ""

    # ========================================
    # Launch Training
    # ========================================
    model=flow

    export FLOW_SFT_LLM_CHECKPOINT="${llm_checkpoint}"
    export TORCH_DISTRIBUTED_TIMEOUT=1800

    echo "Starting training..."
    echo ""

    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
      ../../../cosyvoice/bin/train_sft_Flow_Lynorm_fixs.py \
      --train_engine $train_engine \
      --config conf/cosyvoice2_EmoTra_sft_Flow_Lynorm_fixs.yaml \
      --train_data data/train_sft_Flow_Lynorm_fixs.data.list \
      --cv_data data/dev_sft_Flow_Lynorm_fixs.data.list \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --onnx_path $pretrained_model_dir \
      --model $model \
      --checkpoint $flow_checkpoint \
      --model_dir `pwd`/exp/cosyvoice2_EmoTra_sft_Flow_Lynorm_fixs/$model/$train_engine/${run_name} \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice2_EmoTra_sft_Flow_Lynorm_fixs/$model/$train_engine/${run_name} \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --timeout 120

    echo ""
    echo "========================================================"
    echo "Training Complete!"
    echo "========================================================"
    echo "Model: exp/cosyvoice2_EmoTra_sft_Flow_Lynorm_fixs/$model/$train_engine/${run_name}"
    echo "TensorBoard: tensorboard/cosyvoice2_EmoTra_sft_Flow_Lynorm_fixs/$model/$train_engine/${run_name}"
    echo ""
fi

echo "========================================================"
echo "sft_Flow_Lynorm_fixs pipeline completed at: $(date)"
echo "Log: $LOG_FILE"
echo "========================================================"
