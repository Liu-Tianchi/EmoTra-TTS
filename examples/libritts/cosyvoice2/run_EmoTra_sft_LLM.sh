#!/bin/bash
# EmoTra SFT_LLM - Standalone training script (no V-series dependencies)
#
# Full pipeline:
#   Stage 0: (External) step3_generate_transition_data_w_ASR_check.py — already completed
#   Stage 1: Convert step3 JSONL → Kaldi-style data dir
#   Stage 2: Extract speaker embeddings
#   Stage 3: Extract speech tokens
#   Stage 4: Generate parquet
#   Stage 5: Train LLM
#   Stage 6: TensorBoard

. ./path.sh || exit 1;

stage=${stage:-5}
stop_stage=${stop_stage:-5}

# Data paths — step2 output directory
step3_output_dir=../../../data/transition_data_asr/ash
dataset_name=ash
pretrained_model_dir=../../../pretrained_models/CosyVoice2-0.5B

# Training hyperparameters
hidden_loss_weight=0.1
vad_change_threshold=0.35
num_vad_tokens=5
exp_tag="sft_LLM"

# Data directory
sft_data_dir=data/transition_${dataset_name}_sft_LLM

# Stage 1: Convert step3 JSONL → Kaldi-style data dir
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "=========================================="
  echo "Stage 1: Convert step3 output to Kaldi-style data"
  echo "  step3 output: ${step3_output_dir}"
  echo "  Output:       ${sft_data_dir}"
  echo "=========================================="
  mkdir -p ${sft_data_dir}
  python local/prepare_transition_data_EmoTra_sft_LLM_fast.py \
    --jsonl_main ${step3_output_dir}/transition_data_asr_filtered.jsonl \
    --jsonl_hidden ${step3_output_dir}/transition_data_asr_hidden_states.jsonl \
    --output_dir ${sft_data_dir}
  echo "Stage 1 Complete! Utterances: $(wc -l < ${sft_data_dir}/wav.scp)"
fi

# Stage 2: Extract speaker embeddings
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "=========================================="
  echo "Stage 2: Extract campplus speaker embedding"
  echo "=========================================="
  python ../../../tools/extract_embedding.py \
    --dir ${sft_data_dir} \
    --onnx_path $pretrained_model_dir/campplus.onnx
  echo "Stage 2 Complete!"
fi

# Stage 3: Extract speech tokens
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "=========================================="
  echo "Stage 3: Extract discrete speech tokens"
  echo "=========================================="
  python ../../../tools/extract_speech_token.py \
    --dir ${sft_data_dir} \
    --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx
  echo "Stage 3 Complete!"
fi

# Stage 4: Generate parquet
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "=========================================="
  echo "Stage 4: Generate parquet"
  echo "=========================================="
  mkdir -p ${sft_data_dir}/parquet
  python ../../../tools/make_parquet_list_EmoTra_sft_LLM.py \
    --num_utts_per_parquet 100 \
    --num_processes 4 \
    --src_dir ${sft_data_dir} \
    --des_dir ${sft_data_dir}/parquet
  echo "Stage 4 Complete!"
  echo "  Parquet files: $(ls ${sft_data_dir}/parquet/*.tar 2>/dev/null | wc -l)"
fi

# Train LLM
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=2029
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "========================================================"
  echo "Stage 5: Train LLM (SFT_LLM)"
  echo "========================================================"
  echo "Dataset: ${dataset_name}"
  echo "Training engine: $train_engine"
  echo "Number of GPUs: $num_gpus"
  echo "Hidden loss weight: ${hidden_loss_weight}"
  echo "VAD tokens: ${num_vad_tokens} (1×start + $((num_vad_tokens-2))×interp + 1×end)"
  [ -n "$exp_tag" ] && echo "Experiment tag: ${exp_tag}"
  echo ""
  echo "TensorBoard:"
  echo "  tensorboard --logdir=tensorboard/cosyvoice2_EmoTra_sft_LLM --port=6006 --bind_all"
  echo ""

  timestamp=$(date +"%Y%m%d_%H%M%S")
  weight_str=$(echo $hidden_loss_weight | sed 's/\.//g')
  threshold_str=$(echo $vad_change_threshold | sed 's/\.//g')

  if [ -n "$exp_tag" ]; then
    run_name="${dataset_name}_${exp_tag}_weight${weight_str}_vadth${threshold_str}_vt${num_vad_tokens}_${timestamp}"
  else
    run_name="${dataset_name}_sft_LLM_weight${weight_str}_vadth${threshold_str}_vt${num_vad_tokens}_${timestamp}"
  fi

  echo "Run name: ${run_name}"

  # Verify parquet exists
  if [ ! -f "${sft_data_dir}/parquet/data.list" ]; then
    echo "ERROR: ${sft_data_dir}/parquet/data.list not found. Run Stage 1-4 first."
    exit 1
  fi

  total_lines=$(wc -l < ${sft_data_dir}/parquet/data.list)
  train_lines=$((total_lines * 9 / 10))

  echo "Data dir: ${sft_data_dir}"
  echo "Total: $total_lines | Train: $train_lines | Val: $((total_lines - train_lines))"

  shuf ${sft_data_dir}/parquet/data.list > data/shuffled_${dataset_name}_sft_LLM.list
  head -n $train_lines data/shuffled_${dataset_name}_sft_LLM.list > data/train_EmoTra_sft_LLM.data.list
  tail -n +$((train_lines + 1)) data/shuffled_${dataset_name}_sft_LLM.list > data/dev_EmoTra_sft_LLM.data.list
  rm data/shuffled_${dataset_name}_sft_LLM.list

  model=llm
  torchrun --nnodes=1 --nproc_per_node=$num_gpus \
      --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
    ../../../cosyvoice/bin/train.py \
    --train_engine $train_engine \
    --config conf/cosyvoice2_EmoTra_sft_LLM.yaml \
    --train_data data/train_EmoTra_sft_LLM.data.list \
    --cv_data data/dev_EmoTra_sft_LLM.data.list \
    --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
    --onnx_path $pretrained_model_dir \
    --model $model \
    --checkpoint $pretrained_model_dir/$model.pt \
    --model_dir `pwd`/exp/cosyvoice2_EmoTra_sft_LLM/$model/$train_engine/${run_name} \
    --tensorboard_dir `pwd`/tensorboard/cosyvoice2_EmoTra_sft_LLM/$model/$train_engine/${run_name} \
    --ddp.dist_backend $dist_backend \
    --num_workers ${num_workers} \
    --prefetch ${prefetch} \
    --pin_memory \
    --override "hidden_loss_weight=${hidden_loss_weight} num_vad_tokens=${num_vad_tokens}"

  echo ""
  echo "========================================================"
  echo "Training Complete!"
  echo "========================================================"
fi

echo "SFT_LLM pipeline completed!"

# Stage 6: TensorBoard
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Stage 6: Launch TensorBoard"
  echo "  tensorboard --logdir=./tensorboard/cosyvoice2_EmoTra_sft_LLM --port=6006 --bind_all"
fi
