# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
# Copyright (c) 2026 Tianchi Liu
#
# Derived from CosyVoice (https://github.com/FunAudioLLM/CosyVoice)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env python3
# Pre-compute LLM speech tokens offline using sft_LLM checkpoint.
#
# Loads Qwen2LM_EmoTra_SFT_LLM, reads parquet files from sft_LLM data,
# runs autoregressive inference per sample, and adds `llm_speech_token`
# column to new parquet files for Flow SFT training.
#
# Key difference from precompute_llm_tokens_v4.py:
#   - sft_LLM uses N uniformly-spaced alpha VAD tokens (alpha = k/(N-1))
#   - V4 used 10 fixed tokens (4:3:3 format)
#
# Usage:
#   python tools/precompute_llm_tokens_sft_LLM.py \
#       --config conf/cosyvoice2_EmoTra_sft_Flow_Lynorm_fixs.yaml \
#       --llm_checkpoint exp/.../epoch_7_whole.pt \
#       --qwen_pretrain_path pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN \
#       --input_data_list data/transition_ash_sft_LLM/parquet/data.list \
#       --output_dir data/transition_ash_sft_Flow_Lynorm_fixs/parquet \
#       --gpu 0
#
# Multi-GPU parallel:
#   for rank in 0 1 2 3; do
#       python tools/precompute_llm_tokens_sft_LLM.py ... --rank $rank --world_size 4 --gpu $rank &
#   done; wait

import argparse
import logging
import os
import sys
import time
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'third_party', 'Matcha-TTS'))


def load_llm_from_checkpoint(config_path, llm_checkpoint, qwen_pretrain_path, device):
    """
    Load sft_LLM from config + checkpoint, freeze, move to device.
    """
    override_dict = {
        'qwen_pretrain_path': qwen_pretrain_path,
        'flow': None,
        'hift': None,
        'hifigan': None,
    }
    with open(config_path, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)

    llm_model = configs['llm']

    logging.info(f"Loading LLM checkpoint: {llm_checkpoint}")
    checkpoint = torch.load(llm_checkpoint, map_location='cpu')

    if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    non_param_keys = {'epoch', 'step', 'global_step'}
    state_dict = {k: v for k, v in state_dict.items()
                  if k not in non_param_keys and isinstance(v, torch.Tensor)}

    model_keys = set(llm_model.state_dict().keys())

    # Hybrid key mapping: model.xxx -> llm.model.xxx
    hybrid_state = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            hybrid_state['llm.' + key] = value
        else:
            hybrid_state[key] = value

    hybrid_matched = len(set(hybrid_state.keys()) & model_keys)
    direct_matched = len(set(state_dict.keys()) & model_keys)

    if hybrid_matched >= direct_matched:
        best_state_dict = hybrid_state
        logging.info(f"  Using hybrid key mapping: {hybrid_matched}/{len(model_keys)} matched")
    else:
        best_state_dict = state_dict
        logging.info(f"  Using direct key mapping: {direct_matched}/{len(model_keys)} matched")

    missing, unexpected = llm_model.load_state_dict(best_state_dict, strict=False)
    actually_loaded = len(best_state_dict) - len(unexpected)
    logging.info(f"  Loaded {actually_loaded} parameters")
    if missing:
        logging.warning(f"  Missing keys: {missing[:5]}... ({len(missing)} total)")
    if unexpected:
        logging.warning(f"  Unexpected keys: {unexpected[:5]}... ({len(unexpected)} total)")

    for param in llm_model.parameters():
        param.requires_grad = False

    llm_model = llm_model.to(device).eval()
    logging.info(f"  LLM frozen and moved to {device}")

    # Log sft_LLM specific info
    num_vad_tokens = getattr(llm_model, 'num_vad_tokens', 5)
    logging.info(f"  num_vad_tokens: {num_vad_tokens}")
    logging.info(f"  use_vad: {getattr(llm_model, 'use_vad', False)}")

    return llm_model, configs


def generate_tokens_for_sample(llm, text_token, text_token_len, vad_a, vad_b,
                                gt_speech_token_len, device):
    """
    Generate speech tokens for a single sample using sft_LLM.

    sft_LLM sequence: [SOS] [VAD_0...VAD_{N-1}] [text] [task_id]
    → autoregressive generation → speech token IDs

    VAD token layout (N tokens, uniformly spaced alpha):
      alpha_k = k / (N-1) for k in 0..N-1
      VAD_k = (1 - alpha_k) * vad_a + alpha_k * vad_b
    """
    N = getattr(llm, 'num_vad_tokens', 5)

    t = text_token[:, :text_token_len]  # [1, text_len]
    t_emb = llm.llm.model.model.embed_tokens(t)  # [1, text_len, 896]

    # Build N VAD embeddings with uniform alpha spacing
    vad_embs = []
    for k in range(N):
        alpha = k / (N - 1) if N > 1 else 0.0
        vad_interp = (1 - alpha) * vad_a + alpha * vad_b  # [1, 3]
        vad_emb = llm.vad_projection(vad_interp)  # [1, 896]
        vad_embs.append(vad_emb.unsqueeze(1))  # [1, 1, 896]

    # SOS and task_id
    sos_emb = llm.llm_embedding.weight[llm.sos].reshape(1, 1, -1)
    task_id_emb = llm.llm_embedding.weight[llm.task_id].reshape(1, 1, -1)

    # Construct lm_input: [SOS] [VAD_0 ... VAD_{N-1}] [text] [task_id]
    lm_input = torch.cat(
        [sos_emb] + vad_embs + [t_emb, task_id_emb],
        dim=1
    )  # [1, 1+N+text_len+1, 896]

    # Generation length limits based on GT length
    gt_len = gt_speech_token_len
    max_len = int(gt_len * 1.5)
    min_len = max(1, int(gt_len * 0.5))

    # Autoregressive generation via inference_wrapper
    generated = []
    for tok in llm.inference_wrapper(lm_input, sampling=25, min_len=min_len, max_len=max_len, uuid=''):
        if isinstance(tok, torch.Tensor):
            generated.append(tok.item())
        else:
            generated.append(int(tok))

    if len(generated) == 0:
        generated = [0] * gt_len

    return generated


def truncate_or_pad_tokens(generated, gt_len):
    """Truncate or pad generated tokens to match GT length."""
    gen_len = len(generated)
    if gen_len >= gt_len:
        return generated[:gt_len]
    else:
        pad_val = generated[-1] if gen_len > 0 else 0
        return generated + [pad_val] * (gt_len - gen_len)


def process_parquet(llm, input_parquet, output_parquet, tokenizer_fn, device, min_token_len=100):
    """Process a single parquet file: read, generate LLM tokens, filter short, save."""
    df = pd.read_parquet(input_parquet)
    num_samples = len(df)

    llm_speech_tokens = []
    keep_indices = []

    for idx in range(num_samples):
        row = df.iloc[idx]

        # Get text tokens
        text = row['text']
        text_tokens = tokenizer_fn(text)
        text_token = torch.tensor([text_tokens], dtype=torch.int32, device=device)
        text_token_len = len(text_tokens)

        # Get VAD
        vad_a_raw = row.get('vad_a', [0.0, 0.0, 0.0])
        vad_b_raw = row.get('vad_b', [0.0, 0.0, 0.0])
        if isinstance(vad_a_raw, np.ndarray):
            vad_a_raw = vad_a_raw.tolist()
        if isinstance(vad_b_raw, np.ndarray):
            vad_b_raw = vad_b_raw.tolist()
        vad_a = torch.tensor([vad_a_raw], dtype=torch.float32, device=device)
        vad_b = torch.tensor([vad_b_raw], dtype=torch.float32, device=device)

        # Get GT speech token length
        speech_token = row.get('speech_token', None)
        if speech_token is not None:
            if isinstance(speech_token, (np.ndarray, list)):
                gt_len = len(speech_token)
            else:
                gt_len = 100
        else:
            gt_len = 100

        # Skip if GT itself is too short
        if gt_len < min_token_len:
            continue

        # Generate tokens
        with torch.no_grad():
            generated = generate_tokens_for_sample(
                llm, text_token, text_token_len, vad_a, vad_b, gt_len, device
            )

        # Skip if generation failed or too short
        if len(generated) < min_token_len:
            utt = row.get('utt', f'row_{idx}')
            logging.warning(f"  Skipping {utt}: generated {len(generated)} tokens < {min_token_len}")
            continue

        # Truncate/pad to GT length
        final_tokens = truncate_or_pad_tokens(generated, gt_len)
        llm_speech_tokens.append(final_tokens)
        keep_indices.append(idx)

    if len(keep_indices) == 0:
        logging.warning(f"  All samples filtered from {input_parquet}")
        return 0, num_samples - len(keep_indices)

    # Keep only valid rows
    df = df.iloc[keep_indices].reset_index(drop=True)
    df['llm_speech_token'] = llm_speech_tokens

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_parquet)), exist_ok=True)
    df.to_parquet(output_parquet)
    return len(keep_indices), num_samples - len(keep_indices)


def main():
    parser = argparse.ArgumentParser(
        description='Pre-compute LLM speech tokens using sft_LLM for Flow SFT training'
    )
    parser.add_argument('--config', required=True,
                        help='sft_Flow config yaml (for LLM model definition)')
    parser.add_argument('--llm_checkpoint', required=True,
                        help='sft_LLM checkpoint path (e.g. epoch_7_whole.pt)')
    parser.add_argument('--qwen_pretrain_path', required=True,
                        help='Qwen pretrain path')
    parser.add_argument('--input_data_list', required=True,
                        help='Input data.list (sft_LLM parquet files)')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for parquet files with llm_speech_token')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device id')
    parser.add_argument('--rank', type=int, default=0,
                        help='Worker rank for multi-GPU parallel')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Total number of parallel workers')
    parser.add_argument('--min_token_len', type=int, default=100,
                        help='Minimum llm_speech_token length to keep (default: 100 ≈ 4s)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s [rank{args.rank}] %(levelname)s %(message)s'
    )

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}, rank={args.rank}/{args.world_size}")

    # Load LLM
    llm, configs = load_llm_from_checkpoint(
        args.config, args.llm_checkpoint, args.qwen_pretrain_path, device
    )

    # Setup tokenizer
    get_tokenizer = configs['get_tokenizer']
    tokenizer = get_tokenizer()
    allowed_special = configs.get('allowed_special', 'all')

    def tokenize_text(text):
        if hasattr(tokenizer, 'encode'):
            return tokenizer.encode(text, allowed_special=allowed_special)
        else:
            return tokenizer(text)

    # Read input parquet list
    with open(args.input_data_list, 'r') as f:
        all_parquets = [line.strip() for line in f if line.strip()]

    # Split by rank
    input_parquets = [pq for i, pq in enumerate(all_parquets)
                      if i % args.world_size == args.rank]

    logging.info(f"Total parquet files: {len(all_parquets)}, "
                 f"this worker: {len(input_parquets)} files")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each parquet file
    total_samples = 0
    total_filtered = 0
    output_parquets = []
    start_time = time.time()

    for i, input_pq in enumerate(tqdm(input_parquets, desc=f"[rank{args.rank}] Processing")):
        basename = os.path.basename(input_pq)
        output_pq = os.path.join(args.output_dir, basename)

        n_kept, n_removed = process_parquet(llm, input_pq, output_pq, tokenize_text, device,
                                            min_token_len=args.min_token_len)
        total_samples += n_kept
        total_filtered += n_removed

        if n_kept > 0:
            output_parquets.append(output_pq)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            logging.info(f"  Processed {i+1}/{len(input_parquets)} files, "
                         f"{total_samples} kept, {total_filtered} filtered, {elapsed:.1f}s elapsed")

    # Write data.list
    if args.world_size > 1:
        partial_list_path = os.path.join(args.output_dir, f'data.list.rank{args.rank}')
        with open(partial_list_path, 'w') as f:
            for pq in output_parquets:
                f.write(pq + '\n')
        logging.info(f"Wrote partial data.list: {partial_list_path}")
    else:
        data_list_path = os.path.join(args.output_dir, 'data.list')
        with open(data_list_path, 'w') as f:
            for pq in output_parquets:
                f.write(pq + '\n')
        logging.info(f"Wrote data.list: {data_list_path}")

    elapsed = time.time() - start_time
    logging.info("=" * 60)
    logging.info(f"Pre-computation complete (rank {args.rank})!")
    logging.info(f"  Total samples kept: {total_samples}")
    logging.info(f"  Total samples filtered: {total_filtered}")
    logging.info(f"  Min token length: {args.min_token_len}")
    logging.info(f"  Total time: {elapsed:.1f}s")
    logging.info(f"  Output directory: {args.output_dir}")
    logging.info("=" * 60)


if __name__ == '__main__':
    main()
