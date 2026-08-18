#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
# Modified by Tianchi Liu, March 2026 — EmoTra SFT_LLM data preparation
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

"""
Prepare Kaldi-style training data from Step 3 output (transition_data_asr_filtered.jsonl).

Input:
    --jsonl_main:   transition_data_asr_filtered.jsonl (from step3)
    --jsonl_hidden: transition_data_asr_hidden_states.jsonl (from step3)

Output directory:
    wav.scp                   - utt → GT audio relative path
    text                      - utt → gt_text (sample_a.text + " " + sample_b.text)
    utt2spk                   - utt → speaker
    spk2utt                   - speaker → utts
    vad.json                  - {utt: {vad_a: [A,V,D], vad_b: [A,V,D]}}
    hidden_states.json        - {utt: {hidden_states_a: ..., hidden_states_b: ...}}
    prompt_info.json          - {utt: {gt_text: ...}} (for make_parquet compatibility)
    utt2vad_a.pt              - {utt: [A,V,D]} (torch format)
    utt2vad_b.pt              - {utt: [A,V,D]} (torch format)
    utt2hidden_states_a.pt    - {utt: hidden_states} (torch format)
    utt2hidden_states_b.pt    - {utt: hidden_states} (torch format)
"""

import argparse
import json
import logging
import os
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_single_sample(main_sample, hidden_states_dict):
    """Process a single sample — no prompt, gt_text only."""
    try:
        utt = main_sample['key']

        # GT audio path
        audio_path = main_sample['audio_path']

        # GT text = sample_a.text + " " + sample_b.text
        gt_text = main_sample['sample_a']['text'] + ' ' + main_sample['sample_b']['text']

        # VAD features
        vad_a = main_sample['sample_a']['vad']
        vad_b = main_sample['sample_b']['vad']
        vad_a_list = [vad_a['arousal'], vad_a['valence'], vad_a['dominance']]
        vad_b_list = [vad_b['arousal'], vad_b['valence'], vad_b['dominance']]

        # Hidden states
        hidden_sample = hidden_states_dict.get(utt)
        if hidden_sample is None:
            logger.warning(f"Hidden states not found for {utt}")
            return None

        hidden_states_a = hidden_sample['hidden_states_a']
        hidden_states_b = hidden_sample['hidden_states_b']

        # Speaker
        speaker = main_sample['speaker']

        # Convert path to relative (from examples/libritts/cosyvoice2/)
        audio_path_rel = os.path.join('..', '..', '..', audio_path)

        return {
            'utt': utt,
            'audio_path': audio_path_rel,
            'gt_text': gt_text,
            'speaker': speaker,
            'vad_a': vad_a_list,
            'vad_b': vad_b_list,
            'hidden_states_a': hidden_states_a,
            'hidden_states_b': hidden_states_b,
        }
    except Exception as e:
        logger.error(f"Error processing sample {main_sample.get('key', 'unknown')}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Prepare SFT_LLM transition data (FAST)')
    parser.add_argument('--jsonl_main', required=True,
                       help='Path to transition_data_asr_filtered.jsonl from step3')
    parser.add_argument('--jsonl_hidden', required=True,
                       help='Path to transition_data_asr_hidden_states.jsonl from step3')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for Kaldi-style data')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("SFT_LLM Data Preparation (FAST)")
    logger.info(f"  Main JSONL:   {args.jsonl_main}")
    logger.info(f"  Hidden JSONL: {args.jsonl_hidden}")
    logger.info(f"  Output dir:   {args.output_dir}")
    logger.info("")

    # Load main JSONL
    logger.info(f"Loading main jsonl...")
    main_samples = []
    with open(args.jsonl_main, 'r', encoding='utf-8') as f:
        for line in f:
            main_samples.append(json.loads(line))
    logger.info(f"Loaded {len(main_samples)} samples")

    # Load hidden states JSONL
    logger.info(f"Loading hidden states jsonl...")
    hidden_states_dict = {}
    with open(args.jsonl_hidden, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            hidden_states_dict[sample['key']] = sample
    logger.info(f"Loaded {len(hidden_states_dict)} hidden states")

    # Process samples
    logger.info("Processing samples...")
    results = []
    for sample in tqdm(main_samples):
        result = prepare_single_sample(sample, hidden_states_dict)
        if result is not None:
            results.append(result)

    logger.info(f"Successfully processed {len(results)} samples")

    # Write output files
    wav_scp_path = os.path.join(args.output_dir, 'wav.scp')
    text_path = os.path.join(args.output_dir, 'text')
    utt2spk_path = os.path.join(args.output_dir, 'utt2spk')
    vad_path = os.path.join(args.output_dir, 'vad.json')
    hidden_states_path = os.path.join(args.output_dir, 'hidden_states.json')
    prompt_info_path = os.path.join(args.output_dir, 'prompt_info.json')

    logger.info("Writing output files...")

    vad_data = {}
    hidden_data = {}
    prompt_data = {}
    spk2utt = {}

    with open(wav_scp_path, 'w', encoding='utf-8') as f_wav, \
         open(text_path, 'w', encoding='utf-8') as f_text, \
         open(utt2spk_path, 'w', encoding='utf-8') as f_utt2spk:

        for result in results:
            utt = result['utt']

            # wav.scp: GT audio
            f_wav.write(f"{utt} {result['audio_path']}\n")

            # text: gt_text directly (no prompt prefix)
            f_text.write(f"{utt} {result['gt_text']}\n")

            # utt2spk
            f_utt2spk.write(f"{utt} {result['speaker']}\n")

            # VAD
            vad_data[utt] = {
                'vad_a': result['vad_a'],
                'vad_b': result['vad_b']
            }

            # Hidden states
            hidden_data[utt] = {
                'hidden_states_a': result['hidden_states_a'],
                'hidden_states_b': result['hidden_states_b']
            }

            # prompt_info.json — for make_parquet_list_EmoTra_sft_LLM.py compatibility
            # It reads prompt_info[utt]['gt_text'] to get the text for parquet
            prompt_data[utt] = {
                'gt_text': result['gt_text']
            }

            # spk2utt
            if result['speaker'] not in spk2utt:
                spk2utt[result['speaker']] = []
            spk2utt[result['speaker']].append(utt)

    # Write JSON files
    with open(vad_path, 'w', encoding='utf-8') as f:
        json.dump(vad_data, f, indent=2)
    with open(hidden_states_path, 'w', encoding='utf-8') as f:
        json.dump(hidden_data, f, indent=2)
    with open(prompt_info_path, 'w', encoding='utf-8') as f:
        json.dump(prompt_data, f, indent=2)

    # Write spk2utt
    spk2utt_path = os.path.join(args.output_dir, 'spk2utt')
    with open(spk2utt_path, 'w', encoding='utf-8') as f:
        for spk, utts in sorted(spk2utt.items()):
            f.write(f"{spk} {' '.join(utts)}\n")

    # Save VAD and hidden states as .pt files for parquet generation
    import torch

    utt2vad_a = {r['utt']: r['vad_a'] for r in results}
    utt2vad_b = {r['utt']: r['vad_b'] for r in results}
    utt2hidden_states_a = {r['utt']: r['hidden_states_a'] for r in results}
    utt2hidden_states_b = {r['utt']: r['hidden_states_b'] for r in results}

    torch.save(utt2vad_a, os.path.join(args.output_dir, 'utt2vad_a.pt'))
    torch.save(utt2vad_b, os.path.join(args.output_dir, 'utt2vad_b.pt'))
    torch.save(utt2hidden_states_a, os.path.join(args.output_dir, 'utt2hidden_states_a.pt'))
    torch.save(utt2hidden_states_b, os.path.join(args.output_dir, 'utt2hidden_states_b.pt'))

    logger.info("SFT_LLM data preparation completed!")
    logger.info(f"Output files:")
    logger.info(f"  - wav.scp: {len(results)} GT audio paths")
    logger.info(f"  - text: {len(results)} gt_text entries (no prompt)")
    logger.info(f"  - utt2spk: {len(results)} utterances")
    logger.info(f"  - spk2utt: {len(spk2utt)} speakers")
    logger.info(f"  - vad.json + utt2vad_*.pt: {len(vad_data)} VAD entries")
    logger.info(f"  - hidden_states.json + utt2hidden_states_*.pt: {len(hidden_data)} entries")
    logger.info(f"  - prompt_info.json: {len(prompt_data)} entries (gt_text for parquet)")


if __name__ == '__main__':
    main()
