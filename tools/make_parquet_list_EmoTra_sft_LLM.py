#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
# Modified by Tianchi Liu, March 2026 — EmoTra SFT_LLM parquet generation
# Uses gt_text only, no prompt speech token
#
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
import argparse
import logging
import os
import json
from tqdm import tqdm
import pandas as pd
import multiprocessing
import time
import torch


def job(utt_list, parquet_file, utt2parquet_file, spk2parquet_file):
    start_time = time.time()
    data_list = []
    for utt in tqdm(utt_list):
        data = open(utt2wav[utt], 'rb').read()
        data_list.append(data)

    df = pd.DataFrame()
    df['utt'] = utt_list
    df['audio_data'] = data_list
    df['wav'] = [utt2wav[utt] for utt in utt_list]
    # SFT_LLM: Use gt_text instead of merged text (prompt_text + gt_text)
    df['text'] = [utt2text[utt] for utt in utt_list]
    df['spk'] = [utt2spk[utt] for utt in utt_list]

    if utt2embedding is not None:
        df['utt_embedding'] = [utt2embedding[utt] for utt in utt_list]
    if spk2embedding is not None:
        df['spk_embedding'] = [spk2embedding[utt2spk[utt]] for utt in utt_list]

    # GT speech tokens
    if utt2speech_token is not None:
        df['speech_token'] = [utt2speech_token[utt] for utt in utt_list]

    # VAD and hidden states
    if utt2vad_a is not None:
        df['vad_a'] = [utt2vad_a[utt] for utt in utt_list]
    if utt2vad_b is not None:
        df['vad_b'] = [utt2vad_b[utt] for utt in utt_list]
    if utt2hidden_states_a is not None:
        df['hidden_states_a'] = [utt2hidden_states_a[utt] for utt in utt_list]
    if utt2hidden_states_b is not None:
        df['hidden_states_b'] = [utt2hidden_states_b[utt] for utt in utt_list]

    if utt2instruct is not None:
        df['instruct'] = [utt2instruct[utt] for utt in utt_list]

    if args.dpo:
        df['reject_speech_token'] = [utt2reject_speech_token.get(utt, None) for utt in utt_list]

    df.to_parquet(parquet_file)
    with open(utt2parquet_file, 'w') as f:
        json.dump({k: parquet_file for k in utt_list}, f, ensure_ascii=False, indent=2)
    with open(spk2parquet_file, 'w') as f:
        json.dump({k: parquet_file for k in list(set(spk_list))}, f, ensure_ascii=False, indent=2)
    logging.info('spend time {}'.format(time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_utts_per_parquet',
                        type=int,
                        default=1000,
                        help='num utts per parquet')
    parser.add_argument('--num_processes',
                        type=int,
                        default=1,
                        help='num processes for make parquets')
    parser.add_argument('--src_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    parser.add_argument('--dpo',
                        action='store_true',
                        default=False,
                        help='Use Direct Preference Optimization')
    args = parser.parse_args()

    # Load standard files
    utt2wav, utt2text, utt2spk = {}, {}, {}
    with open('{}/wav.scp'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]

    # SFT_LLM: Use gt_text from prompt_info.json instead of merged text
    prompt_info_path = '{}/prompt_info.json'.format(args.src_dir)
    if os.path.exists(prompt_info_path):
        logging.info("SFT_LLM: Loading gt_text from prompt_info.json (ignoring merged text)")
        with open(prompt_info_path, 'r') as f:
            prompt_info = json.load(f)
        for utt in utt2wav:
            if utt in prompt_info and 'gt_text' in prompt_info[utt]:
                utt2text[utt] = prompt_info[utt]['gt_text']
            else:
                logging.warning(f"gt_text not found for {utt} in prompt_info.json, using text file")

    # Fill any missing from text file
    if len(utt2text) < len(utt2wav):
        with open('{}/text'.format(args.src_dir)) as f:
            for l in f:
                l = l.replace('\n', '').split()
                utt_id = l[0]
                if utt_id not in utt2text:
                    utt2text[utt_id] = ' '.join(l[1:])

    with open('{}/utt2spk'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2spk[l[0]] = l[1]

    # Load optional instruct file
    if os.path.exists('{}/instruct'.format(args.src_dir)):
        utt2instruct = {}
        with open('{}/instruct'.format(args.src_dir)) as f:
            for l in f:
                l = l.replace('\n', '').split()
                utt2instruct[l[0]] = ' '.join(l[1:])
    else:
        utt2instruct = None

    # Load embeddings
    utt2embedding = torch.load('{}/utt2embedding.pt'.format(args.src_dir)) if os.path.exists('{}/utt2embedding.pt'.format(args.src_dir)) else None
    spk2embedding = torch.load('{}/spk2embedding.pt'.format(args.src_dir)) if os.path.exists('{}/spk2embedding.pt'.format(args.src_dir)) else None

    # Load GT speech tokens
    utt2speech_token = torch.load('{}/utt2speech_token.pt'.format(args.src_dir)) if os.path.exists('{}/utt2speech_token.pt'.format(args.src_dir)) else None

    # Load VAD features
    utt2vad_a = torch.load('{}/utt2vad_a.pt'.format(args.src_dir)) if os.path.exists('{}/utt2vad_a.pt'.format(args.src_dir)) else None
    utt2vad_b = torch.load('{}/utt2vad_b.pt'.format(args.src_dir)) if os.path.exists('{}/utt2vad_b.pt'.format(args.src_dir)) else None

    # Load hidden states
    utt2hidden_states_a = torch.load('{}/utt2hidden_states_a.pt'.format(args.src_dir)) if os.path.exists('{}/utt2hidden_states_a.pt'.format(args.src_dir)) else None
    utt2hidden_states_b = torch.load('{}/utt2hidden_states_b.pt'.format(args.src_dir)) if os.path.exists('{}/utt2hidden_states_b.pt'.format(args.src_dir)) else None

    # DPO support
    if args.dpo:
        utt2reject_speech_token = torch.load('{}_reject/utt2speech_token.pt'.format(args.src_dir)) if os.path.exists('{}_reject/utt2speech_token.pt'.format(args.src_dir)) else {}

    utts = list(utt2wav.keys())

    # Using process pool to speedup
    pool = multiprocessing.Pool(processes=args.num_processes)
    parquet_list, utt2parquet_list, spk2parquet_list = [], [], []
    for i, j in enumerate(range(0, len(utts), args.num_utts_per_parquet)):
        parquet_file = os.path.join(args.des_dir, 'parquet_{:09d}.tar'.format(i))
        utt2parquet_file = os.path.join(args.des_dir, 'utt2parquet_{:09d}.json'.format(i))
        spk2parquet_file = os.path.join(args.des_dir, 'spk2parquet_{:09d}.json'.format(i))
        parquet_list.append(parquet_file)
        utt2parquet_list.append(utt2parquet_file)
        spk2parquet_list.append(spk2parquet_file)
        spk_list = [utt2spk[utt] for utt in utts[j: j + args.num_utts_per_parquet]]
        pool.apply_async(job, (utts[j: j + args.num_utts_per_parquet], parquet_file, utt2parquet_file, spk2parquet_file))
    pool.close()
    pool.join()

    with open('{}/data.list'.format(args.des_dir), 'w', encoding='utf8') as f1, \
            open('{}/utt2data.list'.format(args.des_dir), 'w', encoding='utf8') as f2, \
            open('{}/spk2data.list'.format(args.des_dir), 'w', encoding='utf8') as f3:
        for name in parquet_list:
            f1.write(name + '\n')
        for name in utt2parquet_list:
            f2.write(name + '\n')
        for name in spk2parquet_list:
            f3.write(name + '\n')
