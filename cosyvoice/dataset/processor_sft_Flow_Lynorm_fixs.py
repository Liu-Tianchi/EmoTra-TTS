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
# EmoTra sft_Flow_Lynorm_fixs — Data processor for Flow SFT
# Handles VAD, hidden states, and pre-computed LLM speech tokens.
# No V-series dependencies.

import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np


def parse_vad_hidden_states_sft_Flow(data, mode='train', **kwargs):
    """
    Parse VAD features, hidden states, and pre-computed LLM speech tokens.
    """
    for sample in data:
        if 'vad_a' in sample:
            sample['vad_a'] = torch.tensor(sample['vad_a'], dtype=torch.float32)
        if 'vad_b' in sample:
            sample['vad_b'] = torch.tensor(sample['vad_b'], dtype=torch.float32)

        if 'hidden_states_a' in sample:
            hs_a = sample['hidden_states_a']
            if isinstance(hs_a, np.ndarray):
                if hs_a.dtype == np.object_:
                    hs_a_list = hs_a.tolist()
                    if isinstance(hs_a_list, list) and len(hs_a_list) > 0 and isinstance(hs_a_list[0], list):
                        hs_a_list = hs_a_list[0]
                    hs_a = np.array(hs_a_list, dtype=np.float32)
                    sample['hidden_states_a'] = torch.from_numpy(hs_a)
                else:
                    if hs_a.ndim == 2 and hs_a.shape[0] == 1:
                        hs_a = hs_a.squeeze(0)
                    sample['hidden_states_a'] = torch.from_numpy(hs_a).to(torch.float32)
            elif isinstance(hs_a, list):
                if len(hs_a) > 0 and isinstance(hs_a[0], list):
                    hs_a = hs_a[0]
                hs_a = np.array(hs_a, dtype=np.float32)
                sample['hidden_states_a'] = torch.from_numpy(hs_a)
            else:
                sample['hidden_states_a'] = torch.as_tensor(hs_a, dtype=torch.float32)

            if sample['hidden_states_a'].dim() > 1:
                sample['hidden_states_a'] = sample['hidden_states_a'].squeeze()

        if 'hidden_states_b' in sample:
            hs_b = sample['hidden_states_b']
            if isinstance(hs_b, np.ndarray):
                if hs_b.dtype == np.object_:
                    hs_b_list = hs_b.tolist()
                    if isinstance(hs_b_list, list) and len(hs_b_list) > 0 and isinstance(hs_b_list[0], list):
                        hs_b_list = hs_b_list[0]
                    hs_b = np.array(hs_b_list, dtype=np.float32)
                    sample['hidden_states_b'] = torch.from_numpy(hs_b)
                else:
                    if hs_b.ndim == 2 and hs_b.shape[0] == 1:
                        hs_b = hs_b.squeeze(0)
                    sample['hidden_states_b'] = torch.from_numpy(hs_b).to(torch.float32)
            elif isinstance(hs_b, list):
                if len(hs_b) > 0 and isinstance(hs_b[0], list):
                    hs_b = hs_b[0]
                hs_b = np.array(hs_b, dtype=np.float32)
                sample['hidden_states_b'] = torch.from_numpy(hs_b)
            else:
                sample['hidden_states_b'] = torch.as_tensor(hs_b, dtype=torch.float32)

            if sample['hidden_states_b'].dim() > 1:
                sample['hidden_states_b'] = sample['hidden_states_b'].squeeze()

        # Parse pre-computed LLM speech tokens
        if 'llm_speech_token' in sample:
            llt = sample['llm_speech_token']
            if isinstance(llt, np.ndarray):
                sample['llm_speech_token'] = llt.tolist()
            elif isinstance(llt, torch.Tensor):
                sample['llm_speech_token'] = llt.tolist()

        yield sample


def padding_sft_Flow_Lynorm_fixs(data, use_spk_embedding, mode='train', gan=False, dpo=False):
    """
    Padding function for sft_Flow_Lynorm_fixs.
    Includes llm_speech_token, VAD, and hidden states in batch.
    """
    for sample in data:
        assert isinstance(sample, list)
        order = torch.argsort(torch.tensor([x['speech'].size(1) for x in sample], dtype=torch.int32), descending=True)
        batch = {}
        batch['utts'] = [sample[i]['utt'] for i in order]
        batch['text'] = [sample[i]['text'] for i in order]

        text_token = [torch.tensor(sample[i]['text_token']) for i in order]
        batch['text_token_len'] = torch.tensor([i.size(0) for i in text_token], dtype=torch.int32)
        batch['text_token'] = pad_sequence(text_token, batch_first=True, padding_value=0)

        speech_feat = [sample[i]['speech_feat'] for i in order]
        batch['speech_feat_len'] = torch.tensor([i.size(0) for i in speech_feat], dtype=torch.int32)
        batch['speech_feat'] = pad_sequence(speech_feat, batch_first=True, padding_value=0)

        batch['utt_embedding'] = torch.stack([sample[i]['utt_embedding'] for i in order], dim=0)
        batch['spk_embedding'] = torch.stack([sample[i]['spk_embedding'] for i in order], dim=0)

        if torch.tensor(['instruct_token' in sample[i] for i in order]).all():
            instruct_token = [torch.tensor(sample[i]['instruct_token']) for i in order]
            batch['instruct_token_len'] = torch.tensor([i.size(0) for i in instruct_token], dtype=torch.int32)
            batch['instruct_token'] = pad_sequence(instruct_token, batch_first=True, padding_value=0)

        if torch.tensor(['whisper_feat' in sample[i] for i in order]).all():
            whisper_feat = [sample[i]['whisper_feat'] for i in order]
            batch['whisper_feat_len'] = torch.tensor([i.size(0) for i in whisper_feat], dtype=torch.int32)
            batch['whisper_feat'] = pad_sequence(whisper_feat, batch_first=True, padding_value=0)

        # GT speech tokens
        if torch.tensor(['speech_token' in sample[i] for i in order]).all():
            speech_token = [torch.tensor(sample[i]['speech_token']) for i in order]
            batch['speech_token_len'] = torch.tensor([i.size(0) for i in speech_token], dtype=torch.int32)
            batch['speech_token'] = pad_sequence(speech_token, batch_first=True, padding_value=0)

        # Pre-computed LLM speech tokens
        if torch.tensor(['llm_speech_token' in sample[i] for i in order]).all():
            llm_speech_token = [torch.tensor(sample[i]['llm_speech_token'], dtype=torch.int32) for i in order]
            batch['llm_speech_token_len'] = torch.tensor([i.size(0) for i in llm_speech_token], dtype=torch.int32)
            batch['llm_speech_token'] = pad_sequence(llm_speech_token, batch_first=True, padding_value=0)

        # VAD features
        if torch.tensor(['vad_a' in sample[i] for i in order]).all():
            batch['vad_a'] = torch.stack([sample[i]['vad_a'] for i in order], dim=0)
        if torch.tensor(['vad_b' in sample[i] for i in order]).all():
            batch['vad_b'] = torch.stack([sample[i]['vad_b'] for i in order], dim=0)

        # Hidden states
        if torch.tensor(['hidden_states_a' in sample[i] for i in order]).all():
            batch['hidden_states_a'] = torch.stack([sample[i]['hidden_states_a'] for i in order], dim=0)
        if torch.tensor(['hidden_states_b' in sample[i] for i in order]).all():
            batch['hidden_states_b'] = torch.stack([sample[i]['hidden_states_b'] for i in order], dim=0)

        if gan is True:
            speech = [sample[i]['speech'].squeeze(dim=0) for i in order]
            batch['speech_len'] = torch.tensor([i.size(0) for i in speech], dtype=torch.int32)
            batch['speech'] = pad_sequence(speech, batch_first=True, padding_value=0)
            pitch_feat = [sample[i]['pitch_feat'] for i in order]
            batch['pitch_feat_len'] = torch.tensor([i.size(0) for i in pitch_feat], dtype=torch.int32)
            batch['pitch_feat'] = pad_sequence(pitch_feat, batch_first=True, padding_value=0)

        if dpo is True:
            reject_speech_token = [torch.tensor(sample[i]['reject_speech_token']) for i in order]
            batch['reject_speech_token_len'] = torch.tensor([i.size(0) for i in reject_speech_token], dtype=torch.int32)
            batch['reject_speech_token'] = pad_sequence(reject_speech_token, batch_first=True, padding_value=0)

        if use_spk_embedding is True:
            batch["embedding"] = batch["spk_embedding"]
        else:
            batch["embedding"] = batch["utt_embedding"]

        yield batch
