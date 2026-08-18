# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#               2025 Alibaba Inc (authors: Xiang Lyu, Yabin Li, Qihua, Shengqiang Li)
# Modified by Tianchi Liu, March 2026 — EmoTra SFT_LLM with VAD conditioning
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

import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from cosyvoice.llm.llm import Qwen2LM
from cosyvoice.utils.common import IGNORE_ID, th_accuracy
from cosyvoice.transformer.label_smoothing_loss_safe import SafeLabelSmoothingLoss


class Qwen2LM_EmoTra_SFT_LLM(Qwen2LM):
    """
    Standalone EmoTra LLM for SFT training.
    Directly inherits from Qwen2LM — no V-series intermediate classes.

    Features:
    - VAD Projection: 3D float (valence, arousal, dominance) → 896D embedding via MLP
    - Hidden States Reconstructor: 896D → 1024D auxiliary loss pathway
    - Configurable N VAD tokens with uniform alpha interpolation [0, 1]
    - Sequence: [SOS] [VAD_0...VAD_{N-1}] [gt_text_emb] [task_id_emb] [gt_speech_emb]
    - Teacher forcing on speech tokens only; IGNORE_ID for SOS + VAD + text positions
    """

    def __init__(
            self,
            llm_input_size: int,
            llm_output_size: int,
            speech_token_size: int,
            llm: torch.nn.Module,
            sampling,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            mix_ratio: List[int] = [5, 15],
            use_vad_conditioning: bool = True,
            hidden_loss_weight: float = 0.1,
            num_vad_tokens: int = 5,
    ):
        super().__init__(
            llm_input_size=llm_input_size,
            llm_output_size=llm_output_size,
            speech_token_size=speech_token_size,
            llm=llm,
            sampling=sampling,
            length_normalized_loss=length_normalized_loss,
            lsm_weight=lsm_weight,
            mix_ratio=mix_ratio,
        )

        # Override with safe version to prevent CUDA SIGFPE on all-IGNORE_ID batches
        self.criterion_ce = SafeLabelSmoothingLoss(
            size=speech_token_size + 3,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        self.num_vad_tokens = num_vad_tokens
        self.hidden_loss_weight = hidden_loss_weight

        if use_vad_conditioning:
            # VAD Projection (Main pathway: 3 → 256 → 896)
            self.vad_projection = nn.Sequential(
                nn.Linear(3, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, llm_input_size)  # 896
            )

            # Hidden States Reconstructor (Auxiliary pathway: 896 → 512 → 1024)
            self.hidden_reconstructor = nn.Sequential(
                nn.Linear(llm_input_size, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 1024)
            )

            self.use_vad = True
        else:
            self.use_vad = False

    def prepare_lm_input_target_vad(
        self, sos_emb,
        vad_embeddings,  # List[Tensor]: N VAD embeddings, each [B, 896]
        text_token, text_token_emb, text_token_len,
        task_id_emb,
        speech_token, speech_token_emb, speech_token_len
    ):
        """
        Prepare LM input/target with VAD embeddings.

        Input sequence:
            [SOS] [VAD_0] [VAD_1] ... [VAD_{N-1}] [gt_text_emb] [task_id_emb] [gt_speech_emb]

        Target sequence:
            [IGNORE] × (1 + N + gt_text_len) [gt_speech_tokens] [EOS]

        Note: task_id position target is the first gt_speech_token (not IGNORE).
        """
        num_vad = len(vad_embeddings)
        lm_target, lm_input = [], []

        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        text_token_emb = unpad_sequence(text_token_emb, text_token_len.cpu(), batch_first=True)
        speech_token_emb = unpad_sequence(speech_token_emb, speech_token_len.cpu(), batch_first=True)

        for i in range(len(text_token)):
            # Ignore: SOS(1) + N VAD + gt_text_len
            num_ignore = 1 + num_vad + text_token_len[i].item()

            this_lm_target = torch.tensor(
                [IGNORE_ID] * num_ignore +
                speech_token[i].tolist() +
                [self.eos_token]
            )

            # Build input: [SOS] + [VAD_0 ... VAD_{N-1}] + [text] + [task_id] + [speech]
            vad_embs = [vad_embeddings[j][i].unsqueeze(0) for j in range(num_vad)]
            this_lm_input = torch.concat(
                [sos_emb.squeeze(dim=0)] +
                vad_embs +
                [text_token_emb[i],
                 task_id_emb.squeeze(dim=0),
                 speech_token_emb[i]],
                dim=0
            )

            lm_target.append(this_lm_target)
            lm_input.append(this_lm_input)

        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID)

        return lm_target, lm_input, lm_input_len

    def forward(self, batch: dict, device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
        """
        Forward pass with VAD conditioning.

        Generates num_vad_tokens VAD embeddings with uniformly spaced alpha in [0, 1].
        Hidden loss is computed on start (index 0) and end (index -1) VAD embeddings.
        """
        if not self.use_vad:
            return super().forward(batch, device)

        # 1. Encode text_token (gt_text only)
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        text_token_emb = self.llm.model.model.embed_tokens(text_token)

        # 2. Encode speech_token (GT audio)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        speech_token_emb = self.speech_embedding(speech_token)

        # 3. Process VAD features — Generate N VAD embeddings with uniform alpha spacing
        vad_a = batch['vad_a'].to(device)  # [B, 3] - start
        vad_b = batch['vad_b'].to(device)  # [B, 3] - end

        N = self.num_vad_tokens
        vad_embeddings = []
        for k in range(N):
            alpha = k / (N - 1) if N > 1 else 0.0
            vad_interp = (1 - alpha) * vad_a + alpha * vad_b  # [B, 3]
            vad_embeddings.append(self.vad_projection(vad_interp))  # [B, 896]

        # 4. Get SOS and task_id embeddings
        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 5. Prepare LM input/target
        lm_target, lm_input, lm_input_len = self.prepare_lm_input_target_vad(
            sos_emb, vad_embeddings,
            text_token, text_token_emb, text_token_len,
            task_id_emb,
            speech_token, speech_token_emb, speech_token_len
        )
        lm_target = lm_target.to(device)

        # 6. Run LLM forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)

        # 7. CE loss (only on gt_speech + EOS)
        ce_loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1, self.llm_decoder.out_features),
                         lm_target, ignore_label=IGNORE_ID)

        # 8. Hidden states reconstruction loss
        hidden_loss = torch.tensor(0.0, device=device)

        if 'hidden_states_a' in batch and 'hidden_states_b' in batch:
            hidden_start_pred = self.hidden_reconstructor(vad_embeddings[0])   # [B, 1024]
            hidden_end_pred = self.hidden_reconstructor(vad_embeddings[-1])    # [B, 1024]

            hidden_a_target = batch['hidden_states_a'].to(device)
            hidden_b_target = batch['hidden_states_b'].to(device)

            if hidden_a_target.dim() == 3:
                hidden_a_target = hidden_a_target.squeeze(1)
            if hidden_b_target.dim() == 3:
                hidden_b_target = hidden_b_target.squeeze(1)

            hidden_loss = (F.l1_loss(hidden_start_pred, hidden_a_target) +
                          F.l1_loss(hidden_end_pred, hidden_b_target))

            hidden_start_mse = F.mse_loss(hidden_start_pred, hidden_a_target)
            hidden_end_mse = F.mse_loss(hidden_end_pred, hidden_b_target)

        # 9. Total loss
        total_loss = ce_loss + self.hidden_loss_weight * hidden_loss

        return_dict = {
            'loss': total_loss,
            'acc': acc,
            'ce_loss': ce_loss,
            'hidden_loss': hidden_loss,
        }

        if 'hidden_states_a' in batch and 'hidden_states_b' in batch:
            return_dict.update({
                'hidden_start_mse': hidden_start_mse,
                'hidden_end_mse': hidden_end_mse,
                'weighted_hidden_loss': self.hidden_loss_weight * hidden_loss,
            })

        return return_dict
