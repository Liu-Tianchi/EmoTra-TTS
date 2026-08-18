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
# EmoTra sft_Flow_Lynorm_fixs — Flow model with LayerNorm + Fixed Scale
# Direction-Magnitude Decoupling: MLP learns direction, LayerNorm locks magnitude.
# Directly inherits from CausalMaskedDiffWithXvec, no V-series dependencies.

import logging
import os
import random
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import repeat
from cosyvoice.flow.flow import CausalMaskedDiffWithXvec
from cosyvoice.utils.mask import make_pad_mask


class CausalMaskedDiffWithXvec_sft_Flow_Lynorm_fixs(CausalMaskedDiffWithXvec):
    """
    Flow SFT with MLP + LayerNorm + fixed scale (Direction-Magnitude Decoupling).

    Architecture:
      Speaker: normalize(emb) → affine(192→80) → spks [B, 80]
      Emotion: vad_interp → vad_projection(3→896, frozen) →
               hidden_reconstructor(896→1024, frozen) →
               MLP vad_downsample(1024→256→ReLU→80, trainable, last layer zero-init)
               → LayerNorm(80) → emo_direction [B, 80, T]  (norm locked)
      Scale: emo_cond = FIXED_SCALE * emo_direction (default 0.07)
      Combine: spks_expanded[B, 80, T] + emo_cond → spks_with_emo
      Decoder (FROZEN): spks=spks_with_emo (3D), cond=zeros → in_channels=240

    Trainable params: MLP vad_downsample (~280K) + LayerNorm(80) (160)
    """

    DEFAULT_NEUTRAL_EMB_PATH = os.path.join(
        os.path.dirname(__file__), '..', '..', 'pretrained_models', 'neutral_sage_embedding.pt'
    )

    FIXED_EMO_SCALE = 0.07

    def __init__(self, *args, neutral_emb_path=None, **kwargs):
        kwargs.pop('llm_model', None)
        super().__init__(*args, **kwargs)

        # Placeholders — will be created by training script
        self.vad_projection = None
        self.vad_hidden_reconstructor = None
        self.vad_downsample = None
        self.emo_layer_norm = None

        # Load fixed neutral speaker embedding
        emb_path = neutral_emb_path or self.DEFAULT_NEUTRAL_EMB_PATH
        if os.path.exists(emb_path):
            neutral_emb = torch.load(emb_path, map_location='cpu')
            assert neutral_emb.dim() == 1 and neutral_emb.shape[0] == 192, \
                f"Expected [192] embedding, got {neutral_emb.shape}"
            assert neutral_emb.abs().sum().item() > 0, "Neutral embedding is all zeros!"
            self.register_buffer('neutral_embedding', neutral_emb)
            logging.info(f"[sft_Flow_Lynorm_fixs] Loaded neutral embedding from {emb_path}, "
                         f"norm={neutral_emb.norm().item():.4f}")
        else:
            self.neutral_embedding = None
            logging.warning(f"[sft_Flow_Lynorm_fixs] Neutral embedding not found at {emb_path}, "
                            f"will use batch embedding as fallback")

    def _create_vad_emo_embedding(self, vad_a, vad_b, feat_len, device):
        """
        Create per-frame emotion embedding via 40%/30%/30% interpolation + VAD pipeline.
        MLP → LayerNorm → fixed_scale
        """
        actual_model = self.module if hasattr(self, 'module') else self

        if actual_model.vad_projection is None or \
           actual_model.vad_hidden_reconstructor is None or \
           actual_model.vad_downsample is None or \
           actual_model.emo_layer_norm is None:
            batch_size = vad_a.shape[0]
            max_T = feat_len.max().item()
            logging.error("[sft_Flow_Lynorm_fixs] VAD modules not initialized!")
            return torch.zeros(batch_size, 80, max_T, device=device)

        batch_size = vad_a.shape[0]
        max_T = feat_len.max().item()

        # Alpha interpolation: 40% pure A, 30% transition, 30% pure B
        t_start = int(max_T * 0.4)
        t_mid = int(max_T * 0.7)
        mid_len = t_mid - t_start
        alpha = torch.zeros(max_T, device=device)
        if mid_len > 0:
            alpha[t_start:t_mid] = torch.linspace(0, 1, mid_len, device=device)
        alpha[t_mid:] = 1.0
        alpha = alpha.unsqueeze(0)

        # VAD interpolation: [B, 3, max_T]
        vad_a_exp = vad_a.unsqueeze(2)
        vad_b_exp = vad_b.unsqueeze(2)
        vad_interp = (1 - alpha.unsqueeze(1)) * vad_a_exp + alpha.unsqueeze(1) * vad_b_exp

        # Flatten: [B*max_T, 3]
        vad_flat = vad_interp.permute(0, 2, 1).reshape(-1, 3)

        # VAD pipeline: 3 → 896 → 1024 → MLP → LayerNorm → scaled
        vad_896 = actual_model.vad_projection(vad_flat)
        vad_1024 = actual_model.vad_hidden_reconstructor(vad_896)
        vad_80_raw = actual_model.vad_downsample(vad_1024)
        vad_80_normed = actual_model.emo_layer_norm(vad_80_raw)
        vad_80 = self.FIXED_EMO_SCALE * vad_80_normed

        # Reshape: [B, 80, max_T]
        emo_cond = vad_80.reshape(batch_size, max_T, 80).permute(0, 2, 1)

        # Mask padding
        for i in range(batch_size):
            if feat_len[i] < max_T:
                emo_cond[i, :, feat_len[i]:] = 0

        return emo_cond

    def forward(self, batch: dict, device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
        """
        Flow SFT forward pass.
        MLP → LayerNorm → fixed_scale → additive injection via spks channel.
        """
        # Read pre-computed LLM speech tokens
        token = batch['llm_speech_token'].to(device)
        token_len = batch['llm_speech_token_len'].to(device)

        feat = batch['speech_feat'].to(device)
        feat_len = batch['speech_feat_len'].to(device)

        # Fixed neutral speaker embedding
        batch_size = feat.shape[0]
        if self.neutral_embedding is not None:
            embedding = self.neutral_embedding.unsqueeze(0).expand(batch_size, -1).to(device)
        else:
            embedding = batch['embedding'].to(device)

        # VAD
        vad_a = batch.get('vad_a', torch.zeros(batch_size, 3)).to(device)
        vad_b = batch.get('vad_b', torch.zeros(batch_size, 3)).to(device)

        # Diagnosis (once)
        if not hasattr(self, '_diagnosed'):
            self._diagnosed = False

        if not self._diagnosed:
            logging.info(f"[sft_Flow_Lynorm_fixs] Direction-Magnitude Decoupling: MLP + LayerNorm + fixed_scale={self.FIXED_EMO_SCALE}")
            logging.info(f"[sft_Flow_Lynorm_fixs] Using pre-computed LLM tokens, token shape={token.shape}")
            logging.info(f"[sft_Flow_Lynorm_fixs] vad_a[0]={vad_a[0].tolist()}, vad_b[0]={vad_b[0].tolist()}")
            emb_src = "fixed neutral" if self.neutral_embedding is not None else "batch"
            logging.info(f"[sft_Flow_Lynorm_fixs] Speaker embedding: {emb_src}, norm={embedding.norm(dim=1)[0].item():.4f}")

        # Streaming mode
        streaming = True if random.random() < 0.5 else False

        # Speaker embedding: [B, 192] → [B, 80]
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # Token embedding
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # Encoder
        h, h_lengths = self.encoder(token, token_len, streaming=streaming)
        h = self.encoder_proj(h)

        # Get actual frame lengths
        actual_feat_len = h_lengths.sum(dim=-1).squeeze(dim=1)

        # Create per-frame emotion embedding with LayerNorm: [B, 80, T]
        emo_cond = self._create_vad_emo_embedding(vad_a, vad_b, actual_feat_len, device)

        # Combine: spks_expanded + emo_cond (already scaled)
        T = emo_cond.shape[2]
        spks_expanded = repeat(embedding, "b c -> b c t", t=T)
        spks_with_emo = spks_expanded + emo_cond

        # cond = zeros
        cond_zeros = torch.zeros(batch_size, 80, T, device=device, dtype=h.dtype)

        # Diagnosis
        if not self._diagnosed:
            emo_norm = emo_cond[0, :, 0].norm().item() if T > 0 else 0
            spks_norm = embedding.norm(dim=1).mean().item()
            logging.info(f"[sft_Flow_Lynorm_fixs] fixed_scale={self.FIXED_EMO_SCALE}")
            logging.info(f"[sft_Flow_Lynorm_fixs] spks_norm={spks_norm:.4f}")
            logging.info(f"[sft_Flow_Lynorm_fixs] emo_cond_norm (after LN+scale)={emo_norm:.4f}")
            logging.info(f"[sft_Flow_Lynorm_fixs] ratio (emo/spks)={emo_norm/max(spks_norm, 1e-8):.2f}x")
            logging.info(f"[sft_Flow_Lynorm_fixs] cond=zeros, decoder sees 240-dim input (base compatible)")
            logging.info(f"[sft_Flow_Lynorm_fixs] Decoder FROZEN — base weights preserved")

        # Flow Matching loss (decoder frozen)
        mask = (~make_pad_mask(actual_feat_len)).to(h)
        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            spks_with_emo,
            cond=cond_zeros,
            streaming=streaming,
        )

        if not self._diagnosed:
            logging.info(f"[sft_Flow_Lynorm_fixs] Loss: {loss.item():.4f} — diagnosis complete")
            self._diagnosed = True

        # Compute monitoring metrics (detached)
        with torch.no_grad():
            emo_eff_norm = emo_cond[:, :, 0].norm(dim=1).mean() if T > 0 else torch.tensor(0.0)

        return {
            'loss': loss,
            'emo_eff_norm': emo_eff_norm.detach(),
        }
