#!/usr/bin/env python3
# Copyright (c) 2026 Tianchi Liu
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
# -*- coding: utf-8 -*-
"""
EmoTra sft_Flow_Lynorm_fixs Inference — LayerNorm + Fixed Scale (Direction-Magnitude Decoupling)

Architecture:
  LLM: Qwen2LM_EmoTra_SFT_LLM (5 VAD tokens, uniform alpha [0, 0.25, 0.5, 0.75, 1.0])
  Flow: CausalMaskedDiffWithXvec_sft_Flow_Lynorm_fixs
  Decoder/CFM: sft_Flow variants (3D spks support, no V-series dependencies)
  Emotion: MLP → LayerNorm → fixed_scale (Direction-Magnitude Decoupling)
  Speaker: Fixed neutral embedding (ash)

Usage:
  python simple_inference_EmoTra_TTS.py \
      --text "I was really disappointed about what happened" \
      --vad_start 0.25,0.27,0.31 --vad_end 0.20,0.85,0.86 \
      --emo_scale 0.07
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.append('third_party/Matcha-TTS')

import argparse
import torch
import torch.nn as nn
import torchaudio
from torch.nn import functional as F
from einops import repeat
from typing import Generator

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.llm.llm_EmoTra_sft_LLM import Qwen2LM_EmoTra_SFT_LLM
from cosyvoice.utils.mask import make_pad_mask


# ============================================================
# Inference LLM Wrapper for SFT_LLM (5 VAD tokens, uniform alpha)
# ============================================================
class Qwen2LM_EmoTra_SFT_LLM_Inference(Qwen2LM_EmoTra_SFT_LLM):
    """
    Inference wrapper for SFT_LLM with 5 VAD tokens and no prompt speech.

    VAD token layout (uniform alpha spacing):
      [0]: alpha=0.00  (start)
      [1]: alpha=0.25
      [2]: alpha=0.50  (mid)
      [3]: alpha=0.75
      [4]: alpha=1.00  (end)

    LM input sequence: [SOS] [VAD×5] [text] [task_id]
    Auto-regressive generation starts after task_id.

    Requires vad_start and vad_end to be set externally before calling:
        model.vad_start = torch.FloatTensor([[v, a, d]])  # [1, 3]
        model.vad_end = torch.FloatTensor([[v, a, d]])    # [1, 3]
    """

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            uuid: str = '',
    ) -> Generator[int, None, None]:
        """
        SFT_LLM inference with 5 VAD tokens (uniform alpha) and no prompt speech.

        prompt_text, prompt_text_len, prompt_speech_token,
        prompt_speech_token_len, embedding are accepted for API compatibility
        but are NOT used. Only text (gt_text) and VAD values are used.
        """
        device = text.device

        # 1. Encode text
        text_emb = self.llm.model.model.embed_tokens(text)

        # 2. Get base embeddings
        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 3. Build 5 VAD embeddings with uniform alpha spacing
        if not (hasattr(self, 'vad_start') and hasattr(self, 'vad_end')):
            raise RuntimeError(
                "VAD start and end values not set! "
                "Please set model.vad_start and model.vad_end before calling inference()."
            )

        N = self.num_vad_tokens  # 5
        vad_embeddings = []
        for k in range(N):
            alpha = k / (N - 1) if N > 1 else 0.0
            vad_interp = (1 - alpha) * self.vad_start + alpha * self.vad_end  # [1, 3]
            vad_emb = self.vad_projection(vad_interp)  # [1, 896]
            vad_embeddings.append(vad_emb.unsqueeze(1))  # [1, 1, 896]

        # 4. Construct input sequence: [SOS] [VAD×5] [text] [task_id]
        lm_input = torch.concat(
            [sos_emb] + vad_embeddings + [text_emb, task_id_emb],
            dim=1,
        )

        # 5. Calculate generation length limits
        min_len = int(text_len.item() * min_token_text_ratio)
        max_len = int(text_len.item() * max_token_text_ratio)

        # 6. Auto-regressive generation
        for token in self.inference_wrapper(lm_input, sampling, min_len, max_len, uuid):
            yield token


# ============================================================
# Checkpoint Loading
# ============================================================
def load_flow_sft_checkpoint_and_vad(flow_model, flow_sft_checkpoint_path, llm_checkpoint_path, device):
    """Load Flow SFT checkpoint and VAD modules (MLP + LayerNorm + fixed scale)."""
    print(f"\n[Loading sft_Flow_Lynorm_fixs checkpoint + VAD modules]")

    # Create VAD module structures if not already present
    if not hasattr(flow_model, 'vad_projection') or flow_model.vad_projection is None:
        flow_model.vad_projection = nn.Sequential(
            nn.Linear(3, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 896))
    if not hasattr(flow_model, 'vad_hidden_reconstructor') or flow_model.vad_hidden_reconstructor is None:
        flow_model.vad_hidden_reconstructor = nn.Sequential(
            nn.Linear(896, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 1024))
    if not hasattr(flow_model, 'vad_downsample') or flow_model.vad_downsample is None:
        flow_model.vad_downsample = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 80))
    if not hasattr(flow_model, 'emo_layer_norm') or flow_model.emo_layer_norm is None:
        flow_model.emo_layer_norm = nn.LayerNorm(80)

    # Load Flow SFT checkpoint
    print(f"  Loading Flow SFT checkpoint: {flow_sft_checkpoint_path}")
    flow_ckpt = torch.load(flow_sft_checkpoint_path, map_location='cpu')
    flow_state = flow_ckpt['model'] if 'model' in flow_ckpt and isinstance(flow_ckpt['model'], dict) else flow_ckpt
    flow_state = {k: v for k, v in flow_state.items() if isinstance(v, torch.Tensor)}

    vad_proj_in_flow = any('vad_projection' in k for k in flow_state)
    hidden_recon_in_flow = any('hidden_reconstructor' in k for k in flow_state)

    print(f"  Flow SFT checkpoint: {len(flow_state)} tensor params")
    print(f"    Contains vad_downsample: {any('vad_downsample' in k for k in flow_state)}")
    print(f"    Contains emo_layer_norm: {any('emo_layer_norm' in k for k in flow_state)}")

    flow_model.load_state_dict(flow_state, strict=False)

    # Load VAD projection / hidden_reconstructor from LLM checkpoint if not in Flow
    if not vad_proj_in_flow or not hidden_recon_in_flow:
        print(f"  Loading VAD modules from LLM checkpoint: {llm_checkpoint_path}")
        llm_ckpt = torch.load(llm_checkpoint_path, map_location='cpu')
        llm_state = llm_ckpt['model'] if 'model' in llm_ckpt and isinstance(llm_ckpt['model'], dict) else llm_ckpt
        llm_state = {k: v for k, v in llm_state.items() if isinstance(v, torch.Tensor)}

        if not vad_proj_in_flow:
            vad_proj_keys = {}
            for k, v in llm_state.items():
                if 'vad_projection' in k:
                    vad_proj_keys[k[k.index('vad_projection.') + len('vad_projection.'):]] = v
            if vad_proj_keys:
                flow_model.vad_projection.load_state_dict(vad_proj_keys, strict=False)
                print(f"    Loaded vad_projection from LLM ({len(vad_proj_keys)} params)")

        if not hidden_recon_in_flow:
            hidden_keys = {}
            for k, v in llm_state.items():
                if 'hidden_reconstructor' in k:
                    hidden_keys[k[k.index('hidden_reconstructor.') + len('hidden_reconstructor.'):]] = v
            if hidden_keys:
                flow_model.vad_hidden_reconstructor.load_state_dict(hidden_keys, strict=False)
                print(f"    Loaded hidden_reconstructor from LLM ({len(hidden_keys)} params)")

    # Freeze VAD projection and hidden_reconstructor
    for param in flow_model.vad_projection.parameters():
        param.requires_grad = False
    for param in flow_model.vad_hidden_reconstructor.parameters():
        param.requires_grad = False

    # Diagnostic
    fixed_scale = flow_model.FIXED_EMO_SCALE if hasattr(flow_model, 'FIXED_EMO_SCALE') else 0.07
    with torch.no_grad():
        test_vad = torch.tensor([[0.5, 0.5, 0.5]])
        test_80_raw = flow_model.vad_downsample(flow_model.vad_hidden_reconstructor(flow_model.vad_projection(test_vad)))
        test_80_normed = flow_model.emo_layer_norm(test_80_raw)
        print(f"  LN output norm: {test_80_normed.norm().item():.4f} (expect ≈8.94)")
        print(f"  After scale({fixed_scale}): {(fixed_scale * test_80_normed).norm().item():.4f}")

    print("  sft_Flow_Lynorm_fixs loaded successfully")
    return flow_model


def load_emotra_sft_flow_lynorm_fixs_model(base_model_dir, llm_checkpoint_path,
                                                  flow_sft_checkpoint_path, neutral_emb_path=None):
    """Load full EmoTra sft_Flow_Lynorm_fixs model."""
    print("\n" + "=" * 60)
    print("[Loading EmoTra sft_Flow_Lynorm_fixs Model]")
    print("Method: MLP + LayerNorm + fixed scale (Direction-Magnitude Decoupling)")
    print("LLM: SFT_LLM (5 VAD tokens, uniform alpha)")
    print("=" * 60)

    cosyvoice = CosyVoice2(base_model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
    device = cosyvoice.model.device

    # ---- LLM: SFT_LLM with 5 VAD tokens ----
    llm_checkpoint = torch.load(llm_checkpoint_path, map_location='cpu')
    llm_state_dict = llm_checkpoint['model'] if 'model' in llm_checkpoint else llm_checkpoint

    # Determine speech_token_size from checkpoint
    if 'llm.speech_embedding.weight' in llm_state_dict:
        checkpoint_emb_size = llm_state_dict['llm.speech_embedding.weight'].shape[0]
    elif 'speech_embedding.weight' in llm_state_dict:
        checkpoint_emb_size = llm_state_dict['speech_embedding.weight'].shape[0]
    else:
        raise ValueError("Cannot find speech_embedding.weight in LLM checkpoint")
    speech_token_size = checkpoint_emb_size - 3

    original_llm = cosyvoice.model.llm
    sft_llm_inference = Qwen2LM_EmoTra_SFT_LLM_Inference(
        llm_input_size=original_llm.speech_embedding.embedding_dim,
        llm_output_size=original_llm.speech_embedding.embedding_dim,
        speech_token_size=speech_token_size, llm=original_llm.llm,
        sampling=original_llm.sampling, length_normalized_loss=True, lsm_weight=0,
        use_vad_conditioning=True, hidden_loss_weight=0.1, num_vad_tokens=5)

    llm_params = {k: v for k, v in llm_state_dict.items()
                  if isinstance(v, torch.Tensor) and k not in ('epoch', 'step')
                  and not k.startswith('flow.') and not k.startswith('hift.')}
    sft_llm_inference.load_state_dict(llm_params, strict=False)
    cosyvoice.model.llm = sft_llm_inference.to(device).eval()

    # ---- Flow: sft_Flow_Lynorm_fixs ----
    from cosyvoice.flow.flow_sft_Flow_Lynorm_fixs import CausalMaskedDiffWithXvec_sft_Flow_Lynorm_fixs
    from cosyvoice.flow.flow_matching_sft_Flow import CausalConditionalCFM_sft_Flow
    from cosyvoice.flow.decoder_sft_Flow import CausalConditionalDecoder_sft_Flow
    from omegaconf import DictConfig

    original_flow = cosyvoice.model.flow
    original_decoder = original_flow.decoder

    # Build sft_Flow decoder (3D spks support)
    sft_estimator = CausalConditionalDecoder_sft_Flow(
        in_channels=original_decoder.estimator.in_channels, out_channels=original_decoder.estimator.out_channels,
        channels=[256], dropout=0.0, attention_head_dim=64, n_blocks=4, num_mid_blocks=12, num_heads=8,
        act_fn='gelu', static_chunk_size=original_decoder.estimator.static_chunk_size, num_decoding_left_chunks=-1)
    sft_estimator.load_state_dict(original_decoder.estimator.state_dict())

    sft_cfm = CausalConditionalCFM_sft_Flow(
        in_channels=original_decoder.n_feats, n_spks=original_decoder.n_spks,
        spk_emb_dim=original_decoder.spk_emb_dim,
        cfm_params=DictConfig({'sigma_min': original_decoder.sigma_min, 'solver': original_decoder.solver,
                               't_scheduler': original_decoder.t_scheduler,
                               'training_cfg_rate': original_decoder.training_cfg_rate,
                               'inference_cfg_rate': original_decoder.inference_cfg_rate}),
        estimator=sft_estimator)

    sft_flow = CausalMaskedDiffWithXvec_sft_Flow_Lynorm_fixs(
        input_size=original_flow.input_embedding.weight.shape[1], output_size=80, spk_embed_dim=192,
        output_type='mel', vocab_size=original_flow.input_embedding.weight.shape[0],
        input_frame_rate=original_flow.input_frame_rate, only_mask_loss=True,
        token_mel_ratio=original_flow.token_mel_ratio, pre_lookahead_len=original_flow.pre_lookahead_len,
        encoder=original_flow.encoder, decoder=sft_cfm, neutral_emb_path=neutral_emb_path)

    sft_flow.input_embedding.load_state_dict(original_flow.input_embedding.state_dict())
    sft_flow.encoder_proj.load_state_dict(original_flow.encoder_proj.state_dict())
    sft_flow.spk_embed_affine_layer.load_state_dict(original_flow.spk_embed_affine_layer.state_dict())

    cosyvoice.model.flow = sft_flow
    cosyvoice.model.flow = load_flow_sft_checkpoint_and_vad(
        cosyvoice.model.flow, flow_sft_checkpoint_path, llm_checkpoint_path, device)
    cosyvoice.model.flow = cosyvoice.model.flow.to(device).eval()

    # Ensure neutral embedding is loaded
    flow = cosyvoice.model.flow
    if not (hasattr(flow, 'neutral_embedding') and flow.neutral_embedding is not None):
        emb_path = neutral_emb_path or str(Path(__file__).parent / 'pretrained_models' / 'gpt4o_6212_neutral_ash_embedding.pt')
        if Path(emb_path).exists():
            neutral_emb = torch.load(emb_path, map_location='cpu')
            flow.register_buffer('neutral_embedding', neutral_emb.to(device))
        else:
            raise RuntimeError(f"Neutral embedding not found: {emb_path}")

    print(f"\n[sft_Flow_Lynorm_fixs Model Summary]")
    print(f"  LLM: {type(cosyvoice.model.llm).__name__} (5 VAD tokens, uniform alpha)")
    print(f"  Flow: {type(flow).__name__}, FIXED_EMO_SCALE={getattr(flow, 'FIXED_EMO_SCALE', 0.07)}")
    return cosyvoice


# ============================================================
# Flow Inference with LayerNorm + Fixed Scale
# ============================================================
def _flow_inference_with_ln_scale(flow, token, embedding, vad_a, vad_b, device, finalize, emo_scale):
    """Run Flow model inference with emotion injection via MLP + LN + scale."""
    token = token.to(device, dtype=torch.int32)
    token_len = torch.tensor([token.shape[1]], dtype=torch.int32).to(device)
    embedding = embedding.to(device)
    vad_a, vad_b = vad_a.to(device), vad_b.to(device)

    embedding = F.normalize(embedding, dim=1)
    spks_80 = flow.spk_embed_affine_layer(embedding)

    mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(spks_80)
    token_emb = flow.input_embedding(torch.clamp(token, min=0)) * mask

    if finalize:
        h, h_lengths = flow.encoder(token_emb, token_len, streaming=False)
    else:
        token_main = token_emb[:, :-flow.pre_lookahead_len]
        context = token_emb[:, -flow.pre_lookahead_len:]
        h, h_lengths = flow.encoder(token_main, token_len, context=context, streaming=False)

    h = flow.encoder_proj(h)
    num_frames = h.shape[1]
    batch_size = vad_a.shape[0]

    # Alpha interpolation: 40% pure A, 30% transition, 30% pure B
    t_start = int(num_frames * 0.4)
    t_mid = int(num_frames * 0.7)
    mid_len = t_mid - t_start
    alpha = torch.zeros(num_frames, device=device)
    if mid_len > 0:
        alpha[t_start:t_mid] = torch.linspace(0, 1, mid_len, device=device)
    alpha[t_mid:] = 1.0
    alpha = alpha.unsqueeze(0)

    vad_interp = (1 - alpha.unsqueeze(1)) * vad_a.unsqueeze(2) + alpha.unsqueeze(1) * vad_b.unsqueeze(2)
    vad_flat = vad_interp.permute(0, 2, 1).reshape(-1, 3)

    with torch.no_grad():
        vad_896 = flow.vad_projection(vad_flat)
        vad_1024 = flow.vad_hidden_reconstructor(vad_896)
        vad_80_raw = flow.vad_downsample(vad_1024)
        vad_80_normed = flow.emo_layer_norm(vad_80_raw)
        vad_80 = emo_scale * vad_80_normed

    emo_cond = vad_80.reshape(batch_size, num_frames, 80).permute(0, 2, 1)
    print(f"    MLP raw norm: {vad_80_raw[0].norm().item():.4f}, LN norm: {vad_80_normed[0].norm().item():.4f}, "
          f"scaled({emo_scale:.4f}): {vad_80[0].norm().item():.4f}")

    spks_expanded = repeat(spks_80, "b c -> b c t", t=num_frames)
    spks_with_emo = spks_expanded + emo_cond
    cond_zeros = torch.zeros(batch_size, 80, num_frames, device=device, dtype=h.dtype)

    feat_mask = (~make_pad_mask(torch.tensor([num_frames]))).to(h)
    feat, _ = flow.decoder(mu=h.transpose(1, 2).contiguous(), mask=feat_mask.unsqueeze(1),
                           spks=spks_with_emo, cond=cond_zeros, n_timesteps=10, streaming=False)
    return feat.float()


# ============================================================
# End-to-End Inference
# ============================================================
def inference_sft_flow_lynorm_fixs(cosyvoice, tts_text, vad_start, vad_end,
                                    emo_scale=0.07, output_path='output_sft_flow_lynorm_fixs.wav', speed=1.0):
    """Run full TTS inference: LLM → Flow → HiFT."""
    print(f"\nsft_Flow_Lynorm_fixs: text='{tts_text}', vad={vad_start}→{vad_end}, scale={emo_scale}")
    device = cosyvoice.model.device
    vad_a = torch.FloatTensor([vad_start]).to(device)
    vad_b = torch.FloatTensor([vad_end]).to(device)
    cosyvoice.model.llm.vad_start = vad_a
    cosyvoice.model.llm.vad_end = vad_b

    flow = cosyvoice.model.flow
    if not (hasattr(flow, 'neutral_embedding') and flow.neutral_embedding is not None):
        raise RuntimeError("Neutral embedding not loaded!")
    embedding = flow.neutral_embedding.unsqueeze(0)

    text_token, text_token_len = cosyvoice.frontend._extract_text_token(tts_text)

    print("[Step 1: SFT_LLM Token Generation (5 VAD tokens)]")
    llm = cosyvoice.model.llm
    tts_speech_tokens = []
    with torch.cuda.amp.autocast(cosyvoice.model.fp16 is True and not hasattr(llm, 'vllm')):
        for tok in llm.inference(
            text=text_token.to(device), text_len=torch.tensor([text_token.shape[1]], dtype=torch.int32).to(device),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32).to(device),
            prompt_text_len=torch.tensor([0], dtype=torch.int32).to(device),
            prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32).to(device),
            prompt_speech_token_len=torch.tensor([0], dtype=torch.int32).to(device),
            embedding=embedding.to(device)):
            tts_speech_tokens.append(tok)
    print(f"  Generated {len(tts_speech_tokens)} tokens")
    if not tts_speech_tokens:
        print("  ERROR: No tokens!"); return

    print(f"[Step 2: sft_Flow_Lynorm_fixs — MLP + LN + scale={emo_scale}]")
    tts_mel = _flow_inference_with_ln_scale(
        flow, torch.tensor(tts_speech_tokens).unsqueeze(0), embedding,
        vad_a.unsqueeze(0) if vad_a.dim() == 1 else vad_a,
        vad_b.unsqueeze(0) if vad_b.dim() == 1 else vad_b,
        device, True, emo_scale)
    if speed != 1.0:
        tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')

    print("[Step 3: HiFT Vocoding]")
    tts_speech, _ = cosyvoice.model.hift.inference(speech_feat=tts_mel, cache_source=torch.zeros(1, 1, 0))
    if tts_speech is not None and tts_speech.numel() > 0:
        torchaudio.save(output_path, tts_speech.cpu(), cosyvoice.sample_rate)
        print(f"  Saved: {output_path} ({tts_speech.shape[1]/cosyvoice.sample_rate:.2f}s)")
    else:
        print("ERROR: No audio generated!")


def parse_vad(vad_str):
    values = [float(x.strip()) for x in vad_str.split(',')]
    if len(values) != 3:
        raise ValueError("VAD must have exactly 3 values")
    if not all(0.0 <= v <= 1.0 for v in values):
        raise ValueError("VAD values must be in range [0, 1]")
    return values


def main():
    parser = argparse.ArgumentParser(description="EmoTra sft_Flow_Lynorm_fixs Inference")
    parser.add_argument('--base_model_dir', type=str, default='pretrained_models/CosyVoice2-0.5B')
    parser.add_argument('--llm_checkpoint', type=str,
        default='examples/libritts/cosyvoice2/exp/cosyvoice2_EmoTra_sft_LLM/llm/torch_ddp/ash_sft_LLM_weight01_vadth035_vt5_20260310_114606/epoch_7_whole.pt')
    parser.add_argument('--flow_sft_checkpoint', type=str,
        default='examples/libritts/cosyvoice2/exp/cosyvoice2_EmoTra_sft_Flow_Lynorm_fixs/flow/torch_ddp/sft_Flow_Lynorm_fixs_ash_20260311_214130/epoch_3_step_3400.pt',
        help='sft_Flow_Lynorm_fixs checkpoint path')
    parser.add_argument('--neutral_emb_path', type=str,
        default='pretrained_models/gpt4o_6212_neutral_ash_embedding.pt')
    parser.add_argument('--text', type=str,
        default='The weather was so bad that day, and I was sadly walking home when I suddenly saw something that made me extremely angry',
        help='Target text to synthesize')
    parser.add_argument('--vad_start', type=str, default='0.25,0.27,0.31')
    parser.add_argument('--vad_end', type=str, default='0.20,0.85,0.86')
    parser.add_argument('--emo_scale', type=float, default=0.07,
        help='Emotion scale (default 0.07). Sweep: 0.03, 0.05, 0.07, 0.10')
    parser.add_argument('--output', type=str, default='demo_EmoTra_TTS.wav')
    parser.add_argument('--speed', type=float, default=1.0)
    args = parser.parse_args()

    try:
        vad_start, vad_end = parse_vad(args.vad_start), parse_vad(args.vad_end)
    except ValueError as e:
        print(f"Error: {e}"); sys.exit(1)

    print("=" * 60)
    print(f"EmoTra sft_Flow_Lynorm_fixs Inference | emo_scale={args.emo_scale}")
    print(f"LLM: SFT_LLM (5 VAD tokens, uniform alpha)")
    print("=" * 60)

    try:
        cosyvoice = load_emotra_sft_flow_lynorm_fixs_model(
            args.base_model_dir, args.llm_checkpoint, args.flow_sft_checkpoint, args.neutral_emb_path)
        inference_sft_flow_lynorm_fixs(cosyvoice, args.text, vad_start, vad_end,
                                        emo_scale=args.emo_scale, output_path=args.output, speed=args.speed)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback; traceback.print_exc(); sys.exit(1)
    print("\nDone!")


if __name__ == '__main__':
    main()
