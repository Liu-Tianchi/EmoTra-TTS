#!/usr/bin/env python3
"""
Pre-compute and save the mean speaker embedding from N randomly sampled
reference audios (sage speaker, across all emotions).

Output: a .pt file containing {'embedding': tensor(1, D), 'meta': {...}}

Usage (run from the repo root):
  conda run -n sim python tools/precompute_ref_embedding.py --num-refs 200 --gpu 0
  # => saves pretrained_models/CosyVoice2-0.5B/ref_embedding_sage.pt

  # Custom pool and output
  conda run -n sim python tools/precompute_ref_embedding.py \
    --ref-pool-dir ./data/EmoVoice-DB/audio \
    --num-refs 200 --seed 42 --gpu 0 \
    --output ./pretrained_models/CosyVoice2-0.5B/ref_embedding_sage.pt
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import librosa
from torchaudio.transforms import Resample

import importlib.util

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent  # this script lives in tools/, repo root is one level up
DEFAULT_REF_POOL = str(REPO_ROOT / 'data' / 'EmoVoice-DB' / 'audio')
DEFAULT_CHECKPOINT = str(
    REPO_ROOT / 'third_party' / 'seed-tts-eval' / 'pretrained_models' / 'wavlm_large_finetune.pth'
)
DEFAULT_OUTPUT = str(REPO_ROOT / 'pretrained_models' / 'CosyVoice2-0.5B' / 'ref_embedding_sage.pt')

SV_MODELS_DIR = REPO_ROOT / 'third_party' / 'seed-tts-eval' / 'thirdparty' / 'UniSpeech' / 'downstreams' / 'speaker_verification'


def _import_ecapa_tdnn():
    # Load the upstream (pristine) UniSpeech file without modifying it on disk.
    # The only obstacle to a standalone import is the top-level relative import
    # `from .utils import UpstreamExpert` (which needs fairseq/s3prl and a package
    # context). Our SIM path uses config_path=None (fbank) and never calls
    # UpstreamExpert, so we neutralize that import in memory at load time.
    module_path = SV_MODELS_DIR / 'models' / 'ecapa_tdnn.py'
    source = module_path.read_text().replace(
        'from .utils import UpstreamExpert', 'UpstreamExpert = None')
    ns = {'__name__': 'ecapa_tdnn_local', '__file__': str(module_path)}
    exec(compile(source, str(module_path), 'exec'), ns)
    return ns

ECAPA_TDNN = _import_ecapa_tdnn()['ECAPA_TDNN']


def _import_wavlm():
    wavlm_dir = Path.home() / '.cache' / 'torch' / 'hub' / 's3prl_s3prl_main' / 's3prl' / 'upstream' / 'wavlm'
    modules_path = wavlm_dir / 'modules.py'
    spec_m = importlib.util.spec_from_file_location('wavlm_modules', str(modules_path))
    modules_mod = importlib.util.module_from_spec(spec_m)
    sys.modules['wavlm_modules'] = modules_mod
    spec_m.loader.exec_module(modules_mod)
    wavlm_path = wavlm_dir / 'WavLM.py'
    source = wavlm_path.read_text()
    source = source.replace('from .modules import', 'from wavlm_modules import')
    code = compile(source, str(wavlm_path), 'exec')
    ns = {'__name__': 'WavLM_local', '__file__': str(wavlm_path)}
    exec(code, ns)
    return ns['WavLM'], ns['WavLMConfig']


class WavLMFeatureExtractor(torch.nn.Module):
    def __init__(self, wavlm_cls, wavlm_cfg_cls):
        super().__init__()
        self._wavlm_cls = wavlm_cls
        self._wavlm_cfg_cls = wavlm_cfg_cls
        self.model = None

    def init_from_state_dict(self, sd):
        layer_ids = set()
        for k in sd:
            if k.startswith('encoder.layers.'):
                parts = k.split('.')
                if len(parts) > 2:
                    try: layer_ids.add(int(parts[2]))
                    except ValueError: pass
        cfg_dict = {
            'extractor_mode': 'layer_norm', 'encoder_layers': len(layer_ids),
            'encoder_embed_dim': 1024, 'encoder_ffn_embed_dim': 4096,
            'encoder_attention_heads': 16, 'final_dim': 768,
            'layer_norm_first': True, 'feature_grad_mult': 0.0,
            'normalize': True, 'encoder_layerdrop': 0.0,
            'dropout': 0.0, 'attention_dropout': 0.0, 'activation_dropout': 0.0,
            'dropout_input': 0.0, 'dropout_features': 0.0,
            'conv_pos': 128, 'conv_pos_groups': 16,
            'relative_position_embedding': True, 'num_buckets': 320, 'max_distance': 800,
            'num_negatives': 100, 'codebook_negatives': 0, 'sample_distance': None,
            'cross_sample_negatives': 0,
            'mask_length': 10, 'mask_prob': 0.65, 'mask_selection': 'static',
            'mask_other': 0, 'no_mask_overlap': False,
            'mask_channel_length': 10, 'mask_channel_prob': 0.0,
            'mask_channel_selection': 'static', 'mask_channel_other': 0,
            'no_mask_channel_overlap': False, 'mask_min_space': 1, 'mask_channel_min_space': 1,
            'conv_feature_layers': '[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2',
            'logit_temp': 0.1, 'quantize_targets': False,
            'latent_vars': 0, 'latent_groups': 0, 'latent_dim': 0,
            'untie_final_proj': True, 'feature_layer_norm_affine': True,
            'input_feat_per_channel': None,
        }
        cfg = self._wavlm_cfg_cls(cfg_dict)
        self.model = self._wavlm_cls(cfg)
        self.model.load_state_dict(sd, strict=False)
        self.model.feature_grad_mult = 0.0
        self.model.encoder.layerdrop = 0.0
        for layer in self.model.encoder.layers:
            if hasattr(layer.self_attn, 'fp32_attention'):
                layer.self_attn.fp32_attention = False

    def forward(self, wavs):
        if self.model.cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]
        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device), wav_lengths.unsqueeze(1))
        padded_wav = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)
        with torch.no_grad():
            features = self.model.feature_extractor(padded_wav)
        features = features.transpose(1, 2)
        features = self.model.layer_norm(features)
        padding_mask = self.model.forward_padding_mask(features, wav_padding_mask) if wav_padding_mask is not None else None
        if self.model.post_extract_proj is not None:
            features = self.model.post_extract_proj(features)
        features = self.model.dropout_input(features)
        x = features
        x_conv = self.model.encoder.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv
        if not self.model.encoder.layer_norm_first:
            x = self.model.encoder.layer_norm(x)
        x = x.transpose(0, 1)
        hidden_states = []
        pos_bias = None
        for layer in self.model.encoder.layers:
            hidden_states.append(x.transpose(0, 1))
            x, z, pos_bias = layer(x, self_attn_padding_mask=padding_mask, need_weights=False, pos_bias=pos_bias)
        hidden_states.append(x.transpose(0, 1))
        return {'hidden_states': hidden_states}


def init_model(checkpoint_path, device):
    sd = torch.load(checkpoint_path, map_location='cpu')
    model_state = sd['model']
    prefix = 'feature_extract.model.'
    feat_state = {k[len(prefix):]: v for k, v in model_state.items() if k.startswith(prefix)}
    WavLM_cls, WavLMConfig_cls = _import_wavlm()
    fe = WavLMFeatureExtractor(WavLM_cls, WavLMConfig_cls)
    fe.init_from_state_dict(feat_state)
    model = ECAPA_TDNN(feat_dim=1024, channels=512, emb_dim=256, feat_type='fbank',
                       sr=16000, feature_selection='hidden_states', update_extract=False, config_path=None)
    model.feature_extract = fe
    model.feat_type = 'wavlm_large'
    model.feat_num = 25
    model.feature_weight = torch.nn.Parameter(torch.zeros(25))
    model.load_state_dict(model_state, strict=False)
    for p in model.feature_extract.parameters():
        p.requires_grad = False
    return model.to(device).eval()


def load_audio(path):
    wav, sr = librosa.load(path, sr=None, mono=False)
    if len(wav.shape) == 2:
        wav = wav[0, :]
    wav = torch.from_numpy(wav).unsqueeze(0).float()
    if sr != 16000:
        wav = Resample(orig_freq=sr, new_freq=16000)(wav)
    return wav


def main():
    parser = argparse.ArgumentParser(description='Pre-compute mean reference speaker embedding')
    parser.add_argument('--ref-pool-dir', type=str, default=DEFAULT_REF_POOL,
                        help='Root dir of reference audio pool')
    parser.add_argument('--ref-pattern', type=str, default='*_sage.wav')
    parser.add_argument('--num-refs', type=int, default=200)
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ref_pool_dir = Path(args.ref_pool_dir)
    all_refs = sorted([f for f in ref_pool_dir.rglob(args.ref_pattern)
                       if not any(p.startswith('.') or p.startswith('__MACOSX') for p in f.parts)])
    print(f"Reference pool: {len(all_refs)} files in {args.ref_pool_dir}")

    num_refs = min(args.num_refs, len(all_refs))
    selected = random.sample(all_refs, num_refs)

    # Show emotion distribution
    emo_count = {}
    for f in selected:
        emo = f.parent.name
        emo_count[emo] = emo_count.get(emo, 0) + 1
    print(f"Sampling {num_refs} refs (seed={args.seed}), emotion distribution:")
    for emo, cnt in sorted(emo_count.items()):
        print(f"  {emo:<12s}: {cnt}")

    device = f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'

    print(f"\nLoading model...", end=' ', flush=True)
    model = init_model(args.checkpoint, device)
    print("done")

    print(f"Extracting {num_refs} embeddings...")
    embeddings = []
    ref_files_used = []
    for i, ref_path in enumerate(selected, 1):
        try:
            wav = load_audio(str(ref_path)).to(device)
            with torch.no_grad():
                emb = model(wav)  # (1, D)
            embeddings.append(emb.cpu())
            ref_files_used.append(str(ref_path))
            if i % 20 == 0 or i == num_refs:
                print(f"  [{i:3d}/{num_refs}] done")
        except Exception as e:
            print(f"  [{i:3d}/{num_refs}] ERROR {ref_path.name}: {e}")

    print(f"\n{len(embeddings)}/{num_refs} embeddings extracted successfully")

    # Mean embedding
    all_embs = torch.cat(embeddings, dim=0)  # (N, D)
    mean_emb = all_embs.mean(dim=0, keepdim=True)  # (1, D)

    # Save
    save_data = {
        'embedding': mean_emb,           # (1, D) mean embedding
        'all_embeddings': all_embs,       # (N, D) individual embeddings (optional, for analysis)
        'meta': {
            'num_refs': len(embeddings),
            'seed': args.seed,
            'ref_pattern': args.ref_pattern,
            'ref_pool_dir': str(args.ref_pool_dir),
            'emotion_distribution': emo_count,
            'ref_files': ref_files_used,
            'emb_dim': mean_emb.shape[1],
        }
    }
    torch.save(save_data, args.output)
    print(f"\nSaved to: {args.output}")
    print(f"  Mean embedding shape: {mean_emb.shape}")
    print(f"  Embedding dim: {mean_emb.shape[1]}")


if __name__ == '__main__':
    main()
