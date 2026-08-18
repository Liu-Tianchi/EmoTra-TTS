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
# Extract neutral speaker embedding for Flow SFT training.
#
# Two modes:
#   1. From existing spk2embedding.pt: extract a specific speaker's embedding
#   2. From audio file: use CamPPlus ONNX to extract embedding
#
# Output: a 1D [192] tensor saved as .pt file

import argparse
import os
import sys
import torch
import onnxruntime
import torchaudio
import torchaudio.compliance.kaldi as kaldi


def extract_from_spk2embedding(spk2emb_path, speaker_name):
    """Extract embedding from pre-computed spk2embedding.pt"""
    spk2emb = torch.load(spk2emb_path, map_location='cpu')

    if speaker_name not in spk2emb:
        available = list(spk2emb.keys())
        print(f"Error: Speaker '{speaker_name}' not found in {spk2emb_path}")
        print(f"Available speakers: {available[:20]}")
        return None

    emb = spk2emb[speaker_name]
    if isinstance(emb, list):
        emb = torch.tensor(emb, dtype=torch.float32)
    elif isinstance(emb, torch.Tensor):
        emb = emb.float()

    if emb.dim() > 1:
        emb = emb.squeeze()

    assert emb.dim() == 1 and emb.shape[0] == 192, \
        f"Expected [192] embedding, got {emb.shape}"
    return emb


def extract_from_audio(audio_path, onnx_path):
    """Extract embedding from audio file using CamPPlus ONNX model."""
    audio, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)

    feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    ort_session = onnxruntime.InferenceSession(
        onnx_path, sess_options=option, providers=["CPUExecutionProvider"]
    )

    embedding = ort_session.run(
        None, {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()}
    )[0].flatten()

    emb = torch.tensor(embedding, dtype=torch.float32)
    assert emb.dim() == 1 and emb.shape[0] == 192, \
        f"Expected [192] embedding, got {emb.shape}"
    return emb


def main():
    parser = argparse.ArgumentParser(description='Extract neutral speaker embedding')
    parser.add_argument('--output', required=True, help='Output .pt file path')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--spk2emb', type=str,
                       help='Path to spk2embedding.pt (mode 1)')
    group.add_argument('--audio', type=str,
                       help='Path to neutral audio file (mode 2)')

    parser.add_argument('--speaker', type=str, default=None,
                        help='Speaker name (required for --spk2emb mode)')
    parser.add_argument('--onnx_path', type=str, default=None,
                        help='CamPPlus ONNX model path (required for --audio mode)')
    args = parser.parse_args()

    if args.spk2emb:
        if args.speaker is None:
            print("Error: --speaker is required when using --spk2emb mode")
            sys.exit(1)
        print(f"Extracting from spk2embedding: {args.spk2emb}, speaker={args.speaker}")
        emb = extract_from_spk2embedding(args.spk2emb, args.speaker)
    else:
        if args.onnx_path is None:
            print("Error: --onnx_path is required when using --audio mode")
            sys.exit(1)
        print(f"Extracting from audio: {args.audio}")
        print(f"CamPPlus ONNX: {args.onnx_path}")
        emb = extract_from_audio(args.audio, args.onnx_path)

    if emb is None:
        sys.exit(1)

    assert emb.abs().sum().item() > 0, "Extracted embedding is all zeros!"

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(emb, args.output)

    print(f"Saved neutral embedding to: {args.output}")
    print(f"  Shape: {emb.shape}")
    print(f"  Norm: {emb.norm().item():.4f}")
    print(f"  Min: {emb.min().item():.4f}, Max: {emb.max().item():.4f}")


if __name__ == '__main__':
    main()
