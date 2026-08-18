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
"""
ASR Evaluation Tool: Evaluate all audio files in a folder using Whisper.

Supports two modes:
  1. With ground-truth text: compute WER/CER and report quality metrics
  2. Without ground-truth text: transcribe all audio files and save results

For English TTS evaluation, WER (Word Error Rate) is recommended over CER because:
  - English is word-based; WER aligns with human perception of speech quality
  - CER is too lenient for English (a single misspelled letter yields very low CER)
  - Major TTS papers (VALL-E, VoiceCraft, CosyVoice) use WER for English evaluation

For Chinese TTS evaluation, CER is recommended because:
  - Chinese has no clear word boundaries; character is the natural unit
  - Word segmentation introduces additional noise into the metric

Ground-truth text formats supported:
  1. --text-file with pipe-delimited format (one line per sample, 1-indexed):
       text | emotionA | emotionB | (vadA) | (vadB)
     Audio filenames must contain an integer ID matching the line number, e.g.:
       sage_cy2ins_80_angry_happy.wav -> line 80
       speaker_systemname_ID_emotionA_emotionB.wav

  2. --text-file with plain text (one line per audio, matched by sorted filename order)

  3. --text-jsonl with JSONL (each line has "audio_path"/"key" and "text" fields)

Usage examples:
  # Transcribe-only mode (no ground truth)
  python eval_asr_folder.py --audio-dir ./generated_audio --language en

  # Pipe-delimited ground-truth (ID-based matching from filename)
  python eval_asr_folder.py --audio-dir ./for_blind_test --text-file text_samples_ori.txt --language en

  # Plain text ground-truth (order-based matching)
  python eval_asr_folder.py --audio-dir ./generated_audio --text-file texts.txt --language en --match-by-order

  # JSONL ground-truth
  python eval_asr_folder.py --audio-dir ./generated_audio --text-jsonl metadata.jsonl --language en

  # Specify GPU, Whisper model, and threshold
  python eval_asr_folder.py --audio-dir ./for_blind_test --text-file text_samples_ori.txt --gpu 0 --whisper-model large-v3 --threshold 0.1

Dependencies:
  pip install openai-whisper jiwer torchaudio
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torchaudio
import whisper
from jiwer import cer as compute_cer, wer as compute_wer


AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.opus', '.webm'}


def normalize_text_chinese(text: str) -> str:
    """Normalize Chinese text: remove punctuation and whitespace, lowercase."""
    text = re.sub(r'[，。！？、；：""''（）《》【】…—~·\s]', '', text)
    text = re.sub(r'[,\.!?\-;:\'"()\[\]\s]', '', text)
    return text.lower()


def normalize_text_english(text: str) -> str:
    """Normalize English text: remove punctuation, collapse whitespace, lowercase."""
    text = re.sub(r'[,\.!?\-;:\'"()\[\]]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


def normalize_text(text: str, language: str) -> str:
    if language == 'zh':
        return normalize_text_chinese(text)
    return normalize_text_english(text)


def collect_audio_files(audio_dir: str, recursive: bool = False) -> list:
    """Collect all audio files from directory, sorted by name."""
    audio_dir = Path(audio_dir)
    if recursive:
        files = [f for f in sorted(audio_dir.rglob('*')) if f.suffix.lower() in AUDIO_EXTENSIONS]
    else:
        files = [f for f in sorted(audio_dir.iterdir()) if f.suffix.lower() in AUDIO_EXTENSIONS]
    return files


def extract_id_from_filename(filename: str) -> int:
    """
    Extract the numeric ID from a filename like:
      sage_cy2ins_80_angry_happy.wav  -> 80
      sage_1_sad_happy.wav            -> 1
      sage_cy2cat_142_sad_happy.wav   -> 142
    
    Pattern: speaker_[systemname_]ID_emotionA_emotionB.ext
    Strategy: find all integers in filename, the one before emotion labels is the ID.
    """
    stem = Path(filename).stem
    # Split by underscore
    parts = stem.split('_')
    
    # Known emotion labels
    emotions = {'happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'neutral', 'contempt'}
    
    # Walk backwards to find the first emotion part, then the number right before it is the ID
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() in emotions:
            # Look for the nearest integer before this emotion
            for j in range(i - 1, -1, -1):
                if parts[j].isdigit():
                    return int(parts[j])
    
    # Fallback: find any integer in the filename
    numbers = re.findall(r'(\d+)', stem)
    if numbers:
        return int(numbers[-1])
    
    return -1


def parse_pipe_delimited_line(line: str) -> dict:
    """
    Parse a pipe-delimited ground-truth line:
      text | emotionA | emotionB | (vadA) | (vadB)
    Returns dict with 'text', 'emotion_a', 'emotion_b', etc.
    """
    parts = [p.strip() for p in line.split('|')]
    result = {'text': parts[0]}
    if len(parts) >= 2:
        result['emotion_a'] = parts[1].strip()
    if len(parts) >= 3:
        result['emotion_b'] = parts[2].strip()
    if len(parts) >= 4:
        result['vad_a'] = parts[3].strip()
    if len(parts) >= 5:
        result['vad_b'] = parts[4].strip()
    return result


def load_pipe_delimited_text(text_file: str) -> dict:
    """
    Load pipe-delimited ground-truth file.
    Lines are 1-indexed: line 1 -> ID 1, line 2 -> ID 2, ...
    Returns: dict mapping ID (int) -> parsed line info (with 'text' key)
    """
    mapping = {}
    with open(text_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parsed = parse_pipe_delimited_line(line)
            mapping[line_num] = parsed
    return mapping


def is_pipe_delimited_format(text_file: str) -> bool:
    """Check if the text file uses pipe-delimited format by examining the first non-empty line."""
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                return '|' in line
    return False


def load_text_file_plain(text_file: str) -> list:
    """Load ground-truth texts from a plain text file (one line per audio)."""
    with open(text_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def load_text_jsonl(jsonl_path: str) -> dict:
    """
    Load ground-truth texts from JSONL.
    Returns: dict mapping filename stem -> text
    """
    mapping = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            text = item.get('text', '')
            audio_path = item.get('audio_path', item.get('audio_filename', ''))
            if audio_path:
                key = Path(audio_path).stem
                mapping[key] = text
            if 'key' in item:
                mapping[item['key']] = text
    return mapping


def transcribe_audio(model, audio_path: str, language: str, device: str) -> str:
    """Transcribe a single audio file using Whisper."""
    result = model.transcribe(
        str(audio_path),
        language=language,
        task='transcribe',
        fp16=(device != 'cpu')
    )
    return result['text'].strip()


def compute_metrics(reference: str, hypothesis: str, language: str) -> dict:
    """
    Compute ASR metrics.
    - English: primary metric is WER, also reports CER
    - Chinese: primary metric is CER, also reports WER
    """
    ref_norm = normalize_text(reference, language)
    hyp_norm = normalize_text(hypothesis, language)

    if len(ref_norm) == 0:
        cer_val = 1.0 if len(hyp_norm) > 0 else 0.0
        wer_val = cer_val
    else:
        cer_val = compute_cer(ref_norm, hyp_norm)
        if language == 'zh':
            wer_val = cer_val
        else:
            wer_val = compute_wer(ref_norm, hyp_norm)

    primary_metric = 'cer' if language == 'zh' else 'wer'
    primary_value = cer_val if language == 'zh' else wer_val

    return {
        'wer': wer_val,
        'cer': cer_val,
        'primary_metric': primary_metric,
        'primary_value': primary_value,
        'ref_normalized': ref_norm,
        'hyp_normalized': hyp_norm,
    }


def main():
    parser = argparse.ArgumentParser(
        description='ASR Evaluation: transcribe and optionally evaluate all audio in a folder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--audio-dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--text-file', type=str, default=None,
                        help='Ground-truth text file. Auto-detects pipe-delimited format '
                             '(text|emotionA|emotionB|vadA|vadB) and matches by ID in filename. '
                             'Use --match-by-order to force order-based matching.')
    parser.add_argument('--text-jsonl', type=str, default=None,
                        help='JSONL file with ground-truth (fields: audio_path/key, text)')
    parser.add_argument('--match-by-order', action='store_true',
                        help='Force matching text lines to audio files by sorted order '
                             '(instead of by ID extracted from filename)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSONL path (default: <audio-dir>/asr_eval_results.jsonl)')
    parser.add_argument('--whisper-model', type=str, default='large-v3',
                        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                        help='Whisper model size (default: large-v3)')
    parser.add_argument('--language', type=str, default='en', choices=['zh', 'en'],
                        help='Language (en=English uses WER, zh=Chinese uses CER)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--recursive', action='store_true',
                        help='Recursively search subdirectories for audio files')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Error rate threshold for pass/fail reporting (e.g., 0.1 = 10%%)')

    args = parser.parse_args()

    # Validate inputs
    audio_dir = Path(args.audio_dir)
    if not audio_dir.is_dir():
        print(f"Error: {args.audio_dir} is not a valid directory")
        sys.exit(1)

    if args.text_file and args.text_jsonl:
        print("Error: specify only one of --text-file or --text-jsonl, not both")
        sys.exit(1)

    has_ground_truth = args.text_file is not None or args.text_jsonl is not None

    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        device = 'cuda'
    else:
        device = 'cpu'

    # Collect audio files
    audio_files = collect_audio_files(args.audio_dir, recursive=args.recursive)
    if not audio_files:
        print(f"No audio files found in {args.audio_dir}")
        sys.exit(1)

    # Build ground-truth mapping: audio_path.stem -> ground_truth_text
    gt_texts = {}  # stem -> text
    gt_mode = 'none'

    if args.text_jsonl:
        gt_texts = load_text_jsonl(args.text_jsonl)
        gt_mode = 'jsonl'

    elif args.text_file:
        if not args.match_by_order and is_pipe_delimited_format(args.text_file):
            # Pipe-delimited format: match by ID extracted from filename
            gt_mode = 'pipe-id'
            pipe_data = load_pipe_delimited_text(args.text_file)
            # Map each audio file's ID to its text
            matched = 0
            unmatched_files = []
            for audio_path in audio_files:
                file_id = extract_id_from_filename(audio_path.name)
                if file_id > 0 and file_id in pipe_data:
                    gt_texts[audio_path.stem] = pipe_data[file_id]['text']
                    matched += 1
                else:
                    unmatched_files.append(audio_path.name)

            print(f"Ground-truth: pipe-delimited format detected")
            print(f"  Text file lines  : {len(pipe_data)}")
            print(f"  Matched by ID    : {matched}/{len(audio_files)}")
            if unmatched_files and len(unmatched_files) <= 10:
                print(f"  Unmatched files  : {unmatched_files}")
            elif unmatched_files:
                print(f"  Unmatched files  : {len(unmatched_files)} (showing first 5: {unmatched_files[:5]})")
        else:
            # Plain text: match by sorted order
            gt_mode = 'plain-order'
            text_list = load_text_file_plain(args.text_file)
            if len(text_list) != len(audio_files):
                print(f"Warning: text file has {len(text_list)} lines but found {len(audio_files)} audio files")
                print(f"  Will match by order (min of both)")
            for i, audio_path in enumerate(audio_files):
                if i < len(text_list):
                    gt_texts[audio_path.stem] = text_list[i]

    print(f"\n{'='*70}")
    print(f"ASR Evaluation Tool")
    print(f"{'='*70}")
    print(f"  Audio directory : {args.audio_dir}")
    print(f"  Audio files     : {len(audio_files)}")
    print(f"  Language        : {'Chinese' if args.language == 'zh' else 'English'}")
    print(f"  Primary metric  : {'CER' if args.language == 'zh' else 'WER'}")
    print(f"  Whisper model   : {args.whisper_model}")
    print(f"  Device          : {device}")
    print(f"  Ground truth    : {gt_mode} ({len(gt_texts)} matched)" if has_ground_truth else f"  Ground truth    : No (transcribe-only mode)")
    if args.threshold is not None:
        print(f"  Threshold       : {args.threshold*100:.1f}%")
    print(f"{'='*70}\n")

    # Load Whisper model
    print(f"Loading Whisper {args.whisper_model}...", end=' ', flush=True)
    t0 = time.time()
    model = whisper.load_model(args.whisper_model, device=device)
    print(f"done ({time.time()-t0:.1f}s)\n")

    # Process audio files
    results = []
    total_wer = 0.0
    total_cer = 0.0
    num_evaluated = 0
    num_passed = 0
    num_unmatched = 0
    start_time = time.time()

    for i, audio_path in enumerate(audio_files, 1):
        t_start = time.time()

        try:
            transcription = transcribe_audio(model, str(audio_path), args.language, device)
        except Exception as e:
            print(f"  [{i:4d}/{len(audio_files)}] ERROR {audio_path.name}: {e}")
            results.append({
                'file': str(audio_path),
                'filename': audio_path.name,
                'error': str(e),
            })
            continue

        elapsed = time.time() - t_start
        result = {
            'file': str(audio_path),
            'filename': audio_path.name,
            'transcription': transcription,
            'time_seconds': round(elapsed, 2),
        }

        # Compute metrics if ground truth is available for this file
        gt = gt_texts.get(audio_path.stem, None)
        if gt is not None:
            metrics = compute_metrics(gt, transcription, args.language)
            result['ground_truth'] = gt
            result['wer'] = round(metrics['wer'], 4)
            result['cer'] = round(metrics['cer'], 4)
            result['primary_metric'] = metrics['primary_metric']
            result['primary_value'] = round(metrics['primary_value'], 4)

            total_wer += metrics['wer']
            total_cer += metrics['cer']
            num_evaluated += 1

            if args.threshold is not None:
                passed = metrics['primary_value'] <= args.threshold
                result['passed'] = passed
                if passed:
                    num_passed += 1

            metric_name = metrics['primary_metric'].upper()
            metric_val = metrics['primary_value']
            status = ''
            if args.threshold is not None:
                status = ' PASS' if result['passed'] else ' FAIL'

            # Print progress
            if i % 10 == 0 or i == len(audio_files) or i <= 3:
                eta_sec = (time.time() - start_time) / i * (len(audio_files) - i)
                print(f"  [{i:4d}/{len(audio_files)}] {audio_path.name:<50s} "
                      f"{metric_name}={metric_val:.3f}{status}  ({elapsed:.1f}s)  "
                      f"ETA: {timedelta(seconds=int(eta_sec))}")
        else:
            if has_ground_truth:
                num_unmatched += 1
            # Transcribe-only mode progress
            if i % 10 == 0 or i == len(audio_files) or i <= 3:
                eta_sec = (time.time() - start_time) / i * (len(audio_files) - i)
                preview = transcription[:50] + ('...' if len(transcription) > 50 else '')
                gt_note = ' [no GT]' if has_ground_truth else ''
                print(f"  [{i:4d}/{len(audio_files)}] {audio_path.name:<50s} "
                      f"\"{preview}\"{gt_note}  ({elapsed:.1f}s)  "
                      f"ETA: {timedelta(seconds=int(eta_sec))}")

        results.append(result)

    total_time = time.time() - start_time

    # Save results
    output_path = args.output or str(audio_dir / 'asr_eval_results.jsonl')
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Print summary
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Total audio files  : {len(audio_files)}")
    print(f"  Total time         : {timedelta(seconds=int(total_time))}")
    print(f"  Avg time per file  : {total_time/len(audio_files):.2f}s")

    if num_evaluated > 0:
        avg_wer = total_wer / num_evaluated
        avg_cer = total_cer / num_evaluated
        primary_name = 'CER' if args.language == 'zh' else 'WER'
        primary_avg = avg_cer if args.language == 'zh' else avg_wer

        print(f"\n  Evaluated (with GT) : {num_evaluated}")
        if num_unmatched > 0:
            print(f"  No GT match        : {num_unmatched}")
        print(f"  Average WER        : {avg_wer:.4f} ({avg_wer*100:.2f}%)")
        print(f"  Average CER        : {avg_cer:.4f} ({avg_cer*100:.2f}%)")
        print(f"  >>> Primary ({primary_name})   : {primary_avg:.4f} ({primary_avg*100:.2f}%)")

        if args.threshold is not None:
            pass_rate = num_passed / num_evaluated * 100
            print(f"\n  Threshold          : {args.threshold*100:.1f}%")
            print(f"  Passed             : {num_passed}/{num_evaluated} ({pass_rate:.1f}%)")
            print(f"  Failed             : {num_evaluated - num_passed}/{num_evaluated}")

        # Distribution summary
        if num_evaluated >= 5:
            primary_values = [r['primary_value'] for r in results if 'primary_value' in r]
            primary_values.sort()
            n = len(primary_values)
            print(f"\n  {primary_name} Distribution:")
            print(f"    Min    : {primary_values[0]:.4f}")
            print(f"    25th % : {primary_values[n//4]:.4f}")
            print(f"    Median : {primary_values[n//2]:.4f}")
            print(f"    75th % : {primary_values[3*n//4]:.4f}")
            print(f"    Max    : {primary_values[-1]:.4f}")

        # Show worst cases
        evaluated_results = [r for r in results if 'primary_value' in r]
        evaluated_results.sort(key=lambda x: x['primary_value'], reverse=True)
        worst_n = min(5, len(evaluated_results))
        if worst_n > 0:
            print(f"\n  Top {worst_n} worst cases:")
            for r in evaluated_results[:worst_n]:
                print(f"    {r['filename']:<50s} {r['primary_metric'].upper()}={r['primary_value']:.4f}")
                print(f"      GT:  {r['ground_truth'][:80]}")
                print(f"      ASR: {r['transcription'][:80]}")
    else:
        print(f"\n  Mode: Transcribe-only (no ground truth matched)")
        print(f"  Transcribed: {len([r for r in results if 'transcription' in r])} files")

    print(f"\n  Results saved to: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
