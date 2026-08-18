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
Step 1: Filter EmoVoice-DB by VAD scores

Filters audio samples based on VAD (Valence, Arousal, Dominance) scores
that match expected emotional ranges.
"""

import json
import sys
import argparse
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# VADPredictor is our own wrapper around the w2v2-vad ONNX model (lives in tools/)
sys.path.insert(0, str(Path(__file__).resolve().parent / 'tools'))
from vad_wrapper import VADPredictor


# VAD ranges for each emotion (Arousal, Valence, Dominance)
EMOTION_RANGES = {
    'angry': {
        'valence': (0.00, 0.35),
        'arousal': (0.75, 1.00),
        'dominance': (0.70, 1.00)
    },
    'happy': {
        'valence': (0.65, 1.00),
        'arousal': (0.50, 1.00),
        'dominance': (0.55, 1.00)
    },
    'sad': {
        'valence': (0.00, 0.35),
        'arousal': (0.00, 0.50),
        'dominance': (0.00, 0.50)
    },
    'surprised': {
        'valence': (0.35, 0.80),
        'arousal': (0.60, 1.00),
        'dominance': (0.35, 0.75)
    },
    'fearful': {
        'valence': (0.00, 0.45),
        'arousal': (0.4, 0.9),
        'dominance': (0.10, 0.55)
    },
    'disgusted': {
        'valence': (0.00, 0.35),
        'arousal': (0.30, 0.70),
        'dominance': (0.30, 0.65)
    },
    'neutral': {
        'valence': (0.40, 0.60),
        'arousal': (0.40, 0.60),
        'dominance': (0.40, 0.60)
    }
}

# DEBUG: Set to number of samples for testing, None for full run
DEBUG_LIMIT = 500


# ============================================================
# Functions
# ============================================================

def extract_speaker_from_key(key):
    """Extract speaker name from key (last part after final underscore)"""
    return key.split('_')[-1]


def has_unicode_escape(text):
    """Check if text contains unicode escape sequences like \\u2019"""
    return '\\u' in text


def check_vad_in_range(arousal, valence, dominance, emotion):
    """Check if VAD scores fall within emotion's expected range"""
    if emotion.lower() not in EMOTION_RANGES:
        return False
    
    ranges = EMOTION_RANGES[emotion.lower()]
    
    # Check each dimension
    a_min, a_max = ranges['arousal']
    v_min, v_max = ranges['valence']
    d_min, d_max = ranges['dominance']
    
    return (a_min <= arousal <= a_max and 
            v_min <= valence <= v_max and 
            d_min <= dominance <= d_max)


def load_jsonl(path):
    """Load JSONL file"""
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Filter EmoVoice-DB by VAD scores')
    parser.add_argument('--jsonl', type=str, 
                       default="./data/EmoVoice-DB/train.jsonl",
                       help='Path to JSONL file')
    parser.add_argument('--audio-dir', type=str,
                       default="./data/EmoVoice-DB/audio",
                       help='Path to audio directory')
    parser.add_argument('--output-dir', type=str,
                       default="./data/filtered",
                       help='Output directory for filtered data')
    parser.add_argument('--debug-limit', type=int, default=0,
                       help='Number of samples to randomly sample for a quick test (0 = process all)')
    parser.add_argument('--speaker', type=str, default=None,
                       help='Only process this speaker (parsed from key, e.g. "sage"). Default: all speakers')
    args = parser.parse_args()
    
    # Configuration
    JSONL_PATH = args.jsonl
    AUDIO_DIR = args.audio_dir
    OUTPUT_DIR = args.output_dir
    DEBUG_LIMIT = args.debug_limit if args.debug_limit > 0 else None
    
    print("=" * 60)
    print("Step 1: Filter EmoVoice-DB by VAD scores")
    print("=" * 60)
    print()
    
    # Check paths
    jsonl_path = Path(JSONL_PATH)
    audio_dir = Path(AUDIO_DIR)
    output_dir = Path(OUTPUT_DIR)
    
    if not jsonl_path.exists():
        print(f"Error: JSONL file not found: {jsonl_path}")
        exit(1)
    
    if not audio_dir.exists():
        print(f"Error: Audio directory not found: {audio_dir}")
        exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"JSONL: {jsonl_path}")
    print(f"Audio Dir: {audio_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Load samples
    print("Loading JSONL...")
    samples = load_jsonl(jsonl_path)
    
    if DEBUG_LIMIT:
        # Randomly sample instead of taking first N
        if DEBUG_LIMIT < len(samples):
            random.seed(42)  # Set seed for reproducibility
            samples = random.sample(samples, DEBUG_LIMIT)
        print(f"DEBUG MODE: Randomly sampled {len(samples)} samples")
    
    print(f"Total samples to process: {len(samples)}")
    print()
    
    # Initialize VAD predictor
    print("Initializing VAD predictor...")
    predictor = VADPredictor()
    print("✓ VAD predictor ready")
    print()
    
    # Statistics
    stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'passed': 0}))
    filtered_by_speaker = defaultdict(list)
    skipped_unicode = 0
    skipped_audio_missing = 0
    
    # Process samples
    print("Processing samples...")
    for sample in tqdm(samples, desc="Filtering"):
        key = sample['key']
        emotion = sample['emotion']
        text = sample.get('target_text', sample.get('source_text', ''))
        audio_path_rel = sample['target_wav']
        
        # Clean text: remove surrounding quotes if present
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        
        # Extract speaker
        speaker = extract_speaker_from_key(key)
        
        # Optional speaker filter (skip before VAD prediction to save compute)
        if args.speaker is not None and speaker != args.speaker:
            continue
        
        # Check for unicode escapes
        if has_unicode_escape(text):
            skipped_unicode += 1
            continue
        
        # Get full audio path
        # target_wav is relative like "audio/angry/xxx.wav"
        # We need to remove "audio/" prefix since AUDIO_DIR already points to audio directory
        audio_path_rel_fixed = audio_path_rel.replace('audio/', '', 1)
        audio_path = audio_dir / audio_path_rel_fixed
        if not audio_path.exists():
            skipped_audio_missing += 1
            continue
        
        # Update statistics
        stats[speaker][emotion]['total'] += 1
        
        try:
            # Predict VAD scores with hidden states
            # NOTE: w2v2-vad returns dict with arousal, dominance, valence, and hidden_states
            vad_result = predictor.predict(str(audio_path), return_dict=True, return_hidden=True)
            
            arousal = vad_result['arousal']
            dominance = vad_result['dominance']
            valence = vad_result['valence']
            hidden_states = vad_result.get('hidden_states', None)
            
            # Check if VAD scores match emotion range
            if check_vad_in_range(arousal, valence, dominance, emotion):
                # Passed filter - save this sample
                filtered_sample = {
                    'key': key,
                    'audio_path': str(audio_path),
                    'emotion': emotion,
                    'speaker': speaker,
                    'text': text,
                    'vad': {
                        'arousal': float(arousal),
                        'valence': float(valence),
                        'dominance': float(dominance)
                    }
                }
                
                # Add hidden_states if available
                if hidden_states is not None:
                    # Convert to list for JSON serialization
                    if hasattr(hidden_states, 'tolist'):
                        filtered_sample['vad']['hidden_states'] = hidden_states.tolist()
                    else:
                        filtered_sample['vad']['hidden_states'] = hidden_states
                
                filtered_by_speaker[speaker].append(filtered_sample)
                stats[speaker][emotion]['passed'] += 1
        
        except Exception as e:
            print(f"\nError processing {key}: {e}")
            continue
    
    print()
    print("=" * 60)
    print("Filtering Complete!")
    print("=" * 60)
    print()
    
    # Save filtered data by speaker
    print("Saving filtered data...")
    for speaker, samples_list in filtered_by_speaker.items():
        output_file = output_dir / f"{speaker}_filtered.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples_list:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"  - {speaker}: {len(samples_list)} samples -> {output_file}")
    
    print()
    
    # Save statistics
    print("Statistics:")
    print()
    
    stats_output = output_dir / "filtering_stats.json"
    stats_report = {}
    
    for speaker in sorted(stats.keys()):
        print(f"Speaker: {speaker}")
        speaker_stats = {}
        
        for emotion in sorted(stats[speaker].keys()):
            total = stats[speaker][emotion]['total']
            passed = stats[speaker][emotion]['passed']
            pass_rate = (passed / total * 100) if total > 0 else 0
            
            speaker_stats[emotion] = {
                'total': total,
                'passed': passed,
                'pass_rate': pass_rate
            }
            
            print(f"  {emotion:12s}: {passed:4d}/{total:4d} ({pass_rate:5.1f}%)")
        
        stats_report[speaker] = speaker_stats
        print()
    
    # Overall statistics
    total_processed = len(samples) - skipped_unicode - skipped_audio_missing
    total_passed = sum(len(s) for s in filtered_by_speaker.values())
    overall_pass_rate = (total_passed / total_processed * 100) if total_processed > 0 else 0
    
    print(f"Overall:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Skipped (unicode): {skipped_unicode}")
    print(f"  Skipped (audio missing): {skipped_audio_missing}")
    print(f"  Processed: {total_processed}")
    print(f"  Passed: {total_passed} ({overall_pass_rate:.1f}%)")
    
    # Save stats to JSON
    stats_report['_summary'] = {
        'total_samples': len(samples),
        'skipped_unicode': skipped_unicode,
        'skipped_audio_missing': skipped_audio_missing,
        'processed': total_processed,
        'passed': total_passed,
        'overall_pass_rate': overall_pass_rate
    }
    
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(stats_report, f, indent=2, ensure_ascii=False)
    
    print(f"\nStatistics saved to: {stats_output}")
    
    print()
    print("=" * 60)
    print("Step 1 Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
