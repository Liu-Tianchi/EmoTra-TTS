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
Step 3: Generate Emotion Transition Training Data (Multi-GPU Parallel + ASR Quality Check)

Dependencies:
    pip install openai-whisper jiwer

Features:
1. Multi-GPU parallel processing with subprocess isolation
2. ASR quality verification using Whisper
3. Dynamic pair sampling to reach target count
4. Language-specific text normalization (Chinese/English)
5. Comprehensive ASR failure logging
"""

import json
import random
import argparse
import sys
import os
import time
import subprocess
import re
from pathlib import Path
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_jsonl(jsonl_path):
    """Load JSONL file"""
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def calculate_vad_change(vad_a, vad_b):
    """
    Calculate maximum VAD change across 3 dimensions (L-inf / Chebyshev distance).
    Consistent with tools/filter_vad_change.py.
    
    Returns:
        float: max(|VAD_B - VAD_A|) across arousal, valence, dominance
    """
    arousal_diff = abs(vad_b['arousal'] - vad_a['arousal'])
    valence_diff = abs(vad_b['valence'] - vad_a['valence'])
    dominance_diff = abs(vad_b['dominance'] - vad_a['dominance'])
    return max(arousal_diff, valence_diff, dominance_diff)


def save_jsonl(data, jsonl_path):
    """Save data to JSONL file"""
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def launch_gpu_worker(gpu_id, tasks_file, model_path, output_dir, whisper_model, cer_threshold, language):
    """Launch a worker subprocess for a specific GPU"""
    cmd = [
        sys.executable,
        __file__,
        '--worker-mode',
        '--worker-gpu', str(gpu_id),
        '--worker-tasks-file', str(tasks_file),
        '--model-path', model_path,
        '--output-dir', output_dir,
        '--whisper-model', whisper_model,
        '--cer-threshold', str(cer_threshold),
        '--language', language
    ]
    
    print(f"[GPU-{gpu_id}] Launching worker subprocess...", flush=True)
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    return {
        'gpu_id': gpu_id,
        'returncode': result.returncode,
        'tasks_file': tasks_file
    }


def main():
    parser = argparse.ArgumentParser(description='Step 3: Generate Transition Data (Parallel Multi-GPU + ASR)')
    parser.add_argument('--input-jsonl', type=str, 
                       default='./data/filtered/sage_filtered.jsonl')
    parser.add_argument('--num-pairs', type=int, default=50000,
                       help='Target number of successful pairs (after ASR filtering)')
    parser.add_argument('--output-dir', type=str,
                       default='./data/transition_data_asr/sage')
    parser.add_argument('--model-path', type=str,
                       default='pretrained_models/CosyVoice2-0.5B')
    parser.add_argument('--gpu-ids', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--whisper-model', type=str, default='large-v3',
                       help='Whisper model (tiny/base/small/medium/large/large-v3)')
    parser.add_argument('--cer-threshold', type=float, default=0.10,
                       help='CER threshold for ASR check (default: 0.10 = 10%%)')
    parser.add_argument('--language', type=str, default='en', choices=['zh', 'en'],
                       help='Language for ASR and text processing (zh=Chinese, en=English)')
    parser.add_argument('--oversample-ratio', type=float, default=2.0,
                       help='Initial sampling ratio (e.g., 2.0 = sample 2x pairs initially). Not all sampled pairs will be generated; the process stops once the predefined num-pairs limit is reached.')
    parser.add_argument('--vad-change-threshold', type=float, default=0.35,
                       help='Minimum VAD change (L-inf) between A and B to form a valid pair (default: 0.35)')
    
    # Internal args for worker mode
    parser.add_argument('--worker-mode', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--worker-gpu', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--worker-tasks-file', type=str, help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    # Worker mode - run single GPU worker
    if args.worker_mode:
        run_worker(args.worker_gpu, args.worker_tasks_file, args.model_path, 
                  args.output_dir, args.whisper_model, args.cer_threshold, args.language)
        return
    
    # Main coordinator mode
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    num_gpus = len(gpu_ids)
    
    random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / 'audio'
    audio_dir.mkdir(exist_ok=True)
    temp_dir = output_dir / 'temp'
    temp_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("Step 3: Generate Emotion Transition Data (Multi-GPU + ASR)")
    print(f"Multi-GPU Parallel Mode: {num_gpus} GPUs {gpu_ids}")
    print(f"Language: {'Chinese' if args.language == 'zh' else 'English'}")
    print(f"ASR Model: Whisper {args.whisper_model}")
    print(f"CER Threshold: {args.cer_threshold*100:.1f}%")
    print(f"VAD Change Threshold: {args.vad_change_threshold} (L-inf)")
    print("="*70)
    
    # Load data
    print(f"\n[1/4] Loading: {args.input_jsonl}")
    samples = load_jsonl(args.input_jsonl)
    print(f"✓ Loaded {len(samples)} samples")
    
    # Oversample pairs to account for ASR failures, with VAD change filtering
    print(f"\n[2/4] Sampling pairs with {args.oversample_ratio}x oversample ratio...")
    print(f"  VAD change threshold: {args.vad_change_threshold} (pairs with max VAD diff <= threshold will be skipped)")
    initial_pairs_count = int(args.num_pairs * args.oversample_ratio)
    
    pairs = []
    sampling_attempts = 0
    max_sampling_attempts = initial_pairs_count * 20  # safety limit to avoid infinite loop
    
    while len(pairs) < initial_pairs_count and sampling_attempts < max_sampling_attempts:
        sampling_attempts += 1
        sample_a = random.choice(samples)
        sample_b = random.choice(samples)
        
        # Check VAD distance between A and B
        vad_change = calculate_vad_change(sample_a['vad'], sample_b['vad'])
        if vad_change > args.vad_change_threshold:
            pairs.append((sample_a, sample_b))
    
    vad_pass_rate = len(pairs) / sampling_attempts * 100 if sampling_attempts > 0 else 0
    print(f"  Sampling attempts: {sampling_attempts}")
    print(f"  VAD filter pass rate: {vad_pass_rate:.1f}%")
    if len(pairs) < initial_pairs_count:
        print(f"  WARNING: Only found {len(pairs)} valid pairs (target: {initial_pairs_count}). "
              f"Consider lowering --vad-change-threshold or providing more diverse samples.")
    print(f"✓ Sampled {len(pairs)} initial pairs (target: {args.num_pairs} successful)")
    
    # Distribute tasks across GPUs
    print(f"\n[3/4] Distributing tasks across {num_gpus} GPUs...")
    tasks_per_gpu = [[] for _ in range(num_gpus)]
    for idx, (a, b) in enumerate(pairs):
        gpu_idx = idx % num_gpus
        tasks_per_gpu[gpu_idx].append({
            'idx': idx,
            'sample_a': a,
            'sample_b': b
        })
    
    print(f"\nTask distribution:")
    for i, gpu_id in enumerate(gpu_ids):
        print(f"  GPU {gpu_id}: {len(tasks_per_gpu[i])} tasks")
    
    # Save tasks to temp files
    tasks_files = []
    for i, gpu_id in enumerate(gpu_ids):
        if len(tasks_per_gpu[i]) > 0:
            tasks_file = temp_dir / f'tasks_gpu_{gpu_id}.jsonl'
            save_jsonl(tasks_per_gpu[i], tasks_file)
            tasks_files.append((gpu_id, tasks_file))
    
    # Launch all GPU workers in parallel
    print(f"\n{'='*70}")
    print(f"LAUNCHING {num_gpus} GPU WORKERS IN PARALLEL")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for gpu_id, tasks_file in tasks_files:
            future = executor.submit(
                launch_gpu_worker,
                gpu_id,
                str(tasks_file),
                args.model_path,
                args.output_dir,
                args.whisper_model,
                args.cer_threshold,
                args.language
            )
            futures.append(future)
        
        # Wait for all workers to complete
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
            gpu_id = result['gpu_id']
            if result['returncode'] == 0:
                print(f"\n✓ GPU {gpu_id} worker completed successfully", flush=True)
            else:
                print(f"\n✗ GPU {gpu_id} worker failed with code {result['returncode']}", flush=True)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"ALL GPU WORKERS COMPLETED")
    print(f"Total parallel execution time: {timedelta(seconds=int(elapsed))}")
    print(f"{'='*70}\n")
    
    # Cleanup temp files
    for gpu_id, tasks_file in tasks_files:
        if tasks_file.exists():
            tasks_file.unlink()
    
    # Collect results
    print(f"[4/4] Collecting results...")
    
    # Scan audio directory for successful outputs
    output_files = list(audio_dir.glob("*.wav"))
    
    # Load ASR results from workers
    asr_results_file = temp_dir / 'asr_results.jsonl'
    asr_results = []
    if asr_results_file.exists():
        asr_results = load_jsonl(asr_results_file)
        asr_results_file.unlink()
    
    # Build metadata for successful pairs
    output_samples = []
    hidden_states_samples = []
    for idx, (sample_a, sample_b) in enumerate(pairs):
        output_filename = f"{sample_a['speaker']}_{idx:05d}_{sample_a['key']}_to_{sample_b['key']}.wav"
        output_path = audio_dir / output_filename
        
        if output_path.exists():
            # Find ASR result for this pair
            asr_info = next((r for r in asr_results if r['idx'] == idx and r.get('passed')), None)
            
            # Remove hidden_states from vad data for main metadata
            vad_a = {k: v for k, v in sample_a['vad'].items() if k != 'hidden_states'}
            vad_b = {k: v for k, v in sample_b['vad'].items() if k != 'hidden_states'}
            
            # Convert to relative path using os.path.relpath (handles cross-mount paths)
            output_path_rel = os.path.relpath(output_path, Path.cwd()) if output_path.is_absolute() else str(output_path)
            sample_a_path_rel = os.path.relpath(sample_a['audio_path'], Path.cwd()) if Path(sample_a['audio_path']).is_absolute() else sample_a['audio_path']
            sample_b_path_rel = os.path.relpath(sample_b['audio_path'], Path.cwd()) if Path(sample_b['audio_path']).is_absolute() else sample_b['audio_path']
            
            output_sample = {
                'key': f"transition_{idx:05d}",
                'audio_path': output_path_rel,
                'speaker': sample_a['speaker'],
                'sample_a': {
                    'audio_filename': Path(sample_a['audio_path']).name,
                    'emotion': sample_a['emotion'],
                    'vad': vad_a,
                    'text': sample_a['text'],
                    'audio_path': sample_a_path_rel
                },
                'sample_b': {
                    'audio_filename': Path(sample_b['audio_path']).name,
                    'emotion': sample_b['emotion'],
                    'vad': vad_b,
                    'text': sample_b['text'],
                    'audio_path': sample_b_path_rel
                }
            }
            
            if asr_info:
                output_sample['asr_check'] = {
                    'passed': True,
                    'cer': asr_info['cer'],
                    'transcription': asr_info['transcription']
                }
            
            output_samples.append(output_sample)
            
            # Save hidden_states separately
            hidden_states_sample = {
                'key': f"transition_{idx:05d}",
                'audio_path': output_path_rel,
                'hidden_states_a': sample_a['vad'].get('hidden_states', None),
                'hidden_states_b': sample_b['vad'].get('hidden_states', None)
            }
            hidden_states_samples.append(hidden_states_sample)
    
    # Truncate to target if we have more
    if len(output_samples) > args.num_pairs:
        output_samples = output_samples[:args.num_pairs]
        hidden_states_samples = hidden_states_samples[:args.num_pairs]
    
    # Save main metadata (without hidden_states)
    output_jsonl = output_dir / 'transition_data_asr_filtered.jsonl'
    output_samples.sort(key=lambda x: x['key'])
    save_jsonl(output_samples, output_jsonl)
    
    # Save hidden_states separately
    hidden_states_jsonl = output_dir / 'transition_data_asr_hidden_states.jsonl'
    hidden_states_samples.sort(key=lambda x: x['key'])
    save_jsonl(hidden_states_samples, hidden_states_jsonl)
    
    # Load and save ASR failure log
    asr_failures = [r for r in asr_results if not r.get('passed', True)]
    if asr_failures:
        asr_log_path = output_dir / 'asr_failures.jsonl'
        save_jsonl(asr_failures, asr_log_path)
    
    # Cleanup temp dir
    try:
        temp_dir.rmdir()
    except:
        pass
    
    # Calculate statistics
    total_successful = len(output_samples)
    total_failed = len(asr_failures)
    asr_pass_rate = total_successful / (total_successful + total_failed) * 100 if (total_successful + total_failed) > 0 else 0
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"  Target pairs: {args.num_pairs}")
    print(f"  Successful (saved): {total_successful}")
    print(f"  ASR failures: {total_failed}")
    print(f"  Total attempts: {total_successful + total_failed}")
    print(f"  ASR Pass Rate: {asr_pass_rate:.2f}%")
    print(f"\n  Total time: {timedelta(seconds=int(elapsed))}")
    print(f"  Throughput: {total_successful/elapsed*3600:.1f} pairs/hour")
    print(f"\n  Audio: {audio_dir}")
    print(f"  Metadata: {output_jsonl}")
    print(f"  Hidden States: {hidden_states_jsonl}")
    if asr_failures:
        print(f"  ASR failures log: {output_dir / 'asr_failures.jsonl'}")
    print("="*70)
    print("✓ Complete!")
    print("="*70)


def run_worker(gpu_id, tasks_file, model_path, output_dir, whisper_model, cer_threshold, language):
    """Worker function that runs in isolated subprocess"""
    # Set GPU BEFORE any imports
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['TQDM_DISABLE'] = '1'
    
    worker_id = f"GPU-{gpu_id}"
    print(f"\n[{worker_id}] ==================== WORKER STARTING ====================", flush=True)
    print(f"[{worker_id}] PID: {os.getpid()}", flush=True)
    print(f"[{worker_id}] Physical GPU ID: {gpu_id}", flush=True)
    print(f"[{worker_id}] Language: {'Chinese' if language == 'zh' else 'English'}", flush=True)
    print(f"[{worker_id}] CER Threshold: {cer_threshold*100:.1f}%", flush=True)
    
    if not Path(tasks_file).exists():
        print(f"[{worker_id}] ERROR: Tasks file not found: {tasks_file}", flush=True)
        sys.exit(1)
    
    print(f"[{worker_id}] =========================================================\n", flush=True)
    
    # Now import torch and other libraries
    import torch
    import torchaudio
    import whisper
    from jiwer import cer as calculate_cer
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Add CosyVoice to path
    sys.path.insert(0, str(Path(__file__).parent))
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.cli.model_w_mid_state import CosyVoice2ModelWithMidState
    
    def resample_audio(audio, orig_sr, target_sr=24000):
        if orig_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
            audio = resampler(audio)
        return audio
    
    def load_and_resample_audio(audio_path, target_sr=24000):
        audio, sr = torchaudio.load(audio_path)
        audio = resample_audio(audio, sr, target_sr)
        return audio, target_sr
    
    def normalize_text_chinese(text):
        """Normalize Chinese text for comparison"""
        # Remove all punctuation
        text = re.sub(r'[，。！？、；：""''（）《》【】…—~·\s]', '', text)
        text = re.sub(r'[,\.!?;:\'"()\[\]\-\s]', '', text)
        return text.lower()
    
    def normalize_text_english(text):
        """Normalize English text for comparison"""
        # Remove punctuation
        text = re.sub(r'[,\.!?;:\'"()\[\]\-]', '', text)
        # Normalize spaces
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
    
    def check_asr_quality(audio_tensor, sample_rate, expected_text, whisper_model, cer_threshold, language):
        """Check if generated audio passes ASR quality check"""
        try:
            # Convert to mono if stereo
            if audio_tensor.shape[0] > 1:
                audio_mono = audio_tensor.mean(dim=0, keepdim=True)
            else:
                audio_mono = audio_tensor
            
            # Resample to 16kHz for Whisper
            if sample_rate != 16000:
                audio_16k = resample_audio(audio_mono, sample_rate, 16000)
            else:
                audio_16k = audio_mono
            
            # Convert to numpy for Whisper
            audio_np = audio_16k.squeeze(0).numpy()
            
            # Transcribe with Whisper
            result = whisper_model.transcribe(
                audio_np,
                language=language,
                task='transcribe',
                fp16=torch.cuda.is_available()
            )
            transcription = result['text'].strip()
            
            # Normalize texts based on language
            if language == 'zh':
                expected_normalized = normalize_text_chinese(expected_text)
                transcribed_normalized = normalize_text_chinese(transcription)
            else:  # en
                expected_normalized = normalize_text_english(expected_text)
                transcribed_normalized = normalize_text_english(transcription)
            
            # Calculate Character Error Rate
            if len(expected_normalized) == 0:
                cer_score = 1.0 if len(transcribed_normalized) > 0 else 0.0
            else:
                cer_score = calculate_cer(expected_normalized, transcribed_normalized)
            
            # Pass if CER is below threshold
            passed = cer_score <= cer_threshold
            
            return passed, cer_score, transcription
            
        except Exception as e:
            print(f"[{worker_id}] ASR check error: {e}", flush=True)
            return False, 1.0, ""
    
    def generate_transition_audio(cosyvoice, sample_a, sample_b):
        """Generate transition audio"""
        for audio_data in cosyvoice.inference_zero_shot_w_mid_state(
            tts_text=sample_a['text'],
            prompt_text_A=sample_a['text'],
            prompt_text_B=sample_b['text'],
            prompt_speech_16k_A=sample_a['audio_path'],
            prompt_speech_16k_B=sample_b['audio_path'],
            stream=False
        ):
            generated_audio = audio_data['tts_speech']
            if cosyvoice.sample_rate != 24000:
                generated_audio = resample_audio(generated_audio, cosyvoice.sample_rate, 24000)
            return generated_audio
        return None
    
    # Load tasks
    tasks = load_jsonl(tasks_file)
    audio_dir = Path(output_dir) / 'audio'
    temp_dir = Path(output_dir) / 'temp'
    
    print(f"[{worker_id}] Tasks to process: {len(tasks)}", flush=True)
    
    if torch.cuda.is_available():
        print(f"[{worker_id}] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
        torch.cuda.empty_cache()
    
    # Load models
    try:
        print(f"[{worker_id}] Loading CosyVoice2 model...", flush=True)
        start = time.time()
        
        cosyvoice = CosyVoice2(
            model_path,
            load_jit=False,
            load_trt=False,
            load_vllm=False,
            fp16=False
        )
        
        cosyvoice.model.__class__ = CosyVoice2ModelWithMidState
        print(f"[{worker_id}] ✓ CosyVoice2 loaded ({time.time()-start:.1f}s)", flush=True)
        
        print(f"[{worker_id}] Loading Whisper model '{whisper_model}'...", flush=True)
        start = time.time()
        whisper_model_obj = whisper.load_model(whisper_model)
        print(f"[{worker_id}] ✓ Whisper loaded ({time.time()-start:.1f}s)", flush=True)
        
    except Exception as e:
        print(f"[{worker_id}] ✗ Model load failed: {e}", flush=True)
        sys.exit(1)
    
    # Process tasks
    print(f"[{worker_id}] Starting to process {len(tasks)} tasks...\n", flush=True)
    success = 0
    failed_asr = 0
    failed_gen = 0
    asr_results = []
    processing_start_time = time.time()
    
    for task_num, task in enumerate(tasks, 1):
        idx = task['idx']
        sample_a = task['sample_a']
        sample_b = task['sample_b']
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        task_start = time.time()
        
        try:
            # Generate with retry
            generated_audio = None
            for attempt in range(2):
                try:
                    generated_audio = generate_transition_audio(cosyvoice, sample_a, sample_b)
                    if generated_audio is not None:
                        break
                except Exception as e:
                    if attempt == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        time.sleep(0.5)
                    else:
                        raise
            
            if generated_audio is None:
                failed_gen += 1
                print(f"[{worker_id}] [{task_num:3d}/{len(tasks)}] ✗ Pair {idx:3d} - Generation failed", flush=True)
                continue
            
            # ASR quality check
            asr_passed, cer_score, transcription = check_asr_quality(
                generated_audio, 24000, sample_a['text'], 
                whisper_model_obj, cer_threshold, language
            )
            
            # Record ASR result
            asr_result = {
                'idx': idx,
                'passed': asr_passed,
                'cer': cer_score,
                'transcription': transcription,
                'expected': sample_a['text']
            }
            asr_results.append(asr_result)
            
            if not asr_passed:
                failed_asr += 1
                print(f"[{worker_id}] [{task_num:3d}/{len(tasks)}] ✗ Pair {idx:3d} - ASR failed (CER: {cer_score:.3f})", flush=True)
                print(f"[{worker_id}]   Expected: {sample_a['text']}", flush=True)
                print(f"[{worker_id}]   ASR Got:  {transcription}", flush=True)
                continue
            
            # Load B audio
            audio_b, _ = load_and_resample_audio(sample_b['audio_path'], target_sr=24000)
            
            # Concatenate
            if generated_audio.dim() == 1:
                generated_audio = generated_audio.unsqueeze(0)
            if audio_b.dim() == 1:
                audio_b = audio_b.unsqueeze(0)
            
            if generated_audio.shape[0] != audio_b.shape[0]:
                if generated_audio.shape[0] == 1:
                    generated_audio = generated_audio.repeat(audio_b.shape[0], 1)
                elif audio_b.shape[0] == 1:
                    audio_b = audio_b.repeat(generated_audio.shape[0], 1)
            
            concatenated_audio = torch.cat([generated_audio, audio_b], dim=1)
            
            # Save
            output_filename = f"{sample_a['speaker']}_{idx:05d}_{sample_a['key']}_to_{sample_b['key']}.wav"
            output_path = audio_dir / output_filename
            torchaudio.save(str(output_path), concatenated_audio, 24000)
            
            success += 1
            elapsed = time.time() - task_start
            
            # Print progress every 5 tasks or at milestones
            if task_num % 5 == 0 or task_num == len(tasks):
                # Calculate ETA
                total_elapsed = time.time() - processing_start_time
                avg_time_per_task = total_elapsed / task_num
                remaining_tasks = len(tasks) - task_num
                eta_seconds = avg_time_per_task * remaining_tasks
                eta_str = str(timedelta(seconds=int(eta_seconds)))
                
                print(f"[{worker_id}] [{task_num:3d}/{len(tasks)}] ✓ Pair {idx:3d} ({elapsed:.1f}s) CER:{cer_score:.3f} | Success: {success}, ASR-Fail: {failed_asr}, Gen-Fail: {failed_gen} | ETA: {eta_str}", flush=True)
            
        except Exception as e:
            failed_gen += 1
            print(f"[{worker_id}] [{task_num:3d}/{len(tasks)}] ✗ Pair {idx:3d} - Error: {str(e)[:60]}", flush=True)
    
    # Save ASR results
    asr_results_file = temp_dir / f'asr_results_gpu_{gpu_id}.jsonl'
    save_jsonl(asr_results, asr_results_file)
    
    # Merge into global ASR results
    global_asr_file = temp_dir / 'asr_results.jsonl'
    if global_asr_file.exists():
        existing = load_jsonl(global_asr_file)
        asr_results = existing + asr_results
    save_jsonl(asr_results, global_asr_file)
    asr_results_file.unlink()
    
    asr_pass_rate = success / (success + failed_asr) * 100 if (success + failed_asr) > 0 else 0
    
    print(f"\n[{worker_id}] ==================== WORKER COMPLETE ====================", flush=True)
    print(f"[{worker_id}] Total: {len(tasks)} | Success: {success} | ASR-Fail: {failed_asr} | Gen-Fail: {failed_gen}", flush=True)
    print(f"[{worker_id}] ASR Pass Rate: {asr_pass_rate:.1f}%", flush=True)
    print(f"[{worker_id}] =========================================================\n", flush=True)


if __name__ == '__main__':
    main()
