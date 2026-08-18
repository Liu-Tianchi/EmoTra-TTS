#!/usr/bin/env python3
"""
VAD Wrapper for CosyVoice Integration

This module provides a Python API to call the w2v2-vad model
without requiring direct command-line execution.

Usage:
    from tools.vad_wrapper import VADPredictor

    predictor = VADPredictor()
    arousal, dominance, valence = predictor.predict('/path/to/audio.wav')
"""

import os
import sys
import numpy as np
from typing import Tuple, List, Dict, Union

# Add the current directory to path to import w2v2-vad dependencies
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import audeer
import audonnx
import audresample
import audiofile


class VADPredictor:
    """
    Wrapper class for w2v2-vad model prediction.
    
    This class handles model loading and prediction of VAD scores
    (Valence, Arousal, Dominance) from audio files.
    """
    
    def __init__(self, model_root: str = None):
        """
        Initialize VAD predictor.
        
        Args:
            model_root: Path to store/load the model.
                       Defaults to <repo>/third_party/models/w2v2-vad
                       (i.e. inside this repository, not the home directory).
        """
        if model_root is None:
            # tools/vad_wrapper.py -> <repo>/third_party/models/w2v2-vad
            _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_root = os.path.join(_repo_root, 'third_party', 'models', 'w2v2-vad')
        
        # Create directories
        if not os.path.exists(model_root):
            os.makedirs(model_root, exist_ok=True)
        
        self.model_root = audeer.mkdir(os.path.join(model_root, 'model'))
        self.cache_root = audeer.mkdir(os.path.join(model_root, 'cache'))
        
        # Download model if not exists
        self._download_model_if_needed()
        
        # Load model
        self.model = audonnx.load(self.model_root)
        self.target_sr = 16000
    
    def _download_model_if_needed(self):
        """Download w2v2-vad model if not already present."""
        mdl_path = os.path.join(self.model_root, 'model.onnx')
        
        if not os.path.exists(mdl_path):
            print(f"Downloading w2v2-vad model to {self.model_root}...")
            url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
            dst_path = os.path.join(self.cache_root, 'model.zip')
            
            audeer.download_url(url, dst_path, verbose=True)
            audeer.extract_archive(dst_path, self.model_root, verbose=True)
            print("Model download complete.")
    
    def predict(self, audio_path: str, return_dict: bool = False, return_hidden: bool = False) -> Union[Tuple[float, float, float], Dict[str, float]]:
        """
        Predict VAD scores for a single audio file.
        
        Args:
            audio_path: Path to audio file
            return_dict: If True, return dict with keys ['arousal', 'dominance', 'valence']
                        If False, return tuple (arousal, dominance, valence)
            return_hidden: If True, include hidden_states in returned dict (requires return_dict=True)
        
        Returns:
            VAD scores in range [0, 1]
        """
        # Read audio
        wav, fs = audiofile.read(audio_path)
        
        # Resample if needed
        if fs != self.target_sr:
            wav = audresample.resample(wav, fs, self.target_sr)
        
        # Predict
        pred = self.model(wav, self.target_sr)
        logits = pred['logits'].flatten()
        
        arousal, dominance, valence = float(logits[0]), float(logits[1]), float(logits[2])
        
        if return_dict:
            result = {
                'arousal': arousal,
                'dominance': dominance,
                'valence': valence
            }
            if return_hidden and 'hidden_states' in pred:
                result['hidden_states'] = pred['hidden_states']
            return result
        else:
            return arousal, dominance, valence
    
    def predict_chunks(self, audio_path: str, chunk_duration: int = 10) -> List[Dict[str, float]]:
        """
        Predict VAD scores for audio file split into chunks.
        
        Args:
            audio_path: Path to audio file
            chunk_duration: Duration of each chunk in seconds
        
        Returns:
            List of dicts, each containing {'arousal', 'dominance', 'valence', 'start_time', 'end_time'}
        """
        # Read audio
        wav, fs = audiofile.read(audio_path)
        
        # Resample if needed
        if fs != self.target_sr:
            wav = audresample.resample(wav, fs, self.target_sr)
        
        results = []
        chunk_samples = self.target_sr * chunk_duration
        num_full_chunks = wav.shape[0] // chunk_samples
        
        # Process full chunks
        for i in range(num_full_chunks):
            start_sample = i * chunk_samples
            end_sample = (i + 1) * chunk_samples
            chunk_wav = wav[start_sample:end_sample]
            
            pred = self.model(chunk_wav, self.target_sr)
            logits = pred['logits'].flatten()
            
            results.append({
                'arousal': float(logits[0]),
                'dominance': float(logits[1]),
                'valence': float(logits[2]),
                'start_time': i * chunk_duration,
                'end_time': (i + 1) * chunk_duration
            })
        
        # Process last chunk if exists
        remaining_samples = wav.shape[0] % chunk_samples
        if remaining_samples != 0:
            chunk_wav = wav[-remaining_samples:]
            pred = self.model(chunk_wav, self.target_sr)
            logits = pred['logits'].flatten()
            
            start_time = num_full_chunks * chunk_duration
            end_time = start_time + remaining_samples / self.target_sr
            
            results.append({
                'arousal': float(logits[0]),
                'dominance': float(logits[1]),
                'valence': float(logits[2]),
                'start_time': start_time,
                'end_time': end_time
            })
        
        return results


def predict_vad_simple(audio_path: str) -> Tuple[float, float, float]:
    """
    Simple function to predict VAD scores without needing to instantiate class.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        (arousal, dominance, valence) tuple
    """
    predictor = VADPredictor()
    return predictor.predict(audio_path)


if __name__ == '__main__':
    # Test the wrapper
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict VAD scores using wrapper')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input audio file')
    parser.add_argument('-c', '--chunks', action='store_true', help='Process in chunks')
    parser.add_argument('-d', '--duration', type=int, default=10, help='Chunk duration in seconds')
    args = parser.parse_args()
    
    predictor = VADPredictor()
    
    if args.chunks:
        results = predictor.predict_chunks(args.input, args.duration)
        for i, result in enumerate(results):
            print(f"Chunk {i} ({result['start_time']:.2f}s - {result['end_time']:.2f}s):")
            print(f"  Arousal: {result['arousal']:.4f}")
            print(f"  Dominance: {result['dominance']:.4f}")
            print(f"  Valence: {result['valence']:.4f}")
    else:
        arousal, dominance, valence = predictor.predict(args.input)
        print(f"Arousal: {arousal:.4f}")
        print(f"Dominance: {dominance:.4f}")
        print(f"Valence: {valence:.4f}")
