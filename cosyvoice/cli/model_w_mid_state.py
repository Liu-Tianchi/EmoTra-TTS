# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
#               2026 Continuous Emotion Transfer Extension (w_mid_state)
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
CosyVoice2Model with w_mid_state Continuous Emotion Transfer Extension

This extends the original CosyVoice2Model with zero-loss continuous emotion 
transfer capability using in-place mel blending.
"""

import torch
import torch.nn.functional as F
import numpy as np
from cosyvoice.cli.model import CosyVoice2Model


class CosyVoice2ModelWithMidState(CosyVoice2Model):
    """
    CosyVoice2Model extended with w_mid_state continuous emotion transfer.
    
    Key features:
    - Zero-loss architecture: Generate 3 mels from same token sequence
    - In-place blending: Smooth transitions without concatenation
    - Auto transition calculation: Smart duration allocation
    """
    
    def _ensure_attributes(self):
        """Ensure required attributes exist (for dynamic class replacement)."""
        if not hasattr(self, 'mel_overlap_dict'):
            self.mel_overlap_dict = {}
        if not hasattr(self, 'flow_cache_dict'):
            self.flow_cache_dict = {}
        if not hasattr(self, 'mel_overlap_len'):
            # Calculate from flow parameters
            self.mel_overlap_len = int(20 / 12.5 * 22050 / 256)  # ~35 frames
        if not hasattr(self, 'mel_window'):
            self.mel_window = np.hamming(2 * self.mel_overlap_len)
    
    def _calculate_auto_transitions(self, total_duration):
        """
        Auto-calculate transition points and crossfade duration.
        
        New Rules:
        - Stage1: 60% of total duration
        - Stage2: 15% of total duration  
        - Stage3: 25% of total duration
        - Crossfade: max 0.4s, max 10% of total, symmetric (±duration/2)
        - Sigmoid curve with steepness=12
        
        Args:
            total_duration: Total audio duration in seconds
            
        Returns:
            dict with transition1_sec, transition2_sec, crossfade_duration
        """
        # Fixed percentages
        stage1_duration = total_duration * 0.60
        stage2_duration = total_duration * 0.15
        stage3_duration = total_duration * 0.25
        
        # Calculate crossfade duration with constraints
        crossfade_duration = min(0.4, total_duration * 0.10)
        
        # Ensure crossfades don't overlap in stage2
        # Two crossfades each extend ±duration/2, total span = 2 * crossfade_duration
        if 2 * crossfade_duration > stage2_duration:
            crossfade_duration = stage2_duration / 2.0
        
        # Calculate transition points (centers)
        transition1_sec = stage1_duration
        transition2_sec = stage1_duration + stage2_duration
        
        return {
            'transition1_sec': transition1_sec,
            'transition2_sec': transition2_sec,
            'crossfade_duration': crossfade_duration,
            'stage1_duration': stage1_duration,
            'stage2_duration': stage2_duration,
            'stage3_duration': stage3_duration,
        }
    
    def tts(self,
            text=torch.zeros(1, 0, dtype=torch.int32),
            flow_embedding=torch.zeros(0, 192),
            llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80),
            source_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_embedding_B=torch.zeros(0, 192),
            prompt_text_B=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat_B=torch.zeros(1, 0, 80),
            flow_prompt_speech_token_B=torch.zeros(1, 0, dtype=torch.int32),
            stream=False,
            speed=1.0,
            **kwargs):
        """
        TTS with optional w_mid_state continuous emotion transfer.
        
        If flow_embedding_B, prompt_speech_feat_B, flow_prompt_speech_token_B 
        are provided, enables w_mid_state dual-emotion mode with auto transition.
        Otherwise, falls back to standard single-emotion mode.
        """
        # Ensure required attributes exist (for dynamic class replacement)
        self._ensure_attributes()
        
        # Check if dual-emotion mode is enabled
        use_dual_emotion = (flow_embedding_B.numel() > 0 and 
                           prompt_speech_feat_B.numel() > 0 and 
                           flow_prompt_speech_token_B.numel() > 0)
        
        if not use_dual_emotion:
            # Fall back to original single-emotion mode
            return super().tts(
                text=text,
                flow_embedding=flow_embedding,
                llm_embedding=llm_embedding,
                prompt_text=prompt_text,
                llm_prompt_speech_token=llm_prompt_speech_token,
                flow_prompt_speech_token=flow_prompt_speech_token,
                prompt_speech_feat=prompt_speech_feat,
                source_speech_token=source_speech_token,
                stream=stream,
                speed=speed,
                **kwargs
            )
        
        # w_mid_state dual-emotion mode
        import uuid
        this_uuid = str(uuid.uuid1())
        
        with self.lock:
            self.tts_speech_token_dict[this_uuid] = []
            self.llm_end_dict[this_uuid] = False
            self.hift_cache_dict[this_uuid] = None
            self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
            self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
        
        # Generate complete token sequence once
        if source_speech_token.shape[1] == 0:
            with self.llm_context:
                for i in self.llm.inference(text=text.to(self.device),
                                           text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                                           prompt_text=prompt_text.to(self.device),
                                           prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                           prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                           prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                           embedding=llm_embedding.to(self.device),
                                           uuid=this_uuid):
                    self.tts_speech_token_dict[this_uuid].append(i)
            self.llm_end_dict[this_uuid] = True
            tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
        else:
            tts_speech_token = source_speech_token
            self.llm_end_dict[this_uuid] = True
        
        # Generate 3 mel spectrograms from same token sequence
        device = self.device
        
        # Mel A: Emotion A (prompt_A + embedding_A)
        with torch.cuda.amp.autocast(self.fp16):
            mel_A, _ = self.flow.inference(
                token=tts_speech_token.to(device, dtype=torch.int32),
                token_len=torch.tensor([tts_speech_token.shape[1]], dtype=torch.int32).to(device),
                prompt_token=flow_prompt_speech_token.to(device),
                prompt_token_len=torch.tensor([flow_prompt_speech_token.shape[1]], dtype=torch.int32).to(device),
                prompt_feat=prompt_speech_feat.to(device),
                prompt_feat_len=torch.tensor([prompt_speech_feat.shape[1]], dtype=torch.int32).to(device),
                embedding=flow_embedding.to(device),
                streaming=False,
                finalize=True
            )
        
        # Mel AB: Mixed emotion (prompt_A + embedding_B)
        with torch.cuda.amp.autocast(self.fp16):
            mel_AB, _ = self.flow.inference(
                token=tts_speech_token.to(device, dtype=torch.int32),
                token_len=torch.tensor([tts_speech_token.shape[1]], dtype=torch.int32).to(device),
                prompt_token=flow_prompt_speech_token.to(device),
                prompt_token_len=torch.tensor([flow_prompt_speech_token.shape[1]], dtype=torch.int32).to(device),
                prompt_feat=prompt_speech_feat.to(device),
                prompt_feat_len=torch.tensor([prompt_speech_feat.shape[1]], dtype=torch.int32).to(device),
                embedding=flow_embedding_B.to(device),
                streaming=False,
                finalize=True
            )
        
        # Mel B: Emotion B (prompt_B + embedding_B)
        with torch.cuda.amp.autocast(self.fp16):
            mel_B, _ = self.flow.inference(
                token=tts_speech_token.to(device, dtype=torch.int32),
                token_len=torch.tensor([tts_speech_token.shape[1]], dtype=torch.int32).to(device),
                prompt_token=flow_prompt_speech_token_B.to(device),
                prompt_token_len=torch.tensor([flow_prompt_speech_token_B.shape[1]], dtype=torch.int32).to(device),
                prompt_feat=prompt_speech_feat_B.to(device),
                prompt_feat_len=torch.tensor([prompt_speech_feat_B.shape[1]], dtype=torch.int32).to(device),
                embedding=flow_embedding_B.to(device),
                streaming=False,
                finalize=True
            )
        
        # Auto-calculate transition points
        total_mel_frames = mel_A.shape[2]
        mel_fps = 50.0  # 24000 Hz / 480 samples per frame
        total_duration = total_mel_frames / mel_fps
        
        trans_params = self._calculate_auto_transitions(total_duration)
        stage1_duration = trans_params['stage1_duration']
        stage2_duration = trans_params['stage2_duration']
        stage3_duration = trans_params['stage3_duration']
        crossfade_duration = trans_params['crossfade_duration']
        transition1_sec = trans_params['transition1_sec']
        transition2_sec = trans_params['transition2_sec']
        
        # import logging
        # logging.info(f"[W_MID_STATE] ===== CALCULATED PARAMS =====")
        # logging.info(f"[W_MID_STATE] total_mel_frames: {total_mel_frames}, total_duration: {total_duration:.2f}s")
        # logging.info(f"[W_MID_STATE] stage1_duration: {stage1_duration:.2f}s")
        # logging.info(f"[W_MID_STATE] stage2_duration: {stage2_duration:.2f}s")
        # logging.info(f"[W_MID_STATE] stage3_duration: {stage3_duration:.2f}s")
        # logging.info(f"[W_MID_STATE] crossfade_duration: {crossfade_duration:.2f}s")
        # logging.info(f"[W_MID_STATE] transition1_sec: {transition1_sec:.2f}s")
        # logging.info(f"[W_MID_STATE] transition2_sec: {transition2_sec:.2f}s")
        
        # Convert durations to frame indices
        stage1_frames = int(stage1_duration * mel_fps)
        stage2_frames = int(stage2_duration * mel_fps)
        crossfade_frames = int(crossfade_duration * mel_fps)
        
        # logging.info(f"[W_MID_STATE] stage1_frames: {stage1_frames}")
        # logging.info(f"[W_MID_STATE] stage2_frames: {stage2_frames}")
        # logging.info(f"[W_MID_STATE] crossfade_frames: {crossfade_frames}")
        
        # Calculate transition points (centers) in frames
        transition1_frame = int(transition1_sec * mel_fps)
        transition2_frame = int(transition2_sec * mel_fps)
        
        # Calculate crossfade boundaries (symmetric around transition points: ±duration/2)
        half_crossfade = crossfade_frames // 2
        
        # Crossfade 1: A -> AB (centered at transition1)
        fade1_start = max(0, transition1_frame - half_crossfade)
        fade1_end = min(total_mel_frames, transition1_frame + half_crossfade)
        
        # Crossfade 2: AB -> B (centered at transition2)
        fade2_start = max(0, transition2_frame - half_crossfade)
        fade2_end = min(total_mel_frames, transition2_frame + half_crossfade)
        
        # logging.info(f"[W_MID_STATE] ===== TRANSITION POINTS =====")
        # logging.info(f"[W_MID_STATE] transition1_frame: {transition1_frame} ({transition1_sec:.2f}s)")
        # logging.info(f"[W_MID_STATE] transition2_frame: {transition2_frame} ({transition2_sec:.2f}s)")
        # logging.info(f"[W_MID_STATE] Crossfade1: {fade1_start} - {fade1_end} frames ({fade1_start/mel_fps:.2f} - {fade1_end/mel_fps:.2f}s)")
        # logging.info(f"[W_MID_STATE] Crossfade2: {fade2_start} - {fade2_end} frames ({fade2_start/mel_fps:.2f} - {fade2_end/mel_fps:.2f}s)")
        
        # Build final mel by concatenating segments with crossfades
        tts_mel_segments = []
        
        # Segment 1: Pure A (before crossfade1)
        if fade1_start > 0:
            tts_mel_segments.append(mel_A[:, :, :fade1_start])
            # logging.info(f"[W_MID_STATE] Seg1 pure: 0 - {fade1_start} frames (0.00 - {fade1_start/mel_fps:.2f}s)")
        
        # Crossfade 1: A -> AB
        fade1_len = fade1_end - fade1_start
        if fade1_len > 0:
            t1 = torch.linspace(0, 1, fade1_len, device=device).view(1, 1, -1)
            steepness = 12
            fade1_curve = 1 / (1 + torch.exp(-steepness * (t1 - 0.5)))
            
            crossfade1_seg = (
                mel_A[:, :, fade1_start:fade1_end] * (1 - fade1_curve) +
                mel_AB[:, :, fade1_start:fade1_end] * fade1_curve
            )
            tts_mel_segments.append(crossfade1_seg)
            # logging.info(f"[W_MID_STATE] Crossfade1 applied: {fade1_start} - {fade1_end} frames")
        
        # Segment 2: Pure AB (between two crossfades)
        seg2_start = fade1_end
        seg2_end = fade2_start
        
        if seg2_end > seg2_start:
            tts_mel_segments.append(mel_AB[:, :, seg2_start:seg2_end])
            # logging.info(f"[W_MID_STATE] Seg2 pure: {seg2_start} - {seg2_end} frames ({seg2_start/mel_fps:.2f} - {seg2_end/mel_fps:.2f}s)")
        
        # Crossfade 2: AB -> B
        fade2_len = fade2_end - fade2_start
        if fade2_len > 0:
            t2 = torch.linspace(0, 1, fade2_len, device=device).view(1, 1, -1)
            fade2_curve = 1 / (1 + torch.exp(-steepness * (t2 - 0.5)))
            
            crossfade2_seg = (
                mel_AB[:, :, fade2_start:fade2_end] * (1 - fade2_curve) +
                mel_B[:, :, fade2_start:fade2_end] * fade2_curve
            )
            tts_mel_segments.append(crossfade2_seg)
            # logging.info(f"[W_MID_STATE] Crossfade2 applied: {fade2_start} - {fade2_end} frames")
        
        # Segment 3: Pure B (after crossfade2)
        if fade2_end < total_mel_frames:
            tts_mel_segments.append(mel_B[:, :, fade2_end:total_mel_frames])
            # logging.info(f"[W_MID_STATE] Seg3 pure: {fade2_end} - {total_mel_frames} frames ({fade2_end/mel_fps:.2f} - {total_mel_frames/mel_fps:.2f}s)")
        
        # Concatenate all segments
        tts_mel = torch.cat(tts_mel_segments, dim=2)
        
        # logging.info(f"[W_MID_STATE] ===== FINAL MEL =====")
        # logging.info(f"[W_MID_STATE] Final mel frames: {tts_mel.shape[2]}, duration: {tts_mel.shape[2]/mel_fps:.2f}s")
        
        # Hift vocoding
        if speed != 1.0:
            tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
        
        tts_speech, _ = self.hift.inference(speech_feat=tts_mel, cache_source=torch.zeros(1, 1, 0))
        
        yield {'tts_speech': tts_speech.cpu()}
