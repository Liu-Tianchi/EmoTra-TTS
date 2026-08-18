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
# EmoTra sft_Flow — CausalConditionalDecoder with 3D spks [B, C, T] support
# Directly inherits from CausalConditionalDecoder, no V-series dependencies.

import torch
from einops import pack, rearrange, repeat
from cosyvoice.utils.common import mask_to_bias
from cosyvoice.utils.mask import add_optional_chunk_mask
from cosyvoice.flow.decoder import CausalConditionalDecoder


class CausalConditionalDecoder_sft_Flow(CausalConditionalDecoder):
    """CausalConditionalDecoder with 3D spks support."""

    def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
        t = self.time_embeddings(t).to(t.dtype)
        t = self.time_mlp(t)

        x = pack([x, mu], "b * t")[0]

        if spks is not None:
            if spks.dim() == 2:
                spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            # else: already [B, C, T]
            x = pack([x, spks], "b * t")[0]
        if cond is not None:
            x = pack([x, cond], "b * t")[0]

        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            if streaming is True:
                attn_mask = add_optional_chunk_mask(x, mask_down.bool(), False, False, 0, self.static_chunk_size, -1)
            else:
                attn_mask = add_optional_chunk_mask(x, mask_down.bool(), False, False, 0, 0, -1).repeat(1, x.size(1), 1)
            attn_mask = mask_to_bias(attn_mask, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t").contiguous()
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            if streaming is True:
                attn_mask = add_optional_chunk_mask(x, mask_mid.bool(), False, False, 0, self.static_chunk_size, -1)
            else:
                attn_mask = add_optional_chunk_mask(x, mask_mid.bool(), False, False, 0, 0, -1).repeat(1, x.size(1), 1)
            attn_mask = mask_to_bias(attn_mask, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t").contiguous()

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = pack([x[:, :, :skip.shape[-1]], skip], "b * t")[0]
            x = resnet(x, mask_up, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            if streaming is True:
                attn_mask = add_optional_chunk_mask(x, mask_up.bool(), False, False, 0, self.static_chunk_size, -1)
            else:
                attn_mask = add_optional_chunk_mask(x, mask_up.bool(), False, False, 0, 0, -1).repeat(1, x.size(1), 1)
            attn_mask = mask_to_bias(attn_mask, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t").contiguous()
            x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask
