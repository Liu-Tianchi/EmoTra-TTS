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
# EmoTra sft_Flow — CausalConditionalCFM with 3D spks [B, C, T] support
# Directly inherits from CausalConditionalCFM, no V-series dependencies.

import torch
import torch.nn.functional as F
from cosyvoice.flow.flow_matching import CausalConditionalCFM
from cosyvoice.utils.common import set_all_random_seed


class CausalConditionalCFM_sft_Flow(CausalConditionalCFM):
    """
    CausalConditionalCFM with support for 3D spks [B, 80, T].

    When spks is 2D [B, 80]:  behaviour is 100% identical to base class.
    When spks is 3D [B, 80, T]: CFG dropout / euler solver handle it correctly.
    """

    def solve_euler(self, x, t_span, mu, mask, spks, cond, streaming=False):
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        sol = []

        x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=spks.dtype)
        mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=spks.dtype)
        # spks_in shape depends on whether spks is 2D or 3D
        if spks.dim() == 3:
            spks_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        else:
            spks_in = torch.zeros([2, 80], device=x.device, dtype=spks.dtype)
        cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)

        for step in range(1, len(t_span)):
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = spks
            cond_in[0] = cond
            dphi_dt = self.forward_estimator(
                x_in, mask_in,
                mu_in, t_in,
                spks_in,
                cond_in,
                streaming
            )
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].float()

    def compute_loss(self, x1, mask, mu, spks=None, cond=None, streaming=False):
        b, _, t = mu.shape

        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            if spks.dim() == 2:
                spks = spks * cfg_mask.view(-1, 1)
            else:
                spks = spks * cfg_mask.view(-1, 1, 1)
            cond = cond * cfg_mask.view(-1, 1, 1)

        pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond, streaming=streaming)
        denom = (torch.sum(mask) * u.shape[1]).clamp(min=1.0)
        loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / denom
        return loss, y
