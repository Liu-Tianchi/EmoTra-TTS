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
# Safe LabelSmoothingLoss — prevents CUDA SIGFPE from division by zero
# when all tokens in a batch are IGNORE_ID (denom=0).

from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
import torch


class SafeLabelSmoothingLoss(LabelSmoothingLoss):
    """LabelSmoothingLoss with denom=max(denom,1) to prevent divide-by-zero."""

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.size - 1))
        ignore = target == self.padding_idx
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        denom = max(denom, 1)  # prevent SIGFPE
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom
