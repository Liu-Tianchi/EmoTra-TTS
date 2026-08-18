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
# EmoTra sft_Flow_Lynorm_fixs Training Script
# MLP + LayerNorm + fixed scale (Direction-Magnitude Decoupling)
# No V-series dependencies.

from __future__ import print_function
import argparse
import datetime
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from copy import deepcopy
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import deepspeed

from hyperpyyaml import load_hyperpyyaml

from torch.distributed.elastic.multiprocessing.errors import record

from cosyvoice.utils.executor_sft_Flow import Executor_sft_Flow as Executor
from cosyvoice.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config)


def get_args():
    parser = argparse.ArgumentParser(description='EmoTra sft_Flow_Lynorm_fixs Training')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser.add_argument('--model', required=True, help='model which will be trained')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--qwen_pretrain_path', required=False, help='qwen pretrain path')
    parser.add_argument('--onnx_path', required=False, help='onnx path')
    parser.add_argument('--checkpoint', help='checkpoint model (Flow base)')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--timeout',
                        default=120,
                        type=int,
                        help='timeout (in seconds) of cosyvoice_join.')
    parser.add_argument('--override',
                        default='',
                        type=str,
                        help='Override config parameters')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def load_vad_modules_from_llm_checkpoint(flow_model, checkpoint_path):
    """
    Load ONLY VAD-related tensors (12 total) from LLM checkpoint.
    Creates vad_projection (3→896) and vad_hidden_reconstructor (896→1024),
    then freezes them.
    """
    logging.info(f"Loading VAD modules from LLM checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    vad_keys = [k for k in state_dict.keys()
                if ('vad_projection' in k or 'hidden_reconstructor' in k)
                and isinstance(state_dict[k], torch.Tensor)]

    if len(vad_keys) == 0:
        raise RuntimeError(f"No VAD keys found in checkpoint. First 30 keys: {sorted(state_dict.keys())[:30]}")

    logging.info(f"  Found {len(vad_keys)} VAD tensors")

    # Create modules matching LLM architecture
    flow_model.vad_projection = nn.Sequential(
        nn.Linear(3, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 896),
    )
    flow_model.vad_hidden_reconstructor = nn.Sequential(
        nn.Linear(896, 512),
        nn.LayerNorm(512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 1024),
    )

    # Load weights (auto-strip prefix)
    vad_proj_state = {}
    hidden_recon_state = {}
    for k in vad_keys:
        if 'vad_projection.' in k:
            idx = k.index('vad_projection.')
            vad_proj_state[k[idx + len('vad_projection.'):]] = state_dict[k]
        elif 'hidden_reconstructor.' in k:
            idx = k.index('hidden_reconstructor.')
            hidden_recon_state[k[idx + len('hidden_reconstructor.'):]] = state_dict[k]

    if vad_proj_state:
        flow_model.vad_projection.load_state_dict(vad_proj_state, strict=False)
        logging.info(f"  vad_projection loaded ({len(vad_proj_state)} tensors)")
    else:
        raise RuntimeError("No vad_projection weights found!")

    if hidden_recon_state:
        flow_model.vad_hidden_reconstructor.load_state_dict(hidden_recon_state, strict=False)
        logging.info(f"  hidden_reconstructor loaded ({len(hidden_recon_state)} tensors)")
    else:
        raise RuntimeError("No hidden_reconstructor weights found!")

    # Freeze
    for param in flow_model.vad_projection.parameters():
        param.requires_grad = False
    for param in flow_model.vad_hidden_reconstructor.parameters():
        param.requires_grad = False

    logging.info(f"  vad_projection: FROZEN")
    logging.info(f"  hidden_reconstructor: FROZEN")


def setup_flow_sft_lynorm_fixs(flow_model):
    """
    Setup: MLP vad_downsample + LayerNorm + fixed scale.

    Steps:
    1. Create MLP vad_downsample (1024→256→ReLU→80), last layer zero-init
    2. Create LayerNorm(80) for magnitude locking
    3. Freeze ALL parameters
    4. Unfreeze MLP vad_downsample + emo_layer_norm
    """
    logging.info("=" * 60)
    logging.info("Setting up sft_Flow_Lynorm_fixs (MLP + LayerNorm + fixed scale)")
    logging.info("=" * 60)

    # Verify VAD modules
    if flow_model.vad_projection is None:
        raise ValueError("vad_projection not loaded!")
    if flow_model.vad_hidden_reconstructor is None:
        raise ValueError("vad_hidden_reconstructor not loaded!")

    # Step 1: Create MLP vad_downsample
    if not hasattr(flow_model, 'vad_downsample') or flow_model.vad_downsample is None:
        flow_model.vad_downsample = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 80),
        )
        nn.init.zeros_(flow_model.vad_downsample[2].weight)
        nn.init.zeros_(flow_model.vad_downsample[2].bias)
        logging.info("  Created MLP vad_downsample (1024→256→ReLU→80): TRAINABLE (last layer zero-init)")

    # Step 2: Create LayerNorm(80) for magnitude locking
    if not hasattr(flow_model, 'emo_layer_norm') or flow_model.emo_layer_norm is None:
        flow_model.emo_layer_norm = nn.LayerNorm(80)
        logging.info("  Created emo_layer_norm(80): TRAINABLE (locks norm ≈ √80 ≈ 8.94)")

    mlp_params_count = sum(p.numel() for p in flow_model.vad_downsample.parameters())
    ln_params_count = sum(p.numel() for p in flow_model.emo_layer_norm.parameters())

    # Step 3: Freeze ALL parameters
    for param in flow_model.parameters():
        param.requires_grad = False

    # Step 4: Unfreeze MLP vad_downsample + emo_layer_norm
    trainable_params = []
    for param in flow_model.vad_downsample.parameters():
        param.requires_grad = True
        trainable_params.append(param)
    for param in flow_model.emo_layer_norm.parameters():
        param.requires_grad = True
        trainable_params.append(param)

    # Statistics
    total_params = sum(p.numel() for p in flow_model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    fixed_scale = flow_model.FIXED_EMO_SCALE if hasattr(flow_model, 'FIXED_EMO_SCALE') else 0.07

    logging.info("=" * 60)
    logging.info("Parameter Statistics:")
    logging.info(f"  Total Flow params: {total_params:,}")
    logging.info(f"  MLP vad_downsample params: {mlp_params_count:,}")
    logging.info(f"  emo_layer_norm params: {ln_params_count:,}")
    logging.info(f"  Total trainable params: {trainable_count:,} ({trainable_count/max(1,total_params)*100:.4f}%)")
    logging.info(f"  Strategy: MLP + LayerNorm + fixed_scale={fixed_scale}")
    logging.info(f"  Expected effective norm: {fixed_scale} × 8.94 ≈ {fixed_scale * 8.94:.3f}")
    logging.info(f"  NO LoRA — decoder base weights preserved")
    logging.info("=" * 60)

    return trainable_params


@record
def main():
    args = get_args()
    os.environ['onnx_path'] = args.onnx_path
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    gan = False

    override_dict = {k: None for k in ['llm', 'flow', 'hift', 'hifigan'] if k != args.model}
    if args.qwen_pretrain_path is not None:
        override_dict['qwen_pretrain_path'] = args.qwen_pretrain_path

    if args.override:
        for item in args.override.split():
            if '=' in item:
                key, value = item.split('=', 1)
                try:
                    value = float(value)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                override_dict[key] = value

    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)
    configs['train_conf'].update(vars(args))

    init_distributed(args)

    nccl_timeout = datetime.timedelta(seconds=max(args.timeout * 2, 3600))
    if args.train_engine == 'torch_ddp' and dist.is_initialized():
        pg = dist.distributed_c10d._get_default_group()
        nccl_backend = pg._get_backend(torch.device('cuda'))
        if hasattr(nccl_backend, '_set_default_timeout'):
            nccl_backend._set_default_timeout(nccl_timeout)
            logging.info(f"NCCL timeout set to {nccl_timeout.total_seconds():.0f}s")

    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, gan, False)

    if int(os.environ.get('RANK', 0)) == 0:
        try:
            with open(args.train_data, 'r') as f:
                train_files = [l.strip() for l in f if l.strip()]
            with open(args.cv_data, 'r') as f:
                cv_files = [l.strip() for l in f if l.strip()]
            logging.info(f"Train parquet files: {len(train_files)}, CV: {len(cv_files)}")
        except Exception as e:
            logging.warning(f"Could not read data statistics: {e}")

    configs = check_modify_and_save_config(args, configs)
    writer = init_summarywriter(args)

    model = configs[args.model]
    start_step, start_epoch = 0, -1

    # Load Flow base checkpoint
    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            state_dict = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            if 'step' in state_dict:
                start_step = state_dict['step']
            if 'epoch' in state_dict:
                start_epoch = state_dict['epoch']
            logging.info(f"Flow checkpoint loaded: {args.checkpoint}")
        else:
            logging.warning(f'Flow checkpoint not found: {args.checkpoint}')

    # Load VAD modules from LLM checkpoint
    llm_checkpoint = os.environ.get('FLOW_SFT_LLM_CHECKPOINT')
    if not llm_checkpoint or not os.path.exists(llm_checkpoint):
        raise RuntimeError(f"LLM checkpoint required for VAD modules! "
                          f"Set FLOW_SFT_LLM_CHECKPOINT. Got: {llm_checkpoint}")

    logging.info("=" * 60)
    logging.info("sft_Flow_Lynorm_fixs: MLP + LayerNorm + fixed scale")
    logging.info(f"  LLM checkpoint (VAD only): {llm_checkpoint}")
    logging.info("=" * 60)

    load_vad_modules_from_llm_checkpoint(model, llm_checkpoint)
    trainable_params = setup_flow_sft_lynorm_fixs(model)

    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters!")

    model = wrap_cuda_model(args, model)

    optim_type = configs['train_conf']['optim']
    optim_conf = configs['train_conf']['optim_conf']
    if optim_type == 'adam':
        optimizer = torch.optim.Adam(trainable_params, **optim_conf)
    elif optim_type == 'adamw':
        optimizer = torch.optim.AdamW(trainable_params, **optim_conf)
    else:
        raise ValueError(f"Unsupported optimizer: {optim_type}")

    if configs['train_conf']['scheduler'] == 'warmuplr':
        from cosyvoice.utils.scheduler import WarmupLR
        scheduler = WarmupLR(optimizer, **configs['train_conf']['scheduler_conf'])
    elif configs['train_conf']['scheduler'] == 'constantlr':
        from cosyvoice.utils.scheduler import ConstantLR
        scheduler = ConstantLR(optimizer)
    else:
        raise ValueError(f"Unknown scheduler: {configs['train_conf']['scheduler']}")

    scheduler.set_step(start_step)
    logging.info(f"  Optimizer: {optim_type}, lr={optim_conf['lr']}")

    info_dict = deepcopy(configs['train_conf'])
    info_dict['step'] = start_step
    info_dict['epoch'] = start_epoch
    save_model(model, 'init', info_dict)

    executor = Executor(gan=False)
    executor.step = start_step

    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    if int(os.environ.get('RANK', 0)) == 0:
        logging.info(f"Training starts: step {start_step}, epoch {start_epoch + 1}")
        logging.info(f"Target: {info_dict['max_epoch']} epochs")

    for epoch in range(start_epoch + 1, info_dict['max_epoch']):
        if int(os.environ.get('RANK', 0)) == 0:
            logging.info(f"Epoch {epoch}/{info_dict['max_epoch']} starting...")

        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        dist.barrier()

        group_join = dist.new_group(backend="gloo",
                                    timeout=datetime.timedelta(seconds=args.timeout))

        executor.train_one_epoc(model, optimizer, scheduler,
                                train_data_loader, cv_data_loader,
                                writer, info_dict, scaler, group_join)

        dist.destroy_process_group(group_join)


if __name__ == '__main__':
    main()
