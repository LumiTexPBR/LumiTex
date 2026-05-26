import argparse
import functools
import math
import os
import random
from tqdm import tqdm
import wandb

import lpips
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers.optimization import get_cosine_schedule_with_warmup

from data import NVSDataset
from model import LaCTLVSM

parser = argparse.ArgumentParser()
# Basic info
parser.add_argument("--config", type=str, default="config/lact")
parser.add_argument("--expname", type=str, default="default")
parser.add_argument("--path", type=str, default="./dataset_example")
parser.add_argument("--load", type=str, default=None)
parser.add_argument("--save_every", type=int, default=1000)
parser.add_argument("--log_every", type=int, default=500)

# Training
parser.add_argument("--compile", action="store_true")
parser.add_argument("--actckpt", action="store_true")
parser.add_argument("--bs_per_gpu", type=int, default=2)
parser.add_argument("--num_all_views", type=int, default=16)
parser.add_argument("--num_input_views", type=int, default=6)
parser.add_argument("--num_target_views", type=int, default=10)  
parser.add_argument("--image_size", nargs=2, type=int, default=[768, 768], help="Image size W, H")

# Optimizer
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--warmup", type=int, default=500)
parser.add_argument("--steps", type=int, default=20000)
parser.add_argument("--weight_decay", type=float, default=0.05)
parser.add_argument("--lpips_start", type=int, default=0, help="Iteration to start LPIPS loss")

args = parser.parse_args()
model_config = omegaconf.OmegaConf.load(args.config)
output_dir = f"outputs_orth_768/{args.expname}"
os.makedirs(output_dir, exist_ok=True)

dist.init_process_group(backend="nccl")
ddp_local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank() % 8))
torch.cuda.set_device(ddp_local_rank)

# Seed everything
rank_specific_seed = 95 + dist.get_rank()
torch.manual_seed(rank_specific_seed)
np.random.seed(rank_specific_seed)
random.seed(rank_specific_seed)
dataloader_seed_generator = torch.Generator()
dataloader_seed_generator.manual_seed(rank_specific_seed)

model = LaCTLVSM(**model_config).cuda()

# Optimizers
decay_params = [p for p in model.parameters() if p.dim() >= 2]
nodecay_params = [p for p in model.parameters() if p.dim() < 2]
optim_groups = [
    {"params": decay_params, "weight_decay": args.weight_decay},
    {"params": nodecay_params, "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), fused=True)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=args.warmup,
    num_training_steps=args.steps,
)

# Load checkpoint
now_iters = 0

print('loading checkpoint...')
print('finished loading checkpoint')

for try_load_path in [output_dir, args.load]:
    # Always try to load from output_dir first to resume training
    if try_load_path is None: continue
    try:
        if os.path.isdir(try_load_path):
            checkpoints = [f for f in os.listdir(try_load_path) if f.startswith("model_") and f.endswith(".pth")]
            if not checkpoints: continue
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))
            checkpoint_path = os.path.join(try_load_path, latest_checkpoint)
        else:
            checkpoint_path = try_load_path
        
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        now_iters = checkpoint["now_iters"]
        break
    except:
        continue

print('finished loading checkpoint')
        
model = DDP(model, device_ids=[ddp_local_rank])

# This activation checkpointing wrapper supports torch.compile
if args.actckpt:
    torch._dynamo.config.optimize_ddp = False
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper as ptd_checkpoint_wrapper,
        apply_activation_checkpointing,
    )

    wrapper = functools.partial(ptd_checkpoint_wrapper, preserve_rng_state=False)

    def _check_fn(submodule) -> bool:
        from model import Block
        return isinstance(submodule, Block)

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper,
        check_fn=_check_fn,
    )

if args.compile:
    model = torch.compile(model)  

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        key = key.replace("_checkpoint_wrapped_module.", "")
        key = key.replace("_orig_mod.", "")
        while key.startswith("module."):
            key = key[len("module."):]
        new_state_dict[key] = value
    return new_state_dict

# Data
dataset = NVSDataset(args.path, args.num_all_views, tuple(args.image_size))
datasampler = DistributedSampler(dataset)

dataloader = DataLoader(
    dataset,
    batch_size=args.bs_per_gpu,
    shuffle=False,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    prefetch_factor=2,
    sampler=datasampler,
    generator=dataloader_seed_generator,    # This ensures deterministic dataloader
)

if dist.get_rank() == 0:
    print(model)
    print(optimizer)
    print(lr_scheduler)
    print(f"Start training from iter {now_iters}...")
    wandb.init(
        project="lvsm",      
        name=args.expname,            
        config=vars(args),         
        dir=output_dir,
    )



remaining_steps = args.steps - now_iters
lpips_loss_module = lpips.LPIPS(net="vgg").cuda().eval()
for epoch in range((remaining_steps - 1) // len(dataloader) + 1):
    if dist.get_rank() == 0:
        data_iter = tqdm(dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True)
    else:
        data_iter = dataloader

    for data_dict in data_iter:
        data_dict = {key: value.cuda() for key, value in data_dict.items() if isinstance(value, torch.Tensor)}
        input_data_dict = {key: value[:, :args.num_input_views] for key, value in data_dict.items()}
        target_data_dict = {key: value[:, -args.num_target_views:] for key, value in data_dict.items()}

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True):
            rendering = model(input_data_dict, target_data_dict)
            target = target_data_dict["image"]

            l2_loss = F.mse_loss(rendering, target)
            psnr = -10.0 * torch.log10(l2_loss).item()
            if now_iters >= args.lpips_start:
                lpips_loss = lpips_loss_module(rendering.flatten(0, 1), target.flatten(0, 1), normalize=True).mean()
            else:
                lpips_loss = 0.0
            loss = l2_loss + lpips_loss
        loss.backward()
        if dist.get_rank() == 0 and now_iters % 10 == 0:
            x = rendering[0].detach().to(torch.float32).clamp(0, 1)
            x_cat = torch.cat([x[i] for i in range(x.shape[0])], dim=-1)  # [3, H, N*W]
            save_image(x_cat, 'test/output.png')

            gt = target[0].detach().to(torch.float32).clamp(0, 1)
            gt_cat = torch.cat([gt[i] for i in range(gt.shape[0])], dim=-1)
            save_image(gt_cat, 'test/gt.png')

        # Gradident safeguard
        skip_optimizer_step = False
        if now_iters > 1000:
            global_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()

            if not math.isfinite(global_grad_norm):
                skip_optimizer_step = True
            elif global_grad_norm > 4.0:
                skip_optimizer_step = True

        if not skip_optimizer_step:
            optimizer.step()
        lr_scheduler.step()     # Always step the lr scheduler and iters
        now_iters += 1


        if dist.get_rank() == 0:
            data_iter.set_postfix({
                "psnr": "%.2f" % psnr,
                "LPIPS": "%.4f" % lpips_loss,
            })

            if now_iters % args.log_every == 0:
                wandb.log({
                    "iter": now_iters,
                    "psnr": psnr,
                    "lpips": float(lpips_loss),
                    "output_img": wandb.Image(x_cat, caption="Output, iter %d" % now_iters),
                    "gt_img": wandb.Image(gt_cat, caption="GT, iter %d" % now_iters)
                }, step=now_iters)

            else:
                wandb.log({
                    "iter": now_iters,
                    "psnr": psnr,
                    "lpips": float(lpips_loss),
                }, step=now_iters)

            if now_iters % args.save_every == 0:
                torch.save({
                    "model": remove_module_prefix(model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "now_iters": now_iters,
                    "epoch": epoch,
                }, f"{output_dir}/model_{now_iters:07d}.pth")
            

        if now_iters == args.steps:
            break




        
        
    