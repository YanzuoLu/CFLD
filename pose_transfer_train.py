"""
@author: Yanzuo Lu
@email:  luyz5@mail2.sysu.edu.cn
"""

import argparse
import datetime
import logging
import os
import sys
import time
import warnings

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.tracking import TensorBoardTracker, WandBTracker
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from diffusers import (DDIMInverseScheduler, DDIMScheduler, DDPMScheduler,
                       EulerDiscreteScheduler, PNDMScheduler)
from torch.utils.data import DataLoader

from datasets import PisTrainDeepFashion
from defaults import pose_transfer_C as cfg
from lr_scheduler import LinearWarmupMultiStepDecayLRScheduler
from models import (AppearanceEncoder, Decoder, PoseEncoder, UNet,
                    VariationalAutoencoder, build_backbone, build_metric)
from pose_transfer_test import build_test_loader, eval
from utils import AverageMeter

warnings.filterwarnings("ignore")
logger = logging.getLogger()


class build_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pose_query = cfg.MODEL.DECODER_CONFIG.POSE_QUERY

        self.backbone = build_backbone(
            img_size=cfg.INPUT.COND.IMG_SIZE,
            embed_dim=cfg.MODEL.COND_STAGE_CONFIG.EMBED_DIM,
            depths=cfg.MODEL.COND_STAGE_CONFIG.DEPTHS,
            num_heads=cfg.MODEL.COND_STAGE_CONFIG.NUM_HEADS,
            window_size=cfg.MODEL.COND_STAGE_CONFIG.WINDOW_SIZE,
            drop_path_rate=cfg.MODEL.COND_STAGE_CONFIG.DROP_PATH_RATE,
            mask=len(cfg.INPUT.COND.PRED_RATIO) > 0,
            last_norm=cfg.MODEL.COND_STAGE_CONFIG.LAST_NORM,
            pretrained_path=cfg.MODEL.COND_STAGE_CONFIG.PRETRAINED_PATH
        )

        self.appearance_encoder = AppearanceEncoder(
            attn_residual_block_idx=cfg.MODEL.APPEARANCE_GUIDANCE_CONFIG.ATTN_RESIDUAL_BLOCK_IDX,
            inner_dims=cfg.MODEL.APPEARANCE_GUIDANCE_CONFIG.INNER_DIMS,
            ctx_dims=cfg.MODEL.APPEARANCE_GUIDANCE_CONFIG.CTX_DIMS,
            embed_dims=cfg.MODEL.APPEARANCE_GUIDANCE_CONFIG.EMBED_DIMS,
            heads=cfg.MODEL.APPEARANCE_GUIDANCE_CONFIG.HEADS,
            depth=cfg.MODEL.APPEARANCE_GUIDANCE_CONFIG.DEPTH,
            to_self_attn=cfg.MODEL.APPEARANCE_GUIDANCE_CONFIG.TO_SELF_ATTN,
            to_queries=cfg.MODEL.APPEARANCE_GUIDANCE_CONFIG.TO_QUERIES,
            to_keys=cfg.MODEL.APPEARANCE_GUIDANCE_CONFIG.TO_KEYS,
            to_values=cfg.MODEL.APPEARANCE_GUIDANCE_CONFIG.TO_VALUES,
            aspect_ratio=cfg.INPUT.COND.IMG_SIZE[0] // cfg.INPUT.COND.IMG_SIZE[1],
            detach_input=cfg.MODEL.APPEARANCE_GUIDANCE_CONFIG.DETACH_INPUT,
            convin_kernel_size=cfg.MODEL.APPEARANCE_GUIDANCE_CONFIG.CONVIN_KERNEL_SIZE,
            convin_stride=cfg.MODEL.APPEARANCE_GUIDANCE_CONFIG.CONVIN_STRIDE,
            convin_padding=cfg.MODEL.APPEARANCE_GUIDANCE_CONFIG.CONVIN_PADDING
        )

        self.pose_encoder = PoseEncoder(
            downscale_factor=cfg.MODEL.POSE_GUIDANCE_CONFIG.DOWNSCALE_FACTOR,
            pose_channels=cfg.MODEL.POSE_GUIDANCE_CONFIG.POSE_CHANNELS,
            in_channels=cfg.MODEL.POSE_GUIDANCE_CONFIG.IN_CHANNELS,
            channels=cfg.MODEL.POSE_GUIDANCE_CONFIG.CHANNELS
        )

        self.decoder = Decoder(
            n_ctx=cfg.MODEL.DECODER_CONFIG.N_CTX,
            ctx_dim=cfg.MODEL.DECODER_CONFIG.CTX_DIM,
            heads=cfg.MODEL.DECODER_CONFIG.HEADS,
            depth=cfg.MODEL.DECODER_CONFIG.DEPTH,
            last_norm=cfg.MODEL.COND_STAGE_CONFIG.LAST_NORM,
            img_size=cfg.INPUT.COND.IMG_SIZE,
            embed_dim=cfg.MODEL.COND_STAGE_CONFIG.EMBED_DIM,
            depths=cfg.MODEL.COND_STAGE_CONFIG.DEPTHS,
            pose_query=cfg.MODEL.DECODER_CONFIG.POSE_QUERY,
            pose_channel=cfg.MODEL.POSE_GUIDANCE_CONFIG.CHANNELS[-1]
        )

        self.learnable_vector = nn.Parameter(torch.randn((1, cfg.MODEL.DECODER_CONFIG.N_CTX, cfg.MODEL.DECODER_CONFIG.CTX_DIM)))
        self.u_cond_percent = cfg.MODEL.U_COND_PERCENT
        self.u_cond_down_block_guidance = cfg.MODEL.U_COND_DOWN_BLOCK_GUIDANCE
        self.u_cond_up_block_guidance = cfg.MODEL.U_COND_UP_BLOCK_GUIDANCE

    def forward(self, batched_inputs):
        mask = batched_inputs["mask"] if "mask" in batched_inputs else None
        x, features = self.backbone(batched_inputs["img_cond"], mask=mask)
        up_block_additional_residuals = self.appearance_encoder(features)

        bsz = x.shape[0]
        if self.training:
            bsz = bsz * 2
            down_block_additional_residuals = self.pose_encoder(torch.cat([batched_inputs["pose_img_src"], batched_inputs["pose_img_tgt"]]))
            up_block_additional_residuals = {k: torch.cat([v, v]) for k, v in up_block_additional_residuals.items()}
            c = self.decoder(x, features, down_block_additional_residuals)
            if not self.pose_query:
                c = torch.cat([c, c])

            u_cond_prop = torch.rand(bsz, 1, 1)
            u_cond_prop = (u_cond_prop < self.u_cond_percent).to(dtype=x.dtype, device=x.device)
            c = self.learnable_vector.expand(bsz, -1, -1).to(dtype=x.dtype) * u_cond_prop + c * (1 - u_cond_prop)
            if self.u_cond_down_block_guidance:
                down_block_additional_residuals = [torch.zeros_like(sample) * u_cond_prop.unsqueeze(1) + \
                                                   sample * (1 - u_cond_prop.unsqueeze(1)) \
                                                   for sample in down_block_additional_residuals]
            if self.u_cond_up_block_guidance:
                up_block_additional_residuals = {k: torch.zeros_like(v) * u_cond_prop + v * (1 - u_cond_prop) \
                                                 for k, v in up_block_additional_residuals.items()}
        else:
            down_block_additional_residuals = self.pose_encoder(batched_inputs["pose_img"])
            c = self.decoder(x, features, down_block_additional_residuals)
            c = torch.cat([self.learnable_vector.expand(bsz, -1, -1).to(dtype=x.dtype), c], dim=0)

        return c, down_block_additional_residuals, up_block_additional_residuals


def main(cfg):
    project_dir = os.path.join("outputs", cfg.ACCELERATE.PROJECT_NAME)
    run_dir = os.path.join(project_dir, cfg.ACCELERATE.RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)

    accelerator = Accelerator(
        log_with=["wandb", "tensorboard"],
        project_dir=project_dir,
        mixed_precision=cfg.ACCELERATE.MIXED_PRECISION,
        gradient_accumulation_steps=cfg.ACCELERATE.GRADIENT_ACCUMULATION_STEPS,
        kwargs_handlers=[DistributedDataParallelKwargs(bucket_cap_mb=200, gradient_as_bucket_view=True)]
    )
    torch.backends.cuda.matmul.allow_tf32 = cfg.ACCELERATE.ALLOW_TF32
    set_seed(cfg.ACCELERATE.SEED)

    if accelerator.is_main_process:
        accelerator.trackers = []
        accelerator.trackers.append(WandBTracker(
            cfg.ACCELERATE.PROJECT_NAME, name=cfg.ACCELERATE.RUN_NAME, config=cfg, dir=project_dir))
        accelerator.trackers.append(TensorBoardTracker(cfg.ACCELERATE.RUN_NAME, project_dir))

        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            f.write(cfg.dump())
    accelerator.wait_for_everyone()

    fmt = "[%(asctime)s %(filename)s:%(lineno)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        filename=f"{run_dir}/log_rank{accelerator.process_index}.txt",
        filemode="a"
    )
    if accelerator.is_main_process:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(console_handler)

    logger.info(f"running with config:\n{str(cfg)}")

    logger.info("preparing datasets...")
    train_data = PisTrainDeepFashion(
        root_dir=cfg.INPUT.ROOT_DIR,
        gt_img_size=cfg.INPUT.GT.IMG_SIZE,
        pose_img_size=cfg.INPUT.POSE.IMG_SIZE,
        cond_img_size=cfg.INPUT.COND.IMG_SIZE,
        min_scale=cfg.INPUT.COND.MIN_SCALE,
        log_aspect_ratio=cfg.INPUT.COND.PRED_ASPECT_RATIO,
        pred_ratio=cfg.INPUT.COND.PRED_RATIO,
        pred_ratio_var=cfg.INPUT.COND.PRED_RATIO_VAR,
        psz=cfg.INPUT.COND.MASK_PATCH_SIZE
    )
    train_loader = DataLoader(
        train_data,
        cfg.INPUT.BATCH_SIZE // accelerator.num_processes // cfg.ACCELERATE.GRADIENT_ACCUMULATION_STEPS,
        shuffle = True,
        drop_last = True,
        num_workers = cfg.INPUT.NUM_WORKERS,
        pin_memory = True
    )

    test_loader, fid_real_loader, test_data, fid_real_data = build_test_loader(cfg)

    logger.info("preparing model...")
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # not trained, move to 16-bit to save memory
    vae = VariationalAutoencoder(
        pretrained_path=cfg.MODEL.FIRST_STAGE_CONFIG.PRETRAINED_PATH
    ).to(accelerator.device, dtype=weight_dtype)

    if cfg.MODEL.SCHEDULER_CONFIG.NAME == "euler":
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(cfg.MODEL.SCHEDULER_CONFIG.PRETRAINED_PATH)
    elif cfg.MODEL.SCHEDULER_CONFIG.NAME == "pndm":
        noise_scheduler = PNDMScheduler.from_pretrained(cfg.MODEL.SCHEDULER_CONFIG.PRETRAINED_PATH)
    elif cfg.MODEL.SCHEDULER_CONFIG.NAME == "ddim":
        noise_scheduler = DDIMScheduler.from_pretrained(cfg.MODEL.SCHEDULER_CONFIG.PRETRAINED_PATH)
    elif cfg.MODEL.SCHEDULER_CONFIG.NAME == "ddpm":
        noise_scheduler = DDPMScheduler.from_pretrained(cfg.MODEL.SCHEDULER_CONFIG.PRETRAINED_PATH)

    inverse_noise_scheduler = DDIMInverseScheduler(
        num_train_timesteps=noise_scheduler.num_train_timesteps,
        beta_start=noise_scheduler.beta_start,
        beta_end=noise_scheduler.beta_end,
        beta_schedule=noise_scheduler.beta_schedule,
        trained_betas=noise_scheduler.trained_betas,
        clip_sample=noise_scheduler.clip_sample,
        set_alpha_to_one=noise_scheduler.set_alpha_to_one,
        steps_offset=noise_scheduler.steps_offset,
        prediction_type=noise_scheduler.prediction_type,
        timestep_spacing=noise_scheduler.timestep_spacing
    )

    model = build_model(cfg)
    unet = UNet(cfg)
    metric = build_metric().to(accelerator.device)
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad] + \
                           [p.numel() for p in unet.parameters() if p.requires_grad])
    logger.info(f"number of trainable parameters: {trainable_params}")

    logger.info("preparing optimizer...")
    lr = cfg.OPTIMIZER.LR * cfg.INPUT.BATCH_SIZE if cfg.OPTIMIZER.SCALE_LR else cfg.OPTIMIZER.LR
    params = [p for p in model.parameters() if p.requires_grad] + \
             [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    logger.info("preparing accelerator...")
    model, unet, optimizer, train_loader, test_loader, fid_real_loader = accelerator.prepare(
        model, unet, optimizer, train_loader, test_loader, fid_real_loader)

    last_epoch = cfg.MODEL.LAST_EPOCH
    if cfg.MODEL.PRETRAINED_PATH:
        logger.info(f"loading states from {cfg.MODEL.PRETRAINED_PATH}")
        accelerator.load_state(cfg.MODEL.PRETRAINED_PATH)
    global_step = last_epoch * len(train_loader)

    logger.info("preparing lr scheduler...")
    lr_scheduler = LinearWarmupMultiStepDecayLRScheduler(
        optimizer, cfg.OPTIMIZER.WARMUP_STEPS, cfg.OPTIMIZER.WARMUP_RATE, cfg.OPTIMIZER.DECAY_RATE,
        cfg.OPTIMIZER.EPOCHS, cfg.OPTIMIZER.DECAY_EPOCHS, len(train_loader),
        last_epoch=len(train_loader)*last_epoch-1, override_lr=cfg.OPTIMIZER.OVERRIDE_LR)

    logger.info("start training...")
    start_time = time.time()
    end_time = time.time()

    for epoch in range(last_epoch, cfg.OPTIMIZER.EPOCHS, 1):
        model.train()
        unet.train()

        epoch_time = time.time()
        logger.info(f"epoch {epoch + 1} start")
        batch_time = AverageMeter()
        total_loss = AverageMeter()

        for i, batch in enumerate(train_loader):
            with accelerator.accumulate(model, unet):
                optimizer.zero_grad()

                # Convert images to latent space
                with accelerator.autocast():
                    latents = vae.encode(torch.cat([batch["img_src"], batch["img_tgt"]]))

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                if cfg.MODEL.SCHEDULER_CONFIG.CUBIC_SAMPLING:
                    # Cubic sampling to sample a random timestep for each image
                    timesteps = torch.rand((bsz, ), device=accelerator.device)
                    timesteps = (1 - timesteps**3) * noise_scheduler.config.num_train_timesteps
                    timesteps = timesteps.long()
                    timesteps = torch.clamp(timesteps, 0, noise_scheduler.config.num_train_timesteps - 1)
                else:
                    # Uniform sampling to sample a random timestep for each image
                    timesteps = torch.randint(noise_scheduler.config.num_train_timesteps, (bsz, ), device=accelerator.device)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # get embedding
                c, down_block_additional_residuals, up_block_additional_residuals = model(batch)
                down_block_additional_residuals = [sample.to(dtype=weight_dtype) for sample in down_block_additional_residuals]
                up_block_additional_residuals = {k: v.to(dtype=weight_dtype) for k, v in up_block_additional_residuals.items()}

                # predict
                with accelerator.autocast():
                    encoder_hidden_states = c.to(dtype=weight_dtype)
                    model_pred = unet(
                        sample=noisy_latents, timestep=timesteps, encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=down_block_additional_residuals,
                        up_block_additional_residuals=up_block_additional_residuals)

                loss_simple = (noise - model_pred) ** 2
                loss_simple = loss_simple.mean()
                loss = loss_simple / cfg.ACCELERATE.GRADIENT_ACCUMULATION_STEPS
                if torch.isnan(loss).any():
                    accelerator.set_trigger()
                if accelerator.check_trigger():
                    logger.info("loss is nan, stop training")
                    accelerator.end_training()
                    time.sleep(86400) # waiting for...

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    global_step += 1
                optimizer.step()
                lr_scheduler.step()

            total_loss.update(loss_simple.item())
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if (i + 1) % cfg.ACCELERATE.LOG_PERIOD == 0 or i == len(train_loader) - 1:
                accelerator.log({
                    "loss": loss_simple.item(),
                    "loss_avg": total_loss.avg,
                    "lr": optimizer.param_groups[-1]["lr"]
                }, step=global_step)

                etas = batch_time.avg * (len(train_loader) - 1 - i)
                logger.info(
                    f"Train [{epoch+1}/{cfg.OPTIMIZER.EPOCHS}]({i+1}/{len(train_loader)})  "
                    f"Time {batch_time.val:.4f}({batch_time.avg:.4f})  "
                    f"Loss {total_loss.val:.4f}({total_loss.avg:.4f})  "
                    f"Lr {optimizer.param_groups[-1]['lr']:.8f}  "
                    f"Eta {datetime.timedelta(seconds=int(etas))}")

        logger.info(f"epoch {epoch + 1} finished, running time {datetime.timedelta(seconds=int(time.time() - epoch_time))}")
        save_dir = os.path.join(run_dir, f"epochs_{(epoch+1):03d}")

        if (epoch + 1) % cfg.ACCELERATE.EVAL_PERIOD == 0:
            accelerator.save_state(os.path.join(save_dir, "checkpoints"))
            save_dir = os.path.join(save_dir, "log_images")
            os.makedirs(save_dir, exist_ok=True)

            eval(
                cfg=cfg,
                model=model,
                test_loader=test_loader,
                fid_real_loader=fid_real_loader,
                weight_dtype=weight_dtype,
                save_dir=save_dir,
                test_data=test_data,
                fid_real_data=fid_real_data,
                global_step=None,
                accelerator=accelerator,
                metric=metric,
                noise_scheduler=noise_scheduler,
                inverse_noise_scheduler=inverse_noise_scheduler,
                vae=vae,
                unet=unet
            )

    train_time = time.time() - start_time
    logger.info(f'training completed, running time {datetime.timedelta(seconds=int(train_time))}')
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose Transfer Training")
    parser.add_argument("--config_file", type=str, default="", help="path to config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help=
                        "modify config options using the command-line")
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    main(cfg)