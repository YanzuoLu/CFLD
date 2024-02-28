"""
@author: Yanzuo Lu
@author: oliveryanzuolu@gmail.com
"""

import argparse
import copy
import datetime
import logging
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.tracking import TensorBoardTracker, WandBTracker
from accelerate.utils import set_seed
from diffusers import (DDIMInverseScheduler, DDIMScheduler, DDPMScheduler,
                       EulerDiscreteScheduler, PNDMScheduler)
from einops import rearrange
from PIL import Image
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from datasets import FidRealDeepFashion, PisTestDeepFashion
from defaults import pose_transfer_C as cfg
from models import UNet, VariationalAutoencoder, build_metric
from utils import AverageMeter

warnings.filterwarnings("ignore")
logger = logging.getLogger()


def build_test_loader(cfg):
    test_data = PisTestDeepFashion(
        cfg.INPUT.ROOT_DIR, cfg.INPUT.GT.IMG_SIZE, cfg.INPUT.POSE.IMG_SIZE,
        cfg.INPUT.COND.IMG_SIZE, cfg.TEST.IMG_SIZE)
    test_loader = DataLoader(
        test_data,
        cfg.TEST.MICRO_BATCH_SIZE,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=True
    )

    fid_real_data = FidRealDeepFashion(cfg.INPUT.ROOT_DIR, cfg.TEST.IMG_SIZE)
    fid_real_loader = DataLoader(
        fid_real_data,
        cfg.TEST.MICRO_BATCH_SIZE,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=True
    )
    return test_loader, fid_real_loader, test_data, fid_real_data


def eval(cfg, model, test_loader, fid_real_loader, weight_dtype, save_dir,
         test_data, fid_real_data, global_step, accelerator, metric,
         noise_scheduler, inverse_noise_scheduler, vae, unet):
    logger.info("start sampling...")
    model.eval()
    unet.eval()

    gt_out_gathered = []
    pred_out_gathered = []
    lpips_gathered = []
    psnr_gathered = []
    ssim_gathered = []
    ssim_256_gathered = []

    with torch.no_grad():
        end_time = time.time()
        batch_time = AverageMeter()

        for i, test_batch in enumerate(test_loader):
            gt_imgs = test_batch["img_gt"]
            img_size = test_batch["img_tgt"].shape[2:]
            bsz = gt_imgs.shape[0]

            if cfg.TEST.DDIM_INVERSION_STEPS > 0:
                if cfg.TEST.DDIM_INVERSION_DOWN_BLOCK_GUIDANCE:
                    c, down_block_additional_residuals, up_block_additional_residuals = model({
                        "img_cond": test_batch["img_cond_from"], "pose_img": test_batch["pose_img_from"]})
                else:
                    c, down_block_additional_residuals, up_block_additional_residuals = model({
                        "img_cond": test_batch["img_cond_from"], "pose_img": test_batch["pose_img_to"]})

                noisy_latents = inverse_sample(
                    cfg.TEST.DDIM_INVERSION_STEPS, accelerator, inverse_noise_scheduler, vae, unet,
                    test_batch["img_src"], c[:bsz] if cfg.TEST.DDIM_INVERSION_UNCONDITIONAL else c[bsz:],
                    [sample.to(dtype=weight_dtype) for sample in down_block_additional_residuals] if cfg.TEST.DDIM_INVERSION_DOWN_BLOCK_GUIDANCE else None,
                    {k: v.to(dtype=weight_dtype) for k, v in up_block_additional_residuals.items()} if cfg.TEST.DDIM_INVERSION_UP_BLOCK_GUIDANCE else None)
            else:
                c, down_block_additional_residuals, up_block_additional_residuals = model({
                    "img_cond": test_batch["img_cond_from"], "pose_img": test_batch["pose_img_to"]})
                noisy_latents = torch.randn((bsz, 4, img_size[0]//8, img_size[1]//8)).to(accelerator.device)

            if cfg.TEST.DDIM_INVERSION_STEPS > 0 and cfg.TEST.DDIM_INVERSION_DOWN_BLOCK_GUIDANCE:
                c, down_block_additional_residuals, up_block_additional_residuals = model({
                    "img_cond": test_batch["img_cond_from"], "pose_img": test_batch["pose_img_to"]})

            sampling_imgs = sample(
                cfg, weight_dtype, accelerator, noise_scheduler, vae, unet, noisy_latents,
                c, down_block_additional_residuals, up_block_additional_residuals)

            # log one-batch sampling results for visualization
            if i == 0:
                src_imgs = test_batch["img_src"] * 0.5 + 0.5
                tgt_imgs = test_batch["img_tgt"] * 0.5 + 0.5
                pose_imgs = F.interpolate(test_batch["pose_img_to"][:, :3, :, :],
                                          tuple(test_batch["img_src"].shape[2:]),
                                          mode="bicubic", antialias=True)
                save_img = torch.stack([src_imgs, pose_imgs, tgt_imgs, sampling_imgs])
                save_img = postprocess_image(save_img, nrow=save_img.shape[0]*2)
                save_img.save(os.path.join(save_dir, f"inpainting_test_{accelerator.process_index}_{i}.jpg"))

            sampling_imgs = F.interpolate(sampling_imgs, tuple(gt_imgs.shape[2:]), mode="bicubic", antialias=True)
            sampling_imgs = sampling_imgs.float() * 255.0
            sampling_imgs = sampling_imgs.clamp(0, 255).to(dtype=torch.uint8) # can save all images here!!!
            sampling_imgs = sampling_imgs.to(torch.float32) / 255.

            pred_out, lpips, psnr, ssim, ssim_256 = metric(gt_imgs, sampling_imgs)
            pred_out_gathered.append(accelerator.gather_for_metrics(pred_out).cpu().numpy())
            lpips_gathered.append(accelerator.gather_for_metrics(lpips).cpu().numpy())
            psnr_gathered.append(accelerator.gather_for_metrics(psnr).cpu().numpy())
            ssim_gathered.append(accelerator.gather_for_metrics(ssim).cpu().numpy())
            ssim_256_gathered.append(accelerator.gather_for_metrics(ssim_256).cpu().numpy())

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if (i + 1) % cfg.ACCELERATE.LOG_PERIOD == 0 or i == len(test_loader) - 1:
                etas = batch_time.avg * (len(test_loader) - 1 - i)
                logger.info(
                    f"Sampling ({i+1}/{len(test_loader)})  "
                    f"Time {batch_time.val:.4f}({batch_time.avg:.4f})  "
                    f"Eta {datetime.timedelta(seconds=int(etas))}")
                if os.environ.get("WANDB_MODE", None) == "offline":
                    break

        end_time = time.time()
        batch_time = AverageMeter()
        for i, fid_real_imgs in enumerate(fid_real_loader):
            gt_out = metric(fid_real_imgs)
            gt_out_gathered.append(accelerator.gather_for_metrics(gt_out).cpu().numpy())

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if (i + 1) % cfg.ACCELERATE.LOG_PERIOD == 0 or i == len(fid_real_loader) - 1:
                etas = batch_time.avg * (len(fid_real_loader) - 1 - i)
                logger.info(
                    f"FidReal ({i+1}/{len(fid_real_loader)})  "
                    f"Time {batch_time.val:.4f}({batch_time.avg:.4f})  "
                    f"Eta {datetime.timedelta(seconds=int(etas))}")

        if accelerator.is_main_process:
            gt_out_gathered = np.concatenate(gt_out_gathered, axis=0)
            pred_out_gathered = np.concatenate(pred_out_gathered, axis=0)
            lpips_gathered = np.concatenate(lpips_gathered, axis=0)
            psnr_gathered = np.concatenate(psnr_gathered, axis=0)
            ssim_gathered = np.concatenate(ssim_gathered, axis=0)
            ssim_256_gathered = np.concatenate(ssim_256_gathered, axis=0)
            if os.environ.get("WANDB_MODE", None) != "offline":
                assert len(gt_out_gathered) == len(fid_real_data)
                assert len(pred_out_gathered) == len(lpips_gathered) == len(psnr_gathered) == \
                    len(ssim_gathered) == len(ssim_256_gathered) == len(test_data)

            mu1 = np.mean(gt_out_gathered, axis=0)
            sigma1 = np.cov(gt_out_gathered, rowvar=False)
            mu2 = np.mean(pred_out_gathered, axis=0)
            sigma2 = np.cov(pred_out_gathered, rowvar=False)

            mu1 = np.atleast_1d(mu1)
            mu2 = np.atleast_1d(mu2)
            sigma1 = np.atleast_2d(sigma1)
            sigma2 = np.atleast_2d(sigma2)

            diff = mu1 - mu2

            # Product might be almost singular
            covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
            if not np.isfinite(covmean).all():
                msg = ('fid calculation produces singular product; '
                    'adding %s to diagonal of cov estimates') % 1e-6
                logger.info(msg)
                offset = np.eye(sigma1.shape[0]) * 1e-6
                covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

            # Numerical error might give slight imaginary component
            if np.iscomplexobj(covmean):
                if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                    m = np.max(np.abs(covmean.imag))
                    raise ValueError('Imaginary component {}'.format(m))
                covmean = covmean.real

            tr_covmean = np.trace(covmean)

            score_fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
            score_lpips = np.mean(lpips_gathered)
            score_ssim = np.mean(ssim_gathered)
            score_ssim_256 = np.mean(ssim_256_gathered)
            score_psnr = np.mean(psnr_gathered)

            logger.info("Evaluation Results:")
            logger.info(f"FID: {score_fid:.3f}")
            logger.info(f"LPIPS: {score_lpips:.4f}")
            logger.info(f"SSIM: {score_ssim:.4f}")
            logger.info(f"SSIM_256: {score_ssim_256:.4f}")
            logger.info(f"PSNR: {score_psnr:.3f}")

            accelerator.log({
                "score_fid": score_fid,
                "score_lpips": score_lpips,
                "score_ssim": score_ssim,
                "score_ssim_256": score_ssim_256,
                "score_psnr": score_psnr
            }, step=global_step)

        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()


def sample(cfg, weight_dtype, accelerator, noise_scheduler, vae, unet, noisy_latents,
           c_new, down_block_additional_residuals, up_block_additional_residuals):
    bsz = noisy_latents.shape[0]
    noise_scheduler.set_timesteps(cfg.TEST.NUM_INFERENCE_STEPS)

    if cfg.TEST.GUIDANCE_TYPE == "uc_full":
        down_block_additional_residuals = [torch.cat([torch.zeros_like(sample), sample]).to(dtype=weight_dtype) \
                                           for sample in down_block_additional_residuals]
        up_block_additional_residuals = {k: torch.cat([torch.zeros_like(v), v]).to(dtype=weight_dtype) \
                                         for k, v in up_block_additional_residuals.items()}

        for t in noise_scheduler.timesteps:
            inputs = torch.cat([noisy_latents, noisy_latents], dim=0)
            inputs = noise_scheduler.scale_model_input(inputs, timestep=t)
            with accelerator.autocast():
                noise_pred = unet(sample=inputs, timestep=t, encoder_hidden_states=c_new,
                    down_block_additional_residuals=copy.deepcopy(down_block_additional_residuals),
                    up_block_additional_residuals=copy.deepcopy(up_block_additional_residuals))

            noise_pred_uc, noise_pred_full = noise_pred.chunk(2)
            noise_pred = noise_pred_uc + cfg.TEST.FULL_GUIDANCE_SCALE * (noise_pred_full - noise_pred_uc)
            noisy_latents = noise_scheduler.step(noise_pred, t, noisy_latents)[0]

    elif cfg.TEST.GUIDANCE_TYPE == "updown_full":
        down_block_additional_residuals = [torch.cat([sample, sample]).to(dtype=weight_dtype) \
                                           for sample in down_block_additional_residuals]
        up_block_additional_residuals = {k: torch.cat([v, v]).to(dtype=weight_dtype) \
                                         for k, v in up_block_additional_residuals.items()}

        for t in noise_scheduler.timesteps:
            inputs = torch.cat([noisy_latents, noisy_latents], dim=0)
            inputs = noise_scheduler.scale_model_input(inputs, timestep=t)
            with accelerator.autocast():
                noise_pred = unet(sample=inputs, timestep=t, encoder_hidden_states=c_new,
                    down_block_additional_residuals=copy.deepcopy(down_block_additional_residuals),
                    up_block_additional_residuals=copy.deepcopy(up_block_additional_residuals))

            noise_pred_updown, noise_pred_full = noise_pred.chunk(2)
            noise_pred = noise_pred_updown + cfg.TEST.FULL_GUIDANCE_SCALE * (noise_pred_full - noise_pred_updown)
            noisy_latents = noise_scheduler.step(noise_pred, t, noisy_latents)[0]

    elif cfg.TEST.GUIDANCE_TYPE == "down_full":
        down_block_additional_residuals = [torch.cat([sample, sample]).to(dtype=weight_dtype) \
                                           for sample in down_block_additional_residuals]
        up_block_additional_residuals = {k: torch.cat([torch.zeros_like(v), v]).to(dtype=weight_dtype) \
                                         for k, v in up_block_additional_residuals.items()}

        for t in noise_scheduler.timesteps:
            inputs = torch.cat([noisy_latents, noisy_latents], dim=0)
            inputs = noise_scheduler.scale_model_input(inputs, timestep=t)
            with accelerator.autocast():
                noise_pred = unet(sample=inputs, timestep=t, encoder_hidden_states=c_new,
                    down_block_additional_residuals=copy.deepcopy(down_block_additional_residuals),
                    up_block_additional_residuals=copy.deepcopy(up_block_additional_residuals))

            noise_pred_down, noise_pred_full = noise_pred.chunk(2)
            noise_pred = noise_pred_down + cfg.TEST.FULL_GUIDANCE_SCALE * (noise_pred_full - noise_pred_down)
            noisy_latents = noise_scheduler.step(noise_pred, t, noisy_latents)[0]

    elif cfg.TEST.GUIDANCE_TYPE == "uc_down_full":
        c_new = torch.cat([c_new[:bsz], c_new[:bsz], c_new[bsz:]])
        down_block_additional_residuals = [torch.cat([torch.zeros_like(sample), sample, sample]).to(dtype=weight_dtype) \
                                           for sample in down_block_additional_residuals]
        up_block_additional_residuals = {k: torch.cat([torch.zeros_like(v), torch.zeros_like(v), v]).to(dtype=weight_dtype) \
                                         for k, v in up_block_additional_residuals.items()}

        for t in noise_scheduler.timesteps:
            inputs = torch.cat([noisy_latents, noisy_latents, noisy_latents], dim=0)
            inputs = noise_scheduler.scale_model_input(inputs, timestep=t)
            with accelerator.autocast():
                noise_pred = unet(sample=inputs, timestep=t, encoder_hidden_states=c_new,
                    down_block_additional_residuals=copy.deepcopy(down_block_additional_residuals),
                    up_block_additional_residuals=copy.deepcopy(up_block_additional_residuals))

            noise_pred_uc, noise_pred_down, noise_pred_full = noise_pred.chunk(3)
            noise_pred = noise_pred_uc + \
                         cfg.TEST.DOWN_BLOCK_GUIDANCE_SCALE * (noise_pred_down - noise_pred_uc) + \
                         cfg.TEST.FULL_GUIDANCE_SCALE * (noise_pred_full - noise_pred_down)
            noisy_latents = noise_scheduler.step(noise_pred, t, noisy_latents)[0]

    elif cfg.TEST.GUIDANCE_TYPE == "uc_down_updown_cdown":
        c_new = torch.cat([c_new[:bsz], c_new[:bsz], c_new[:bsz], c_new[bsz:]])
        down_block_additional_residuals = [torch.cat([torch.zeros_like(sample), sample, sample, sample]).to(dtype=weight_dtype) \
                                           for sample in down_block_additional_residuals]
        up_block_additional_residuals = {k: torch.cat([torch.zeros_like(v), torch.zeros_like(v), v, torch.zeros_like(v)]).to(dtype=weight_dtype) \
                                         for k, v in up_block_additional_residuals.items()}

        for t in noise_scheduler.timesteps:
            inputs = torch.cat([noisy_latents, noisy_latents, noisy_latents, noisy_latents], dim=0)
            inputs = noise_scheduler.scale_model_input(inputs, timestep=t)
            with accelerator.autocast():
                noise_pred = unet(sample=inputs, timestep=t, encoder_hidden_states=c_new,
                    down_block_additional_residuals=copy.deepcopy(down_block_additional_residuals),
                    up_block_additional_residuals=copy.deepcopy(up_block_additional_residuals))

            noise_pred_uc, noise_pred_down, noise_pred_updown, noise_pred_cdown = noise_pred.chunk(4)
            noise_pred = noise_pred_uc + \
                         cfg.TEST.DOWN_BLOCK_GUIDANCE_SCALE * (noise_pred_down - noise_pred_uc) + \
                         cfg.TEST.ALL_BLOCK_GUIDANCE_SCALE * (noise_pred_updown - noise_pred_down) + \
                         cfg.TEST.GUIDANCE_SCALE * (noise_pred_cdown - noise_pred_down)
            noisy_latents = noise_scheduler.step(noise_pred, t, noisy_latents)[0]

    elif cfg.TEST.GUIDANCE_TYPE == "uc_down_updown_full":
        c_new = torch.cat([c_new[:bsz], c_new[:bsz], c_new[:bsz], c_new[bsz:]])
        down_block_additional_residuals = [torch.cat([torch.zeros_like(sample), sample, sample, sample]).to(dtype=weight_dtype) \
                                           for sample in down_block_additional_residuals]
        up_block_additional_residuals = {k: torch.cat([torch.zeros_like(v), torch.zeros_like(v), v, v]).to(dtype=weight_dtype) \
                                         for k, v in up_block_additional_residuals.items()}

        for t in noise_scheduler.timesteps:
            inputs = torch.cat([noisy_latents, noisy_latents, noisy_latents, noisy_latents], dim=0)
            inputs = noise_scheduler.scale_model_input(inputs, timestep=t)
            with accelerator.autocast():
                noise_pred = unet(sample=inputs, timestep=t, encoder_hidden_states=c_new,
                    down_block_additional_residuals=copy.deepcopy(down_block_additional_residuals),
                    up_block_additional_residuals=copy.deepcopy(up_block_additional_residuals))

            noise_pred_uc, noise_pred_down, noise_pred_updown, noise_pred_full = noise_pred.chunk(4)
            noise_pred = noise_pred_uc + \
                         cfg.TEST.DOWN_BLOCK_GUIDANCE_SCALE * (noise_pred_down - noise_pred_uc) + \
                         cfg.TEST.ALL_BLOCK_GUIDANCE_SCALE * (noise_pred_updown - noise_pred_down) + \
                         cfg.TEST.FULL_GUIDANCE_SCALE * (noise_pred_full - noise_pred_updown)
            noisy_latents = noise_scheduler.step(noise_pred, t, noisy_latents)[0]

    with accelerator.autocast():
        sampling_imgs = vae.decode(noisy_latents) * 0.5 + 0.5 # denormalize
        sampling_imgs = sampling_imgs.clamp(0, 1)
    return sampling_imgs


def inverse_sample(num_inference_steps, accelerator, inverse_noise_scheduler, vae, unet, img_src,
                   c_new, down_block_additional_residuals=None, up_block_additional_residuals=None):
    inverse_noise_scheduler.set_timesteps(num_inference_steps)
    with accelerator.autocast():
        noisy_latents = vae.encode(img_src)

    for t in inverse_noise_scheduler.timesteps:
        inputs = noisy_latents
        with accelerator.autocast():
            noise_pred = unet(sample=inputs, timestep=t, encoder_hidden_states=c_new,
                down_block_additional_residuals=copy.deepcopy(down_block_additional_residuals) if down_block_additional_residuals else None,
                up_block_additional_residuals=copy.deepcopy(up_block_additional_residuals) if up_block_additional_residuals else None)
        noisy_latents = inverse_noise_scheduler.step(noise_pred, t, noisy_latents)[0]

    return noisy_latents


def postprocess_image(tensor, nrow):
    tensor = tensor * 255.
    tensor = torch.clamp(tensor, min=0., max=255.)
    tensor = rearrange(tensor, 'n b c h w -> b n c h w')
    tensor = rearrange(tensor, 'b n c h w -> (b n) c h w')
    tensor = make_grid(tensor, nrow=nrow)
    img = tensor.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    return Image.fromarray(img)


def main(cfg):
    project_dir = os.path.join("outputs", cfg.ACCELERATE.PROJECT_NAME)
    run_dir = os.path.join(project_dir, cfg.ACCELERATE.RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)

    accelerator = Accelerator(
        log_with = ["wandb", "tensorboard"],
        project_dir = project_dir,
        mixed_precision = cfg.ACCELERATE.MIXED_PRECISION
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
        level = logging.INFO,
        format = fmt,
        datefmt = datefmt,
        filename = f"{run_dir}/log_rank{accelerator.process_index}.txt",
        filemode = "a"
    )
    if accelerator.is_main_process:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(console_handler)

    logger.info(f"running with config:\n{str(cfg)}")

    logger.info("preparing datasets...")
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

    from pose_transfer_train import build_model
    model = build_model(cfg)
    unet = UNet(cfg)
    metric = build_metric().to(accelerator.device)

    logger.info(model.load_state_dict(torch.load(
        os.path.join(cfg.MODEL.PRETRAINED_PATH, "pytorch_model.bin"), map_location="cpu"
    ), strict=False))
    logger.info(unet.load_state_dict(torch.load(
        os.path.join(cfg.MODEL.PRETRAINED_PATH, "pytorch_model_1.bin"), map_location="cpu"
    ), strict=False))

    logger.info("preparing accelerator...")
    model, unet, test_loader, fid_real_loader = accelerator.prepare(model, unet, test_loader, fid_real_loader)

    save_dir = os.path.join(run_dir, "log_images")
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

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose Transfer Testing")
    parser.add_argument("--config_file", type=str, default="", help="path to config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help=
                        "modify config options using the command-line")
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    main(cfg)