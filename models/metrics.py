"""
@author: anonymous
@email:  anonymous
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torchvision import models


class build_metric(nn.Module):
    def __init__(self):
        super().__init__()

        # FID
        inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        self.inception_blocks = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        # LPIPS
        self.lpips_model = LPIPS(net="alex", verbose=False)

        # freeze
        self.eval()
        self.requires_grad_(False)

    def forward(self, gt, pred=None):
        if pred is None:
            return self.forward_inception(gt).reshape(gt.shape[0], -1) # fid real

        # inputs should be [0,1] here
        assert gt.shape[0] == pred.shape[0]
        bsz = gt.shape[0]

        # FID
        out = self.forward_inception(pred).reshape(bsz, -1)

        # LPIPS
        lpips = self.lpips_model(pred, gt, normalize=True).reshape(bsz, -1)

        # PSNR & SSIM
        img_gts = gt.cpu().numpy()
        img_preds = pred.cpu().numpy()
        psnr = []
        ssim = []
        ssim_256 = []

        for i in range(bsz):
            img_gt = img_gts[i]
            img_pred = img_preds[i]

            psnr.append(compare_psnr(img_gt, img_pred, data_range=1))
            ssim.append(compare_ssim(img_gt, img_pred, data_range=1, win_size=51, channel_axis=0))

            img_gt_256 = img_gt * 255.0
            img_pred_256 = img_pred * 255.0
            ssim_256.append(compare_ssim(img_gt_256, img_pred_256, gaussian_weights=True, sigma=1.5,
                                         use_sample_covariance=False, channel_axis=0,
                                         data_range=img_pred_256.max() - img_pred_256.min()))

        psnr = torch.tensor(psnr).to(gt.device).reshape(bsz, -1)
        ssim = torch.tensor(ssim).to(gt.device).reshape(bsz, -1)
        ssim_256 = torch.tensor(ssim_256).to(gt.device).reshape(bsz, -1)
        return out, lpips, psnr, ssim, ssim_256

    def forward_inception(self, x):
        x = F.interpolate(x, size=(299, 299), mode='bilinear')
        x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        out = self.inception_blocks(x)
        return out