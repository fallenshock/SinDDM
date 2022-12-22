import math
import copy
import os
import datetime

import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision import transforms, utils
from PIL import Image
from skimage import color, morphology, filters

import numpy as np
from tqdm import tqdm
from einops import rearrange

import matplotlib
from matplotlib import pyplot as plt

from skimage.exposure import match_histograms

from text2live_util.util import get_augmentations_template
matplotlib.use('Agg')

try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False



def dilate_mask(mask, mode):
    if mode == "harmonization":
        element = morphology.disk(radius=7)
    if mode == "editing":
        element = morphology.disk(radius=20)
    mask = mask.permute((1, 2, 0))
    mask = mask[:, :, 0]
    mask = morphology.binary_dilation(mask, selem=element)
    mask = filters.gaussian(mask, sigma=5)
    mask = mask[:, :, None, None]
    mask = mask.transpose(3, 2, 0, 1)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask


# for roi_sampling

def stat_from_bbs(image, bb):
    y_bb, x_bb, h_bb, w_bb = bb
    bb_mean = torch.mean(image[:, :,y_bb:y_bb+h_bb, x_bb:x_bb+w_bb], dim=(2,3), keepdim=True)
    bb_std = torch.std(image[:, :, y_bb:y_bb+h_bb, x_bb:x_bb+w_bb], dim=(2,3), keepdim=True)
    return [bb_mean, bb_std]


def extract_patch(image, bb):
    y_bb, x_bb, h_bb, w_bb = bb
    image_patch = image[:, :,y_bb:y_bb+h_bb, x_bb:x_bb+w_bb]
    return image_patch


# for clip sampling
def thresholded_grad(grad, quantile=0.8):
    grad_energy = torch.norm(grad, dim=1)
    grad_energy_reshape = torch.reshape(grad_energy, (grad_energy.shape[0],-1))
    enery_quant = torch.quantile(grad_energy_reshape, q=quantile, dim=1, interpolation='nearest')[:,None,None] #[batch ,1 ,1]
    gead_energy_minus_energy_quant = grad_energy - enery_quant
    grad_mask = (gead_energy_minus_energy_quant > 0)[:,None,:,:]

    gead_energy_minus_energy_quant_clamp = torch.clamp(gead_energy_minus_energy_quant, min=0)[:,None,:,:]#[b,1,h,w]
    unit_grad_energy = grad / grad_energy[:,None,:,:] #[b,c,h,w]
    unit_grad_energy[torch.isnan(unit_grad_energy)] = 0
    sparse_grad = gead_energy_minus_energy_quant_clamp * unit_grad_energy #[b,c,h,w]
    return sparse_grad, grad_mask

# helper functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)


# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# building block modules

class SinDDMConvBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),

        ) if exists(time_emb_dim) else None

        self.time_reshape = nn.Conv2d(time_emb_dim, dim, 1)
        self.ds_conv = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1)
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            condition = rearrange(condition, 'b c -> b c 1 1')
            condition = self.time_reshape(condition)
            h = h + condition

        h = self.net(h)
        return h + self.res_conv(x)


# denoiser model

class SinDDMNet(nn.Module):
    def __init__(
            self,
            dim,
            out_dim=None,
            channels=3,
            with_time_emb=True,
            multiscale=False,
            device=None
    ):
        super().__init__()
        self.device = device
        self.channels = channels
        self.multiscale = multiscale

        if with_time_emb:
            time_dim = 32

            if multiscale:
                self.SinEmbTime = SinusoidalPosEmb(time_dim)
                self.SinEmbScale = SinusoidalPosEmb(time_dim)
                self.time_mlp = nn.Sequential(
                    nn.Linear(time_dim * 2, time_dim * 4),
                    nn.GELU(),
                    nn.Linear(time_dim * 4, time_dim)
                )
            else:
                self.time_mlp = nn.Sequential(
                    SinusoidalPosEmb(time_dim),
                    nn.Linear(time_dim, time_dim * 4),
                    nn.GELU(),
                    nn.Linear(time_dim * 4, time_dim)
                )
        else:
            time_dim = None
            self.time_mlp = None

        half_dim = int(dim / 2)

        self.l1 = SinDDMConvBlock(channels, half_dim, time_emb_dim=time_dim)
        self.l2 = SinDDMConvBlock(half_dim, dim, time_emb_dim=time_dim)
        self.l3 = SinDDMConvBlock(dim, dim, time_emb_dim=time_dim)
        self.l4 = SinDDMConvBlock(dim, half_dim, time_emb_dim=time_dim)

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            nn.Conv2d(half_dim, out_dim, 1)
        )

    def forward(self, x, time, scale=None):
        
        if exists(self.multiscale):
            scale_tensor = torch.ones(size=time.shape).to(device=self.device) * scale
            t = self.SinEmbTime(time)
            s = self.SinEmbScale(scale_tensor)
            t_s_vec = torch.cat((t, s), dim=1)
            cond_vec = self.time_mlp(t_s_vec)
        else:
            t = self.time_mlp(time) if exists(self.time_mlp) else None
            cond_vec = t

        x = self.l1(x, cond_vec)
        x = self.l2(x, cond_vec)
        x = self.l3(x, cond_vec)
        x = self.l4(x, cond_vec)

        return self.final_conv(x)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


# dataset classes

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, deblur=False, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.deblur = deblur
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        if deblur:
            self.folder_recon = folder + '_recon/'
            self.paths_recon = [p for ext in exts for p in Path(f'{self.folder_recon}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([

            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths) * 128

    def __getitem__(self, index):
        path = self.paths[0]
        img = Image.open(path).convert('RGB')
        if self.deblur:
            path_recon = self.paths_recon[0]
            img_recon = Image.open(path_recon).convert('RGB')
            return self.transform(img), self.transform(img_recon)
        # else
        return self.transform(img)


def create_img_scales(foldername, filename, n_scales=5, scale_step=1.411, image_size=None, create=False, auto_scale=None, const_rf_ratio='small'):
    orig_image = Image.open(foldername + filename)
    # convert to PNG extension for lossless conversion
    filename = filename.rsplit( ".", 1 )[ 0 ] + '.png'
    if image_size is None:
        image_size = (orig_image.size)
    if auto_scale is not None:
        scaler = np.sqrt((image_size[0] * image_size[1])/auto_scale)
        if scaler > 1:
            image_size = (int(image_size[0]/scaler), int(image_size[1]/scaler))
    sizes = []
    downscaled_images = []
    recon_images = []
    rescale_losses = []

    # auto resize
    rf_net = np.asarray(35) # denoiser net RF
    area_rf = rf_net ** 2
    area_scale_0 = 3110  # defined such that area_rf/area_scale0 ~= 40%
    # area_ratio = area_rf/area_scale_0
    s_dim = min(image_size[0], image_size[1])
    l_dim = max(image_size[0], image_size[1])
    scale_0_dim = int(round(np.sqrt(area_scale_0*s_dim/l_dim)))
    # clamp between 42 and 55
    scale_0_dim = 42 if scale_0_dim < 42 else (55 if scale_0_dim > 55 else scale_0_dim )

    if const_rf_ratio == 'small':
        small_val = scale_0_dim
        min_val_image = min(image_size[0], image_size[1])
        n_scales = int(round( (np.log(min_val_image/small_val)) / (np.log(scale_step)) ) + 1)
        scale_step = np.exp((np.log(min_val_image / small_val)) / (n_scales - 1))

    for i in range(n_scales):
        cur_size = (int(round(image_size[0] / np.power(scale_step, n_scales - i - 1))),
                    int(round(image_size[1] / np.power(scale_step, n_scales - i - 1))))
        cur_img = orig_image.resize(cur_size, Image.LANCZOS)
        path_to_save = foldername + 'scale_' + str(i) + '/'
        if create:
            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            cur_img.save(path_to_save + filename)
        downscaled_images.append(cur_img)
        sizes.append(cur_size)
    for i in range(n_scales - 1):
        recon_image = downscaled_images[i].resize(sizes[i + 1], Image.BILINEAR)
        recon_images.append(recon_image)
        rescale_losses.append(
                np.linalg.norm(np.subtract(downscaled_images[i + 1], recon_image)) / np.asarray(recon_image).size)
        if create:
            path_to_save = foldername + 'scale_' + str(i + 1) + '_recon/'
            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            recon_image.save(path_to_save + filename)

    return sizes, rescale_losses, recon_images, scale_step, n_scales


class MultiScaleGaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            *,
            save_interm=False,
            results_folder = '/Results',
            recon_images=None,
            n_scales,
            scale_step,
            image_sizes,
            scale_mul=(1, 1),
            channels=3,
            timesteps=100,
            train_full_t=False,
            scale_losses=None,
            loss_factor=1,
            loss_type='l1',
            betas=None,
            device=None,
            reblurring=True,
            sample_limited_t=False,
            omega=0,
    ):
        super().__init__()
        self.device = device
        self.save_interm = save_interm
        self.results_folder = Path(results_folder)
        self.recon_images = recon_images
        self.channels = channels
        self.n_scales = n_scales
        self.scale_step = scale_step
        self.image_sizes = ()
        self.scale_mul = scale_mul

        self.sample_limited_t = sample_limited_t
        self.reblurring = reblurring

        self.img_prev_upsample = None

        # CLIP guided sampling
        self.clip_guided_sampling = False
        self.guidance_sub_iters = None
        self.stop_guidance = None
        self.quantile = 0.8
        self.clip_model = None
        self.clip_strength = None
        self.clip_text = ''
        self.text_embedds = None
        self.text_embedds_hr = None
        self.text_embedds_lr = None
        self.clip_text_features = None
        self.clip_score = []
        self.clip_mask = None
        self.llambda = 0
        self.x_recon_prev = None

        # for clip_roi
        self.clip_roi_bb = []

        # omega tests
        self.omega = omega

        # ROI guided sampling
        self.roi_guided_sampling = False
        self.roi_bbs = []  # roi_bbs - list of [y,x,h,w]
        self.roi_bbs_stat = []  # roi_bbs_stat - list of [mean_tensor[1,3,1,1], std_tensor[1,3,1,1]]
        self.roi_target_patch = []

        for i in range(n_scales):  # flip xy->hw
            self.image_sizes += ((image_sizes[i][1], image_sizes[i][0]),)

        self.denoise_fn = denoise_fn

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        # self.num_timesteps_trained = int(timesteps_trained) # overwritten if scale_loss is given
        self.num_timesteps_trained = []
        self.num_timesteps_ideal = []
        self.num_timesteps_trained.append(self.num_timesteps)
        self.num_timesteps_ideal.append(self.num_timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        sigma_t = np.sqrt(1. - alphas_cumprod) / np.sqrt(alphas_cumprod) # sigma_t = sqrt_one_minus_alphas_cumprod_div_sqrt_alphas_cumprod

        # flag to force training of all the timesteps across all scales
        if scale_losses is not None:
            for i in range(n_scales - 1):
                self.num_timesteps_ideal.append(
                    int(np.argmax(sigma_t > loss_factor * scale_losses[i])))
                if train_full_t:
                    self.num_timesteps_trained.append(
                        int(timesteps))
                else:
                    self.num_timesteps_trained.append(self.num_timesteps_ideal[i+1])

        # gamma blur schedule
        gammas = torch.zeros(size=(n_scales - 1, self.num_timesteps), device=self.device)
        for i in range(n_scales - 1):
            gammas[i,:] = (torch.tensor(sigma_t, device=self.device) / (loss_factor * scale_losses[i])).clamp(min=0, max=1)

        self.register_buffer('gammas', gammas)

    # for roi_guided_sampling
    #
    def roi_patch_modification(self, x_recon, scale=0, eta=0.8):
        x_modified = x_recon
        for bb in self.roi_bbs:  # bounding box is of shape [y,x,h,w]
            bb = [int(bb_i / np.power(self.scale_step, self.n_scales - scale - 1)) for bb_i in bb]
            bb_y, bb_x, bb_h, bb_w = bb
            target_patch_resize = F.interpolate(self.roi_target_patch[scale], size=(bb_h, bb_w))
            x_modified[:, :, bb_y:bb_y + bb_h, bb_x:bb_x + bb_w] = eta * target_patch_resize + (1-eta) * x_modified[:, :, bb_y:bb_y + bb_h, bb_x:bb_x + bb_w]
        return x_modified

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, s, noise):

        x_recon_ddpm = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

        if not self.reblurring or s == 0:
            return x_recon_ddpm, x_recon_ddpm  # x_t_mix = x_tm1_mix
        else:
            cur_gammas = self.gammas[s - 1].reshape(-1).clamp(0, 0.55)
            x_tm1_mix = (x_recon_ddpm - extract(cur_gammas, t, x_recon_ddpm.shape) * self.img_prev_upsample) / (
                        1 - extract(cur_gammas, t, x_recon_ddpm.shape))
            x_t_mix = x_recon_ddpm
            return x_tm1_mix, x_t_mix


    def q_posterior(self, x_start, x_t_mix, x_t, t, s, pred_noise):  # x_start is x_tm1_mix
        if not self.reblurring or s == 0:
            # regular DDPM
            posterior_mean = (
                    extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                    extract(self.posterior_mean_coef2, t, x_t.shape) * x_t

            )
            posterior_variance = extract(self.posterior_variance, t, x_t.shape)
            posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        elif t[0]>0:
            x_tm1_mix = x_start

            posterior_variance_low = torch.zeros(x_t.shape,
                                                 device=self.device)  # extract(self.posterior_variance, t, x_t.shape)
            posterior_variance_high = 1 - extract(self.alphas_cumprod, t - 1, x_t.shape)
            omega = self.omega
            posterior_variance = (1-omega) * posterior_variance_low + omega * posterior_variance_high
            posterior_log_variance_clipped = torch.log(posterior_variance.clamp(1e-20,None))

            var_t = posterior_variance

            posterior_mean = extract(self.sqrt_alphas_cumprod, t-1, x_t.shape) * x_tm1_mix + \
                             torch.sqrt(1-extract(self.alphas_cumprod, t-1, x_t.shape) - var_t) * \
                             (x_t - extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t_mix) / \
                             extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        else:
            posterior_mean = x_start  # for t==0 no noise added
            posterior_variance = extract(self.posterior_variance, t, x_t.shape)
            posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.enable_grad()
    def p_mean_variance(self, x, t, s, clip_denoised: bool):
        pred_noise = self.denoise_fn(x, t, scale=s)
        x_recon, x_t_mix = self.predict_start_from_noise(x, t=t, s=s, noise=pred_noise)
        pred_noise.clamp_(-1.,1.)
        cur_gammas = self.gammas[s - 1].reshape(-1).clamp(0, 0.55)

        if self.save_interm:
            final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            final_img = (x_recon.clamp(-1., 1.) + 1) * 0.5
            utils.save_image(final_img,
                             str(final_results_folder / f'denoised_t-{t[0]:03}_s-{s}.png'),
                             nrow=4)
        # CLIP guidance
        if self.clip_guided_sampling and (self.stop_guidance <= t[0] or s < self.n_scales - 1) and self.guidance_sub_iters[s] > 0:
            if clip_denoised:
                x_recon.clamp_(-1., 1.)

            # preserve CLIP changes from previous iteration
            if self.clip_mask is not None:
                x_recon = x_recon * (1 - self.clip_mask) + (
                        (1 - self.llambda) * self.x_recon_prev + self.llambda * x_recon) * self.clip_mask
            x_recon.requires_grad_(True)  # for autodiff

            x_recon_renorm = (x_recon + 1) * 0.5
            for i in range(self.guidance_sub_iters[s]):  # can experiment with more than 1 iter. per timestep
                self.clip_model.zero_grad()
                # choose text embedding augmentation (High Res / Low Res)
                if s > 0:
                    score = -self.clip_model.calculate_clip_loss(x_recon_renorm, self.text_embedds_hr)
                else:
                    score = -self.clip_model.calculate_clip_loss(x_recon_renorm, self.text_embedds_lr)

                clip_grad = torch.autograd.grad(score, x_recon, create_graph=False)[0]

                # create CLIP mask depending on the strongest gradients locations
                if self.clip_mask is None:
                    clip_grad, clip_mask = thresholded_grad(grad=clip_grad, quantile=self.quantile)
                    self.clip_mask = clip_mask.float()

                if self.save_interm:
                    final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
                    final_results_folder.mkdir(parents=True, exist_ok=True)
                    final_mask = self.clip_mask.type(torch.float64)

                    utils.save_image(final_mask,
                                     str(final_results_folder / f'clip_mask_s-{s}.png'),
                                     nrow=4)
                    utils.save_image((x_recon.clamp(-1., 1.) + 1) * 0.5,
                                     str(final_results_folder / f'clip_out_s-{s}_t-{t[0]}_subiter_{i}.png'),
                                     nrow=4)

                #normalize gradients
                division_norm = torch.linalg.vector_norm(x_recon * self.clip_mask, dim=(1,2,3), keepdim=True) / torch.linalg.vector_norm(
                    clip_grad * self.clip_mask, dim=(1,2,3), keepdim=True)

                # update clean image
                x_recon += self.clip_strength * division_norm * clip_grad * self.clip_mask

                x_recon.clamp_(-1., 1.)
                # prepare for next sub-iteration
                x_recon_renorm = (x_recon + 1) * 0.5
                # plot score
                self.clip_score.append(score.detach().cpu())

            self.x_recon_prev = x_recon.detach()

            # plot clip loss
            plt.rcParams['figure.figsize'] = [16, 8]
            plt.plot(self.clip_score)
            plt.grid(True)
            # plt.ylim((0, 0.2))
            plt.savefig(str(self.results_folder / 'clip_score'))
            plt.clf()

        # ROI guided sampling
        elif self.roi_guided_sampling and (s < self.n_scales-1):
            x_recon = self.roi_patch_modification(x_recon, scale=s)

        # else normal sampling

        if int(s) > 0 and t[0] > 0 and self.reblurring:
            x_tm1_mix = extract(cur_gammas, t - 1, x_recon.shape) * self.img_prev_upsample + \
                        (1 - extract(cur_gammas, t - 1, x_recon.shape)) * x_recon  # mix blurred and orig
        else:
            x_tm1_mix = x_recon

        if clip_denoised:
            x_tm1_mix.clamp_(-1., 1.)
            x_t_mix.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_tm1_mix, x_t_mix=x_t_mix,
                                                                                  x_t=x, t=t, s=s,
                                                                                  pred_noise=pred_noise)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, s, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device

        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s, clip_denoised=clip_denoised)

        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        nonzero_mask_s = torch.tensor([True], device=self.device).float()

        return model_mean + nonzero_mask_s * nonzero_mask * (0.5 * model_log_variance).exp() * noise


    @torch.no_grad()
    def p_sample_loop(self, shape, s):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        if self.save_interm:
            final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            final_img = (img + 1) * 0.5
            utils.save_image(final_img,
                             str(final_results_folder / f'input_noise_s-{s}.png'),
                             nrow=4)
        if self.sample_limited_t and s < (self.n_scales-1):
            t_min = self.num_timesteps_ideal[s+1]
        else:
            t_min = 0
        for i in tqdm(reversed(range(t_min, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), s)
            if self.save_interm:
                final_img = (img + 1) * 0.5
                utils.save_image(final_img,
                                 str(final_results_folder / f'output_t-{i:03}_s-{s}.png'),
                                 nrow=4)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, scale_0_size=None, s=0):
        if scale_0_size is not None:
            image_size = scale_0_size
        else:
            image_size = self.image_sizes[0]
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size[0], image_size[1]), s=s)

    @torch.no_grad()
    def p_sample_via_scale_loop(self, batch_size, img, s, custom_t=0):
        device = self.betas.device
        if custom_t == 0:
            total_t = self.num_timesteps_ideal[min(s, self.n_scales-1)]-1
        else:
            total_t = custom_t
        b = batch_size
        self.img_prev_upsample = img
        if self.save_interm:
            final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            final_img = (img + 1) * 0.5
            utils.save_image(final_img,
                             str(final_results_folder / f'clean_input_s_{s}.png'),
                             nrow=4)
        img = self.q_sample(x_start=img, t=torch.Tensor.expand(torch.tensor(total_t, device=device), batch_size), noise=None)  # add noise

        if self.save_interm:
            final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            final_img = (img + 1) * 0.5
            utils.save_image(final_img,
                             str(final_results_folder / f'noisy_input_s_{s}.png'),
                             nrow=4)

        if self.clip_mask is not None:
            if s > 0:
                mul_size = [int(self.image_sizes[s][0]* self.scale_mul[0]), int(self.image_sizes[s][1]* self.scale_mul[1])]
                self.clip_mask = F.interpolate(self.clip_mask, size=mul_size, mode='bilinear')#.bool()
                self.x_recon_prev = F.interpolate(self.x_recon_prev, size=mul_size, mode='bilinear')
            else:  # mask created at scale 0 is usually too noisy
                self.clip_mask = None

        if self.sample_limited_t and s < (self.n_scales - 1):
            t_min = self.num_timesteps_ideal[s + 1]
        else:
            t_min = 0
        for i in tqdm(reversed(range(t_min, total_t)), desc='sampling loop time step',
                      total=total_t):

            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), s)
            if self.save_interm:
                final_img = (img + 1) * 0.5
                utils.save_image(final_img,
                                 str(final_results_folder / f'output_t-{i:03}_s-{s}.png'),
                                 nrow=4)
        return img

    @torch.no_grad()
    def sample_via_scale(self, batch_size, img, s, scale_mul=(1, 1), custom_sample=False, custom_img_size_idx=0, custom_t=0, custom_image_size=None):
        if custom_sample:
            if custom_img_size_idx >= self.n_scales:
                size = self.image_sizes[self.n_scales-1] # extrapolate size
                factor = self.scale_step ** (custom_img_size_idx + 1 - self.n_scales)
                size = (int(size[0] * factor), int(size[1] * factor))
            else:
                size = self.image_sizes[custom_img_size_idx]
        else:
            size = self.image_sizes[s]
        image_size = (int(size[0] * scale_mul[0]), int(size[1] * scale_mul[1]))
        if custom_image_size is not None:  # force custom image size
            image_size = custom_image_size

        img = F.interpolate(img, size=image_size, mode='bilinear')
        return self.p_sample_via_scale_loop(batch_size, img, s, custom_t=custom_t)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, s, noise=None, x_orig=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        if int(s) > 0:
            cur_gammas = self.gammas[s - 1].reshape(-1)
            x_mix = extract(cur_gammas, t, x_start.shape) * x_start + \
                    (1 - extract(cur_gammas, t, x_start.shape)) * x_orig  # mix blurred and orig
            x_noisy = self.q_sample(x_start=x_mix, t=t, noise=noise)  # add noise
            x_recon = self.denoise_fn(x_noisy, t, s)

        else:
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            x_recon = self.denoise_fn(x_noisy, t, s)

        if self.loss_type == 'l1':

            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        elif self.loss_type == 'l1_pred_img':
            if int(s) > 0:
                cur_gammas = self.gammas[s - 1].reshape(-1)
                if t[0]>0:
                    x_mix_prev = extract(cur_gammas, t-1, x_start.shape) * x_start + \
                            (1 - extract(cur_gammas, t-1, x_start.shape)) * x_orig  # mix blurred and orig
                else:
                    x_mix_prev = x_orig
            else:
                x_mix_prev = x_start
            loss = (x_mix_prev-x_recon).abs().mean()
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, s, *args, **kwargs):
        if int(s) > 0:  # no deblurring in scale=0
            x_orig = x[0]
            x_recon = x[1]
            b, c, h, w = x_orig.shape
            device = x_orig.device
            img_size = self.image_sizes[s]
            assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
            t = torch.randint(0, self.num_timesteps_trained[s], (b,), device=device).long()
            return self.p_losses(x_recon, t, s, x_orig=x_orig, *args, **kwargs)

        else:

            b, c, h, w = x[0].shape
            device = x[0].device
            img_size = self.image_sizes[s]
            assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
            t = torch.randint(0, self.num_timesteps_trained[s], (b,), device=device).long()
            return self.p_losses(x[0], t, s, *args, **kwargs)


class MultiscaleTrainer(object):

    def __init__(
            self,
            ms_diffusion_model,
            folder,
            *,
            ema_decay=0.995,
            n_scales=None,
            scale_step=1,
            image_sizes=None,
            train_batch_size=32,
            train_lr=2e-5,
            train_num_steps=100000,
            gradient_accumulate_every=2,
            fp16=False,
            step_start_ema=2000,
            update_ema_every=10,
            save_and_sample_every=25000,
            avg_window=100,
            sched_milestones=None,
            results_folder='./results',
            args=None,
            device=None
    ):
        super().__init__()
        self.device = device
        if sched_milestones is None:
            self.sched_milestones = [10000, 30000, 60000, 80000, 90000]
        else:
            self.sched_milestones = sched_milestones
        if image_sizes is None:
            image_sizes = []
        n_pixels = []
        for i in range(n_scales):
            n_pixels.append(image_sizes[i][0]*image_sizes[i][1])
        self.n_pixels = torch.Tensor(n_pixels).to(self.device)
        self.sqrt_n_pixels = torch.sqrt(self.n_pixels)
        self.pix_cumsum = torch.cumsum(self.n_pixels, dim=0)
        self.sqrt_pix_cumsum = torch.cumsum(self.sqrt_n_pixels, dim=0)
        self.tot_pix = torch.sum(self.n_pixels).int()
        self.tot_sqrt_pix = torch.sum(self.sqrt_n_pixels).int()
        self.model = ms_diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.avg_window = avg_window

        self.batch_size = train_batch_size
        self.n_scales = n_scales
        self.scale_step = scale_step
        self.image_sizes = ms_diffusion_model.image_sizes
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.input_paths = []
        self.output_paths = []
        self.results_folders = []
        self.ds_list = []
        self.dl_list = []
        self.data_list = []

        for i in range(n_scales):
            self.input_paths.append(folder + 'scale_' + str(i))
            self.output_paths.append(results_folder + '/scale_' + str(i))
            deblur = True if i > 0 else False
            self.ds_list.append(Dataset(self.input_paths[i], image_sizes[i], deblur))
            self.dl_list.append(
                cycle(data.DataLoader(self.ds_list[i], batch_size=train_batch_size, shuffle=True, pin_memory=True)))

            self.results_folders.append(Path(results_folder))
            self.results_folders[i].mkdir(parents=True, exist_ok=True)

            if i > 0:
                Data = next(self.dl_list[i])
                self.data_list.append((Data[0].to(self.device), Data[1].to(self.device)))
            else:
                self.data_list.append(
                    (next(self.dl_list[i]).to(self.device), next(self.dl_list[i]).to(self.device)))  # just duplicate orig over deblur

        self.opt = Adam(ms_diffusion_model.parameters(), lr=train_lr)

        self.scheduler = MultiStepLR(self.opt, milestones=self.sched_milestones, gamma=0.5)

        self.step = 0
        self.running_loss = []
        self.running_scale = []
        self.avg_t = []

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt,
                                                                    opt_level='O1')

        self.reset_parameters()
        self.args = args

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'sched': self.scheduler.state_dict(),
            'running_loss': self.running_loss,
            'running_scale': self.running_scale
        }
        torch.save(data, str(self.results_folders[0] / f'model-{milestone}.pt'))
        plt.rcParams['figure.figsize'] = [16, 8]

        plt.plot(self.running_loss)
        plt.grid(True)
        plt.ylim((0, 0.2))
        plt.savefig(str(self.results_folders[0] / 'running_loss'))
        plt.clf()

    def load(self, milestone):
        data = torch.load(str(self.results_folders[0] / f'model-{milestone}.pt'), map_location=self.device)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        self.scheduler.load_state_dict(data['sched'])
        self.running_loss = data['running_loss']
    #    self.running_scale = data['running_scale']

    def train(self):

        backwards = partial(loss_backwards, self.fp16)
        loss_avg = 0
        s_weights = torch.tensor(self.model.num_timesteps_trained, device=self.device, dtype=torch.float)
        while self.step < self.train_num_steps:

            # t weighted multinomial sampling
            s = torch.multinomial(input=s_weights, num_samples=1)  # uniform when train_full_t = True
            for i in range(self.gradient_accumulate_every):
                data = self.data_list[s]
                loss = self.model(data, s)
                loss_avg += loss.item()

                backwards(loss / self.gradient_accumulate_every, self.opt)
            if self.step % self.avg_window == 0:
                print(f'step:{self.step} loss:{loss_avg/self.avg_window}')
                self.running_loss.append(loss_avg/self.avg_window)
                loss_avg = 0
            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
            self.scheduler.step()
            self.step += 1
            if self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                batches = num_to_groups(16, self.batch_size)
                all_images_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5
                utils.save_image(all_images, str(self.results_folders[s] / f'sample-{milestone}.png'), nrow=4)
                self.save(milestone)

        print('training completed')

    def sample_scales(self, scale_mul=None, batch_size=16, custom_sample=False, custom_image_size_idxs=None,
                      custom_scales=None, image_name='', start_noise= False, custom_t_list=None, desc=None, save_unbatched=True):
        if desc is None:
            desc = f'sample_{str(datetime.datetime.now()).replace(":", "_")}'
        if self.ema_model.reblurring:
            desc = desc + '_rblr'
        if self.ema_model.sample_limited_t:
            desc = desc + '_t_lmtd'
        if custom_t_list is None:
            custom_t_list = self.ema_model.num_timesteps_trained[1:]
        if custom_scales is None:
            custom_scales = [*range(self.n_scales)]  # [0, 1, 2, 3, 4, 5, 6]
            n_scales = self.n_scales
        else:
            n_scales = len(custom_scales)
        if custom_image_size_idxs is None:
            custom_image_size_idxs = [*range(self.n_scales)]  # [0, 1, 2, 3, 4, 5, 6]

        samples_from_scales = []
        final_results_folder = Path(str(self.results_folders[0] / 'final_samples'))
        final_results_folder.mkdir(parents=True, exist_ok=True)
        if scale_mul is not None:
            scale_0_size = (
                int(self.model.image_sizes[custom_image_size_idxs[0]][0] * scale_mul[0]),
                int(self.model.image_sizes[custom_image_size_idxs[0]][1] * scale_mul[1]))
        else:
            scale_0_size = None
        t_list = [self.ema_model.num_timesteps_trained[0]] + custom_t_list
        res_sub_folder = '_'.join(str(e) for e in t_list)
        final_img = None
        for i in range(n_scales):
            # if (custom_image_size_idxs[i] == 0 or start_noise) and i == 0:
            if start_noise and i == 0:
                samples_from_scales.append(
                    self.ema_model.sample(batch_size=batch_size, scale_0_size=scale_0_size, s=custom_scales[i]))
            elif i == 0:
                orig_sample_0 = Image.open((self.input_paths[custom_scales[i]] + '/' + image_name)).convert("RGB")

                samples_from_scales.append((transforms.ToTensor()(orig_sample_0) * 2 - 1).repeat(batch_size, 1, 1, 1).to(self.device))
            else:

                samples_from_scales.append(self.ema_model.sample_via_scale(batch_size,
                                                                           samples_from_scales[i - 1],
                                                                           s=custom_scales[i],
                                                                           scale_mul=scale_mul,
                                                                           custom_sample=custom_sample,
                                                                           custom_img_size_idx=custom_image_size_idxs[i],
                                                                           custom_t=custom_t_list[int(custom_scales[i])-1],
                                                                           # custom_t=custom_t_list[i-1],
                                                                           ))
            final_img = (samples_from_scales[i] + 1) * 0.5

            utils.save_image(final_img, str(final_results_folder / res_sub_folder) + f'_out_s{i}_{desc}_sm_{scale_mul[0]}_{scale_mul[1]}.png', nrow=4)
        if save_unbatched:
            final_results_folder = Path(str(self.results_folders[0] / f'final_samples_unbatched_{desc}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            for b in range(batch_size):
                utils.save_image(final_img[b], str(final_results_folder / res_sub_folder) + f'_out_b{b}.png')


    def image2image(self, input_folder='', input_file='', mask='', hist_ref_path='', image_name='', start_s=1, custom_t=None, batch_size=16, scale_mul=(1, 1), device=None, use_hist=False, save_unbatched=True, auto_scale=None, mode=None):
        if custom_t is None:
            custom_t = [0, 0, 0, 0, 0, 0, 0] # 0 - use default sampling t
        orig_image = self.data_list[self.n_scales-1][0][0][None,:,:,:]
        input_path = os.path.join(input_folder, input_file)
        input_img = Image.open(input_path).convert("RGB")
        if mode == 'harmonization':
            mask_path = os.path.join(input_folder, mask)
            mask_img = Image.open(mask_path).convert("RGB")
            mask_img = transforms.ToTensor()(mask_img)
            mask_img = dilate_mask(mask_img, mode=mode)
            mask_img = torch.from_numpy(mask_img).to(self.device)
        else:
            mask_img = 1
        image_size = (input_img.size)
        if auto_scale is not None:
            scaler = np.sqrt((image_size[0] * image_size[1]) / auto_scale)
            if scaler > 1:
                image_size = (int(image_size[0] / scaler), int(image_size[1] / scaler))
                input_img = input_img.resize(image_size, Image.LANCZOS)

        if use_hist:
            image_name = image_name.rsplit(".", 1)[0] + '.png'
            orig_sample_0 = Image.open((hist_ref_path + image_name)).convert("RGB")  # next(self.dl_list[0])
            input_img_ds_matched_arr = match_histograms(image=np.array(input_img), reference=np.array(orig_sample_0), channel_axis=2)
            input_img = Image.fromarray(input_img_ds_matched_arr)
        input_img_tensor = (transforms.ToTensor()(input_img) * 2 - 1)  # normalize
        input_size = torch.tensor(input_img_tensor.shape[1:])
        input_img_batch = input_img_tensor.repeat(batch_size, 1, 1, 1).to(device)  # batchify and send to GPU

        samples_from_scales = []

        final_results_folder = Path(str(self.results_folders[0] / 'i2i_final_samples'))
        final_results_folder.mkdir(parents=True, exist_ok=True)
        final_img   =   None
        t_string = '_'.join(str(e) for e in custom_t)

        for i in range(self.n_scales-start_s):
            s = i + start_s
            ds_factor = self.scale_step ** (self.n_scales - s - 1)
            cur_size = input_size/ds_factor
            cur_size = (int(cur_size[0].item()), int(cur_size[1].item()))

            if i == 0:
                samples_from_scales.append(self.ema_model.sample_via_scale(batch_size,
                                                                           input_img_batch,
                                                                           s=s,
                                                                           custom_t=custom_t[s],
                                                                           scale_mul=scale_mul,
                                                                           custom_image_size=cur_size), )
            else:
                samples_from_scales.append(self.ema_model.sample_via_scale(batch_size,
                                                                           samples_from_scales[i - 1],
                                                                           s=s,
                                                                           custom_t=custom_t[s],
                                                                           scale_mul=scale_mul,
                                                                           custom_image_size=cur_size),)
            final_img = (samples_from_scales[i] + 1) * 0.5
            input_file_name = input_file.rsplit(".", 1)[0]
            if i == self.n_scales-start_s - 1:
                input_img_batch_denorm = (input_img_batch + 1) * 0.5
                input_img_batch_denorm.clamp_(0.0, 1.0)
                final_img = mask_img * final_img + (1 - mask_img) * input_img_batch_denorm

            utils.save_image(final_img, str(final_results_folder / f'{input_file_name}_i2i_s_{start_s+i}_t_{t_string}_hist_{"on" if use_hist else "off"}.png'), nrow=4)
        if save_unbatched:
            final_results_folder = Path(str(self.results_folders[0] / f'unbatched_i2i_s{start_s}_t_{t_string}_{str(datetime.datetime.now()).replace(":", "_")}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            for b in range(batch_size):
                utils.save_image(final_img[b], os.path.join(final_results_folder ,input_file + f'_out_b{b}_i2i.png'))


    def clip_sampling(self, clip_model, text_input, strength, sample_batch_size, custom_t_list=None,
                      guidance_sub_iters=None, quantile=0.8, stop_guidance=None, save_unbatched=False, scale_mul=(1,1), llambda=0, start_noise=True, image_name=''):
        if guidance_sub_iters is None:
            guidance_sub_iters = [*reversed(range(self.n_scales))]
        self.ema_model.clip_strength = strength
        self.ema_model.clip_text = text_input
        self.ema_model.text_embedds_hr = clip_model.get_text_embedding(text_input, template=get_augmentations_template('hr'))
        self.ema_model.text_embedds_lr = clip_model.get_text_embedding(text_input, template=get_augmentations_template('lr'))
        self.ema_model.clip_guided_sampling = True
        self.ema_model.guidance_sub_iters = guidance_sub_iters
        self.ema_model.quantile = quantile
        self.ema_model.stop_guidance = stop_guidance
        self.ema_model.clip_model = clip_model
        self.ema_model.clip_score = []
        self.ema_model.llambda = llambda
        strength_string = f'{strength}'
        gsi_string = '_'.join(str(e) for e in guidance_sub_iters)
        n_aug = self.ema_model.clip_model.cfg["n_aug"]
        desc = f"clip_{text_input.replace(' ', '_')}_n_aug{n_aug}_str_" + strength_string + "_gsi_" + gsi_string + \
               f'_ff{1-quantile}' + f'_{str(datetime.datetime.now()).replace(":", "_")}'

        if not start_noise:
            # start from last scale
            custom_scales = [self.n_scales - 2, self.n_scales - 1]
            custom_image_size_idxs = [self.n_scales - 2, self.n_scales - 1]
            # custom_t_list = [self.ema_model.num_timesteps_ideal[-2], self.ema_model.num_timesteps_ideal[-1]]
            self.sample_scales(scale_mul=scale_mul,  # H,W
                               custom_sample=True,
                               custom_scales=custom_scales,
                               custom_image_size_idxs=custom_image_size_idxs,
                               image_name=image_name,
                               batch_size=sample_batch_size,
                               custom_t_list=custom_t_list,
                               desc=desc,
                               save_unbatched=save_unbatched,
                               start_noise=start_noise,
                               )
        else:
            self.sample_scales(scale_mul=scale_mul,  # H,W
                               custom_sample=False,
                               image_name='',
                               batch_size=sample_batch_size,
                               custom_t_list=custom_t_list,
                               desc=desc,
                               save_unbatched=save_unbatched,
                               start_noise=start_noise,
                               )
        self.ema_model.clip_guided_sampling = False

    def clip_roi_sampling(self, clip_model, text_input, strength, sample_batch_size, custom_t_list=None,
                      num_clip_iters=100, num_denoising_steps=2, clip_roi_bb=None, save_unbatched=False, full_grad=False):

        text_embedds = clip_model.get_text_embedding(text_input, template=get_augmentations_template('lr'))
        strength_string = f'{strength}'
        n_aug = clip_model.cfg["n_aug"]
        desc = f"clip_roi_{text_input.replace(' ', '_')}_n_aug{n_aug}_str_" + strength_string + "_n_iters_" + str(num_clip_iters) + \
               f'_{str(datetime.datetime.now()).replace(":", "_")}'
        image = self.data_list[self.n_scales-1][0][0][None,:,:,:]
        image = image.repeat(sample_batch_size, 1, 1, 1)

        if full_grad:
            image_roi = image.clone()
        else:
            image_roi = image[:,:,clip_roi_bb[0]:clip_roi_bb[0]+clip_roi_bb[2],clip_roi_bb[1]:clip_roi_bb[1]+clip_roi_bb[3]].clone()

        image_roi.requires_grad_(True)
        image_roi_renorm = (image_roi + 1) * 0.5
        interm_results_folder = Path(str(self.ema_model.results_folder / f'interm_samples_clip_roi'))
        interm_results_folder.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(num_clip_iters)):
            clip_model.zero_grad()
            score = -clip_model.calculate_clip_loss(image_roi_renorm, text_embedds)
            clip_grad = torch.autograd.grad(score, image_roi, create_graph=False)[0]
            if self.ema_model.save_interm:
                utils.save_image((image_roi.clamp(-1., 1.) + 1) * 0.5,
                                 str(interm_results_folder / f'iter_{i}.png'),
                                 nrow=4)

            image_roi_prev_norm = torch.linalg.vector_norm(image_roi, dim=(1, 2, 3), keepdim=True)
            division_norm = torch.linalg.vector_norm(image_roi, dim=(1,2,3), keepdim=True) / torch.linalg.vector_norm(
                clip_grad, dim=(1,2,3), keepdim=True)
            image_roi_prev = image_roi
            image_roi = image_roi_prev + strength* division_norm * clip_grad
            image_roi_norm = torch.linalg.vector_norm(image_roi, dim=(1, 2, 3), keepdim=True)
            keep_norm = False
            if keep_norm:
                image_roi *= (image_roi_prev_norm) / (image_roi_norm)

            image_roi.clamp_(-1., 1.)
            image_roi_renorm = (image_roi + 1) * 0.5

        # insert patch into original image
        if full_grad:
            image[:, :, clip_roi_bb[0]:clip_roi_bb[0] + clip_roi_bb[2],clip_roi_bb[1]:clip_roi_bb[1] + clip_roi_bb[3]] = image_roi[:,:,clip_roi_bb[0]:clip_roi_bb[0]+clip_roi_bb[2],clip_roi_bb[1]:clip_roi_bb[1]+clip_roi_bb[3]]
        else:
            image[:, :, clip_roi_bb[0]:clip_roi_bb[0] + clip_roi_bb[2],clip_roi_bb[1]:clip_roi_bb[1] + clip_roi_bb[3]] = image_roi

        final_image = self.ema_model.sample_via_scale(sample_batch_size,
                                                      image,
                                                      s=self.n_scales-1,
                                                      custom_t=num_denoising_steps,
                                                      scale_mul=(1,1))
        final_img_renorm = (final_image + 1) * 0.5
        final_results_folder = Path(str(self.ema_model.results_folder / f'final_samples'))
        final_results_folder.mkdir(parents=True, exist_ok=True)
        utils.save_image(final_img_renorm, str(final_results_folder / (desc + '.png')), nrow=4)
        final_results_folder = Path(str(self.results_folders[0] / f'final_samples_unbatched_{desc}'))
        final_results_folder.mkdir(parents=True, exist_ok=True)

        if save_unbatched:
            final_results_folder = Path(str(self.results_folders[0] / f'final_samples_unbatched_{desc}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            for b in range(sample_batch_size):
                utils.save_image(final_img_renorm[b], os.path.join(final_results_folder, f'{desc}_out_b{b}.png'))

    def roi_guided_sampling(self, custom_t_list=None, target_roi=None, roi_bb_list=None, save_unbatched=False, batch_size=4 ,scale_mul=(1, 1)):
        self.ema_model.roi_guided_sampling = True
        self.ema_model.roi_bbs = roi_bb_list
        target_bb = target_roi
        for scale in range(self.n_scales):

            target_bb_rescaled = [int(bb_i / np.power(self.scale_step, self.n_scales - scale - 1)) for bb_i in target_bb]
            self.ema_model.roi_target_patch.append(extract_patch(self.data_list[scale][0][0][None, :,:,:], target_bb_rescaled))

        self.sample_scales(scale_mul=scale_mul,  # H,W
                           custom_sample=False,
                           image_name='',
                           batch_size=batch_size,
                           custom_t_list=custom_t_list,
                           desc=f'roi_{str(datetime.datetime.now()).replace(":", "_")}',
                           save_unbatched=save_unbatched,
                           start_noise=True,
                           )
        self.ema_model.roi_guided_sampling = False
