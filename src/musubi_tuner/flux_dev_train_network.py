#!/usr/bin/env python3
"""
FLUX Dev LoRA Training Network - Musubi Tuner Integration

This module integrates sd-scripts FLUX dev training capabilities with musubi-tuner's
infrastructure, providing full FLUX.1 dev LoRA training support while maintaining
all of musubi-tuner's advanced features like memory offloading and optimization.

Based on sd-scripts-sd3/flux_train_network.py but adapted for musubi-tuner framework.
"""

import argparse
import copy
import math
import random
from typing import Any, Optional, Union

import torch
from accelerate import Accelerator

from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    clean_memory_on_device,
    setup_parser_common,
    read_config_from_file,
)

# Import sd-scripts FLUX components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../sd-scripts-sd3'))

from library import (
    flux_train_utils,
    flux_utils as sd_scripts_flux_utils,
    strategy_base,
    strategy_flux,
    train_util,
)

# Use musubi-tuner FLUX components (supports supports_backward parameter)
from musubi_tuner.flux import flux_models, flux_utils

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FluxDevNetworkTrainer(NetworkTrainer):
    """
    FLUX Dev LoRA Training Network Trainer

    Integrates sd-scripts FLUX dev training capabilities with musubi-tuner's infrastructure.
    Supports FLUX.1 dev LoRA training with advanced memory management and optimization.
    """

    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self.is_schnell: Optional[bool] = None
        self.is_swapping_blocks: bool = False
        self.model_type: str = "flux"  # Always flux for this trainer
        self.use_clip_l = True
        self.train_clip_l = True
        self.train_t5xxl = False
        self.noise_scheduler_copy = None

    @property
    def architecture(self) -> str:
        return "flux_dev"

    @property
    def architecture_full_name(self) -> str:
        return "flux_dev"

    def handle_model_specific_args(self, args):
        """Handle FLUX dev specific arguments"""
        self.dit_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16

    def setup_noise_scheduler(self, args, device):
        """Setup FLUX noise scheduler"""
        if self.noise_scheduler_copy is None:
            self.get_noise_scheduler(args, device)

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        """Initialize FLUX flow matching noise scheduler"""
        from library import sd3_train_utils

        # Set default guidance scale for FLUX dev
        if not hasattr(args, 'guidance_scale'):
            args.guidance_scale = 3.5  # Default for FLUX dev

        # Enable necessary FLUX features
        if not hasattr(args, 'apply_t5_attn_mask'):
            args.apply_t5_attn_mask = False

        # Set default model prediction type
        if not hasattr(args, 'model_prediction_type'):
            args.model_prediction_type = "raw"

        # Set default weighting scheme
        if not hasattr(args, 'weighting_scheme'):
            args.weighting_scheme = "logit_normal"

        # Set default timestep sampling
        if not hasattr(args, 'timestep_sampling'):
            args.timestep_sampling = "uniform"

        noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=args.discrete_flow_shift
        )
        self.noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        return noise_scheduler

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        """Load FLUX VAE/AutoEncoder"""
        logger.info(f"Loading FLUX AutoEncoder from {args.vae}")
        ae = flux_utils.load_ae(args.vae, dtype=vae_dtype, device="cpu", disable_mmap=True)
        return ae

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        """Load FLUX transformer model"""
        logger.info(f"Loading FLUX transformer from {args.dit}")

        # Use exact same model loading as FLUX kontext (working implementation)
        model = flux_utils.load_flow_model(
            ckpt_path=args.dit,
            dtype=None,  # FLUX kontext always uses None
            device=loading_device,
            disable_mmap=True,
            attn_mode=attn_mode,
            split_attn=split_attn,
            loading_device=loading_device,
            fp8_scaled=args.fp8_scaled,
        )

        # Apply musubi-tuner optimizations
        if hasattr(args, 'blocks_to_swap') and args.blocks_to_swap > 0:
            self.is_swapping_blocks = True

        return model

    def get_tokenize_strategy(self, args):
        """Get FLUX tokenization strategy"""
        return strategy_flux.FluxTokenizeStrategy(args.t5_max_token_length)

    def get_text_encoding_strategy(self, args):
        """Get FLUX text encoding strategy"""
        return strategy_flux.FluxTextEncodingStrategy(apply_t5_attn_mask=args.apply_t5_attn_mask)

    def get_latents_caching_strategy(self, args):
        """Get FLUX latents caching strategy"""
        return strategy_flux.FluxLatentsCachingStrategy(args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check)

    def scale_shift_latents(self, latents):
        """FLUX doesn't need latent scaling/shifting"""
        return latents

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ):
        """Call FLUX DiT for training step"""
        model: flux_models.Flux = transformer

        # FLUX-specific fix: Ensure latents have correct dimensions for FLUX
        # Cached latents can be [C,H,W] or [B,C,1,H,W] - convert to [B,C,H,W]
        if latents.dim() == 3:  # [C,H,W] -> [B,C,H,W]
            latents = latents.unsqueeze(0)
        elif latents.dim() == 5 and latents.shape[2] == 1:  # [B,C,1,H,W] -> [B,C,H,W]
            latents = latents.squeeze(2)

        if noise.dim() == 3:  # [C,H,W] -> [B,C,H,W]
            noise = noise.unsqueeze(0)
        elif noise.dim() == 5 and noise.shape[2] == 1:  # [B,C,1,H,W] -> [B,C,H,W]
            noise = noise.squeeze(2)

        bsz = latents.shape[0]

        # Ensure noise scheduler is initialized
        self.setup_noise_scheduler(args, accelerator.device)

        # Get noisy model input and timesteps using sd-scripts utilities
        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, self.noise_scheduler_copy, latents, noise, accelerator.device, network_dtype
        )

        # Pack latents and get img_ids
        # Ensure noisy_model_input is 4D before packing (squeeze out any singleton dimensions)
        if noisy_model_input.dim() == 5 and noisy_model_input.shape[2] == 1:
            noisy_model_input = noisy_model_input.squeeze(2)  # Remove singleton time dimension
        packed_noisy_model_input = sd_scripts_flux_utils.pack_latents(noisy_model_input)
        packed_latent_height, packed_latent_width = noisy_model_input.shape[2] // 2, noisy_model_input.shape[3] // 2
        img_ids = sd_scripts_flux_utils.prepare_img_ids(bsz, packed_latent_height, packed_latent_width).to(device=accelerator.device)

        # Get guidance
        guidance_vec = torch.full((bsz,), float(args.guidance_scale), device=accelerator.device)

        # Get text encoder outputs from batch
        if "text_encoder_outputs" in batch:
            text_encoder_outputs = batch["text_encoder_outputs"]
            l_pooled = text_encoder_outputs[0]
            t5_out = text_encoder_outputs[1]
            txt_ids = text_encoder_outputs[2]
            t5_attn_mask = text_encoder_outputs[3] if len(text_encoder_outputs) > 3 else None
        else:
            # Fallback to individual components
            l_pooled = batch.get("clip_l_pooler")
            t5_out = batch.get("t5_vec")
            t5_attn_mask = None

        # Ensure t5_out is 3D - squeeze extra batch dimension if present
        if t5_out.ndim == 4 and t5_out.shape[1] == 1:
            t5_out = t5_out.squeeze(1)  # Remove the extra dimension: [B, 1, S, H] -> [B, S, H]

        # Create txt_ids after ensuring proper t5_out shape
        if "text_encoder_outputs" not in batch:
            txt_ids = torch.zeros(t5_out.shape[0], t5_out.shape[1], 3, device=accelerator.device)

        if not args.apply_t5_attn_mask:
            t5_attn_mask = None

        # Enable gradients for gradient checkpointing
        if args.gradient_checkpointing:
            packed_noisy_model_input.requires_grad_(True)
            if t5_out is not None:
                t5_out.requires_grad_(True)
            if l_pooled is not None:
                l_pooled.requires_grad_(True)

        # Ensure all tensors are on the correct device (like FLUX kontext does)
        packed_noisy_model_input = packed_noisy_model_input.to(device=accelerator.device, dtype=network_dtype)
        img_ids = img_ids.to(device=accelerator.device)
        t5_out = t5_out.to(device=accelerator.device, dtype=network_dtype)
        txt_ids = txt_ids.to(device=accelerator.device)
        l_pooled = l_pooled.to(device=accelerator.device, dtype=network_dtype)
        timesteps = timesteps.to(device=accelerator.device)
        guidance_vec = guidance_vec.to(device=accelerator.device, dtype=network_dtype)

        # Call FLUX model
        with accelerator.autocast():
            model_pred = model(
                img=packed_noisy_model_input,
                img_ids=img_ids,
                txt=t5_out,
                txt_ids=txt_ids,
                y=l_pooled,
                timesteps=timesteps / 1000,  # FLUX expects timesteps in [0, 1]
                guidance=guidance_vec,
            )

        # Unpack latents
        model_pred = sd_scripts_flux_utils.unpack_latents(model_pred, packed_latent_height, packed_latent_width)

        # Apply model prediction type
        model_pred, weighting = flux_train_utils.apply_model_prediction_type(args, model_pred, noisy_model_input, sigmas)

        # Flow matching loss (different from SD3)
        target = noise - latents

        return model_pred, target

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=None,
        control_video_path=None,
    ):
        """FLUX dev inference for sampling"""
        # Use sd-scripts sampling utilities
        return flux_train_utils.sample_images(
            accelerator, args, 0, 0, transformer, vae, None, self.sample_prompts_te_outputs
        )


def flux_dev_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """FLUX dev specific parser setup - only arguments not in base parser"""
    # Add FLUX-specific arguments that the framework passes but aren't in base parser
    parser.add_argument("--text_encoder", type=str, default=None, help="text encoder (CLIP-L) checkpoint path")
    parser.add_argument("--text_encoder2", type=str, default=None, help="text encoder (T5-XXL) checkpoint path")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT")
    parser.add_argument("--fp8_t5", action="store_true", help="use fp8 for T5 text encoder")

    # Input Perturbation Noise arguments (optional regularization)
    parser.add_argument("--ip_noise_gamma", type=float, default=None, help="input perturbation noise gamma (e.g. 0.1)")
    parser.add_argument("--ip_noise_gamma_random_strength", action="store_true", help="use random strength between 0~ip_noise_gamma")
    return parser


def main():
    parser = setup_parser_common()
    parser = flux_dev_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    # Set FLUX dev specific defaults
    args.dit_dtype = None  # set from mixed_precision
    if args.vae_dtype is None:
        args.vae_dtype = "float32" if args.mixed_precision == "fp16" else "bfloat16"

    # Set FLUX dev architecture for cache file lookup
    from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_FLUX_DEV
    args.architecture = ARCHITECTURE_FLUX_DEV  # "fd" - matches cache file naming

    trainer = FluxDevNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()