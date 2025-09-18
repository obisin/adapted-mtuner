#!/usr/bin/env python3
"""
FLUX Dev Latents Caching Script - Musubi Tuner Integration

Cache latents for FLUX dev training using sd-scripts utilities
with musubi-tuner's infrastructure.
"""

import argparse
import sys
import os

# Add sd-scripts path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../sd-scripts-sd3'))

from library import flux_utils, strategy_flux
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.cache_latents import encode_datasets, setup_parser_common
from musubi_tuner.dataset.image_video_dataset import save_latent_cache_common, ARCHITECTURE_FLUX_DEV, ARCHITECTURE_FLUX_DEV_FULL
from musubi_tuner.utils.model_utils import str_to_dtype, dtype_to_str

import torch
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def save_latent_cache_flux_dev(item_info, latent):
    """FLUX dev architecture - saves 3D latents (C, H, W)"""
    assert latent.dim() == 3, "latent should be 3D tensor (channel, height, width)"
    C, H, W = latent.shape
    F = 1  # Single frame for images
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}
    save_latent_cache_common(item_info, sd, ARCHITECTURE_FLUX_DEV_FULL)


def encode_and_save_batch(ae, batch):
    """Encode batch to latents and save cache for FLUX"""
    # Stack batch into tensor (B,H,W,C) in RGB order
    contents = []
    for item in batch:
        # item.content: target image (H, W, C) as np.ndarray
        contents.append(torch.from_numpy(item.content))

    contents = torch.stack(contents, dim=0)  # B, H, W, C
    contents = contents.permute(0, 3, 1, 2)  # B, H, W, C -> B, C, H, W
    contents = contents.float() / 127.5 - 1.0  # normalize to [-1, 1]

    # Encode with FLUX VAE
    with torch.no_grad():
        latents = ae.encode(contents.to(ae.device, dtype=ae.dtype))  # B, C, H, W

    # Save cache for each item in the batch
    for b, item in enumerate(batch):
        latent = latents[b]  # C, H, W
        save_latent_cache_flux_dev(item, latent.cpu())


def main():
    parser = setup_parser_common()
    # Add FLUX-specific arguments not in the common parser
    parser.add_argument("--disable_mmap_load_safetensors", action="store_true", help="disable mmap for loading safetensors")

    args = parser.parse_args()

    logger.info(f"Load dataset config from {args.dataset_config}")

    # Generate dataset blueprint
    sanitizer = ConfigSanitizer()
    blueprint_generator = BlueprintGenerator(sanitizer)
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_FLUX_DEV)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    logger.info(f"Loading AE model from {args.vae}")
    weight_dtype = str_to_dtype("float32")
    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    ae = flux_utils.load_ae(args.vae, weight_dtype, device, disable_mmap=args.disable_mmap_load_safetensors)

    logger.info("Encoding dataset [0]")
    encode_datasets(datasets, lambda batch: encode_and_save_batch(ae, batch), args)

    logger.info("Latent caching completed")


if __name__ == "__main__":
    main()