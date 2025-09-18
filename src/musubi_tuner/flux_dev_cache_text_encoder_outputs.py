#!/usr/bin/env python3
"""
FLUX Dev Text Encoder Outputs Caching Script - Musubi Tuner Integration

Cache text encoder outputs for FLUX dev training using sd-scripts utilities
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
from musubi_tuner.dataset.image_video_dataset import save_text_encoder_output_cache_flux_kontext, ARCHITECTURE_FLUX_DEV
from musubi_tuner.utils.model_utils import str_to_dtype
import torch
from transformers import CLIPTokenizer, T5TokenizerFast

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def flux_dev_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, required=True, help="CLIP-L text encoder path")
    parser.add_argument("--text_encoder2", type=str, required=True, help="T5-XXL text encoder path")
    parser.add_argument("--fp8_t5", action="store_true", help="use fp8 for T5")
    parser.add_argument("--disable_mmap_load_safetensors", action="store_true", help="disable mmap for loading safetensors")
    return parser


def main():
    # Import setup_parser_common from cache_text_encoder_outputs
    from musubi_tuner.cache_text_encoder_outputs import setup_parser_common

    parser = setup_parser_common()
    parser = flux_dev_setup_parser(parser)
    args = parser.parse_args()

    logger.info(f"Load dataset config from {args.dataset_config}")

    # Generate dataset blueprint
    sanitizer = ConfigSanitizer()
    blueprint_generator = BlueprintGenerator(sanitizer)
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_FLUX_DEV)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    # Load text encoders and tokenizers
    logger.info(f"Loading CLIP-L from {args.text_encoder}")
    weight_dtype = str_to_dtype("bfloat16")
    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder1 = flux_utils.load_clip_l(args.text_encoder, weight_dtype, device, disable_mmap=args.disable_mmap_load_safetensors)
    tokenizer1 = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    logger.info(f"Loading T5-XXL from {args.text_encoder2}")
    t5_dtype = str_to_dtype("float8_e4m3fn") if args.fp8_t5 else str_to_dtype("bfloat16")
    text_encoder2 = flux_utils.load_t5xxl(args.text_encoder2, t5_dtype, device, disable_mmap=args.disable_mmap_load_safetensors)
    tokenizer2 = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl")

    text_encoders = [text_encoder1, text_encoder2]

    logger.info("Encoding text encoder outputs")

    # Process each dataset
    num_workers = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() - 1)
    for i, dataset in enumerate(datasets):
        logger.info(f"Encoding dataset [{i}]")
        batches = dataset.retrieve_text_encoder_output_cache_batches(num_workers)
        for batch in batches:
            # Skip if existing cache files exist
            if args.skip_existing:
                filtered_batch = [item for item in batch if not os.path.exists(item.text_encoder_output_cache_path)]
                if len(filtered_batch) == 0:
                    continue
                batch = filtered_batch

            # Process batch
            captions = [item.caption for item in batch]

            # Tokenize with CLIP-L
            clip_tokens = tokenizer1(
                captions,
                max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )["input_ids"]

            # Tokenize with T5-XXL
            t5_tokens = tokenizer2(
                captions,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )["input_ids"]

            # Encode with text encoders
            with torch.no_grad():
                clip_output = text_encoder1(clip_tokens.to(text_encoder1.device))["pooler_output"]
                t5_output = text_encoder2(t5_tokens.to(text_encoder2.device))["last_hidden_state"]

            # Save to cache for each item
            for j, item in enumerate(batch):
                # FLUX uses separate T5 and CLIP-L outputs
                t5_vec = t5_output[j:j+1].cpu()  # T5 output
                clip_l_pooler = clip_output[j:j+1].cpu()  # CLIP-L pooler output
                save_text_encoder_output_cache_flux_kontext(item, t5_vec, clip_l_pooler)

    logger.info("Text encoder caching completed")


if __name__ == "__main__":
    main()