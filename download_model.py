import torch
import torchvision
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verify is TORCH_HOME is set
torch_home = os.getenv('TORCH_HOME')

if not torch_home:
    logger.error("TORCH_HOME environment variable is not set.")
    exit(1)
logger.info(f"TORCH_HOME is set to: {torch_home}")

logger.info("Attempting to download/cache ViT_B_16 default weights...")
try:
    # Use the specific weights used in the train script
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    model = torchvision.models.vit_b_16(weights=weights)
    logger.info("Model loaded/downloaded successfully.")
except Exception as e:
    logger.error(f"Failed to download/load the model weights: {e}")
    exit(1)

logger.info("Model download script finished.")