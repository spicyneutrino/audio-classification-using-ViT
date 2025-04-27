import os
from datasets import load_dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dataset_name = "danavery/urbansound8K"

# Verify HF_HOME is set
hf_home = os.environ.get("HF_HOME")
if not hf_home:
    logger.error("HF_HOME environment variable is not set!")
    exit(1)
logger.info(f"Using HF_HOME: {hf_home}")

# Construct the expected cache path on HF_HOME
cache_dir = os.environ.get("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))
logger.info(f"Target cache directory for datasets: {cache_dir}")

logger.info(f"Attempting to download and cache dataset: {dataset_name}")

try:
    ds = load_dataset(
        dataset_name,
        cache_dir=cache_dir,
        # download_mode="force_redownload", # Optional
        # trust_remote_code=True, # Optional
    )

    # --- Log success messages and info only AFTER load_dataset succeeds ---
    logger.info(
        f"Successfully loaded dataset '{dataset_name}'. Cache should be populated."
    )
    # Use single quotes inside f-string expression for dictionary keys
    logger.info(f"Dataset features: {ds['train'].features if 'train' in ds else 'N/A'}")
    logger.info(
        f"Number of rows (train): {len(ds['train']) if 'train' in ds else 'N/A'}"
    )

except Exception as e:
    # This block catches errors FROM load_dataset
    logger.error(
        f"Failed to download/load dataset '{dataset_name}': {e}", exc_info=True
    )
    exit(1)

# --- This message indicates the whole script finished without exiting in the except block ---
logger.info("Download script finished successfully.")
