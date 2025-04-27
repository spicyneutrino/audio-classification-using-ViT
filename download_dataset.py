import os
from datasets import load_dataset, get_dataset_config_names
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
logger.info(f"Using cache directory: {cache_dir}")

logger.info(f"Attempting to download and cache dataset: {dataset_name}")

try:
    logger.info(
        f"Successfully loaded dataset' {dataset_name}'. Cache could be populated."
    )
except Exception as e:
    logger.error(f"Failed to load dataset '{dataset_name}'. Error: {e}", exc_info=True)
    exit(1)

logger.info(f"Download Script completed successfully.")

ds = load_dataset(dataset_name, cache_dir=cache_dir)

logger.info(f"Dataset '{dataset_name}' loaded successfully.")
logger.info(f"Dataset features: {ds["train"].features if "train" in ds else 'N/A'} ")
logger.info(
    f"Number of rows (samples) in train: {len(ds["train"]) if "train" in ds else 'N/A'} "
)
