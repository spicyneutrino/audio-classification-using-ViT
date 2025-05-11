#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --partition=ptolemy
#SBATCH --gres=gpu:a100_1g.10gb:1
#SBATCH --account=research-cse
#SBATCH --nodes=1                   # Number of nodes to use
#SBATCH --ntasks=1                  # Total number of tasks (processes)
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G                   # Request appropriate memory (e.g., 16G, 32G, 64G)
#SBATCH --time=12:00:00             # Maximum runtime (HH:MM:SS)

# --- Environment Setup ---
NUM_EPOCHS=500
NUM_WORKERS=$SLURM_CPUS_PER_TASK
BATCH_SIZE=64
HEAD_LR=8e-5
ENCODER_LR=1e-5
USE_TIME_AUGMENT=True

echo "Job started on $(hostname) at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Number of CPUs allocated: $SLURM_CPUS_PER_TASK" # Verify CPU allocation
echo "Memory allocated: $SLURM_MEM"
echo "Batch size: $BATCH_SIZE"
echo "This job has $NUM_WORKERS worker and $NUM_EPOCHS epochs assigned."
echo "HEAD_LR: $HEAD_LR | ENCODER_LR: $ENCODER_LR | USE_TIME_AUGMENT: $USE_TIME_AUGMENT"

# Define the absolute path to your project directory
export PROJECT_ROOT="/scratch/ptolemy/users/kg1623/projects/deep-learning/audio-classification-using-ViT"
echo "Project Root: $PROJECT_ROOT"

# PYTORCH CACHING
export TORCH_HOME="/scratch/ptolemy/users/kg1623/.cache/torch"
mkdir -p $TORCH_HOME/hub/checkpoints
echo "Setting Torch cache to: $TORCH_HOME"

# HUGGINGFACE CACHING
# Define and ensure the SHARED cache directory exists
export HF_HOME="/scratch/ptolemy/users/kg1623/.cache/huggingface"
mkdir -p $HF_HOME
echo "Setting Hugging Face cache to: $HF_HOME"

# --- Crucial: Change to the project directory ---
cd $PROJECT_ROOT || { echo "Failed to cd into $PROJECT_ROOT"; exit 1; }
echo "Current Directory: $(pwd)"

# Load necessary system modules
echo "Loading modules..."
module load python
module load cuda

# --- Setup Virtual Environment ---
VENV_DIR="venv"
if [ ! -d "$VENV_DIR/bin" ]; then
  echo "Creating virtual environment '$VENV_DIR'..."
  python -m venv $VENV_DIR
  echo "Activating virtual environment..."
  source $VENV_DIR/bin/activate
  echo "Installing dependencies from requirements.txt..."
  pip install --upgrade pip
  pip install -r requirements.txt
else
  echo "Activating existing virtual environment '$VENV_DIR'..."
  source $VENV_DIR/bin/activate
  # Optional: Uncomment below to update packages every time
  # echo "Updating dependencies from requirements.txt..."
  # pip install --upgrade pip
  # pip install -r requirements.txt -q
fi

# Set PYTHONPATH to include the project root (helps find 'modules')
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

# --- Run the Training Script ---
echo "Running train.py script..."

python train.py --num_epochs $NUM_EPOCHS --num_workers $NUM_WORKERS  --batch_size $BATCH_SIZE --head_lr $HEAD_LR --encoder_lr $ENCODER_LR --use_time_augment $USE_TIME_AUGMENT

EXIT_CODE=$? # Capture exit code
echo "Python script finished with exit code $EXIT_CODE at $(date)"

exit $EXIT_CODE
