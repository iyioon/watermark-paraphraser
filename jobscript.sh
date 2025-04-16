#!/bin/bash
#SBATCH --job-name=llm_run            # A descriptive job name
#SBATCH --output=logs/llm_run_%j.out   # Output log file (create a logs/ directory if needed)
#SBATCH --error=logs/llm_run_%j.err    # Error log file
#SBATCH --ntasks=1                   # Number of tasks (usually one for a single script)
#SBATCH --cpus-per-task=4            # Adjust based on your program's multi-threading needs
#SBATCH --time=02:00:00              # Estimated wall time (HH:MM:SS)
#SBATCH --partition=<your_partition> # Replace with the appropriate partition or queue name

# Load the required modules or environment
module load python/3.8               # Modify based on the available module versions
# If you use a virtual environment, activate it here:
# source /path/to/your/venv/bin/activate

# Execute your Python script
python demo/generate.py data/in/sample.txt --model microsoft/phi-2 --key 42 --output data/out/sample_paraphrased.text --verbose
