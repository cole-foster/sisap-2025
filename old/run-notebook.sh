#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time 12:00:00
#SBATCH --mem=64G
#SBATCH --job-name tunnel


## get tunneling info
XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)

## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "
    Copy/Paste this in your local terminal to ssh tunnel with remote
    -----------------------------------------------------------------
    ssh -N -L $ipnport:$ipnip:$ipnport $USER@ssh.ccv.brown.edu
    -----------------------------------------------------------------
    Then open a browser on your local machine to the following address
    ------------------------------------------------------------------
    localhost:$ipnport  (prefix w/ https:// if using password)
    ------------------------------------------------------------------
    "
## start an ipcluster instance and launch jupyter server
module purge
module load anaconda/2023.09-0-7nso27y
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
jupyter-notebook --no-browser --port=$ipnport --ip=$ipnip