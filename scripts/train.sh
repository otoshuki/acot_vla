export DEBUG_MODE=false
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
CONFIG_NAME=${1}
EXP_NAME=${2}

env | sort
/root/.local/bin/uv run python scripts/train.py $CONFIG_NAME --exp-name=$EXP_NAME