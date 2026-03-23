#!/usr/bin/env bash

set -euo pipefail

# Make PYTHONPATH safe even when unset (bash `set -u`).
export PYTHONPATH="./:${PYTHONPATH:-}"

model="${1:-}"
num_gpus="${2:-}"
data_path="${3:-datasets/Barefoot_Dataset}"
output_root="${4:-output_cosine}"

if [[ -z "$model" || -z "$num_gpus" ]]; then
  echo "Usage: sh scripts/train.sh <model> <num_gpus> [data_path] [output_root]"
  echo "  - model: resnet34|resnet152|vgg19|densenet121|densenet161|resnet18|resnet50|resnet101|vgg16|..."
  echo "  - num_gpus: 0=CPU, 1=single GPU, >=2=DDP multi-GPU"
  echo "  - data_path: datasets/Barefoot_Dataset_2|_5|_200 (default: datasets/Barefoot_Dataset)"
  echo "  - output_root: output directory root (default: output_cosine); run folder: <N>p_<model>_YYYYMMDD_HHMMSS_<seed>_<lr>_<opt>_<epochs>_train (N=class count)"
  exit 2
fi

use_port=2681
data_set="Barefoot_Dataset"

# Auto-detect class count from train_cropped_augmented or train_cropped under data_path
data_path="${data_path%/}"
train_aug="${data_path}/train_cropped_augmented"
train_fb="${data_path}/train_cropped"
if [[ -d "$train_aug" ]]; then
  train_dir="$train_aug"
elif [[ -d "$train_fb" ]]; then
  train_dir="$train_fb"
else
  echo "ERROR: Cannot find train_cropped_augmented or train_cropped under: ${data_path}" >&2
  exit 1
fi
num_classes=0
shopt -s nullglob
for d in "${train_dir}"/*/; do
  [[ -d "$d" ]] || continue
  num_classes=$((num_classes + 1))
done
shopt -u nullglob
if [[ "$num_classes" -eq 0 ]]; then
  echo "ERROR: No class subdirectories in: ${train_dir}" >&2
  exit 1
fi


train_batch_size=64
test_batch_size=128

# Paper-aligned defaults (shared across backbones in the paper's settings)
seed=1028
opt=adam
lr=1e-4

warmup_epochs=5
decay_epochs=3
decay_rate=0.2
sched=step
epochs=20
input_size=224
dim=64

# Loss / module hyper-params
features_lr="$lr"
add_on_layers_lr=3e-3
prototype_vectors_lr=3e-3
activation_weight_lr=1e-6

use_ortho_loss=True
ortho_coe=1e-4
consis_coe=0.50
consis_thresh=0.10
num_prototypes_per_class=10

ft=train
# One folder per run: <N>p_<base_architecture>_YYYYMMDD_HHMMSS_<seed>_<lr>_<opt>_<epochs>_train
date_part="$(date '+%Y%m%d')"
time_part="$(date '+%H%M%S')"
run_name="${num_classes}p_${model}_${date_part}_${time_part}_${seed}_${lr}_${opt}_${epochs}_${ft}"
output_dir="${output_root}/${run_name}"

common_args=(
  --seed="$seed"
  --output_dir="$output_dir"
  --data_set="$data_set"
  --data_path="$data_path"
  --train_batch_size="$train_batch_size"
  --test_batch_size="$test_batch_size"
  --base_architecture="$model"
  --input_size="$input_size"
  --num_prototypes_per_class="$num_prototypes_per_class"
  --prototype_activation_function=log
  --add_on_layers_type=regular
  --use_ortho_loss="$use_ortho_loss"
  --ortho_coe="$ortho_coe"
  --consis_coe="$consis_coe"
  --consis_thresh="$consis_thresh"
  --opt="$opt"
  --sched="$sched"
  --lr="$lr"
  --features_lr="$features_lr"
  --add_on_layers_lr="$add_on_layers_lr"
  --prototype_vectors_lr="$prototype_vectors_lr"
  --activation_weight_lr="$activation_weight_lr"
  --epochs="$epochs"
  --warmup_epochs="$warmup_epochs"
  --decay_epochs="$decay_epochs"
  --decay_rate="$decay_rate"
)

if [[ "$num_gpus" -eq 0 ]]; then
  echo ">>> num_gpus=0: CPU single-process"
  python main.py --device cpu "${common_args[@]}"
elif [[ "$num_gpus" -eq 1 ]]; then
  echo ">>> num_gpus=1: single-GPU single-process"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  python main.py --device cuda "${common_args[@]}"
else
  echo ">>> num_gpus=${num_gpus}: multi-GPU DDP (paper logic)"
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # default to 0..num_gpus-1
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((num_gpus-1)))"
    export CUDA_VISIBLE_DEVICES
  fi
  if command -v torchrun >/dev/null 2>&1; then
    torchrun --nproc_per_node="$num_gpus" --master_port="$use_port" main.py "${common_args[@]}"
  else
    python -m torch.distributed.launch --nproc_per_node="$num_gpus" --master_port="$use_port" --use_env main.py "${common_args[@]}"
  fi
fi