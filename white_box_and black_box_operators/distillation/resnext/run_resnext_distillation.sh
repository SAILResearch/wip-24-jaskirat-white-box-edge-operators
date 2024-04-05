#!/bin/bash
set -x

function main {
  init_params "$@"
  run_distillation

}

# init params
function init_params {
  output_model='saved_results'

  for var in "$@"
  do
    case $var in
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      --teacher=*)
          teacher=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo $var |cut -f2 -d=)
      ;;
          
    esac
  done

}

# run_tuning
function run_distillation {
    python -m torch.distributed.launch --master_addr=localhost --nproc_per_node=8 --master_port=8041 ./resnext_distillation.py \
           --topology=${topology} \
           --teacher=${teacher} \
           --dataset=${dataset_location} \
           --distillation \
           --pretrained \
           --output_model=${output_model}
}

main "$@"
