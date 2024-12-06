#! /bin/bash

data_root=../data
data_name="zx"

data_dir="${data_root}/${data_name}"

data=${data_dir}/${data_name}.example.pauses

for seed in 0 1 3; do
prob=${data_dir}/probs/seed${seed}/${data_name}.example.prob

  for threshold in 0.0 0.1 0.5 0.9; do
    out=${data_dir}/filtered/seed${seed}/${data_name}.filter_${threshold}.txt
    mkdir -p "${data_dir}/filtered/seed${seed}/"
    echo ${out}

    python scripts/analysis/filter_according_marg_prob.py \
            ${data} \
            ${prob} \
            ${threshold} \
            ${out} \
  
  done

done
