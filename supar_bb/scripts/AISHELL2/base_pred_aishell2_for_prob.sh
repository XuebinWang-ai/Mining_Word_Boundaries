#! /bin/bash

data_root=../data
data_name="AISHELL2"

data_dir="${data_root}/${data_name}"
mode="predict"

for seed in 0 ;  do
device=$seed

mkdir -p "${data_dir}/probs/seed${seed}/"

pred=${data_dir}/probs/seed${seed}/${data_name}.example.prob

base_model_path=../exp-bitag/ctb5_wo_punc/bi.wo_punc.ctb5.crf-cws.bert.epc=5.seed${seed}/model

# compute probs on AISHELL2 partial-label data (data with pauses)
CUDA_VISIBLE_DEVICES=$device python -u -m supar.cmds.crf_cws $mode \
        --device $device \
        --path $base_model_path \
        --data ${data_dir}/${data_name}.example.pauses \
        --compute_marg_probs \
        --pred ${pred} \

done
