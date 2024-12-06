#! /bin/bash

# w_punc Baseline to predict prob on partial_label_data
data_root=../data

data_dir="${data_root}/zx"

encoder="bert"
mode="predict"

partial_label_data=${data_dir}/zx.example.pauses

for seed in 0 1 3; do
device=$seed

mkdir -p "${data_dir}/probs/seed${seed}/"
method="bi.segment.crf-cws.${encoder}.epc=5.seed${seed}"
base_model_path=../exp-bitag/ctb5/${method}/model.wxb

pred=${data_dir}/probs/seed${seed}/${data_name}.example.prob

CUDA_VISIBLE_DEVICES=$device python -u -m supar.cmds.crf_cws $mode \
            --device $device \
            --path $base_model_path \
            --data ${partial_label_data} \
            --compute_marg_probs \
            --pred ${pred_out} \

done
